"""Mem0 memory backend for ppl.

``PplMem0`` subclasses mem0's ``MemoryBase`` so it drops into any code that
expects a Mem0-compatible memory store, and adds Mem0-style ``add``/``search``
convenience methods with the same signatures and result shapes as
``mem0.Memory``.

Memories are stored as notes on a ppl contact's timeline (the same storage as
the LangGraph adapter, with the same ``[ppl-memory ...]`` markers, so the two
adapters interoperate). The Mem0 ``user_id`` selects the ppl contact (id or
name); ``default_contact`` is used when no ``user_id`` is given.

Note: ``add(..., infer=True)`` is accepted but inference is a no-op here.
Mem0's LLM fact extraction needs an LLM configured; this adapter stores the
message text as the memory verbatim, which is what ppl's ``remember`` does.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from mem0.memory.base import MemoryBase
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ppl-memory[mem0] requires mem0ai. Install with: pip install ppl-memory[mem0]"
    ) from exc

from .client import PplClient, PplError

_MARKER_RE = re.compile(r'\[ppl-memory key="(.*?)" ns="(.*?)"\]')


def _marker(key: str, namespace: str) -> str:
    return f'[ppl-memory key="{key}" ns="{namespace}"]'


def _strip_marker(body: str) -> str:
    return _MARKER_RE.sub("", body).strip()


def _messages_to_text(messages: Any) -> str:
    """Accept a string or a list of {role, content} dicts; return plain text."""
    if isinstance(messages, str):
        return messages.strip()
    if isinstance(messages, list):
        parts = []
        for m in messages:
            if isinstance(m, dict):
                content = m.get("content", "")
                role = m.get("role", "")
                parts.append(f"{role}: {content}".strip(": ").strip())
            else:
                parts.append(str(m))
        return "\n".join(p for p in parts if p).strip()
    return str(messages).strip()


def _memory_dict(note_id: Any, body: str, note: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    note = note or {}
    return {
        "id": str(note_id),
        "memory": _strip_marker(body),
        "hash": note.get("hash"),
        "created_at": note.get("created_at"),
        "updated_at": note.get("updated_at"),
        "metadata": {"ppl_note_id": note_id},
    }


class PplMem0(MemoryBase):
    """A Mem0-compatible memory store backed by ppl (https://withppl.com)."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        base_url: Optional[str] = None,
        default_contact: Optional[int | str] = None,
    ) -> None:
        self.client = PplClient(api_token=api_token, base_url=base_url or "https://withppl.com")
        self.default_contact = default_contact
        self._contact_cache: Dict[str, int] = {}

    # -- contact resolution ---------------------------------------------

    def _namespace_for(self, user_id: Optional[str]) -> str:
        ref = user_id or self.default_contact
        return f"mem0.{ref}" if ref is not None else "mem0"

    def _resolve_contact(self, user_id: Optional[str] = None) -> int:
        ref = user_id if user_id is not None else self.default_contact
        if ref is None:
            raise PplError(
                "No contact for this memory. Pass user_id= (a ppl contact id or name) "
                "or set default_contact= on PplMem0."
            )
        cache_key = str(ref)
        if cache_key in self._contact_cache:
            return self._contact_cache[cache_key]
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            contact_id = int(ref)
        else:
            contact = self.client.find_contact(str(ref))
            if not contact or "id" not in contact:
                raise PplError(f"No ppl contact found matching '{ref}'.")
            contact_id = int(contact["id"])
        self._contact_cache[cache_key] = contact_id
        return contact_id

    def _find_note(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Find a note by ppl note id, or by mem0 memory key in markers."""
        if str(memory_id).isdigit():
            try:
                return self.client.get_note(int(memory_id))
            except PplError:
                return None
        return None

    def _find_note_by_key(self, key: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        want = f'[ppl-memory key="{key}"'
        contact_id = self._resolve_contact(user_id)
        for note in self.client.contact_notes(contact_id, limit=200):
            if want in str(note.get("body", "")):
                return note
        return None

    # -- MemoryBase abstract methods ------------------------------------

    def get(self, memory_id):
        note = self._find_note(str(memory_id))
        if not note:
            return None
        return _memory_dict(note["id"], str(note.get("body", "")), note)

    def get_all(self, *, user_id: Optional[str] = None, limit: int = 100, **kwargs):
        contact_id = self._resolve_contact(user_id)
        notes = self.client.contact_notes(contact_id, limit=limit)
        return {
            "results": [
                _memory_dict(n["id"], str(n.get("body", "")), n) for n in notes
            ]
        }

    def update(self, memory_id, data):
        note = self._find_note(str(memory_id))
        if not note:
            raise PplError(f"No memory found with id '{memory_id}'.")
        body = str(note.get("body", ""))
        m = _MARKER_RE.search(body)
        marker = m.group(0) if m else ""
        # ppl notes have no in-place update in the client; delete + re-add.
        self.client.delete_note(int(note["id"]))
        new_text = _messages_to_text(data)
        result = self.client.remember(int(note.get("contact_id") or self._resolve_contact()), f"{new_text}\n\n{marker}".strip())
        new_id = result.get("note", {}).get("id") or result.get("id")
        return {"id": str(new_id), "memory": new_text, "event": "UPDATE"}

    def delete(self, memory_id):
        note = self._find_note(str(memory_id))
        if note:
            self.client.delete_note(int(note["id"]))
        return {"message": "Memory deleted successfully."}

    def history(self, memory_id):
        # ppl notes are append-only; there is no per-note revision history.
        # Return the current state as the single known event.
        note = self._find_note(str(memory_id))
        if not note:
            return []
        body = str(note.get("body", ""))
        return [
            {
                "memory_id": str(note["id"]),
                "old_memory": None,
                "new_memory": _strip_marker(body),
                "event": "ADD",
                "created_at": note.get("created_at"),
                "updated_at": note.get("updated_at"),
            }
        ]

    # -- Mem0-style add / search -----------------------------------------

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = False,
        **kwargs,
    ):
        """Store messages as a memory. Returns {"results": [{id, memory, event}]}.

        ``infer`` is accepted for signature compatibility but is a no-op:
        message text is stored verbatim as the memory.
        """
        _ = (agent_id, run_id, metadata, infer, kwargs)  # reserved for future use
        contact_id = self._resolve_contact(user_id)
        text = _messages_to_text(messages)
        if not text:
            raise PplError("add() requires non-empty messages.")
        key = uuid.uuid4().hex[:12]
        body = f"{text}\n\n{_marker(key, self._namespace_for(user_id))}"
        result = self.client.remember(contact_id, body)
        note_id = result.get("note", {}).get("id") or result.get("id") or key
        return {"results": [{"id": str(note_id), "memory": text, "event": "ADD"}]}

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Search memories with ppl's ranked retrieval. Returns {"results": [...]}."""
        _ = (filters, kwargs)  # reserved for future use
        results = self.client.ask(query, limit=top_k)
        items: List[Dict[str, Any]] = []
        for r in results:
            text = r.get("title") or r.get("text") or r.get("body") or json.dumps(r)
            items.append(
                {
                    "id": str(r.get("id", "")),
                    "memory": text,
                    "score": r.get("score"),
                    "contact": r.get("contact") or r.get("contact_name"),
                    "user_id": user_id,
                }
            )
        return {"results": items}

    def delete_all(self, *, user_id: Optional[str] = None, **kwargs):
        """Delete all memories for a user. ppl never wipes CRM history silently,
        so this is a deliberate no-op returning an empty result."""
        _ = (user_id, kwargs)
        return {"message": "delete_all is a no-op on ppl; delete memories individually."}

    def reset(self):
        """No-op: ppl is the system of record and is never wiped by an adapter."""
        return {"message": "reset is a no-op on ppl."}
