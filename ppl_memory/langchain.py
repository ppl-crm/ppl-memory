"""LangGraph store backend for ppl.

Maps the LangGraph BaseStore key-value interface onto ppl's memory layer:
memories are stored as notes on a contact's timeline.

Namespace convention: ("memories", <contact_ref>), where <contact_ref> is a
ppl contact id or the contact's name. A bare ("memories",) namespace uses the
default contact passed to the store (or PPL_DEFAULT_CONTACT).

Each stored value becomes one note. The fact text is the human-readable body;
a small machine marker at the end carries the key/namespace so get() and
delete() can find the exact note again. search() uses ppl's ranked retrieval
(POST /api/agent/ask), which understands natural language.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

try:
    from langgraph.store.base import (
        BaseStore,
        GetOp,
        Item,
        ListNamespacesOp,
        Op,
        PutOp,
        Result,
        SearchItem,
        SearchOp,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ppl-memory[langchain] requires langgraph. Install with: pip install ppl-memory[langchain]"
    ) from exc

from .client import PplClient, PplError

_MARKER_RE = re.compile(r"\[ppl-memory key=\"(.*?)\" ns=\"(.*?)\"\]")
DEFAULT_NAMESPACE = ("memories",)


def _marker(key: str, namespace: tuple[str, ...]) -> str:
    return f'[ppl-memory key="{key}" ns="{".".join(namespace)}"]'


def _fact_text(value: dict[str, Any]) -> str:
    for field in ("fact", "text", "content", "memory"):
        if isinstance(value.get(field), str) and value[field].strip():
            return value[field].strip()
    return json.dumps(value, ensure_ascii=False)


class PplStore(BaseStore):
    """A LangGraph BaseStore backed by ppl (https://ppl.gift)."""

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
        default_contact: int | str | None = None,
    ) -> None:
        self.client = PplClient(api_token=api_token, base_url=base_url or "https://ppl.gift")
        self.default_contact = default_contact
        self._contact_cache: dict[str, int] = {}

    # -- contact resolution ---------------------------------------------

    def _resolve_contact(self, namespace: tuple[str, ...]) -> int:
        ref: int | str | None = namespace[1] if len(namespace) > 1 else self.default_contact
        if ref is None:
            raise PplError(
                "No contact for this namespace. Use namespace ('memories', <contact id or name>) "
                "or pass default_contact= to PplStore."
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

    # -- op implementations ----------------------------------------------

    def _do_put(self, op: PutOp) -> None:
        if op.value is None:
            self._do_delete(op.namespace, op.key)
            return
        contact_id = self._resolve_contact(op.namespace)
        body = _fact_text(op.value) + "\n\n" + _marker(op.key, op.namespace)
        self.client.remember(contact_id, body)

    def _find_note_id(self, namespace: tuple[str, ...], key: str) -> int | None:
        want = _marker(key, namespace)
        contact_id = self._resolve_contact(namespace)
        for note in self.client.contact_notes(contact_id, limit=200):
            if want in str(note.get("body", "")):
                return int(note["id"])
        return None

    def _do_get(self, op: GetOp) -> Item | None:
        note_id = self._find_note_id(op.namespace, op.key)
        if note_id is None:
            return None
        note = self.client.get_note(note_id)
        body = str(note.get("body", ""))
        fact = _MARKER_RE.sub("", body).strip()
        now = datetime.now(timezone.utc)
        created = note.get("created_at") or now.isoformat()
        return Item(
            value={"fact": fact, **({"note_id": note_id} if note_id else {})},
            key=op.key,
            namespace=op.namespace,
            created_at=created,
            updated_at=note.get("updated_at") or created,
        )

    def _do_delete(self, namespace: tuple[str, ...], key: str) -> None:
        note_id = self._find_note_id(namespace, key)
        if note_id is not None:
            self.client.delete_note(note_id)

    def _do_search(self, op: SearchOp) -> list[SearchItem]:
        query = op.query or ""
        results = self.client.ask(query, limit=op.limit or 10)
        if op.offset:
            results = results[op.offset:]
        items: list[SearchItem] = []
        now = datetime.now(timezone.utc)
        for r in results:
            text = r.get("title") or r.get("text") or r.get("body") or json.dumps(r)
            items.append(
                SearchItem(
                    namespace=op.namespace_prefix,
                    key=str(r.get("id", "")),
                    value={
                        "fact": text,
                        "score": r.get("score"),
                        "contact": r.get("contact") or r.get("contact_name"),
                    },
                    created_at=now,
                    updated_at=now,
                    score=r.get("score"),
                )
            )
        return items

    def _do_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        namespaces: set[tuple[str, ...]] = set()
        payload = self.client._request("GET", "/api/notes", params={"limit": 100})
        notes = self.client._unwrap(payload) or []
        for note in notes if isinstance(notes, list) else []:
            m = _MARKER_RE.search(str(note.get("body", "")))
            if m:
                namespaces.add(tuple(m.group(2).split(".")))
        return sorted(namespaces)[: (op.limit or 100)]

    # -- BaseStore interface ----------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, PutOp):
                self._do_put(op)
                results.append(None)
            elif isinstance(op, GetOp):
                results.append(self._do_get(op))
            elif isinstance(op, SearchOp):
                results.append(self._do_search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._do_list_namespaces(op))
            else:
                raise PplError(f"Unsupported store operation: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))
