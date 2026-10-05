"""OpenAI Agents SDK session backend for ppl.

``PplSession`` implements the ``agents`` ``Session`` protocol, so it drops
straight into ``Runner.run``:

.. code-block:: python

    from agents import Agent, Runner
    from ppl_memory.openai_agents import PplSession

    agent = Agent(name="assistant", instructions="You are helpful.")
    session = PplSession(session_id="chat-123", default_contact="Jane Smith")
    result = await Runner.run(agent, "Remember that Jane likes tea.", session=session)
    # a later run with the same session_id replays the conversation

Mapping:

- ``add_items()`` stores items as a note on the contact's timeline
  (POST /api/notes), marked ``[ppl-agents session="<session_id>"]`` so
  sessions stay separate.
- ``get_items()`` replays the session's notes as input items, in
  chronological order.
- ``pop_item()`` removes the most recent item of this session (delete +
  re-add of the note, since ppl notes are append-only).
- ``clear_session()`` is a deliberate no-op: ppl holds real CRM history and
  is never wiped by a session reset.

The contact the session is anchored to comes from ``default_contact`` (or
the PPL_DEFAULT_CONTACT env var), or ``contact_id``/``contact`` passed to
the constructor.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

try:
    from agents.memory.session import Session
    from agents.items import TResponseInputItem
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ppl-memory[openai-agents] requires openai-agents. "
        "Install with: pip install ppl-memory[openai-agents]"
    ) from exc

from .client import PplClient, PplError

_MARKER_RE = re.compile(r'\[ppl-agents session="([^"]+)"\]')


def _marker(session_id: str) -> str:
    return f'[ppl-agents session="{session_id}"]'


def _item_to_json(item: Any) -> str:
    if isinstance(item, dict):
        return json.dumps(item, ensure_ascii=False)
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        return json.dumps(to_dict(), ensure_ascii=False)
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        return json.dumps(model_dump(), ensure_ascii=False)
    return json.dumps({"content": str(item)}, ensure_ascii=False)


def _item_text(item: dict[str, Any]) -> str:
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                texts.append(str(part.get("text") or part.get("transcript") or ""))
            else:
                texts.append(str(part))
        return " ".join(t for t in texts if t)
    return str(content)


class PplSession(Session):
    """An OpenAI Agents SDK ``Session`` backed by ppl (https://withppl.com).

    Args:
        session_id: id for this conversation; notes for different sessions
            stay separate via a marker.
        api_token: ppl API token (or PPL_API_TOKEN env var).
        base_url: ppl base URL, defaults to https://withppl.com.
        default_contact: contact id or name the session is anchored to.
            Falls back to PPL_DEFAULT_CONTACT env var.
        contact_id / contact: explicit anchor, overrides default_contact.
    """

    session_id: str
    session_settings = None

    def __init__(
        self,
        session_id: str,
        api_token: str | None = None,
        base_url: str | None = None,
        default_contact: int | str | None = None,
        contact_id: int | str | None = None,
        contact: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.client = PplClient(
            api_token=api_token, base_url=base_url or "https://withppl.com"
        )
        self._contact_ref = (
            contact_id
            or contact
            or default_contact
            or os.environ.get("PPL_DEFAULT_CONTACT")
        )
        self._contact_cache: dict[str, int] = {}

    # -- helpers --------------------------------------------------------

    def _resolve_contact(self) -> int:
        ref = self._contact_ref
        if ref is None:
            raise PplError(
                "No contact to anchor this session to. Pass default_contact= "
                "(or PPL_DEFAULT_CONTACT), contact=, or contact_id=."
            )
        cache_key = str(ref)
        if cache_key in self._contact_cache:
            return self._contact_cache[cache_key]
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            contact_id = int(ref)
        else:
            found = self.client.find_contact(str(ref))
            if not found or "id" not in found:
                raise PplError(f"No ppl contact found matching '{ref}'.")
            contact_id = int(found["id"])
        self._contact_cache[cache_key] = contact_id
        return contact_id

    def _session_notes(self) -> list[dict[str, Any]]:
        """All notes for this session, oldest first."""
        contact_id = self._resolve_contact()
        notes = self.client.contact_notes(contact_id, limit=500)
        want = _marker(self.session_id)
        return [n for n in notes if want in str(n.get("body", ""))]

    @staticmethod
    def _note_items(note: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for line in str(note.get("body", "")).splitlines():
            line = line.strip()
            if not line or line.startswith("[ppl-agents"):
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                parsed = {"role": "user", "content": line}
            if isinstance(parsed, dict):
                items.append(parsed)
        return items

    def _note_body(self, items: list[dict[str, Any]]) -> str:
        lines = [_marker(self.session_id)]
        lines.extend(_item_to_json(i) for i in items)
        return "\n".join(lines)

    # -- Session protocol -------------------------------------------------

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        notes = await asyncio.to_thread(self._session_notes)
        items: list[dict[str, Any]] = []
        for note in notes:
            items.extend(self._note_items(note))
        if limit is not None:
            items = items[-limit:]
        return items  # type: ignore[return-value]

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        if not items:
            return
        contact_id = await asyncio.to_thread(self._resolve_contact)
        body = self._note_body([i if isinstance(i, dict) else {"role": "user", "content": str(i)} for i in items])  # type: ignore[misc]
        await asyncio.to_thread(self.client.remember, contact_id, body)

    async def pop_item(self) -> TResponseInputItem | None:
        notes = await asyncio.to_thread(self._session_notes)
        if not notes:
            return None
        note = notes[-1]
        items = self._note_items(note)
        if not items:
            return None
        popped = items.pop()
        await asyncio.to_thread(self.client.delete_note, int(note["id"]))
        if items:
            contact_id = await asyncio.to_thread(self._resolve_contact)
            await asyncio.to_thread(self.client.remember, contact_id, self._note_body(items))
        return popped  # type: ignore[return-value]

    async def clear_session(self) -> None:
        # ppl holds the user's real CRM history; clearing a session must not
        # delete it. Replays stay readable in ppl's timeline.
        return None
