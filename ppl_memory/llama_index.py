"""LlamaIndex memory block backend for ppl.

``PplMemoryBlock`` is a LlamaIndex ``BaseMemoryBlock[str]`` backed by ppl
(https://withppl.com). Drop it into a ``Memory`` alongside LlamaIndex's other
blocks:

.. code-block:: python

    from llama_index.core.memory import Memory
    from ppl_memory.llama_index import PplMemoryBlock

    memory = Memory(memory_blocks=[
        PplMemoryBlock(default_contact="Jane Smith"),
    ])
    agent = ReActAgent.from_tools(tools, llm=llm, memory=memory)

Mapping:

- ``_aput()`` writes messages as a note on the contact's timeline
  (POST /api/notes), marked ``[ppl-llamaindex]`` so it round-trips.
- ``_aget()`` runs ppl's ranked retrieval (POST /api/agent/ask) over the
  latest user message and returns the hits as text for the memory template.
- ``atruncate()`` drops the block content; the ppl notes themselves are kept.

The contact a message is stored under comes from ``default_contact`` (or the
PPL_DEFAULT_CONTACT env var), or per-message ``additional_kwargs``
``contact``/``contact_id`` keys.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, List, Optional

try:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.core.memory import BaseMemoryBlock
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ppl-memory[llamaindex] requires llama-index-core. "
        "Install with: pip install ppl-memory[llamaindex]"
    ) from exc

from pydantic import PrivateAttr

from .client import PplClient, PplError

MARKER = "[ppl-llamaindex]"


def _message_text(message: ChatMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            parts.append(text if isinstance(text, str) else str(block))
        return " ".join(p for p in parts if p).strip()
    return str(content).strip()


def _last_user_text(messages: Optional[List[ChatMessage]]) -> str:
    for message in reversed(messages or []):
        if message.role == MessageRole.USER:
            text = _message_text(message)
            if text:
                return text
    return ""


class PplMemoryBlock(BaseMemoryBlock[str]):
    """A LlamaIndex memory block backed by ppl (https://withppl.com).

    Args:
        api_token: ppl API token (or PPL_API_TOKEN env var).
        base_url: ppl base URL, defaults to https://withppl.com.
        default_contact: contact id or name that _aput writes to when a
            message carries no contact reference. Falls back to
            PPL_DEFAULT_CONTACT env var.
        limit: max ranked-retrieval hits returned by _aget.
    """

    name: str = "ppl"
    description: Optional[str] = (
        "Long-term memory about people, stored in ppl (withppl.com)."
    )
    default_contact: Optional[Any] = None
    limit: int = 10

    _client: PplClient = PrivateAttr()
    _contact_cache: dict = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._client = PplClient(
            api_token=api_token, base_url=base_url or "https://withppl.com"
        )
        if self.default_contact is None:
            self.default_contact = os.environ.get("PPL_DEFAULT_CONTACT")

    # -- contact resolution ---------------------------------------------

    def _resolve_contact(self, message: ChatMessage | None = None) -> int:
        ref = None
        if message is not None:
            extra = getattr(message, "additional_kwargs", None) or {}
            ref = extra.get("contact_id") or extra.get("contact")
        if ref is None:
            ref = self.default_contact
        if ref is None:
            raise PplError(
                "No contact to store this memory under. Pass default_contact= "
                "(or PPL_DEFAULT_CONTACT), or set contact/contact_id in the "
                "message additional_kwargs."
            )
        cache_key = str(ref)
        if cache_key in self._contact_cache:
            return self._contact_cache[cache_key]
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            contact_id = int(ref)
        else:
            contact = self._client.find_contact(str(ref))
            if not contact or "id" not in contact:
                raise PplError(f"No ppl contact found matching '{ref}'.")
            contact_id = int(contact["id"])
        self._contact_cache[cache_key] = contact_id
        return contact_id

    # -- BaseMemoryBlock interface ---------------------------------------

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> str:
        query = _last_user_text(messages)
        if not query:
            return ""
        results = await asyncio.to_thread(self._client.ask, query, self.limit)
        lines = []
        for r in results:
            text = r.get("title") or r.get("text") or r.get("body") or json.dumps(r)
            contact = r.get("contact") or r.get("contact_name")
            lines.append(f"- {text}" + (f" (about {contact})" if contact else ""))
        return "\n".join(lines)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        lines = []
        contact_ref: ChatMessage | None = None
        for message in messages:
            text = _message_text(message)
            if not text:
                continue
            extra = getattr(message, "additional_kwargs", None) or {}
            if extra.get("contact_id") or extra.get("contact"):
                contact_ref = message
            lines.append(f"{message.role.value}: {text}")
        if not lines:
            return
        contact_id = await asyncio.to_thread(self._resolve_contact, contact_ref)
        body = "\n".join(lines) + f"\n\n{MARKER}"
        await asyncio.to_thread(self._client.remember, contact_id, body)

    async def atruncate(self, content: str, tokens_to_truncate: int) -> Optional[str]:
        # ppl notes are kept; only the injected block content is dropped.
        return None
