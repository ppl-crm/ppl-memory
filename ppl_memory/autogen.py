"""AutoGen 0.4 memory backend for ppl.

Implements the autogen_core Memory protocol so agents get ppl as long-term
memory: add() stores facts on a contact's timeline, query() runs ppl's ranked
retrieval, and update_context() injects relevant memories into the model
context before the model sees it.

Mapping:
  add()            -> ppl note on the contact's timeline (POST /api/notes)
  query()          -> ppl ranked retrieval (POST /api/agent/ask)
  update_context() -> last user message becomes the query; top memories are
                      added to the context as a system message
  clear()/close()  -> no-ops that keep the user's CRM data safe
                     (clearing ppl would delete real CRM history)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

try:
    from autogen_core import CancellationToken, Component
    from autogen_core.memory import (
        Memory,
        MemoryContent,
        MemoryQueryResult,
        UpdateContextResult,
    )
    from autogen_core.model_context import ChatCompletionContext
    from autogen_core.models import SystemMessage
    from pydantic import BaseModel
    from typing_extensions import Self
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ppl-memory[autogen] requires autogen-core. Install with: pip install ppl-memory[autogen]"
    ) from exc

from .client import PplClient, PplError

RESET_MARKER = "[ppl-autogen]"


class PplMemoryConfig(BaseModel):
    name: str | None = None
    base_url: str = "https://withppl.com"
    default_contact: int | str | None = None


class PplMemory(Memory, Component[PplMemoryConfig]):
    """AutoGen Memory backed by ppl (https://withppl.com).

    Args:
        api_token: ppl API token (or PPL_API_TOKEN env var). Not stored in the
            serializable component config.
        default_contact: contact id or name that add() writes to when the
            content metadata carries no contact reference. Falls back to
            PPL_DEFAULT_CONTACT env var.
        name: component name.
    """

    component_type = "memory"
    component_provider_override = "ppl_memory.autogen.PplMemory"
    component_config_schema = PplMemoryConfig

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
        default_contact: int | str | None = None,
        name: str | None = None,
    ) -> None:
        self._name = name or "ppl_memory"
        self.client = PplClient(api_token=api_token, base_url=base_url or "https://withppl.com")
        self.default_contact = default_contact
        self._contact_cache: dict[str, int] = {}

    # -- helpers ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _content_text(content: MemoryContent) -> str:
        raw = content.content
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            return json.dumps(raw, ensure_ascii=False)
        return str(raw)

    def _resolve_contact(self, metadata: dict[str, Any] | None) -> int:
        metadata = metadata or {}
        ref = metadata.get("contact_id") or metadata.get("contact") or self.default_contact
        if ref is None:
            raise PplError(
                "No contact to store this memory under. Pass contact_id or contact in "
                "MemoryContent metadata, or set default_contact=/PPL_DEFAULT_CONTACT."
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

    @staticmethod
    def _to_memory_content(result: dict[str, Any]) -> MemoryContent:
        text = result.get("title") or result.get("text") or result.get("body") or json.dumps(result)
        return MemoryContent(
            content=str(text),
            mime_type="text/plain",
            metadata={
                "contact": result.get("contact") or result.get("contact_name"),
                "score": result.get("score"),
                "source": "ppl",
            },
        )

    # -- Memory protocol --------------------------------------------------

    async def add(
        self, content: MemoryContent, cancellation_token: CancellationToken | None = None
    ) -> None:
        contact_id = await asyncio.to_thread(self._resolve_contact, content.metadata)
        text = self._content_text(content)
        await asyncio.to_thread(self.client.remember, contact_id, f"{text}\n\n{RESET_MARKER}")

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        query_text = query if isinstance(query, str) else self._content_text(query)
        limit = int(kwargs.get("limit", 10))
        results = await asyncio.to_thread(self.client.ask, query_text, limit)
        return MemoryQueryResult(results=[self._to_memory_content(r) for r in results])

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        messages = await model_context.get_messages()
        query_text = ""
        for message in reversed(messages):
            text = getattr(message, "content", "")
            if isinstance(text, str) and text.strip():
                query_text = text.strip()
                break
        if not query_text:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        query_result = await self.query(query_text)
        if not query_result.results:
            return UpdateContextResult(memories=query_result)
        lines = [
            f"{i}. {self._content_text(m)}"
            for i, m in enumerate(query_result.results, 1)
        ]
        await model_context.add_message(
            SystemMessage(content="Relevant memories from ppl:\n" + "\n".join(lines))
        )
        return UpdateContextResult(memories=query_result)

    async def clear(self) -> None:
        # ppl holds the user's real CRM history; clear() must not wipe it.
        return None

    async def close(self) -> None:
        return None

    # -- Component serialization ------------------------------------------

    @classmethod
    def _from_config(cls, config: PplMemoryConfig) -> Self:
        return cls(
            base_url=config.base_url,
            default_contact=config.default_contact,
            name=config.name,
        )

    def _to_config(self) -> PplMemoryConfig:
        return PplMemoryConfig(
            name=self._name,
            base_url=self.client.base_url,
            default_contact=self.default_contact,
        )
