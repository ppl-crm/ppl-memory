"""CrewAI storage backend for ppl.

Implements CrewAI's unified-memory ``StorageBackend`` protocol
(``crewai.memory.storage.backend``), so a crew's memory can live in ppl
instead of the default local LanceDB store:

.. code-block:: python

    from crewai.memory import Memory
    from ppl_memory.crewai import PplCrewAIStorage

    memory = Memory(storage=PplCrewAIStorage(default_contact="Jane Smith"))
    crew = Crew(agents=[...], tasks=[...], memory=memory)

Mapping:

- ``save(records)`` writes each record as a note on a contact's timeline.
  The record id, scope, categories, and importance travel in a small
  machine-readable marker so records round-trip exactly.
- ``search(embedding, ...)`` scores candidate records with cosine similarity
  against a local embedding cache (``~/.ppl-memory/crewai_embeddings.json``).
  ppl holds the memory content; the cache holds only derived vectors and can
  be rebuilt by re-saving.
- ``delete/update/get_record/list_records`` operate on the ppl notes via
  their markers. ``delete`` only ever removes notes this backend created;
  the user's other CRM data is untouched.

Contact resolution for a record: ``metadata["contact_id"]`` or
``metadata["contact"]`` first, then a scope of the form
``/contacts/<name or id>``, then ``default_contact`` (or PPL_DEFAULT_CONTACT).
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from crewai.memory.storage.backend import StorageBackend  # noqa: F401
    from crewai.memory.types import MemoryRecord, ScopeInfo  # noqa: F401

    _HAVE_CREWAI = True
except ImportError:  # pragma: no cover
    _HAVE_CREWAI = False

    class MemoryRecord:  # type: ignore[no-redef]
        """Minimal stand-in used only when crewai is not installed."""

        def __init__(self, **kwargs: Any) -> None:
            from uuid import uuid4

            self.id: str = kwargs.get("id") or str(uuid4())
            self.content: str = kwargs.get("content", "")
            self.scope: str = kwargs.get("scope", "/")
            self.categories: list[str] = kwargs.get("categories", [])
            self.metadata: dict[str, Any] = kwargs.get("metadata", {})
            self.importance: float = kwargs.get("importance", 0.5)
            self.created_at: datetime = kwargs.get("created_at", datetime.utcnow())
            self.last_accessed: datetime = kwargs.get("last_accessed", datetime.utcnow())
            self.embedding: list[float] | None = kwargs.get("embedding")
            self.source: str | None = kwargs.get("source")
            self.private: bool = kwargs.get("private", False)

    class ScopeInfo:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

from .client import PplClient, PplError

_MARKER_RE = re.compile(r"\[ppl-crewai (\{.*?\})\]", re.DOTALL)
_CACHE_PATH = Path(os.environ.get("PPL_MEMORY_CACHE_DIR", Path.home() / ".ppl-memory")) / "crewai_embeddings.json"


def _marker(record: Any) -> str:
    payload = {
        "id": record.id,
        "scope": record.scope,
        "categories": list(record.categories or []),
        "importance": record.importance,
        "created_at": record.created_at.isoformat() if hasattr(record.created_at, "isoformat") else str(record.created_at),
        "source": record.source,
        "private": bool(record.private),
    }
    return f"[ppl-crewai {json.dumps(payload, separators=(',', ':'))}]"


def _parse_marker(body: str) -> dict[str, Any] | None:
    m = _MARKER_RE.search(body or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.utcnow()


class PplCrewAIStorage:
    """CrewAI unified-memory storage backend backed by ppl (https://ppl.gift).

    Args:
        api_token: ppl API token (or PPL_API_TOKEN env var).
        base_url: ppl base URL, defaults to https://ppl.gift.
        default_contact: contact id or name used when a record carries no
            contact reference. Falls back to PPL_DEFAULT_CONTACT env var.
    """

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
        default_contact: int | str | None = None,
    ) -> None:
        self.client = PplClient(api_token=api_token, base_url=base_url or "https://ppl.gift")
        self.default_contact = default_contact or os.environ.get("PPL_DEFAULT_CONTACT")
        self._contact_cache: dict[str, int] = {}

    # -- contact resolution ---------------------------------------------

    def _resolve_contact(self, record: Any) -> int:
        metadata = getattr(record, "metadata", None) or {}
        ref = metadata.get("contact_id") or metadata.get("contact")
        scope = getattr(record, "scope", "/") or "/"
        if ref is None and scope.startswith("/contacts/"):
            ref = scope[len("/contacts/"):]
        if ref is None:
            ref = self.default_contact
        if ref is None:
            raise PplError(
                "No contact to store this memory under. Set metadata['contact_id'/'contact'], "
                "use a /contacts/<name> scope, or pass default_contact=/PPL_DEFAULT_CONTACT."
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

    # -- embedding cache (derived vectors, rebuildable) -------------------

    def _load_cache(self) -> dict[str, list[float]]:
        try:
            return json.loads(_CACHE_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self, cache: dict[str, list[float]]) -> None:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache))

    # -- note <-> record ---------------------------------------------------

    def _note_to_record(self, note: dict[str, Any]) -> tuple[Any, int] | None:
        meta = _parse_marker(str(note.get("body", "")))
        if not meta:
            return None
        content = _MARKER_RE.sub("", str(note.get("body", ""))).strip()
        kwargs: dict[str, Any] = {
            "id": meta["id"],
            "content": content,
            "scope": meta.get("scope", "/"),
            "categories": meta.get("categories", []),
            "metadata": {"ppl_note_id": note["id"], "contact_id": note.get("contact_id")},
            "importance": meta.get("importance", 0.5),
            "created_at": _parse_dt(meta.get("created_at") or note.get("created_at")),
            "source": meta.get("source"),
            "private": bool(meta.get("private", False)),
        }
        if _HAVE_CREWAI:
            record = MemoryRecord(**kwargs)
        else:
            record = MemoryRecord(**kwargs)
        return record, int(note["id"])

    def _all_marked_notes(self, limit: int = 500) -> list[dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        seen = 0
        while seen < limit:
            payload = self.client._request("GET", "/api/notes", params={"limit": 100})
            page = self.client._unwrap(payload) or []
            if not isinstance(page, list) or not page:
                break
            notes.extend(page)
            seen += len(page)
            if len(page) < 100:
                break
        return notes

    def _find_note_id(self, record_id: str) -> int | None:
        for note in self._all_marked_notes():
            meta = _parse_marker(str(note.get("body", "")))
            if meta and meta.get("id") == record_id:
                return int(note["id"])
        return None

    @staticmethod
    def _matches(
        record: Any,
        scope_prefix: str | None,
        categories: list[str] | None,
        metadata_filter: dict[str, Any] | None,
    ) -> bool:
        if scope_prefix and not str(record.scope).startswith(scope_prefix):
            return False
        if categories and not set(categories) & set(record.categories or []):
            return False
        if metadata_filter:
            md = record.metadata or {}
            if any(md.get(k) != v for k, v in metadata_filter.items()):
                return False
        return True

    # -- StorageBackend protocol -------------------------------------------

    def save(self, records: list[Any]) -> None:
        cache = self._load_cache()
        for record in records:
            contact_id = self._resolve_contact(record)
            old_note_id = self._find_note_id(record.id)
            if old_note_id is not None:
                self.client.delete_note(old_note_id)
            body = f"{record.content}\n\n{_marker(record)}"
            self.client.remember(contact_id, body)
            if record.embedding:
                cache[record.id] = list(record.embedding)
            else:
                cache.pop(record.id, None)
        self._save_cache(cache)

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Any, float]]:
        cache = self._load_cache()
        scored: list[tuple[Any, float]] = []
        for note in self._all_marked_notes():
            parsed = self._note_to_record(note)
            if not parsed:
                continue
            record, _ = parsed
            if not self._matches(record, scope_prefix, categories, metadata_filter):
                continue
            emb = cache.get(record.id)
            score = _cosine(query_embedding, emb) if emb else 0.0
            if score >= min_score:
                scored.append((record, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        cache = self._load_cache()
        deleted = 0
        for note in self._all_marked_notes():
            parsed = self._note_to_record(note)
            if not parsed:
                continue
            record, note_id = parsed
            if record_ids is not None and record.id not in record_ids:
                continue
            if not self._matches(record, scope_prefix, categories, metadata_filter):
                continue
            if older_than is not None and record.created_at >= older_than:
                continue
            self.client.delete_note(note_id)
            cache.pop(record.id, None)
            deleted += 1
        if deleted:
            self._save_cache(cache)
        return deleted

    def update(self, record: Any) -> None:
        self.save([record])

    def get_record(self, record_id: str) -> Any | None:
        for note in self._all_marked_notes():
            parsed = self._note_to_record(note)
            if parsed and parsed[0].id == record_id:
                return parsed[0]
        return None

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Any]:
        records = []
        for note in self._all_marked_notes(limit=limit + offset):
            parsed = self._note_to_record(note)
            if parsed and self._matches(parsed[0], scope_prefix, None, None):
                records.append(parsed[0])
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[offset : offset + limit]

    def get_scope_info(self, scope: str) -> Any:
        records = self.list_records(scope_prefix=scope, limit=10000)
        categories: list[str] = sorted(
            {c for r in records for c in (r.categories or [])}
        )
        created = [r.created_at for r in records]
        info = {
            "path": scope,
            "record_count": len(records),
            "categories": categories,
            "oldest_record": min(created) if created else None,
            "newest_record": max(created) if created else None,
            "child_scopes": self.list_scopes(parent=scope),
        }
        return ScopeInfo(**info)

    def list_scopes(self, parent: str = "/") -> list[str]:
        children: set[str] = set()
        parent = parent.rstrip("/") or "/"
        for note in self._all_marked_notes():
            parsed = self._note_to_record(note)
            if not parsed:
                continue
            scope = str(parsed[0].scope).rstrip("/") or "/"
            if parent == "/":
                first = "/" + scope.lstrip("/").split("/")[0]
                if first != "/":
                    children.add(first)
            elif scope.startswith(parent + "/"):
                rest = scope[len(parent) + 1 :].split("/")[0]
                children.add(parent + "/" + rest)
        return sorted(children)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.list_records(scope_prefix=scope_prefix, limit=10000):
            for c in record.categories or []:
                counts[c] = counts.get(c, 0) + 1
        return counts
