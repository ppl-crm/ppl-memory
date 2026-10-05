"""Semantic Kernel plugin for ppl.

``PplMemoryPlugin`` exposes ppl as kernel functions. Add it to a kernel and
the agent can call ``ppl-remember``, ``ppl-recall``, and ``ppl-briefing``:

.. code-block:: python

    from semantic_kernel import Kernel
    from ppl_memory.semantic_kernel import PplMemoryPlugin

    kernel = Kernel()
    kernel.add_plugin(PplMemoryPlugin(default_contact="Jane Smith"), plugin_name="ppl")

    result = await kernel.invoke(
        kernel.get_function("ppl", "recall"),
        query="What does Jane like to drink?",
    )
    print(result)

Functions:

- ``remember(contact, fact)`` stores a fact as a note on the contact's
  timeline (POST /api/notes). ``contact`` may be a ppl contact id or name.
- ``recall(query)`` runs ppl's ranked retrieval (POST /api/agent/ask) and
  returns the hits as text.
- ``briefing()`` returns the morning briefing (birthdays, reconnects,
  pending tasks, suggested actions).
"""

from __future__ import annotations

import json
import os
from typing import Annotated

try:
    from semantic_kernel.functions import kernel_function
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ppl-memory[semantic-kernel] requires semantic-kernel. "
        "Install with: pip install ppl-memory[semantic-kernel]"
    ) from exc

from .client import PplClient, PplError


class PplMemoryPlugin:
    """A Semantic Kernel plugin backed by ppl (https://withppl.com).

    Args:
        api_token: ppl API token (or PPL_API_TOKEN env var).
        base_url: ppl base URL, defaults to https://withppl.com.
        default_contact: contact id or name used when ``remember`` is called
            without an explicit contact. Falls back to PPL_DEFAULT_CONTACT.
    """

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
        default_contact: int | str | None = None,
    ) -> None:
        self.client = PplClient(
            api_token=api_token, base_url=base_url or "https://withppl.com"
        )
        self.default_contact = default_contact or os.environ.get("PPL_DEFAULT_CONTACT")
        self._contact_cache: dict[str, int] = {}

    # -- helpers --------------------------------------------------------

    def _resolve_contact(self, contact: int | str | None) -> int:
        ref = contact if contact not in (None, "") else self.default_contact
        if ref is None:
            raise PplError(
                "No contact given. Pass contact= (a ppl contact id or name) or "
                "set default_contact=/PPL_DEFAULT_CONTACT."
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

    @staticmethod
    def _format_hits(results: list[dict]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            text = r.get("title") or r.get("text") or r.get("body") or json.dumps(r)
            contact = r.get("contact") or r.get("contact_name")
            lines.append(f"{i}. {text}" + (f" (about {contact})" if contact else ""))
        return "\n".join(lines) if lines else "No memories found."

    # -- kernel functions -------------------------------------------------

    @kernel_function(
        name="remember",
        description="Store a fact about a person in ppl. The fact is saved as a note on their timeline and recalled in later sessions.",
    )
    def remember(
        self,
        fact: Annotated[str, "The fact to remember, in plain text."],
        contact: Annotated[str, "The ppl contact id or name the fact is about."] = "",
    ) -> str:
        """Store a fact about a contact in ppl."""
        if not fact or not fact.strip():
            raise PplError("remember() requires a non-empty fact.")
        contact_id = self._resolve_contact(contact or None)
        note = self.client.remember(contact_id, fact.strip())
        note_id = None
        if isinstance(note, dict):
            note_id = note.get("id") or (note.get("note") or {}).get("id")
        return f"Remembered (note {note_id})."

    @kernel_function(
        name="recall",
        description="Search ppl's memory for facts about people. Returns ranked results with the best matches first.",
    )
    def recall(
        self,
        query: Annotated[str, "The question or topic to search for."],
    ) -> str:
        """Recall memories from ppl with ranked retrieval."""
        return self._format_hits(self.client.ask(query))

    @kernel_function(
        name="briefing",
        description="Get the ppl morning briefing: birthdays, people to reconnect with, pending tasks, and suggested actions.",
    )
    def briefing(self) -> str:
        """Get the ppl morning briefing."""
        data = self.client.get_briefing()
        if not data:
            return "No briefing available."
        return json.dumps(data, ensure_ascii=False, indent=2)
