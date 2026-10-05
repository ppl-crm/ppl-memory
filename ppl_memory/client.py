"""HTTP client for the withppl.com REST API.

ppl is a personal CRM. This client talks to the hosted API at
https://withppl.com/api using a Bearer token (the user's ppl API token,
created at https://withppl.com/settings/agents).

All network I/O is synchronous (built on urllib) and dependency-free.
Framework adapters run it in a thread for async interfaces.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://withppl.com"
TOKEN_ENV_VAR = "PPL_API_TOKEN"


class PplError(Exception):
    """Raised when the ppl API returns an error or is unreachable."""

    def __init__(self, message: str, status: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


class PplClient:
    """Thin Bearer-authenticated client for the withppl.com API."""

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30,
    ) -> None:
        token = api_token or os.environ.get(TOKEN_ENV_VAR)
        if not token:
            raise PplError(
                f"No API token. Pass api_token= or set the {TOKEN_ENV_VAR} "
                "environment variable (create one at https://withppl.com/settings/agents)."
            )
        self.api_token = token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- low level ------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        url = self.base_url + path
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
        body = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except Exception:
                payload = None
            raise PplError(
                f"ppl API {exc.code} on {method} {path}: {payload or exc.reason}",
                status=exc.code,
                payload=payload,
            ) from exc
        except urllib.error.URLError as exc:
            raise PplError(f"Could not reach withppl.com ({exc.reason}).") from exc

        if not raw.strip():
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    @staticmethod
    def _unwrap(payload: Any) -> Any:
        """Laravel resources wrap collections in {"data": [...]}; unwrap for convenience."""
        if isinstance(payload, dict) and "data" in payload and len(payload) <= 3:
            return payload["data"]
        return payload

    # -- high level -----------------------------------------------------

    def remember(self, contact_id: int | str, fact: str) -> dict[str, Any]:
        """Store a fact about a contact. ppl keeps it as a note on their timeline.

        Args:
            contact_id: The ppl contact id.
            fact: The fact to remember (plain text).
        """
        payload = self._request("POST", "/api/notes", data={"contact_id": contact_id, "body": fact})
        return self._unwrap(payload) or {}

    def get_contact(self, contact_id: int | str) -> dict[str, Any]:
        """Fetch a single contact by id."""
        payload = self._request("GET", f"/api/contacts/{contact_id}")
        return self._unwrap(payload) or {}

    def find_contact(self, name: str) -> dict[str, Any] | None:
        """Find the best-matching contact by name. Returns None when no match."""
        payload = self._request("GET", "/api/search/semantic", params={"q": name, "types": "contacts", "limit": 3})
        items = self._unwrap(payload) or []
        if isinstance(items, dict):
            return items
        return items[0] if items else None

    def ask(self, question: str, limit: int = 10) -> list[dict[str, Any]]:
        """Ranked retrieval over everything ppl knows (POST /api/agent/ask).

        Returns ranked results with titles, scores, and contact info.
        """
        payload = self._request("POST", "/api/agent/ask", data={"question": question, "limit": limit})
        results = self._unwrap(payload)
        if isinstance(results, dict):
            for key in ("results", "matches", "items"):
                if key in results:
                    results = results[key]
                    break
        return results if isinstance(results, list) else [results] if results else []

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Semantic search over notes, journal entries, and activities."""
        payload = self._request("GET", "/api/search/semantic", params={"q": query, "limit": limit})
        results = self._unwrap(payload)
        return results if isinstance(results, list) else [results] if results else []

    def get_briefing(self) -> dict[str, Any]:
        """The morning briefing: birthdays, reconnects, pending tasks, suggested actions."""
        payload = self._request("GET", "/api/agent/briefing")
        return self._unwrap(payload) or {}

    def get_note(self, note_id: int | str) -> dict[str, Any]:
        payload = self._request("GET", f"/api/notes/{note_id}")
        return self._unwrap(payload) or {}

    def delete_note(self, note_id: int | str) -> bool:
        self._request("DELETE", f"/api/notes/{note_id}")
        return True

    def contact_notes(self, contact_id: int | str, limit: int = 50) -> list[dict[str, Any]]:
        payload = self._request(
            "GET", f"/api/contacts/{contact_id}/notes", params={"limit": limit}
        )
        results = self._unwrap(payload)
        return results if isinstance(results, list) else []
