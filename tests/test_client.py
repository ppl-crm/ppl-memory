"""Tests for ppl_memory.client. Network is fully mocked; no ppl.gift calls."""

import io
import json
import urllib.error
import urllib.request

import pytest

from ppl_memory import PplClient, PplError
from ppl_memory.client import DEFAULT_BASE_URL, TOKEN_ENV_VAR


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def make_client(monkeypatch, payload, method_check=None, path_check=None):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["method"] = req.get_method()
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        if req.data:
            captured["body"] = json.loads(req.data.decode())
        if method_check:
            method_check(captured["method"])
        if path_check:
            path_check(captured["url"])
        return FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv(TOKEN_ENV_VAR, "test-token")
    return PplClient(), captured


def test_requires_token(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    with pytest.raises(PplError, match="No API token"):
        PplClient()


def test_bearer_header_sent(monkeypatch):
    client, captured = make_client(monkeypatch, {"data": {}})
    client.get_briefing()
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert captured["url"].startswith(DEFAULT_BASE_URL)


def test_remember_posts_note(monkeypatch):
    client, captured = make_client(monkeypatch, {"data": {"id": 1, "body": "likes tea"}})
    note = client.remember(contact_id=42, fact="likes tea")
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/api/notes")
    assert captured["body"] == {"contact_id": 42, "body": "likes tea"}
    assert note["id"] == 1


def test_ask_posts_question(monkeypatch):
    client, captured = make_client(monkeypatch, {"data": [{"title": "Jane", "score": 0.9}]})
    results = client.ask("How do I reach Jane?")
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/api/agent/ask")
    assert captured["body"]["question"] == "How do I reach Jane?"
    assert results[0]["score"] == 0.9


def test_search_gets_semantic(monkeypatch):
    client, captured = make_client(monkeypatch, {"data": [{"text": "birthday note"}]})
    results = client.search("birthday")
    assert captured["method"] == "GET"
    assert "/api/search/semantic" in captured["url"]
    assert "q=birthday" in captured["url"]
    assert results[0]["text"] == "birthday note"


def test_get_briefing(monkeypatch):
    client, captured = make_client(monkeypatch, {"data": {"birthdays": []}})
    briefing = client.get_briefing()
    assert captured["url"].endswith("/api/agent/briefing")
    assert briefing == {"birthdays": []}


def test_delete_note(monkeypatch):
    client, captured = make_client(monkeypatch, {"data": {"deleted": True}})
    assert client.delete_note(7) is True
    assert captured["method"] == "DELETE"
    assert captured["url"].endswith("/api/notes/7")


def test_http_error_raises_ppl_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b'{"error": "nope"}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv(TOKEN_ENV_VAR, "test-token")
    with pytest.raises(PplError) as excinfo:
        PplClient().get_contact(999)
    assert excinfo.value.status == 404


def test_unwrap_plain_list(monkeypatch):
    client, _ = make_client(monkeypatch, [{"id": 1}])
    assert client.contact_notes(5) == [{"id": 1}]
