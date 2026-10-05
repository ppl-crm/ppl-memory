"""Tests for ppl_memory.openai_agents. The PplClient is fully stubbed; no network."""

import asyncio

import pytest
from agents.memory.session import Session

from ppl_memory.client import PplError
from ppl_memory.openai_agents import PplSession, _marker


class FakeClient:
    def __init__(self):
        self.notes = {}
        self.next_id = 100

    def find_contact(self, name):
        return {"id": 42, "name": name}

    def remember(self, contact_id, fact):
        note_id = self.next_id
        self.next_id += 1
        self.notes[note_id] = {
            "id": note_id,
            "contact_id": contact_id,
            "body": fact,
            "created_at": "2026-10-05T00:00:00Z",
        }
        return {"id": note_id}

    def delete_note(self, note_id):
        self.notes.pop(int(note_id), None)
        return True

    def contact_notes(self, contact_id, limit=500):
        return [n for n in self.notes.values() if n["contact_id"] == int(contact_id)][:limit]


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setenv("PPL_API_TOKEN", "test-token")
    s = PplSession(session_id="chat-123", default_contact=42)
    s.client = FakeClient()
    return s


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


def test_is_session_protocol(session):
    assert isinstance(session, Session)
    assert session.session_id == "chat-123"


def test_add_and_get_items_round_trip(session):
    run(session.add_items([user("Jane likes tea."), assistant("Noted.")]))
    items = run(session.get_items())
    assert len(items) == 2
    assert items[0]["role"] == "user"
    assert items[0]["content"] == "Jane likes tea."
    assert items[1]["content"] == "Noted."


def test_add_items_marks_session(session):
    run(session.add_items([user("hello")]))
    bodies = [n["body"] for n in session.client.notes.values()]
    assert len(bodies) == 1
    assert _marker("chat-123") in bodies[0]


def test_sessions_are_separate(monkeypatch):
    monkeypatch.setenv("PPL_API_TOKEN", "test-token")
    s1 = PplSession(session_id="one", default_contact=42)
    s1.client = FakeClient()
    s2 = PplSession(session_id="two", default_contact=42)
    s2.client = s1.client
    run(s1.add_items([user("from session one")]))
    run(s2.add_items([user("from session two")]))
    assert [i["content"] for i in run(s1.get_items())] == ["from session one"]
    assert [i["content"] for i in run(s2.get_items())] == ["from session two"]


def test_get_items_limit_returns_latest(session):
    run(session.add_items([user("first"), user("second"), user("third")]))
    items = run(session.get_items(limit=2))
    assert [i["content"] for i in items] == ["second", "third"]


def test_add_items_empty_is_noop(session):
    run(session.add_items([]))
    assert session.client.notes == {}


def test_pop_item(session):
    run(session.add_items([user("keep me"), user("drop me")]))
    popped = run(session.pop_item())
    assert popped["content"] == "drop me"
    assert [i["content"] for i in run(session.get_items())] == ["keep me"]


def test_pop_item_single_item_note(session):
    run(session.add_items([user("only")]))
    popped = run(session.pop_item())
    assert popped["content"] == "only"
    assert run(session.get_items()) == []
    assert run(session.pop_item()) is None


def test_clear_session_is_noop(session):
    run(session.add_items([user("persistent")]))
    run(session.clear_session())
    assert [i["content"] for i in run(session.get_items())] == ["persistent"]


def test_contact_name_resolution(monkeypatch):
    monkeypatch.setenv("PPL_API_TOKEN", "test-token")
    s = PplSession(session_id="named", contact="Jane Smith")
    s.client = FakeClient()
    run(s.add_items([user("hi")]))
    notes = list(s.client.notes.values())
    assert notes[0]["contact_id"] == 42


def test_no_contact_raises(monkeypatch):
    monkeypatch.setenv("PPL_API_TOKEN", "test-token")
    monkeypatch.delenv("PPL_DEFAULT_CONTACT", raising=False)
    s = PplSession(session_id="orphan")
    s.client = FakeClient()
    with pytest.raises(PplError, match="No contact"):
        run(s.add_items([user("hi")]))
