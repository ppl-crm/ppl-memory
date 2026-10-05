"""Tests for ppl_memory.mem0. The PplClient is fully stubbed; no network."""

import pytest

from ppl_memory.mem0 import PplMem0, _messages_to_text


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
            "updated_at": "2026-10-05T00:00:00Z",
        }
        return {"note": {"id": note_id}}

    def get_note(self, note_id):
        from ppl_memory.client import PplError

        note = self.notes.get(int(note_id))
        if not note:
            raise PplError("not found", status=404)
        return note

    def delete_note(self, note_id):
        self.notes.pop(int(note_id), None)
        return True

    def contact_notes(self, contact_id, limit=50):
        return [n for n in self.notes.values() if n["contact_id"] == int(contact_id)][:limit]

    def ask(self, question, limit=10):
        return [
            {"id": 7, "title": "Jane likes tea", "score": 0.9, "contact_name": "Jane Smith"}
        ][:limit]


@pytest.fixture
def memory(monkeypatch):
    monkeypatch.setenv("PPL_API_TOKEN", "test-token")
    m = PplMem0(default_contact="Jane Smith")
    m.client = FakeClient()
    return m


def test_messages_to_text_string():
    assert _messages_to_text("hello") == "hello"


def test_messages_to_text_list():
    msgs = [
        {"role": "user", "content": "Jane's birthday is Friday."},
        {"role": "assistant", "content": "Noted."},
    ]
    text = _messages_to_text(msgs)
    assert "Jane's birthday is Friday." in text
    assert "Noted." in text


def test_add_returns_mem0_result_shape(memory):
    result = memory.add("Jane likes tea.", user_id="Jane Smith")
    assert "results" in result
    item = result["results"][0]
    assert item["event"] == "ADD"
    assert item["memory"] == "Jane likes tea."
    assert item["id"]


def test_add_message_list(memory):
    result = memory.add(
        [{"role": "user", "content": "Prefers oat milk."}], user_id="Jane Smith"
    )
    assert "oat milk" in result["results"][0]["memory"]


def test_add_empty_raises(memory):
    from ppl_memory.client import PplError

    with pytest.raises(PplError):
        memory.add("   ", user_id="Jane Smith")


def test_get_round_trip(memory):
    added = memory.add("Loves hiking.", user_id="Jane Smith")
    mem_id = added["results"][0]["id"]
    got = memory.get(mem_id)
    assert got["id"] == mem_id
    assert got["memory"] == "Loves hiking."


def test_get_missing_returns_none(memory):
    assert memory.get("999999") is None


def test_get_all(memory):
    memory.add("Fact one.", user_id="Jane Smith")
    memory.add("Fact two.", user_id="Jane Smith")
    result = memory.get_all(user_id="Jane Smith")
    assert len(result["results"]) == 2


def test_update_replaces_memory(memory):
    added = memory.add("Old fact.", user_id="Jane Smith")
    mem_id = added["results"][0]["id"]
    updated = memory.update(mem_id, "New fact.")
    assert updated["event"] == "UPDATE"
    assert updated["memory"] == "New fact."
    assert memory.get(mem_id) is None  # old note id gone
    got = memory.get(updated["id"])
    assert got["memory"] == "New fact."


def test_delete(memory):
    added = memory.add("Temporary.", user_id="Jane Smith")
    mem_id = added["results"][0]["id"]
    result = memory.delete(mem_id)
    assert "deleted" in result["message"].lower()
    assert memory.get(mem_id) is None


def test_history(memory):
    added = memory.add("Some fact.", user_id="Jane Smith")
    mem_id = added["results"][0]["id"]
    hist = memory.history(mem_id)
    assert len(hist) == 1
    assert hist[0]["new_memory"] == "Some fact."
    assert hist[0]["event"] == "ADD"


def test_history_missing_is_empty(memory):
    assert memory.history("999999") == []


def test_search_returns_mem0_shape(memory):
    result = memory.search("tea", user_id="Jane Smith")
    assert "results" in result
    assert result["results"][0]["memory"] == "Jane likes tea"
    assert result["results"][0]["score"] == 0.9


def test_is_memory_base_subclass():
    from mem0.memory.base import MemoryBase

    assert issubclass(PplMem0, MemoryBase)


def test_delete_all_and_reset_are_noops(memory):
    memory.add("Keep me.", user_id="Jane Smith")
    assert "no-op" in memory.delete_all(user_id="Jane Smith")["message"]
    assert "no-op" in memory.reset()["message"]
    # memories still there
    assert memory.get_all(user_id="Jane Smith")["results"]
