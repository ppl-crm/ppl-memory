"""Tests for ppl_memory.llama_index. The PplClient is fully stubbed; no network."""

import asyncio

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from ppl_memory.client import PplError
from ppl_memory.llama_index import MARKER, PplMemoryBlock, _last_user_text, _message_text


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
        return {"id": note_id, "body": fact}

    def ask(self, question, limit=10):
        return [
            {"id": 7, "title": "Jane likes tea", "score": 0.9, "contact_name": "Jane Smith"}
        ][:limit]

    def contact_notes(self, contact_id, limit=200):
        return [n for n in self.notes.values() if n["contact_id"] == int(contact_id)][:limit]


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def block(monkeypatch):
    monkeypatch.setenv("PPL_API_TOKEN", "test-token")
    b = PplMemoryBlock(default_contact="Jane Smith")
    b._client = FakeClient()
    return b


def msg(role, content, **kwargs):
    return ChatMessage(role=role, content=content, additional_kwargs=kwargs)


def test_message_text_string():
    assert _message_text(msg(MessageRole.USER, "hello")) == "hello"


def test_message_text_content_blocks():
    from llama_index.core.base.llms.types import TextBlock

    # ChatMessage merges text blocks into one string; the str path handles it.
    m = msg(MessageRole.USER, [TextBlock(text="piece"), TextBlock(text="two")])
    assert _message_text(m) == "piece\ntwo"


def test_last_user_text():
    messages = [
        msg(MessageRole.USER, "first"),
        msg(MessageRole.ASSISTANT, "reply"),
        msg(MessageRole.USER, "latest"),
    ]
    assert _last_user_text(messages) == "latest"


def test_last_user_text_empty():
    assert _last_user_text([]) == ""
    assert _last_user_text(None) == ""


def test_aput_stores_note_with_marker(block):
    run(block.aput([msg(MessageRole.USER, "Jane likes tea.")]))
    notes = list(block._client.notes.values())
    assert len(notes) == 1
    note = notes[0]
    assert note["contact_id"] == 42
    assert "Jane likes tea." in note["body"]
    assert MARKER in note["body"]
    assert "user: Jane likes tea." in note["body"]


def test_aput_skips_empty_messages(block):
    run(block.aput([msg(MessageRole.USER, "   ")]))
    assert block._client.notes == {}


def test_aput_uses_message_contact_ref(block):
    run(block.aput([msg(MessageRole.USER, "Hi.", contact_id=99)]))
    notes = list(block._client.notes.values())
    assert notes[0]["contact_id"] == 99


def test_aput_no_contact_raises(monkeypatch):
    monkeypatch.setenv("PPL_API_TOKEN", "test-token")
    monkeypatch.delenv("PPL_DEFAULT_CONTACT", raising=False)
    b = PplMemoryBlock()
    b._client = FakeClient()
    with pytest.raises(PplError, match="No contact"):
        run(b.aput([msg(MessageRole.USER, "hello")]))


def test_aget_returns_hits(block):
    text = run(block.aget([msg(MessageRole.USER, "What does Jane drink?")]))
    assert "Jane likes tea" in text
    assert "Jane Smith" in text


def test_aget_no_user_message_returns_empty(block):
    assert run(block.aget([msg(MessageRole.ASSISTANT, "hi")])) == ""
    assert run(block.aget([])) == ""


def test_atruncate_drops_content(block):
    assert run(block.atruncate("some content", 100)) is None


def test_block_fits_llamaindex_memory(block):
    from llama_index.core.memory import Memory

    memory = Memory(memory_blocks=[block], token_limit=30000)
    assert memory.memory_blocks[0] is block
