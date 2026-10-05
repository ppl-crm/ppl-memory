"""Tests for ppl_memory.semantic_kernel. The PplClient is fully stubbed; no network."""

import json

import pytest
from semantic_kernel import Kernel

from ppl_memory.client import PplError
from ppl_memory.semantic_kernel import PplMemoryPlugin


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
        }
        return {"id": note_id, "body": fact}

    def ask(self, question, limit=10):
        return [
            {"id": 7, "title": "Jane likes tea", "score": 0.9, "contact_name": "Jane Smith"}
        ][:limit]

    def get_briefing(self):
        return {"birthdays": ["Jane Smith"], "reconnect": []}


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setenv("PPL_API_TOKEN", "test-token")
    p = PplMemoryPlugin(default_contact="Jane Smith")
    p.client = FakeClient()
    return p


def test_registers_as_plugin(plugin):
    kernel = Kernel()
    kernel.add_plugin(plugin, plugin_name="ppl")
    for name in ("remember", "recall", "briefing"):
        fn = kernel.get_function("ppl", name)
        assert fn.name == name
        assert fn.plugin_name == "ppl"


def test_remember_stores_note(plugin):
    result = plugin.remember(fact="Jane likes tea.", contact="Jane Smith")
    notes = list(plugin.client.notes.values())
    assert len(notes) == 1
    assert notes[0]["contact_id"] == 42
    assert notes[0]["body"] == "Jane likes tea."
    assert "Remembered" in result


def test_remember_uses_default_contact(plugin):
    plugin.remember(fact="Prefers oat milk.")
    notes = list(plugin.client.notes.values())
    assert notes[0]["contact_id"] == 42


def test_remember_numeric_contact_id(plugin):
    plugin.remember(fact="Hi.", contact="7")
    notes = list(plugin.client.notes.values())
    assert notes[0]["contact_id"] == 7


def test_remember_empty_fact_raises(plugin):
    with pytest.raises(PplError, match="non-empty fact"):
        plugin.remember(fact="   ", contact="Jane Smith")


def test_remember_no_contact_raises(monkeypatch):
    monkeypatch.setenv("PPL_API_TOKEN", "test-token")
    monkeypatch.delenv("PPL_DEFAULT_CONTACT", raising=False)
    p = PplMemoryPlugin()
    p.client = FakeClient()
    with pytest.raises(PplError, match="No contact"):
        p.remember(fact="Hello.")


def test_recall_formats_hits(plugin):
    result = plugin.recall(query="What does Jane drink?")
    assert "Jane likes tea" in result
    assert "Jane Smith" in result
    assert result.startswith("1. ")


def test_recall_no_hits(monkeypatch, plugin):
    plugin.client.ask = lambda q, limit=10: []
    assert plugin.recall(query="nothing") == "No memories found."


def test_briefing_returns_json(plugin):
    result = plugin.briefing()
    data = json.loads(result)
    assert data["birthdays"] == ["Jane Smith"]


def test_briefing_empty(monkeypatch, plugin):
    plugin.client.get_briefing = lambda: {}
    assert plugin.briefing() == "No briefing available."


def test_invoke_through_kernel(plugin):
    import asyncio

    kernel = Kernel()
    kernel.add_plugin(plugin, plugin_name="ppl")
    fn = kernel.get_function("ppl", "recall")

    async def go():
        result = await kernel.invoke(fn, query="tea")
        return str(result)

    out = asyncio.run(go())
    assert "Jane likes tea" in out
