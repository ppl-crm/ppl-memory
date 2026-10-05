# ppl-memory

ppl ([withppl.com](https://withppl.com)) as a memory backend for agent frameworks.

ppl is a personal CRM. This package makes it the memory layer for your agents:
agents remember facts about people and recall them in later sessions, without
keeping anything in their own context. The memory lives in the user's ppl
account, so it survives across sessions, agents, and frameworks.

Supported frameworks (one package, framework extras):

| Framework | Extra | Class |
|---|---|---|
| LangGraph | `ppl-memory[langchain]` | `PplStore` (BaseStore) |
| CrewAI | `ppl-memory[crewai]` | `PplCrewAIStorage` (StorageBackend) |
| AutoGen 0.4 | `ppl-memory[autogen]` | `PplMemory` (Memory) |
| Mem0 | `ppl-memory[mem0]` | `PplMem0` (MemoryBase) |

## Setup

1. Create an API token at [withppl.com/settings/agents](https://withppl.com/settings/agents).
2. Export it:

```bash
export PPL_API_TOKEN="your-token"
```

Optionally set a default contact for memories: `export PPL_DEFAULT_CONTACT="Jane Smith"` (name or contact id).

3. Install the extra for your framework:

```bash
pip install ppl-memory[langchain]   # or [crewai], [autogen], or [mem0]
```

## LangGraph

`PplStore` implements LangGraph's `BaseStore`. Memories are namespaced per
contact: namespace `("memories", <contact id or name>)`. `put` stores a fact
as a note on that contact's timeline; `get` retrieves it back; `search`
runs ppl's ranked retrieval over everything ppl knows.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from ppl_memory.langchain import PplStore

store = PplStore(default_contact="Jane Smith")

# remember
store.put(("memories", "Jane Smith"), "coffee", {"fact": "Jane drinks oat milk lattes."})

# recall in a later session
item = store.get(("memories", "Jane Smith"), "coffee")
print(item.value["fact"])

# natural-language search across the whole CRM
for hit in store.search(("memories",), query="What does Jane like to drink?"):
    print(hit.value["fact"], hit.score)

# use it in a graph
graph = StateGraph(...).compile(checkpointer=MemorySaver(), store=store)
```

## CrewAI

`PplCrewAIStorage` implements CrewAI's unified-memory `StorageBackend`
protocol. Use it as the storage for the unified `Memory`:

```python
from crewai import Agent, Crew, Task, Process
from crewai.memory import Memory
from ppl_memory.crewai import PplCrewAIStorage

memory = Memory(storage=PplCrewAIStorage(default_contact="Jane Smith"))

crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.sequential,
    memory=memory,
)

memory.remember("The customer prefers email over phone", scope="/clients/acme")
memory.recall("how should we contact acme?", scope="/clients")
```

`save()` writes each record as a note on the contact's timeline (contact from
record metadata, a `/contacts/<name>` scope, or the default contact).
`search()` scores records with cosine similarity against a local embedding
cache (`~/.ppl-memory/crewai_embeddings.json`; derived vectors only, the
memory content lives in ppl). `delete()` removes only notes this backend
created; your other CRM data is untouched.

## AutoGen 0.4

`PplMemory` implements the `autogen_core` `Memory` protocol.

```python
import asyncio
from autogen_core.memory import MemoryContent
from autogen_core.model_context import BufferedChatCompletionContext
from ppl_memory.autogen import PplMemory

async def main():
    memory = PplMemory(default_contact="Jane Smith")

    await memory.add(MemoryContent(
        content="Jane's birthday is Friday.",
        mime_type="text/plain",
        metadata={"contact": "Jane Smith"},
    ))

    result = await memory.query("When is Jane's birthday?")
    print(result.results[0].content)

    # inject relevant memories into the model context
    context = BufferedChatCompletionContext(buffer_size=10)
    await memory.update_context(context)

asyncio.run(main())
```

`update_context()` takes the latest user message, queries ppl for relevant
memories, and adds them as a system message. `clear()` is a deliberate no-op:
ppl holds real CRM history and must not be wiped by a memory reset.

## Mem0

`PplMem0` subclasses mem0's `MemoryBase`, so it works anywhere a Mem0 memory
store is expected, with Mem0-style `add`/`search` result shapes.

```python
from ppl_memory.mem0 import PplMem0

memory = PplMem0(default_contact="Jane Smith")

# add: messages as a string or [{"role": ..., "content": ...}] list
result = memory.add(
    [{"role": "user", "content": "Jane's birthday is Friday."}],
    user_id="Jane Smith",
)
print(result["results"][0]["id"])

# search
hits = memory.search("When is Jane's birthday?", user_id="Jane Smith")
print(hits["results"][0]["memory"])

# get / get_all / update / delete / history follow the MemoryBase interface
mem = memory.get(result["results"][0]["id"])
memory.update(mem["id"], "Jane's birthday is Saturday, not Friday.")
memory.delete(mem["id"])
```

`user_id` selects the ppl contact (id or name); `default_contact` is the
fallback. `delete_all()` and `reset()` are deliberate no-ops: ppl is the
system of record and is never wiped by an adapter.

## Raw client

```python
from ppl_memory import PplClient

client = PplClient()  # reads PPL_API_TOKEN

client.remember(contact_id=123, fact="Jane prefers email over phone.")
client.ask("How should I reach Jane?")     # ranked retrieval
client.search("birthday")                   # semantic search
client.get_briefing()                       # morning briefing
```

## Notes

- Memories are stored as notes on contact timelines in ppl. They are visible
  in the ppl web app, which is the point: the user's memory is inspectable.
- The human never handles the token for framework setup beyond the one-time
  export; the agent drives the rest via [withppl.com/agents](https://withppl.com/agents).
- Not affiliated with LangChain, CrewAI, or Microsoft. ppl is a standalone
  personal CRM by Cumulative Systems.

## License

MIT
