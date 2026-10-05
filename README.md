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
| LlamaIndex | `ppl-memory[llamaindex]` | `PplMemoryBlock` (BaseMemoryBlock) |
| OpenAI Agents SDK | `ppl-memory[openai-agents]` | `PplSession` (Session) |
| Semantic Kernel | `ppl-memory[semantic-kernel]` | `PplMemoryPlugin` (kernel functions) |
| Vercel AI SDK | `npm install ppl-memory` | `createPplTools()` (AI SDK tools, in `ppl-memory-js/`) |

## Setup

1. Create an API token at [withppl.com/settings/agents](https://withppl.com/settings/agents).
2. Export it:

```bash
export PPL_API_TOKEN="your-token"
```

Optionally set a default contact for memories: `export PPL_DEFAULT_CONTACT="Jane Smith"` (name or contact id).

3. Install the extra for your framework:

```bash
pip install ppl-memory[langchain]   # or [crewai], [autogen], [mem0],
                                    # [llamaindex], [openai-agents], or [semantic-kernel]
```

TypeScript users (Vercel AI SDK): `npm install ppl-memory` (same name, same
repo, in `ppl-memory-js/`; see that directory's README).

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

## LlamaIndex

`PplMemoryBlock` is a LlamaIndex `BaseMemoryBlock[str]`. Drop it into a
`Memory` alongside LlamaIndex's other blocks:

```python
from llama_index.core.memory import Memory
from ppl_memory.llama_index import PplMemoryBlock

memory = Memory(
    memory_blocks=[PplMemoryBlock(default_contact="Jane Smith")],
    token_limit=30000,
)
agent = ReActAgent.from_tools(tools, llm=llm, memory=memory)
```

`aput` writes messages as a note on the contact's timeline (marked
`[ppl-llamaindex]`); `aget` runs ppl's ranked retrieval over the latest user
message and returns the hits as text for the memory template. The contact
comes from `default_contact` (or `PPL_DEFAULT_CONTACT`), or per-message
`additional_kwargs` `contact`/`contact_id` keys.

## OpenAI Agents SDK

`PplSession` implements the `agents` `Session` protocol, so it drops straight
into `Runner.run`:

```python
from agents import Agent, Runner
from ppl_memory.openai_agents import PplSession

agent = Agent(name="assistant", instructions="You are helpful.")
session = PplSession(session_id="chat-123", default_contact="Jane Smith")

result = await Runner.run(agent, "Remember that Jane likes tea.", session=session)
# a later run with the same session_id replays the conversation
```

`add_items` stores the conversation as notes on the contact's timeline (marked
per session); `get_items` replays them in order; `pop_item` removes the most
recent item; `clear_session` is a deliberate no-op, since ppl holds real CRM
history and must not be wiped by a session reset.

## Semantic Kernel

`PplMemoryPlugin` exposes ppl as kernel functions (`ppl-remember`,
`ppl-recall`, `ppl-briefing`):

```python
from semantic_kernel import Kernel
from ppl_memory.semantic_kernel import PplMemoryPlugin

kernel = Kernel()
kernel.add_plugin(PplMemoryPlugin(default_contact="Jane Smith"), plugin_name="ppl")

result = await kernel.invoke(
    kernel.get_function("ppl", "recall"),
    query="What does Jane like to drink?",
)
print(result)
```

`remember(contact, fact)` stores a fact on the contact's timeline (contact
may be an id or a name); `recall(query)` returns ranked hits as text;
`briefing()` returns the morning briefing.

## Vercel AI SDK (TypeScript)

The TypeScript side lives in `ppl-memory-js/` in this repo and is published
as `ppl-memory` on npm (same package name, same version). It wraps the same
ppl REST API as AI SDK tools:

```ts
import { generateText } from 'ai';
import { createPplTools } from 'ppl-memory';

const { text } = await generateText({
  model: myModel,
  tools: createPplTools(), // reads PPL_API_TOKEN
  prompt: 'What does Jane like to drink?',
});
```

Tools: `pplRemember`, `pplRecall`, `pplBriefing`, `pplFindContact`. See
`ppl-memory-js/README.md`.

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
- Not affiliated with LangChain, CrewAI, Microsoft, OpenAI, Meta (LlamaIndex),
  or Vercel. ppl is a standalone personal CRM by Cumulative Systems.

## License

MIT
