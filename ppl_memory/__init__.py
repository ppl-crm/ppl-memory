"""ppl as a memory backend for agent frameworks.

ppl (https://withppl.com) is a personal CRM. This package makes it the memory
layer for agents built on LangGraph, CrewAI, AutoGen, Mem0, LlamaIndex, the
OpenAI Agents SDK, or Semantic Kernel: agents remember facts about people,
and recall them in later sessions, without keeping anything in their own
context.

Framework adapters are optional extras, installed per framework:

    pip install ppl-memory[langchain]
    pip install ppl-memory[crewai]
    pip install ppl-memory[autogen]
    pip install ppl-memory[mem0]
    pip install ppl-memory[llamaindex]
    pip install ppl-memory[openai-agents]
    pip install ppl-memory[semantic-kernel]

The Vercel AI SDK (TypeScript) lives in ppl-memory-js/ and is published as
the same "ppl-memory" name on npm.
"""

from .client import PplClient, PplError

__all__ = ["PplClient", "PplError"]
__version__ = "0.3.0"
