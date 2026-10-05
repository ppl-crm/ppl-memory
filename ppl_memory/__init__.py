"""ppl as a memory backend for agent frameworks.

ppl (https://withppl.com) is a personal CRM. This package makes it the memory
layer for agents built on LangGraph, CrewAI, or AutoGen: agents remember
facts about people, and recall them in later sessions, without keeping
anything in their own context.

Framework adapters are optional extras, installed per framework:

    pip install ppl-memory[langchain]
    pip install ppl-memory[crewai]
    pip install ppl-memory[autogen]
"""

from .client import PplClient, PplError

__all__ = ["PplClient", "PplError"]
__version__ = "0.1.0"
