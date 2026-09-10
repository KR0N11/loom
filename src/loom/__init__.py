"""Loom: a multi-agent data-to-decision platform.

A question goes through plan -> retrieve semantics -> execute -> verify -> narrate.
Each stage is a worker agent with a strict Pydantic output contract; the
supervisor (LangGraph) owns the routing and checkpointing.
"""

__version__ = "0.1.0"
