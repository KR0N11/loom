"""Supervisor graph (LangGraph).

Approach: Supervisor/worker pattern. The supervisor is a typed StateGraph over
LoomState that routes planner -> semantic -> executor -> (verifier | narrator)
-> narrator. Workers never call each other; they only read and write slots on
the shared state. SQLite checkpointing means a run can be inspected or resumed
by thread id. Every node runs inside a tracer span.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from loom.agents.executor import ExecutorAgent
from loom.agents.narrator import NarratorAgent
from loom.agents.planner import PlannerAgent
from loom.agents.state import LoomState
from loom.agents.verifier import VerifierAgent
from loom.config import Settings, get_settings
from loom.llm.base import LLMProvider
from loom.sandbox.base import SandboxExecutor
from loom.tracing import Tracer, get_tracer


def _default_semantic_agent(settings: Settings) -> Any:
    from loom.agents.semantic_agent import make_semantic_agent

    return make_semantic_agent(settings)


def _default_sandbox(settings: Settings) -> SandboxExecutor:
    from loom.sandbox import get_executor

    return get_executor(settings)


class NodeFn(Protocol):
    """Shape LangGraph expects for a node: it calls `fn(state=...)` by keyword."""

    def __call__(self, state: LoomState) -> dict[str, Any]: ...


def _traced(name: str, tracer: Tracer, fn: Callable[[LoomState], dict[str, Any]]) -> NodeFn:
    def node(state: LoomState) -> dict[str, Any]:
        with tracer.span(
            name, input={"question": state.question, "dataset": state.dataset}, kind="agent"
        ):
            return fn(state)

    node.__name__ = name
    return node


def _after_executor(state: LoomState) -> str:
    # Nothing to verify if no step produced rows; go straight to the failure memo.
    if not state.executions:
        return "narrator"
    return "verifier"


def build_graph(
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    semantic_agent: Any | None = None,
    executor: SandboxExecutor | None = None,
    tracer: Tracer | None = None,
    checkpointer: Any | None = None,
):
    """Compile the supervisor graph. Injected collaborators make it unit-testable offline."""
    settings = settings or get_settings()
    tracer = tracer or Tracer(None)
    if provider is None:
        from loom.llm import get_provider

        provider = get_provider(settings)
    semantic_agent = semantic_agent or _default_semantic_agent(settings)
    executor = executor or _default_sandbox(settings)

    planner = PlannerAgent(provider, tracer, settings)
    exec_agent = ExecutorAgent(provider, executor, settings, tracer)
    verifier = VerifierAgent(provider, executor, settings, tracer)
    narrator = NarratorAgent(provider, settings, tracer)

    graph = StateGraph(LoomState)
    graph.add_node("planner", _traced("planner", tracer, planner.run))
    graph.add_node("semantic", _traced("semantic", tracer, semantic_agent.run))
    graph.add_node("executor", _traced("executor", tracer, exec_agent.run))
    graph.add_node("verifier", _traced("verifier", tracer, verifier.run))
    graph.add_node("narrator", _traced("narrator", tracer, narrator.run))
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "semantic")
    graph.add_edge("semantic", "executor")
    graph.add_conditional_edges(
        "executor", _after_executor, {"verifier": "verifier", "narrator": "narrator"}
    )
    graph.add_edge("verifier", "narrator")
    graph.add_edge("narrator", END)
    if checkpointer is None:
        checkpointer = _sqlite_checkpointer(settings.checkpoint_path)
    return graph.compile(checkpointer=checkpointer)


def _sqlite_checkpointer(path: Path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph may touch the saver from a worker thread.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def run_question(
    question: str,
    dataset: str,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    injection: str | None = None,
    thread_id: str | None = None,
    semantic_agent: Any | None = None,
    sandbox: SandboxExecutor | None = None,
    tracer: Tracer | None = None,
) -> LoomState:
    """Run one question end to end and return the final typed state."""
    settings = settings or get_settings()
    tracer = tracer or get_tracer(settings)
    thread_id = thread_id or uuid.uuid4().hex[:12]
    app = build_graph(settings, provider, semantic_agent, sandbox, tracer)
    state = LoomState(question=question, dataset=dataset, thread_id=thread_id, injection=injection)
    tracer.start_trace(
        "loom.ask",
        input={"question": question, "dataset": dataset},
        metadata={"thread_id": thread_id, "injection": injection},
    )
    try:
        raw = app.invoke(state, config={"configurable": {"thread_id": thread_id}})
        final = LoomState.model_validate(raw)
    finally:
        tracer.end_trace()
    return final
