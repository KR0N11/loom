"""Typed LangGraph state shared by the supervisor and workers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from loom.agents.schemas import (
    AnalysisPlan,
    ExecutionResult,
    Memo,
    SemanticContext,
    VerificationReport,
)
from loom.llm.base import LLMUsage


class LoomState(BaseModel):
    """Everything the graph knows about one question. Each node fills its own slot."""

    question: str
    dataset: str
    thread_id: str = "default"
    plan: AnalysisPlan | None = None
    semantic: SemanticContext | None = None
    executions: list[ExecutionResult] = Field(default_factory=list)
    verification: VerificationReport | None = None
    memo: Memo | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    errors: list[str] = Field(default_factory=list)
    # Eval-only: when set, the executor output is corrupted before verification.
    injection: str | None = None
