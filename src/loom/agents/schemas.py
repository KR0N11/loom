"""Strict output contracts for every agent.

Approach: each agent returns exactly one of these models. `extra="forbid"` means
an LLM cannot smuggle in fields we do not handle, and contract tests pin the
shape so a prompt change that breaks downstream consumers fails in CI.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Planner ----------


class AnalysisType(StrEnum):
    LOOKUP = "lookup"
    TREND = "trend"
    COMPARISON = "comparison"
    DRIVER = "driver"


class PlanStep(StrictModel):
    step_id: int = Field(ge=1)
    description: str
    needs_tables: list[str] = Field(default_factory=list)
    depends_on: list[int] = Field(default_factory=list)


class AnalysisPlan(StrictModel):
    question: str
    analysis_type: AnalysisType
    dataset: str
    steps: list[PlanStep] = Field(min_length=1)
    metrics_needed: list[str] = Field(default_factory=list)
    time_range_hint: str | None = None
    assumptions: list[str] = Field(default_factory=list)


# ---------- Semantic layer ----------


class RetrievedDoc(StrictModel):
    doc_id: str
    kind: str  # table | column | metric | join
    name: str
    text: str
    score: float


class SemanticContext(StrictModel):
    tables: list[str]
    columns: list[str]
    metrics: list[str]
    joins: list[str]
    docs: list[RetrievedDoc]
    context_text: str


# ---------- Executor ----------


class ExecutionLanguage(StrEnum):
    SQL = "sql"
    PANDAS = "pandas"


class QueryRecord(StrictModel):
    """One executed query, kept for the audit appendix."""

    attempt: int
    language: ExecutionLanguage
    code: str
    ok: bool
    error: str | None = None
    row_count: int = 0
    elapsed_s: float = 0.0


class KeyNumber(StrictModel):
    """A headline figure the verifier will re-derive."""

    name: str
    value: float
    unit: str = ""
    derivation: str = ""


class ExecutionResult(StrictModel):
    step_id: int
    language: ExecutionLanguage
    code: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    key_numbers: list[KeyNumber] = Field(default_factory=list)
    queries: list[QueryRecord] = Field(default_factory=list)
    summary: str = ""


# ---------- Verifier ----------


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIPPED = "skipped"


class VerificationCheck(StrictModel):
    name: str
    status: CheckStatus
    detail: str
    expected: float | None = None
    observed: float | None = None


class VerificationReport(StrictModel):
    overall: CheckStatus
    checks: list[VerificationCheck]
    reconciled_numbers: list[KeyNumber] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    queries: list[QueryRecord] = Field(default_factory=list)


# ---------- Narrator ----------


class ChartSpec(StrictModel):
    title: str
    kind: str  # bar | line
    x: str
    y: str
    path: str | None = None


class Memo(StrictModel):
    title: str
    headline: str
    findings: list[str]
    recommendation: str
    caveats: list[str] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    appendix_queries: list[QueryRecord] = Field(default_factory=list)
    verification_summary: str = ""


# ---------- LLM-facing partial schemas (what the model must emit) ----------


class PlannerOutput(StrictModel):
    analysis_type: AnalysisType
    steps: list[PlanStep] = Field(min_length=1)
    metrics_needed: list[str] = Field(default_factory=list)
    time_range_hint: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class ExecutorOutput(StrictModel):
    language: ExecutionLanguage
    code: str
    rationale: str = ""


class KeyNumbersOutput(StrictModel):
    key_numbers: list[KeyNumber]
    summary: str


class VerifierOutput(StrictModel):
    alternate_sql: str
    rationale: str = ""


class NarratorOutput(StrictModel):
    title: str
    headline: str
    findings: list[str]
    recommendation: str
    caveats: list[str] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
