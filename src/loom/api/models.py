"""Request/response models for the HTTP API.

Approach: the API re-uses the agent output schemas (Memo, VerificationReport,
LLMUsage) verbatim so the wire format is the same contract the agents are
tested against; only the request envelope is API-specific.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from loom.agents.schemas import Memo, VerificationReport
from loom.llm.base import LLMUsage


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    dataset: str
    injection: str | None = None


class AskResponse(BaseModel):
    thread_id: str
    memo: Memo | None
    verification: VerificationReport | None
    usage: LLMUsage
    errors: list[str]
    memo_markdown: str


class DatasetInfo(BaseModel):
    name: str
    description: str


class HealthResponse(BaseModel):
    status: str
    datasets: list[str]
    duckdb_exists: bool
