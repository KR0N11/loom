"""Executor interface (Strategy): the agent hands over code, gets rows or an error back.

Approach: the request/result are plain JSON-able models so the same payload can
go to a local subprocess, a Docker container, or an AWS Lambda without change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class SandboxRequest(BaseModel):
    language: str  # sql | pandas
    code: str
    duckdb_path: str
    row_limit: int = 10_000
    timeout_s: int = 30
    memory_mb: int = 1024
    # For pandas: tables to pre-load as DataFrames named after the table.
    tables: list[str] = Field(default_factory=list)


class SandboxResult(BaseModel):
    ok: bool
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: str | None = None
    elapsed_s: float = 0.0
    backend: str = ""


class SandboxExecutor(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, request: SandboxRequest) -> SandboxResult: ...
