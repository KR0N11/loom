"""Helpers for agent tests: a direct-DuckDB sandbox stand-in and canned state builders."""

from __future__ import annotations

import json
import time
from decimal import Decimal

import duckdb

from loom.agents.schemas import (
    AnalysisPlan,
    AnalysisType,
    ExecutionLanguage,
    ExecutionResult,
    KeyNumber,
    PlanStep,
    QueryRecord,
    RetrievedDoc,
    SemanticContext,
)
from loom.agents.state import LoomState
from loom.sandbox.base import SandboxExecutor, SandboxRequest, SandboxResult


class DirectDuckDBExecutor(SandboxExecutor):
    """Runs SQL straight in DuckDB (no isolation) so agent tests do not depend on the sandbox package."""

    name = "direct"

    def run(self, request: SandboxRequest) -> SandboxResult:
        start = time.perf_counter()
        if request.language != "sql":
            return SandboxResult(
                ok=False, error="direct executor supports sql only", backend=self.name
            )
        conn = duckdb.connect(request.duckdb_path, read_only=True)
        try:
            cur = conn.execute(request.code)
            columns = [d[0] for d in cur.description]
            rows = [list(r) for r in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001 - surface any DuckDB error as a sandbox error
            return SandboxResult(ok=False, error=str(exc), backend=self.name)
        finally:
            conn.close()
        # DuckDB returns Decimal for sums over DECIMAL columns; make cells JSON-native like the sandbox does.
        rows = json.loads(
            json.dumps(rows, default=lambda v: float(v) if isinstance(v, Decimal) else str(v))
        )
        return SandboxResult(
            ok=True,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            elapsed_s=time.perf_counter() - start,
            backend=self.name,
        )


def demo_semantic() -> SemanticContext:
    doc = RetrievedDoc(
        doc_id="demo.column.demo_sales.amount",
        kind="column",
        name="demo_sales.amount",
        text="Column demo_sales.amount (DOUBLE): sale amount. Unit: usd.",
        score=1.0,
    )
    return SemanticContext(
        tables=["demo_sales"],
        columns=["sale_id", "province", "sale_date", "amount"],
        metrics=[],
        joins=[],
        docs=[doc],
        context_text="Table demo_sales(sale_id INT, province VARCHAR, sale_date DATE, amount DOUBLE usd)",
    )


def demo_plan(analysis_type: AnalysisType = AnalysisType.LOOKUP, steps: int = 1) -> AnalysisPlan:
    return AnalysisPlan(
        question="total sales",
        dataset="demo",
        analysis_type=analysis_type,
        steps=[
            PlanStep(step_id=i + 1, description=f"step {i + 1}", needs_tables=["demo_sales"])
            for i in range(steps)
        ],
    )


def demo_execution(
    value: float = 510.0, rows: list[list] | None = None, unit: str = "usd"
) -> ExecutionResult:
    code = "SELECT sum(amount) AS total_sales FROM demo_sales"
    return ExecutionResult(
        step_id=1,
        language=ExecutionLanguage.SQL,
        code=code,
        columns=["total_sales"],
        rows=rows if rows is not None else [[value]],
        row_count=len(rows) if rows is not None else 1,
        key_numbers=[
            KeyNumber(name="total_sales", value=value, unit=unit, derivation="sum(amount)")
        ],
        queries=[
            QueryRecord(attempt=1, language=ExecutionLanguage.SQL, code=code, ok=True, row_count=1)
        ],
        summary="Total sales are 510.",
    )


def demo_state(**overrides) -> LoomState:
    base = dict(question="total sales", dataset="demo", thread_id="t-test")
    base.update(overrides)
    return LoomState(**base)


def j(obj: dict) -> str:
    return json.dumps(obj)
