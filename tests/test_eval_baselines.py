"""Baselines with injected model functions (no weights downloaded)."""

from __future__ import annotations

from pathlib import Path

import duckdb

from loom.config import Settings
from loom.eval.baselines.compare import comparison_table
from loom.eval.baselines.small_sql import QwenSQLBaseline, extract_sql
from loom.eval.baselines.sqlcoder import SQLCoderBaseline
from loom.eval.baselines.tapas import TapasLookupBaseline, tapas_number
from loom.eval.baselines.zero_shot_router import routing_accuracy
from loom.eval.benchmark import Benchmark, BenchmarkQuestion, Difficulty
from loom.eval.runner import EvalResults, EvalRow
from loom.sandbox.base import SandboxExecutor, SandboxRequest, SandboxResult


class DirectSandbox(SandboxExecutor):
    """Runs SQL straight in DuckDB; keeps baseline tests independent of the real sandbox."""

    name = "direct"

    def run(self, request: SandboxRequest) -> SandboxResult:
        conn = duckdb.connect(request.duckdb_path, read_only=True)
        try:
            rel = conn.execute(request.code)
            cols = [d[0] for d in rel.description]
            rows = [list(r) for r in rel.fetchall()]
            return SandboxResult(ok=True, columns=cols, rows=rows, row_count=len(rows))
        except duckdb.Error as exc:
            return SandboxResult(ok=False, error=str(exc))
        finally:
            conn.close()


def _question() -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id="demo-total-on",
        dataset="demo",
        question="What is the total sales amount in ON?",
        difficulty=Difficulty.EASY,
        analysis_type="lookup",
        gold_sql="SELECT sum(amount) AS total_amount FROM demo_sales WHERE province = 'ON'",
        gold_answer={"total_amount": 250.0},
        gold_columns=["total_amount"],
        gold_rows=[[250.0]],
        expected_tables=["demo_sales"],
    )


# Proves SQL extraction handles fences, bare statements and trailing semicolons.
def test_extract_sql() -> None:
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("Sure: SELECT 2 FROM t;") == "SELECT 2 FROM t"
    assert extract_sql("WITH x AS (SELECT 1) SELECT * FROM x").startswith("WITH")


# Proves the Qwen baseline runs generated SQL and scores it like the pipeline.
def test_qwen_baseline_scores(settings: Settings, tiny_db: Path) -> None:
    seen: list[str] = []

    def generate(prompt: str) -> str:
        seen.append(prompt)
        return "```sql\nSELECT sum(amount) AS total FROM demo_sales WHERE province = 'ON'\n```"

    base = QwenSQLBaseline(settings, DirectSandbox(), generate=generate)
    row = base.answer(_question(), "Table demo_sales(province, amount)")
    assert row.exec_ok and row.exec_acc and row.numeric_match and row.error is None
    assert "Table demo_sales" in seen[0] and "total sales amount in ON" in seen[0]


# Proves a bad generation is recorded as a failure row, not an exception.
def test_qwen_baseline_bad_sql(settings: Settings, tiny_db: Path) -> None:
    base = QwenSQLBaseline(
        settings, DirectSandbox(), generate=lambda p: "SELECT nope FROM demo_sales"
    )
    row = base.answer(_question(), "")
    assert not row.exec_ok and row.error


# Proves sqlcoder uses its own prompt format but the same scoring path.
def test_sqlcoder_prompt(settings: Settings, tiny_db: Path) -> None:
    base = SQLCoderBaseline(settings, DirectSandbox(), generate=lambda p: "SELECT 250.0 AS total")
    assert "[QUESTION]" in base.build_prompt("q", "schema")
    assert base.answer(_question(), "").numeric_match


# Proves TAPAS output parsing and the easy-only guard.
def test_tapas_baseline(settings: Settings, tiny_db: Path) -> None:
    assert tapas_number({"aggregator": "SUM", "cells": ["100", "150"]}) == 250.0
    assert tapas_number({"aggregator": "COUNT", "cells": ["a", "b"]}) == 2.0
    assert tapas_number({"aggregator": "NONE", "cells": ["n/a"]}) is None
    calls: list[str] = []

    def fake(table, query):  # type: ignore[no-untyped-def]
        calls.append(query)
        assert len(table) == 5 and table.map(lambda v: isinstance(v, str)).all().all()
        return {"aggregator": "SUM", "cells": ["100.0", "150.0"]}

    base = TapasLookupBaseline(settings, tapas=fake)
    row = base.answer(_question())
    assert row.exec_ok and row.numeric_match and calls
    hard = _question().model_copy(update={"difficulty": Difficulty.HARD})
    assert TapasLookupBaseline(settings, tapas=fake).answer(hard).error.startswith("skipped")


# Proves routing accuracy compares the classifier with the gold analysis type.
def test_routing_accuracy() -> None:
    bench = Benchmark(
        questions=[
            _question(),
            _question().model_copy(update={"id": "b", "analysis_type": "trend"}),
        ]
    )
    assert routing_accuracy(bench, lambda q: "lookup") == 0.5
    assert routing_accuracy(Benchmark(questions=[]), lambda q: "lookup") == 0.0


# Proves the comparison table lists the LLM and each baseline, skipping skipped rows.
def test_comparison_table() -> None:
    results = EvalResults(
        started_at="t",
        model="fake",
        rows=[
            EvalRow(
                id="a", dataset="demo", difficulty="easy", analysis_type="lookup", exec_acc=True
            )
        ],
    )
    from loom.eval.runner import BaselineRow

    results.baseline_rows = [
        BaselineRow(baseline="tapas-wtq", id="a", difficulty="easy", dataset="demo", exec_acc=True),
        BaselineRow(
            baseline="tapas-wtq", id="b", difficulty="hard", dataset="demo", error="skipped: x"
        ),
    ]
    table = comparison_table(results)
    assert "| tapas-wtq | 1 | 100.0% |" in table and "LLM pipeline (fake) | 1 |" in table
