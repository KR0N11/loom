"""TAPAS table-QA baseline for easy lookups.

Approach: TAPAS reads a small table as text, so it only gets a <=100-row slice
of the one table the question needs. It is a cheap lower bound, not a
competitor for joins or trends; medium/hard questions are skipped.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import duckdb
import pandas as pd

from loom.config import Settings
from loom.eval.benchmark import BenchmarkQuestion, Difficulty
from loom.eval.metrics import numeric_match
from loom.eval.runner import BaselineRow

TapasFn = Callable[[pd.DataFrame, str], dict[str, Any]]


def tapas_number(output: dict[str, Any]) -> float | None:
    """Turn a TAPAS pipeline output into one number (aggregator over cells, else first cell)."""
    cells = [c for c in output.get("cells", []) if c not in (None, "")]
    values: list[float] = []
    for c in cells:
        try:
            values.append(float(str(c).replace(",", "")))
        except ValueError:
            continue
    aggregator = str(output.get("aggregator", "NONE")).upper()
    if aggregator == "COUNT":
        return float(len(cells))
    if not values:
        return None
    if aggregator == "SUM":
        return sum(values)
    if aggregator == "AVERAGE":
        return sum(values) / len(values)
    return values[0]


class TapasLookupBaseline:
    name = "tapas-wtq"

    def __init__(
        self,
        settings: Settings,
        tapas: TapasFn | None = None,
        max_rows: int = 100,
    ) -> None:
        self.settings = settings
        self._tapas = tapas
        self.max_rows = max_rows

    def _load(self) -> TapasFn:
        if self._tapas is not None:
            return self._tapas
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers/torch are not installed; install with `uv sync --extra ml`"
            ) from exc
        pipe = pipeline("table-question-answering", model=self.settings.tapas_model)

        def run(table: pd.DataFrame, query: str) -> dict[str, Any]:
            return dict(pipe(table=table, query=query))

        self._tapas = run
        return run

    def table_slice(self, question: BenchmarkQuestion) -> pd.DataFrame:
        table = question.expected_tables[0]
        conn = duckdb.connect(str(self.settings.duckdb_path), read_only=True)
        try:
            df = conn.execute(f"SELECT * FROM {table} LIMIT {self.max_rows}").df()
        finally:
            conn.close()
        # TAPAS expects every cell as a string.
        return df.astype(str)

    def answer(self, question: BenchmarkQuestion, context_text: str = "") -> BaselineRow:
        row = BaselineRow(
            baseline=self.name,
            id=question.id,
            difficulty=question.difficulty.value,
            dataset=question.dataset,
        )
        # Only easy single-table lookups are within TAPAS's design envelope.
        if question.difficulty != Difficulty.EASY or len(question.expected_tables) != 1:
            row.error = "skipped: not an easy single-table lookup"
            return row
        start = time.perf_counter()
        try:
            output = self._load()(self.table_slice(question), question.question)
            value = tapas_number(output)
            row.exec_ok = value is not None
            if value is not None:
                row.numeric_match, _ = numeric_match([value], question.gold_answer)
                row.exec_acc = row.numeric_match
        except Exception as exc:  # noqa: BLE001
            row.error = f"{type(exc).__name__}: {exc}"[:300]
        row.latency_s = time.perf_counter() - start
        return row
