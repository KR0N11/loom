"""Benchmark generation: guaranteed-correct templates plus LLM-proposed candidates.

Approach: a `Template` is a question/SQL pair with `{param}` slots; parameter
values are sampled from the live DuckDB tables so every instance is grounded in
real data, and the SQL is executed to materialise gold rows and key numbers.
`LLMGenerator` asks the model for candidates and keeps only those whose SQL
executes cleanly. Nothing is marked reviewed by generation; a human flips that.
"""

from __future__ import annotations

import datetime as dt
import itertools
import logging
import math
import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import duckdb
from pydantic import BaseModel, ConfigDict, Field

from loom.eval.benchmark import Benchmark, BenchmarkQuestion, Difficulty
from loom.llm.base import LLMMessage, LLMProvider, LLMUsage

log = logging.getLogger(__name__)

MAX_GOLD_ROWS = 50


@dataclass
class Template:
    key: str
    dataset: str
    difficulty: Difficulty
    analysis_type: str
    question: str
    sql: str
    expected_tables: list[str]
    expected_metrics: list[str] = field(default_factory=list)
    # param name -> one-column SQL listing candidate values
    params: dict[str, str] = field(default_factory=dict)
    max_variants: int = 4


class GoldResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    answer: dict[str, float]


class GoldSQLError(ValueError):
    """Gold SQL failed validation (error, empty, too many rows or no numeric column)."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def materialise_gold(conn: duckdb.DuckDBPyConnection, sql: str) -> GoldResult:
    """Run gold SQL and build gold rows plus key numbers from the first row."""
    try:
        rel = conn.execute(sql)
        columns = [d[0] for d in rel.description]
        raw = rel.fetchmany(MAX_GOLD_ROWS + 1)
    except duckdb.Error as exc:
        raise GoldSQLError(f"gold SQL failed: {exc}") from exc
    # Reject queries that cannot be scored deterministically.
    if not raw:
        raise GoldSQLError("gold SQL returned no rows")
    if len(raw) > MAX_GOLD_ROWS:
        raise GoldSQLError(f"gold SQL returned more than {MAX_GOLD_ROWS} rows")
    rows = [[_json_safe(v) for v in row] for row in raw]
    answer: dict[str, float] = {}
    for col, value in zip(columns, rows[0], strict=True):
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        answer[col] = float(value)
    if not answer:
        raise GoldSQLError("gold SQL has no numeric column in its first row")
    return GoldResult(columns=columns, rows=rows, answer=answer)


def _sql_literal(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("'", "''")
    return str(value)


class TemplateGenerator:
    """Instantiate templates against a DuckDB connection."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, seed: int = 0) -> None:
        self.conn = conn
        self.rng = random.Random(seed)

    def _candidates(self, sql: str) -> list[Any]:
        try:
            return [r[0] for r in self.conn.execute(sql).fetchall() if r[0] is not None]
        except duckdb.Error as exc:
            log.warning("param sampler failed (%s): %s", sql, exc)
            return []

    def instantiate(self, template: Template) -> list[BenchmarkQuestion]:
        # Sample every parameter from the live data so questions reference real values.
        pools = {name: self._candidates(sql) for name, sql in template.params.items()}
        if any(not pool for pool in pools.values()):
            log.warning("template %s skipped: empty parameter pool", template.key)
            return []
        names = list(pools)
        combos = list(itertools.product(*(pools[n] for n in names)))
        self.rng.shuffle(combos)
        questions: list[BenchmarkQuestion] = []
        for combo in combos:
            if len(questions) >= template.max_variants:
                break
            values = dict(zip(names, combo, strict=True))
            # Two slots drawn from the same pool must differ (e.g. year_a vs year_b).
            same_pool = [
                (a, b)
                for a, b in itertools.combinations(names, 2)
                if template.params[a] == template.params[b]
            ]
            if any(values[a] == values[b] for a, b in same_pool):
                continue
            # "from {x_a} to {x_b}" must read chronologically, so require a < b.
            if any(
                a.endswith("_a") and b.endswith("_b") and not values[a] < values[b]
                for a, b in same_pool
            ):
                continue
            sql = template.sql.format(**{k: _sql_literal(v) for k, v in values.items()})
            try:
                gold = materialise_gold(self.conn, sql)
            except GoldSQLError as exc:
                log.warning("template %s variant %s rejected: %s", template.key, values, exc)
                continue
            suffix = "_".join(str(v) for v in combo).replace(" ", "-").lower()
            questions.append(
                BenchmarkQuestion(
                    id=f"{template.dataset}-{template.key}-{suffix}"[:80],
                    dataset=template.dataset,
                    question=template.question.format(**values),
                    difficulty=template.difficulty,
                    analysis_type=template.analysis_type,
                    gold_sql=sql.strip(),
                    gold_answer=gold.answer,
                    gold_columns=gold.columns,
                    gold_rows=gold.rows,
                    expected_tables=list(template.expected_tables),
                    expected_metrics=list(template.expected_metrics),
                    generator="template",
                )
            )
        return questions

    def generate(self, templates: list[Template]) -> list[BenchmarkQuestion]:
        out: list[BenchmarkQuestion] = []
        seen: set[str] = set()
        for t in templates:
            for q in self.instantiate(t):
                # Distinct parameterisations can collide on id; keep the first.
                if q.id in seen:
                    continue
                seen.add(q.id)
                out.append(q)
        return out


# ---------- LLM candidate generation ----------


class LLMCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    gold_sql: str
    analysis_type: str
    difficulty: Difficulty
    expected_tables: list[str] = Field(default_factory=list)
    expected_metrics: list[str] = Field(default_factory=list)


class LLMCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[LLMCandidate]


LLM_GEN_SYSTEM = (
    "You write benchmark questions for a data analysis assistant. Given a semantic layer "
    "(tables, columns, metrics, joins) for DuckDB, propose realistic business questions with "
    "the exact DuckDB SQL that answers each one. Rules: use only listed tables and columns; "
    "the SQL must return between 1 and 50 rows and at least one numeric column; the first "
    "row's first numeric column is the headline answer; difficulty easy = single filter or "
    "aggregate lookup, medium = trend over time or A-vs-B comparison, hard = multi-step "
    "driver analysis with CTEs; analysis_type is one of lookup, trend, comparison, driver."
)


class LLMGenerator:
    """Ask the LLM for candidates, keep only the ones whose SQL validates against the data."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, provider: LLMProvider) -> None:
        self.conn = conn
        self.provider = provider
        self.usage = LLMUsage()

    def generate(
        self, dataset: str, context_text: str, n: int, batch: int = 10
    ) -> list[BenchmarkQuestion]:
        questions: list[BenchmarkQuestion] = []
        attempts = 0
        # Ask in batches; stop when we have n valid questions or the model keeps failing.
        while len(questions) < n and attempts < math.ceil(n / batch) * 3:
            attempts += 1
            want = min(batch, n - len(questions))
            out, usage = self.provider.complete_json(
                LLM_GEN_SYSTEM,
                [
                    LLMMessage(
                        role="user",
                        content=f"Dataset: {dataset}\n\nSemantic layer:\n{context_text}\n\n"
                        f"Propose {want} new questions, mixing difficulties.",
                    )
                ],
                LLMCandidates,
            )
            self.usage = self.usage + usage
            for i, cand in enumerate(out.candidates):
                try:
                    gold = materialise_gold(self.conn, cand.gold_sql)
                except GoldSQLError as exc:
                    log.warning("llm candidate rejected: %s", exc)
                    continue
                questions.append(
                    BenchmarkQuestion(
                        id=f"{dataset}-llm-{attempts:02d}-{i:02d}",
                        dataset=dataset,
                        question=cand.question,
                        difficulty=cand.difficulty,
                        analysis_type=cand.analysis_type,
                        gold_sql=cand.gold_sql.strip(),
                        gold_answer=gold.answer,
                        gold_columns=gold.columns,
                        gold_rows=gold.rows,
                        expected_tables=cand.expected_tables,
                        expected_metrics=cand.expected_metrics,
                        generator="llm",
                    )
                )
                if len(questions) >= n:
                    break
        return questions


def build_benchmark(questions: list[BenchmarkQuestion]) -> Benchmark:
    return Benchmark(questions=questions)
