"""Benchmark question schema and YAML persistence.

Approach: one `BenchmarkQuestion` per gold example, tagged by dataset,
difficulty and analysis type, with the gold SQL *and* the materialised gold
rows/numbers so scoring never depends on re-running the gold query. Stratified
sampling keeps a quick smoke run representative of the full set.
"""

from __future__ import annotations

import random
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BenchmarkQuestion(BaseModel):
    id: str
    dataset: str
    question: str
    difficulty: Difficulty
    analysis_type: str
    gold_sql: str
    gold_answer: dict[str, float] = Field(default_factory=dict)
    gold_columns: list[str] = Field(default_factory=list)
    gold_rows: list[list[Any]] = Field(default_factory=list)
    expected_tables: list[str] = Field(default_factory=list)
    expected_metrics: list[str] = Field(default_factory=list)
    reviewed: bool = False
    review_note: str = ""
    generator: str = "template"

    @property
    def main_number(self) -> tuple[str, float] | None:
        """The headline gold figure: first entry of gold_answer (insertion order)."""
        for name, value in self.gold_answer.items():
            return name, value
        return None


class Benchmark(BaseModel):
    questions: list[BenchmarkQuestion]

    def by_id(self, qid: str) -> BenchmarkQuestion:
        for q in self.questions:
            if q.id == qid:
                return q
        raise KeyError(qid)

    def sample(self, n: int | None, seed: int = 0) -> Benchmark:
        """Stratified sample by (dataset, difficulty): round-robin so every stratum is represented."""
        if n is None or n >= len(self.questions):
            return Benchmark(questions=list(self.questions))
        rng = random.Random(seed)
        strata: dict[tuple[str, str], list[BenchmarkQuestion]] = defaultdict(list)
        for q in self.questions:
            strata[(q.dataset, q.difficulty.value)].append(q)
        buckets = [rng.sample(v, len(v)) for _, v in sorted(strata.items())]
        picked: list[BenchmarkQuestion] = []
        # Round-robin across strata until n is reached so small strata are not starved.
        while len(picked) < n and any(buckets):
            for bucket in buckets:
                if bucket and len(picked) < n:
                    picked.append(bucket.pop())
        return Benchmark(questions=picked)

    def composition(self) -> dict[str, dict[str, int]]:
        """Counts per dataset, difficulty, analysis type and review status."""
        out: dict[str, dict[str, int]] = {
            "dataset": defaultdict(int),
            "difficulty": defaultdict(int),
            "analysis_type": defaultdict(int),
            "reviewed": defaultdict(int),
            "generator": defaultdict(int),
        }
        for q in self.questions:
            out["dataset"][q.dataset] += 1
            out["difficulty"][q.difficulty.value] += 1
            out["analysis_type"][q.analysis_type] += 1
            out["reviewed"]["reviewed" if q.reviewed else "unreviewed"] += 1
            out["generator"][q.generator] += 1
        return {k: dict(v) for k, v in out.items()}

    def save(self, path: Path, header: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = yaml.safe_dump(
            {"questions": [q.model_dump(mode="json") for q in self.questions]},
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )
        # A comment header documents provenance (generator, seed, review status) for auditors.
        text = "".join(f"# {line}\n" for line in header.splitlines()) + body if header else body
        path.write_text(text)

    @classmethod
    def load(cls, path: Path) -> Benchmark:
        data = yaml.safe_load(path.read_text()) or {}
        return cls(
            questions=[BenchmarkQuestion.model_validate(q) for q in data.get("questions", [])]
        )
