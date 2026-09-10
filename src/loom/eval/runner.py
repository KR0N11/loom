"""Benchmark runner: drives the supervisor graph per question and scores the result.

Approach: one `EvalRow` per (question, injection mode). Rows are appended to
results.json after every question so a long run can be resumed, and every
exception is captured as a row with `error` rather than aborting the run.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from loom.config import Settings, get_settings
from loom.eval.benchmark import Benchmark, BenchmarkQuestion
from loom.eval.metrics import (
    execution_accuracy,
    numeric_cells,
    numeric_match,
    retrieval_precision_recall,
    verifier_catch,
)
from loom.llm.base import LLMProvider

log = logging.getLogger(__name__)


class EvalRow(BaseModel):
    id: str
    dataset: str
    difficulty: str
    analysis_type: str
    injection: str | None = None
    planner_type: str | None = None
    exec_ok: bool = False
    exec_acc: bool = False
    numeric_match: bool = False
    gold_coverage: float = 0.0
    retrieval_p: float | None = None
    retrieval_r: float | None = None
    verifier_overall: str | None = None
    caught: bool | None = None
    zero_correction: bool = False
    cost_usd: float = 0.0
    latency_s: float = 0.0
    llm_latency_s: float = 0.0
    error: str | None = None


class BaselineRow(BaseModel):
    baseline: str
    id: str
    difficulty: str
    dataset: str
    exec_ok: bool = False
    exec_acc: bool = False
    numeric_match: bool = False
    latency_s: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None


class EvalResults(BaseModel):
    started_at: str
    model: str = ""
    rows: list[EvalRow] = Field(default_factory=list)
    baseline_rows: list[BaselineRow] = Field(default_factory=list)
    routing: dict[str, float] = Field(default_factory=dict)

    # ----- aggregate helpers used by report and gate -----

    def main_rows(self) -> list[EvalRow]:
        return [r for r in self.rows if r.injection is None]

    def injected_rows(self) -> list[EvalRow]:
        return [r for r in self.rows if r.injection is not None]

    def execution_accuracy(self, rows: list[EvalRow] | None = None) -> float:
        rows = self.main_rows() if rows is None else rows
        return _rate([r.exec_acc for r in rows])

    def numeric_match_rate(self, rows: list[EvalRow] | None = None) -> float:
        rows = self.main_rows() if rows is None else rows
        return _rate([r.numeric_match for r in rows])

    def zero_correction_rate(self, rows: list[EvalRow] | None = None) -> float:
        rows = self.main_rows() if rows is None else rows
        return _rate([r.zero_correction for r in rows])

    def catch_rate(self, mode: str | None = None) -> float | None:
        rows = [r for r in self.injected_rows() if mode is None or r.injection == mode]
        flagged = [r.caught for r in rows if r.caught is not None]
        return _rate(flagged) if flagged else None

    def false_alarm_rate(self) -> float | None:
        rows = [r for r in self.main_rows() if r.exec_acc and r.verifier_overall is not None]
        if not rows:
            return None
        return _rate([r.verifier_overall == "fail" for r in rows])

    def summary(self) -> dict[str, Any]:
        rows = self.main_rows()
        precision = [r.retrieval_p for r in rows if r.retrieval_p is not None]
        recall = [r.retrieval_r for r in rows if r.retrieval_r is not None]
        return {
            "n": len(rows),
            "execution_accuracy": self.execution_accuracy(),
            "numeric_match": self.numeric_match_rate(),
            "zero_correction": self.zero_correction_rate(),
            "retrieval_precision": sum(precision) / len(precision) if precision else None,
            "retrieval_recall": sum(recall) / len(recall) if recall else None,
            "verifier_catch_rate": self.catch_rate(),
            "false_alarm_rate": self.false_alarm_rate(),
            "mean_cost_usd": sum(r.cost_usd for r in rows) / len(rows) if rows else 0.0,
            "mean_latency_s": sum(r.latency_s for r in rows) / len(rows) if rows else 0.0,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> EvalResults:
        return cls.model_validate_json(path.read_text())


def _rate(flags: list[bool]) -> float:
    return sum(1 for f in flags if f) / len(flags) if flags else 0.0


def _strip_prefix(name: str, dataset: str) -> str:
    name = name.lower().strip()
    prefix = f"{dataset}_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def score_state(question: BenchmarkQuestion, state: Any, injection: str | None) -> EvalRow:
    """Turn a finished LoomState into a scored row (pure; no I/O)."""
    row = EvalRow(
        id=question.id,
        dataset=question.dataset,
        difficulty=question.difficulty.value,
        analysis_type=question.analysis_type,
        injection=injection,
    )
    plan = getattr(state, "plan", None)
    if plan is not None:
        row.planner_type = str(getattr(plan, "analysis_type", None) or "")
    executions = list(getattr(state, "executions", []) or [])
    row.exec_ok = bool(executions)
    if executions:
        # The last step holds the final answer; earlier steps are intermediates.
        final = executions[-1]
        row.exec_acc = execution_accuracy(list(final.rows), question.gold_rows)
        preds = [kn.value for ex in executions for kn in ex.key_numbers]
        # Numeric cells of the final result count too: a lookup may not name a key number.
        preds += numeric_cells(list(final.rows))
        row.numeric_match, row.gold_coverage = numeric_match(preds, question.gold_answer)
    semantic = getattr(state, "semantic", None)
    if semantic is not None:
        retrieved = {
            _strip_prefix(d.name, question.dataset)
            for d in semantic.docs
            if d.kind in ("table", "metric")
        }
        expected = {
            _strip_prefix(t, question.dataset)
            for t in question.expected_tables + question.expected_metrics
        }
        row.retrieval_p, row.retrieval_r = retrieval_precision_recall(retrieved, expected)
    verification = getattr(state, "verification", None)
    if verification is not None:
        row.verifier_overall = str(verification.overall.value)
    row.caught = verifier_catch(row.verifier_overall, injection is not None)
    row.zero_correction = row.numeric_match and row.verifier_overall not in ("fail", None)
    usage = getattr(state, "usage", None)
    if usage is not None:
        row.cost_usd = float(usage.cost_usd)
        row.llm_latency_s = float(usage.latency_s)
    errors = list(getattr(state, "errors", []) or [])
    if errors and not executions:
        row.error = "; ".join(errors)[:500]
    return row


RunQuestion = Callable[..., Any]


def _default_run_question() -> RunQuestion:
    from loom.agents.supervisor import run_question

    return run_question


def run_benchmark(
    benchmark: Benchmark,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    sample_n: int | None = None,
    seed: int = 0,
    injection_modes: tuple[str, ...] = (),
    injection_sample_n: int | None = None,
    out_dir: Path | None = None,
    resume: bool = False,
    on_progress: Callable[[EvalRow], None] | None = None,
    run_question: RunQuestion | None = None,
) -> EvalResults:
    """Run every (question, injection) pair, scoring and persisting as it goes."""
    settings = settings or get_settings()
    run_question = run_question or _default_run_question()
    out_dir = out_dir or (
        Path(settings.output_dir) / "eval" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    results_path = out_dir / "results.json"
    # Resume picks up the previous file and skips rows already scored.
    if resume and results_path.exists():
        results = EvalResults.load(results_path)
    else:
        results = EvalResults(
            started_at=dt.datetime.now().isoformat(timespec="seconds"),
            model=getattr(provider, "model", "") if provider else settings.llm_model,
        )
    done = {(r.id, r.injection) for r in results.rows}
    questions = benchmark.sample(sample_n, seed).questions
    # Injection runs are expensive, so they use a (possibly smaller) stratified subset.
    injection_ids = {
        q.id for q in Benchmark(questions=questions).sample(injection_sample_n, seed).questions
    }
    jobs: list[tuple[BenchmarkQuestion, str | None]] = []
    for q in questions:
        jobs.append((q, None))
        if q.id in injection_ids:
            jobs.extend((q, mode) for mode in injection_modes)
    for question, mode in jobs:
        if (question.id, mode) in done:
            continue
        row = _run_one(question, mode, settings, provider, run_question)
        results.rows.append(row)
        results.save(results_path)
        if on_progress:
            on_progress(row)
    return results


def _run_one(
    question: BenchmarkQuestion,
    mode: str | None,
    settings: Settings,
    provider: LLMProvider | None,
    run_question: RunQuestion,
) -> EvalRow:
    start = time.perf_counter()
    try:
        state = run_question(
            question.question,
            question.dataset,
            settings=settings,
            provider=provider,
            injection=mode,
            thread_id=f"eval-{question.id}-{mode or 'clean'}",
        )
        row = score_state(question, state, mode)
    except Exception as exc:  # noqa: BLE001 - one bad question must not sink the run
        log.exception("question %s failed", question.id)
        row = EvalRow(
            id=question.id,
            dataset=question.dataset,
            difficulty=question.difficulty.value,
            analysis_type=question.analysis_type,
            injection=mode,
            error=f"{type(exc).__name__}: {exc}"[:500],
        )
    row.latency_s = time.perf_counter() - start
    return row
