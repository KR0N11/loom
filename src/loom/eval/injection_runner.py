"""Verifier catch-rate measurement via deliberately corrupted executor output.

Approach: reuse the normal runner with `injection_modes` set; the supervisor
applies `loom.agents.injection.inject` before the verifier runs. A catch is a
FAIL verdict on an injected run; a false alarm is a FAIL verdict on a clean run
whose answer was actually correct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from loom.config import Settings
from loom.eval.benchmark import Benchmark
from loom.eval.runner import EvalResults, RunQuestion, run_benchmark
from loom.llm.base import LLMProvider

ALL_MODES: tuple[str, ...] = ("scale", "sign", "off_by_year", "drop_rows", "swap_unit", "zero")


def resolve_modes(spec: str | list[str] | None) -> tuple[str, ...]:
    """Accept 'all', a comma list, or a list; validate against the known modes."""
    if spec is None:
        return ()
    if isinstance(spec, str):
        items = (
            ALL_MODES
            if spec.strip().lower() == "all"
            else tuple(s.strip() for s in spec.split(","))
        )
    else:
        items = tuple(spec)
    unknown = [m for m in items if m not in ALL_MODES]
    # A typo in a mode name would silently measure nothing, so fail early.
    if unknown:
        raise ValueError(f"unknown injection modes {unknown}; known: {ALL_MODES}")
    return items


class CatchRateReport(BaseModel):
    per_mode: dict[str, float | None]
    overall: float | None
    false_alarm_rate: float | None
    n_injected: int


def catch_rates(results: EvalResults) -> CatchRateReport:
    modes = sorted({r.injection for r in results.injected_rows() if r.injection})
    return CatchRateReport(
        per_mode={m: results.catch_rate(m) for m in modes},
        overall=results.catch_rate(),
        false_alarm_rate=results.false_alarm_rate(),
        n_injected=len(results.injected_rows()),
    )


def run_injection_suite(
    benchmark: Benchmark,
    settings: Settings,
    provider: LLMProvider | None,
    sample_n: int | None = 20,
    modes: tuple[str, ...] = ALL_MODES,
    out_dir: Path | None = None,
    run_question: RunQuestion | None = None,
    **kwargs: Any,
) -> tuple[EvalResults, CatchRateReport]:
    results = run_benchmark(
        benchmark,
        settings=settings,
        provider=provider,
        sample_n=sample_n,
        injection_modes=modes,
        out_dir=out_dir,
        run_question=run_question,
        **kwargs,
    )
    return results, catch_rates(results)
