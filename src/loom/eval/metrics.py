"""Pure scoring functions used by the runner and the report.

Approach: every metric is a small side-effect-free function over plain data,
so it can be unit-tested exhaustively and reused by baselines and the gate.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any


def _norm_value(value: Any) -> Any:
    """Normalise a cell so DuckDB / pandas / LLM formatting differences do not count as misses."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        # NaN never equals NaN, so map it to None explicitly.
        if isinstance(value, float) and math.isnan(value):
            return None
        if value == 0:
            return 0.0
        return float(f"{float(value):.6g}")
    text = str(value).strip().lower()
    # A numeric string should compare equal to the number it represents.
    try:
        return _norm_value(float(text))
    except ValueError:
        return text


def _norm_rows(rows: list[list[Any]]) -> list[tuple[Any, ...]]:
    return sorted(
        (tuple(_norm_value(v) for v in row) for row in rows),
        key=lambda r: tuple("" if v is None else str(v) for v in r),
    )


def execution_accuracy(pred_rows: list[list[Any]], gold_rows: list[list[Any]]) -> bool:
    """Exact multiset match of normalised rows (column order matters, row order does not)."""
    if not gold_rows:
        return not pred_rows
    return _norm_rows(pred_rows) == _norm_rows(gold_rows)


def numeric_cells(rows: list[list[Any]], max_rows: int = 5) -> list[float]:
    """Numeric values from the first rows (ints, floats, Decimals; never bools)."""
    out: list[float] = []
    for row in rows[:max_rows]:
        for v in row:
            if isinstance(v, bool):
                continue
            if isinstance(v, int | float | Decimal):
                out.append(float(v))
    return out


def within_tolerance(
    pred: float, gold: float, rel_tol: float = 0.01, abs_floor: float = 1e-9
) -> bool:
    if pred is None or gold is None:
        return False
    if isinstance(pred, float) and math.isnan(pred):
        return False
    return abs(pred - gold) <= max(abs(gold) * rel_tol, abs_floor)


def numeric_match(
    pred_numbers: dict[str, float] | list[float],
    gold_answer: dict[str, float],
    rel_tol: float = 0.01,
) -> tuple[bool, float]:
    """Return (main number matched, fraction of gold numbers matched by some prediction)."""
    preds = list(pred_numbers.values()) if isinstance(pred_numbers, dict) else list(pred_numbers)
    if not gold_answer:
        return False, 0.0
    golds = list(gold_answer.values())
    main_ok = any(within_tolerance(p, golds[0], rel_tol) for p in preds)
    covered = sum(1 for g in golds if any(within_tolerance(p, g, rel_tol) for p in preds))
    return main_ok, covered / len(golds)


def retrieval_precision_recall(
    retrieved: set[str], expected: set[str]
) -> tuple[float | None, float | None]:
    """Precision/recall of retrieved table+metric names vs the ones the gold SQL needs."""
    if not expected:
        return None, None
    hits = len(retrieved & expected)
    precision = hits / len(retrieved) if retrieved else 0.0
    return precision, hits / len(expected)


def verifier_catch(overall: str | None, injected: bool) -> bool | None:
    """True when an injected corruption was flagged FAIL; None when nothing was injected."""
    if not injected:
        return None
    return overall == "fail"


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; returns 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[idx]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
