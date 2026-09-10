"""Eval-only corruption of executor output (injection mode).

Approach: to measure the verifier's catch rate we need known-wrong inputs. Each
mode applies one deterministic corruption to the ExecutionResults; the verifier
then re-derives the numbers and should flag the mismatch.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from loom.agents.schemas import ExecutionResult, KeyNumber


class InjectionMode(StrEnum):
    SCALE = "scale"
    SIGN = "sign"
    OFF_BY_YEAR = "off_by_year"
    DROP_ROWS = "drop_rows"
    SWAP_UNIT = "swap_unit"
    ZERO = "zero"


_UNIT_SWAP = {"minutes": "hours", "hours": "minutes", "usd": "cad", "cad": "usd", "": "units"}
_DATE_RE = re.compile(r"^(\d{4})(-\d{2}-\d{2}.*)?$")


def _shift_year(cell: Any) -> Any:
    # Date-like strings (or year ints) get moved one year later; everything else is untouched.
    if isinstance(cell, int) and 1900 <= cell <= 2100:
        return cell + 1
    if isinstance(cell, str):
        m = _DATE_RE.match(cell)
        if m:
            return f"{int(m.group(1)) + 1}{m.group(2) or ''}"
    return cell


def _numeric_cell(cell: Any, fn) -> Any:
    if isinstance(cell, bool):
        return cell
    if isinstance(cell, int | float):
        return fn(cell)
    return cell


def inject(executions: list[ExecutionResult], mode: InjectionMode | str) -> list[ExecutionResult]:
    """Return corrupted copies; the originals are never mutated."""
    mode = InjectionMode(mode)
    out: list[ExecutionResult] = []
    for ex in executions:
        rows = [list(r) for r in ex.rows]
        keys = [k.model_copy() for k in ex.key_numbers]
        if mode == InjectionMode.SCALE:
            rows = [[_numeric_cell(c, lambda v: v * 10) for c in r] for r in rows]
            keys = [k.model_copy(update={"value": k.value * 10}) for k in keys]
        elif mode == InjectionMode.SIGN:
            rows = [[_numeric_cell(c, lambda v: -v) for c in r] for r in rows]
            keys = [k.model_copy(update={"value": -k.value}) for k in keys]
        elif mode == InjectionMode.OFF_BY_YEAR:
            rows = [[_shift_year(c) for c in r] for r in rows]
            keys = [
                k.model_copy(update={"value": k.value + 1}) if _looks_like_year(k) else k
                for k in keys
            ]
        elif mode == InjectionMode.DROP_ROWS:
            rows = rows[: max(1, len(rows) // 2)]
            # Halving the rows should change a count/sum; shave the key numbers to match.
            keys = [k.model_copy(update={"value": k.value / 2}) for k in keys]
        elif mode == InjectionMode.SWAP_UNIT:
            keys = [
                k.model_copy(update={"unit": _UNIT_SWAP.get(k.unit.lower(), "units")}) for k in keys
            ]
        elif mode == InjectionMode.ZERO:
            keys = [k.model_copy(update={"value": 0.0}) for k in keys]
        out.append(
            ex.model_copy(update={"rows": rows, "row_count": len(rows), "key_numbers": keys})
        )
    return out


def _looks_like_year(k: KeyNumber) -> bool:
    return "year" in k.name.lower() and 1900 <= k.value <= 2100
