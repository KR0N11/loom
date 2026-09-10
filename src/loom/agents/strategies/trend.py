"""Trend strategy: a metric over a time grain."""

from __future__ import annotations

from loom.agents.schemas import AnalysisType
from loom.agents.strategies.base import AnalysisStrategy


class TrendStrategy(AnalysisStrategy):
    name = AnalysisType.TREND

    def planner_guidance(self) -> str:
        return (
            "Trend: state the time grain (day, month, year) and the time range explicitly. "
            "Step 1 computes the metric per period; an optional step 2 computes period-over-period change."
        )

    def executor_guidance(self) -> str:
        return (
            "Group by a truncated date (date_trunc) and ORDER BY it ascending; "
            "filter to the requested time range explicitly."
        )

    def verifier_checks(self) -> list[str]:
        return [
            "time_range",
            "sample_size",
            "duplicate_check",
            "row_truncation",
            "unit_check",
            "null_check",
        ]
