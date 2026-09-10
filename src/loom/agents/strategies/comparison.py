"""Comparison strategy: the same metric across two or more groups."""

from __future__ import annotations

from loom.agents.schemas import AnalysisType
from loom.agents.strategies.base import AnalysisStrategy


class ComparisonStrategy(AnalysisStrategy):
    name = AnalysisType.COMPARISON

    def planner_guidance(self) -> str:
        return (
            "Comparison: name the groups being compared and the metric. "
            "Compute the metric per group in one step so both sides use identical filters."
        )

    def executor_guidance(self) -> str:
        return (
            "Return one row per group with the metric and the row count behind it, "
            "so sample sizes are visible."
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
