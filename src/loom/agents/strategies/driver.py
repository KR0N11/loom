"""Driver strategy: which segments explain a total or a change."""

from __future__ import annotations

from loom.agents.schemas import AnalysisType
from loom.agents.strategies.base import AnalysisStrategy


class DriverStrategy(AnalysisStrategy):
    name = AnalysisType.DRIVER

    def planner_guidance(self) -> str:
        return (
            "Driver analysis: step 1 computes the total; step 2 breaks it down by the candidate "
            "driver dimension with each segment's share; step 3 (optional) compares periods."
        )

    def executor_guidance(self) -> str:
        return (
            "Include the segment contribution AND its share of the total in the same query "
            "(use a window SUM() OVER () for the denominator)."
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
