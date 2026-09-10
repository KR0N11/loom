"""Lookup strategy: a single fact or a short ranked list."""

from __future__ import annotations

from loom.agents.schemas import AnalysisType
from loom.agents.strategies.base import AnalysisStrategy


class LookupStrategy(AnalysisStrategy):
    name = AnalysisType.LOOKUP

    def planner_guidance(self) -> str:
        return "Lookup: use one step. The answer is a single value or a short ranked list."

    def executor_guidance(self) -> str:
        return "Return a single value or a short list; always add LIMIT (at most 20 rows)."

    def verifier_checks(self) -> list[str]:
        return ["sample_size", "duplicate_check", "row_truncation", "unit_check", "null_check"]
