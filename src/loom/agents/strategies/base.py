"""Strategy pattern: one object per analysis type carrying the guidance that differs by type.

Approach: the planner, executor and verifier stay generic; what changes between a
lookup and a driver analysis (how to decompose, what to check) lives here.
"""

from __future__ import annotations

from loom.agents.schemas import AnalysisType


class AnalysisStrategy:
    """Base strategy: generic guidance. Subclasses override per analysis type."""

    name: AnalysisType = AnalysisType.LOOKUP

    def planner_guidance(self) -> str:
        return ""

    def executor_guidance(self) -> str:
        return ""

    def verifier_checks(self) -> list[str]:
        """Names of deterministic checks the verifier should run for this type."""
        return ["sample_size", "duplicate_check", "row_truncation", "unit_check", "null_check"]
