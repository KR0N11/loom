"""Strategy per analysis type (Strategy pattern)."""

from __future__ import annotations

from loom.agents.schemas import AnalysisType
from loom.agents.strategies.base import AnalysisStrategy
from loom.agents.strategies.comparison import ComparisonStrategy
from loom.agents.strategies.driver import DriverStrategy
from loom.agents.strategies.lookup import LookupStrategy
from loom.agents.strategies.trend import TrendStrategy

_STRATEGIES: dict[AnalysisType, AnalysisStrategy] = {
    AnalysisType.LOOKUP: LookupStrategy(),
    AnalysisType.TREND: TrendStrategy(),
    AnalysisType.COMPARISON: ComparisonStrategy(),
    AnalysisType.DRIVER: DriverStrategy(),
}


def get_strategy(analysis_type: AnalysisType) -> AnalysisStrategy:
    return _STRATEGIES[analysis_type]


__all__ = ["AnalysisStrategy", "get_strategy"]
