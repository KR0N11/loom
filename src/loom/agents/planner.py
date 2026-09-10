"""Planner worker agent.

Approach: one LLM call classifies the question (lookup/trend/comparison/driver)
and decomposes it into ordered steps over the dataset's tables. The chosen
Strategy's guidance is included so decomposition follows the type's rules.
Pattern: worker agent under the supervisor graph; Strategy per analysis type.
"""

from __future__ import annotations

from loom.agents.prompts import prompt
from loom.agents.schemas import AnalysisPlan, AnalysisType, PlannerOutput
from loom.agents.state import LoomState
from loom.agents.strategies import get_strategy
from loom.config import Settings, get_settings
from loom.llm.base import LLMMessage, LLMProvider
from loom.semantic.models import DatasetSemantics
from loom.tracing import Tracer

_ALL_STRATEGY_GUIDANCE = "\n".join(
    f"- {t.value}: {get_strategy(t).planner_guidance()}" for t in AnalysisType
)


def table_summary(semantics: DatasetSemantics | None) -> str:
    """Compact one-line-per-table listing for the planner prompt."""
    # Without semantics the planner still runs; it just cannot name tables precisely.
    if semantics is None:
        return "(no table metadata available)"
    lines = []
    for t in semantics.tables:
        cols = ", ".join(c.name for c in t.columns[:25])
        lines.append(f"- {t.name} (grain: {t.grain}; {t.row_count} rows): {cols}")
    if semantics.metrics:
        lines.append("Metrics: " + ", ".join(m.name for m in semantics.metrics))
    return "\n".join(lines)


class PlannerAgent:
    def __init__(
        self,
        provider: LLMProvider,
        tracer: Tracer | None = None,
        settings: Settings | None = None,
        semantics_loader=None,
    ) -> None:
        self.provider = provider
        self.tracer = tracer or Tracer(None)
        self.settings = settings or get_settings()
        self._load = semantics_loader or _default_loader

    def run(self, state: LoomState) -> dict:
        semantics = self._load(state.dataset, self.settings)
        version, text = prompt("planner")
        system = text.format(
            strategy_guidance=_ALL_STRATEGY_GUIDANCE, table_summary=table_summary(semantics)
        )
        messages = [LLMMessage(role="user", content=f"Question: {state.question}")]
        out, usage = self.provider.complete_json(system, messages, PlannerOutput)
        self.tracer.generation("planner", version, usage, input=state.question, output=out)
        plan = AnalysisPlan(
            question=state.question,
            dataset=state.dataset,
            analysis_type=out.analysis_type,
            steps=out.steps,
            metrics_needed=out.metrics_needed,
            time_range_hint=out.time_range_hint,
            assumptions=out.assumptions,
        )
        return {"plan": plan, "usage": state.usage + usage}


def _default_loader(dataset: str, settings: Settings) -> DatasetSemantics | None:
    # Lazy import so the planner is testable before the semantic package exists.
    try:
        from loom.semantic.ingest import load_semantics

        return load_semantics(dataset, settings)
    except (ImportError, FileNotFoundError):
        return None
