"""Zero-shot question router (facebook/bart-large-mnli) used by the eval comparison.

Approach: an NLI model scores the question against one description per analysis
type. Cheap baseline to compare against the planner's LLM classification; not
in the main path.
"""

from __future__ import annotations

from loom.agents.schemas import AnalysisType

LABELS: dict[str, AnalysisType] = {
    "a lookup of a single value or a short ranked list": AnalysisType.LOOKUP,
    "a trend of a metric over time": AnalysisType.TREND,
    "a comparison of a metric between groups": AnalysisType.COMPARISON,
    "an analysis of which factors drive a total or a change": AnalysisType.DRIVER,
}


class ZeroShotRouter:
    def __init__(self, model_name: str = "facebook/bart-large-mnli") -> None:
        # transformers is an optional extra; fail with a clear message, not an AttributeError.
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("install the [ml] extra: uv sync --extra ml") from exc
        self._pipe = pipeline("zero-shot-classification", model=model_name)

    def classify(self, question: str) -> AnalysisType:
        out = self._pipe(question, candidate_labels=list(LABELS))
        return LABELS[out["labels"][0]]
