"""Routing-accuracy baseline: zero-shot NLI classifier vs the planner's analysis type."""

from __future__ import annotations

from collections.abc import Callable

from loom.eval.benchmark import Benchmark

Classifier = Callable[[str], str]


def routing_accuracy(benchmark: Benchmark, classify: Classifier) -> float:
    """Share of questions whose predicted analysis type equals the gold tag."""
    if not benchmark.questions:
        return 0.0
    hits = sum(1 for q in benchmark.questions if str(classify(q.question)) == q.analysis_type)
    return hits / len(benchmark.questions)


def zero_shot_classifier(model_name: str | None = None) -> Classifier:
    """Wrap loom.agents.routing.ZeroShotRouter (lazy; needs the ml extra)."""
    from loom.agents.routing import ZeroShotRouter

    router = ZeroShotRouter(model_name) if model_name else ZeroShotRouter()

    def classify(question: str) -> str:
        return str(router.classify(question).value)

    return classify
