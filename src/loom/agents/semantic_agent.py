"""Semantic-layer worker agent.

Approach: a worker in the supervisor/worker pattern. It owns no LLM call; it
turns the question (plus the planner's steps and metric names, when present)
into a retrieval query, runs hybrid search through the `VectorRepository`
(Repository pattern) and returns a `SemanticContext` the executor can trust.
"""

from __future__ import annotations

from collections.abc import Callable

from loom.agents.schemas import SemanticContext
from loom.agents.state import LoomState
from loom.config import Settings, get_settings
from loom.semantic.ingest import get_repository, load_semantics
from loom.semantic.models import DatasetSemantics
from loom.semantic.search import HybridSearch, build_context


class SemanticAgent:
    name = "semantic"

    def __init__(
        self,
        search: HybridSearch,
        semantics_loader: Callable[[str], DatasetSemantics | None],
        top_k: int = 8,
    ) -> None:
        self.search = search
        self.semantics_loader = semantics_loader
        self.top_k = top_k

    def build_query(self, state: LoomState) -> str:
        parts = [state.question]
        # The plan sharpens the query: step text names the tables/measures we actually need.
        if state.plan is not None:
            parts.extend(step.description for step in state.plan.steps)
            parts.extend(state.plan.metrics_needed)
        return " ".join(parts)

    def run(self, state: LoomState) -> dict:
        docs = self.search.retrieve(self.build_query(state), state.dataset, self.top_k)
        context: SemanticContext = build_context(docs, self.semantics_loader(state.dataset))
        return {"semantic": context}


def make_semantic_agent(settings: Settings | None = None, rerank: bool = True) -> SemanticAgent:
    """Wire repository, embedder and reranker from settings."""
    from loom.semantic.embeddings import get_embedder
    from loom.semantic.reranker import get_reranker

    settings = settings or get_settings()
    search = HybridSearch(
        get_repository(settings), get_embedder(settings), get_reranker(settings, rerank)
    )

    def _load(dataset: str) -> DatasetSemantics | None:
        try:
            return load_semantics(dataset, settings)
        except FileNotFoundError:
            return None

    return SemanticAgent(search, _load, top_k=settings.retrieval_top_k)
