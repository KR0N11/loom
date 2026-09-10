"""Thin tracer wrapper over Langfuse v4 (OpenTelemetry-based).

Approach: nodes call `tracer.span(name, ...)` and `tracer.generation(...)`.
With Langfuse keys configured, every node, tool call, prompt version and cost
lands in Langfuse; without keys every call is a no-op so tests and offline runs
never need the SDK to be live.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from loom.config import Settings, get_settings
from loom.llm.base import LLMUsage


class Tracer:
    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self._root: Any | None = None
        self._stack: list[Any] = []

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def start_trace(self, name: str, input: Any = None, metadata: dict | None = None) -> None:
        # Without a client every method is a cheap no-op.
        if not self._client:
            return
        self._root = self._client.start_observation(
            name=name, as_type="agent", input=input, metadata=metadata or {}
        )

    def end_trace(self, output: Any = None) -> None:
        if self._client and self._root is not None:
            self._root.update(output=output)
            self._root.end()
            self._client.flush()
            self._root = None

    def _parent(self) -> Any:
        return self._stack[-1] if self._stack else self._root

    @contextmanager
    def span(
        self, name: str, input: Any = None, metadata: dict | None = None, kind: str = "span"
    ) -> Iterator[None]:
        # No live trace: yield immediately so callers never branch on tracing.
        if not self._client or self._root is None:
            yield
            return
        span = self._parent().start_observation(
            name=name, as_type=kind, input=input, metadata=metadata or {}
        )
        self._stack.append(span)
        try:
            yield
        finally:
            self._stack.pop()
            span.end()

    def generation(
        self,
        name: str,
        prompt_version: str,
        usage: LLMUsage,
        input: Any = None,
        output: Any = None,
    ) -> None:
        """Record one LLM call with its prompt version, tokens and cost."""
        if not self._client or self._root is None:
            return
        gen = self._parent().start_observation(
            name=name,
            as_type="generation",
            model=usage.model,
            input=input,
            output=output,
            version=prompt_version,
            metadata={"prompt_version": prompt_version, "latency_s": usage.latency_s},
            usage_details={"input": usage.input_tokens, "output": usage.output_tokens},
            cost_details={"total": usage.cost_usd},
        )
        gen.end()

    def event(self, name: str, **metadata: Any) -> None:
        """Record a point-in-time fact (e.g. a retry or an injection) inside the current span."""
        if not self._client or self._root is None:
            return
        self._parent().create_event(name=name, metadata=metadata)


def get_tracer(settings: Settings | None = None) -> Tracer:
    settings = settings or get_settings()
    # Only construct the Langfuse client when both keys exist; otherwise trace nothing.
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        from langfuse import Langfuse

        return Tracer(
            Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        )
    return Tracer(None)
