"""Scripted provider for tests: returns canned replies in order, records prompts."""

from __future__ import annotations

from loom.llm.base import LLMMessage, LLMProvider


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, replies: list[str] | None = None, model: str = "fake-model") -> None:
        self.model = model
        self.replies = list(replies or [])
        self.calls: list[tuple[str, list[LLMMessage]]] = []

    def _raw_complete(
        self, system: str, messages: list[LLMMessage], max_tokens: int, temperature: float
    ) -> tuple[str, int, int]:
        self.calls.append((system, messages))
        # Tests script replies; running out is a test bug, so fail loudly.
        if not self.replies:
            raise RuntimeError("FakeProvider has no more scripted replies")
        return self.replies.pop(0), 10, 5
