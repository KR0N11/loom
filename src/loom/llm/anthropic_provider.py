"""Anthropic Messages API provider (default)."""

from __future__ import annotations

from typing import Any, cast

from loom.llm.base import LLMMessage, LLMProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def _raw_complete(
        self, system: str, messages: list[LLMMessage], max_tokens: int, temperature: float
    ) -> tuple[str, int, int]:
        # SDK 1.x for Claude 5 models no longer accepts `temperature`; sampling is
        # controlled server-side, so the argument is accepted here for interface parity only.
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            messages=[{"role": cast(Any, m.role), "content": m.content} for m in messages],
        )
        text = "".join(getattr(block, "text", "") for block in resp.content if block.type == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens
