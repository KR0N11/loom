"""Provider-agnostic LLM interface.

Approach: agents only ever call `provider.complete_json(...)` with a Pydantic
schema, so prompts, retries on bad JSON, and cost accounting live here once.
Concrete providers (Anthropic, Bedrock) implement `_raw_complete` only.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from loom.llm.pricing import estimate_cost_usd

T = TypeVar("T", bound=BaseModel)


class LLMMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    model: str = ""

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_s=self.latency_s + other.latency_s,
            model=self.model or other.model,
        )


class LLMResponse(BaseModel):
    text: str
    usage: LLMUsage


class LLMJSONError(RuntimeError):
    """Raised when the model cannot produce output matching the requested schema."""


class LLMProvider(ABC):
    """Strategy interface for text completion. Subclasses wrap one vendor SDK."""

    name: str = "base"
    model: str = ""

    @abstractmethod
    def _raw_complete(
        self, system: str, messages: list[LLMMessage], max_tokens: int, temperature: float
    ) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens)."""

    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # Time every call so latency shows up in traces and the eval report.
        start = time.perf_counter()
        text, in_tok, out_tok = self._raw_complete(system, messages, max_tokens, temperature)
        usage = LLMUsage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=estimate_cost_usd(self.model, in_tok, out_tok),
            latency_s=time.perf_counter() - start,
            model=self.model,
        )
        return LLMResponse(text=text, usage=usage)

    def complete_json(
        self,
        system: str,
        messages: list[LLMMessage],
        schema: type[T],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_attempts: int = 2,
    ) -> tuple[T, LLMUsage]:
        """Ask for JSON matching `schema`; re-prompt once with the validation error if it fails."""
        schema_hint = json.dumps(schema.model_json_schema(), indent=None)
        system_full = (
            f"{system}\n\nRespond with a single JSON object only, no prose, matching this JSON "
            f"schema:\n{schema_hint}"
        )
        total = LLMUsage(model=self.model)
        convo = list(messages)
        last_error = ""
        # Retry loop: a malformed JSON reply is common enough that one corrective
        # turn recovers most cases without a human in the loop.
        for _ in range(max_attempts):
            resp = self.complete(system_full, convo, max_tokens=max_tokens, temperature=temperature)
            total = total + resp.usage
            try:
                return schema.model_validate(extract_json(resp.text)), total
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                convo = convo + [
                    LLMMessage(role="assistant", content=resp.text),
                    LLMMessage(
                        role="user",
                        content=f"That was not valid. Error: {last_error[:800]}. "
                        "Reply again with only the corrected JSON object.",
                    ),
                ]
        raise LLMJSONError(f"Could not get valid {schema.__name__}: {last_error[:300]}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a reply that may include fences or prose."""
    candidate = text.strip()
    fenced = _FENCE.search(candidate)
    # Models sometimes wrap JSON in a code fence despite instructions.
    if fenced:
        candidate = fenced.group(1).strip()
    # Fall back to the outermost braces if there is leading/trailing prose.
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object found in reply")
        candidate = candidate[start : end + 1]
    return json.loads(candidate)
