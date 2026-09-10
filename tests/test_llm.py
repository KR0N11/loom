"""Tests for the provider abstraction: JSON extraction, retry on bad JSON, cost, factory."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from loom.config import LLMProviderName, Settings
from loom.llm.base import LLMJSONError, LLMMessage, LLMUsage, extract_json
from loom.llm.factory import get_provider
from loom.llm.fake_provider import FakeProvider
from loom.llm.pricing import estimate_cost_usd


class Answer(BaseModel):
    value: int


# extract_json handles bare JSON, fenced JSON, and JSON surrounded by prose.
@pytest.mark.parametrize(
    "text",
    [
        '{"value": 1}',
        '```json\n{"value": 1}\n```',
        'Sure, here it is:\n{"value": 1}\nDone.',
    ],
)
def test_extract_json_variants(text: str) -> None:
    assert extract_json(text) == {"value": 1}


# A reply with no JSON object at all is a hard error, not a silent empty dict.
def test_extract_json_missing() -> None:
    with pytest.raises(ValueError):
        extract_json("no json here")


# complete_json returns the parsed model and accumulates usage over the single call.
def test_complete_json_happy_path() -> None:
    p = FakeProvider(['{"value": 42}'])
    out, usage = p.complete_json("sys", [LLMMessage(role="user", content="q")], Answer)
    assert out.value == 42
    assert usage.input_tokens == 10 and usage.output_tokens == 5


# A malformed first reply triggers one corrective turn that includes the error text.
def test_complete_json_retries_once_then_succeeds() -> None:
    p = FakeProvider(['{"value": "not-an-int"}', '{"value": 7}'])
    out, usage = p.complete_json("sys", [LLMMessage(role="user", content="q")], Answer)
    assert out.value == 7
    assert len(p.calls) == 2
    assert "not valid" in p.calls[1][1][-1].content
    assert usage.input_tokens == 20


# Two bad replies exhaust the attempts and raise a typed error.
def test_complete_json_gives_up() -> None:
    p = FakeProvider(["garbage", "still garbage"])
    with pytest.raises(LLMJSONError):
        p.complete_json("sys", [LLMMessage(role="user", content="q")], Answer)


# Cost uses the longest matching model prefix so Bedrock ids resolve to the same price.
def test_pricing_prefix_match() -> None:
    direct = estimate_cost_usd("claude-sonnet-5", 1_000_000, 0)
    bedrock = estimate_cost_usd("anthropic.claude-sonnet-5-v1:0", 1_000_000, 0)
    assert direct == bedrock == 3.0
    assert estimate_cost_usd("unknown-model", 1000, 1000) == 0.0


# Usage objects add up field by field.
def test_usage_add() -> None:
    total = LLMUsage(input_tokens=1, output_tokens=2, cost_usd=0.5, latency_s=1.0, model="m") + (
        LLMUsage(input_tokens=3, output_tokens=4, cost_usd=0.5, latency_s=2.0)
    )
    assert (total.input_tokens, total.output_tokens, total.cost_usd, total.latency_s) == (
        4,
        6,
        1.0,
        3.0,
    )
    assert total.model == "m"


# The factory honours the provider switch without any other code knowing about vendors.
def test_factory_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    s = Settings(llm_provider=LLMProviderName.ANTHROPIC, anthropic_api_key="k")
    assert get_provider(s).name == "anthropic"

    class StubClient:
        def __init__(self, *a, **k) -> None:
            pass

    monkeypatch.setattr("boto3.client", lambda *a, **k: StubClient())
    s2 = Settings(llm_provider=LLMProviderName.BEDROCK, anthropic_api_key=None)
    assert get_provider(s2).name == "bedrock"
