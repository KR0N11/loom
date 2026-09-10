"""Amazon Bedrock provider: same interface, Anthropic models served via AWS."""

from __future__ import annotations

import json

from loom.llm.base import LLMMessage, LLMProvider


class BedrockProvider(LLMProvider):
    name = "bedrock"

    def __init__(self, model_id: str, region: str) -> None:
        import boto3

        self.model = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def _raw_complete(
        self, system: str, messages: list[LLMMessage], max_tokens: int, temperature: float
    ) -> tuple[str, int, int]:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        raw = self._client.invoke_model(modelId=self.model, body=json.dumps(body))
        payload = json.loads(raw["body"].read())
        text = "".join(b.get("text", "") for b in payload.get("content", []))
        usage = payload.get("usage", {})
        return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
