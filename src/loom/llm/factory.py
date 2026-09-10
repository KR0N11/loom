"""Pick the provider from settings (Factory)."""

from __future__ import annotations

from loom.config import LLMProviderName, Settings, get_settings
from loom.llm.base import LLMProvider


def get_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    # Config switch between vendors; nothing else in the codebase knows which one is live.
    if settings.llm_provider == LLMProviderName.BEDROCK:
        from loom.llm.bedrock_provider import BedrockProvider

        return BedrockProvider(settings.bedrock_model_id, settings.aws_region)
    from loom.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider(settings.llm_model, settings.anthropic_api_key)
