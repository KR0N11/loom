"""Claude Code headless provider: uses the `claude -p` CLI on the user's Claude subscription.

Approach: same `LLMProvider` interface as the API providers, but each call shells
out to `claude --print --output-format json` with tools disabled and a single
turn. Cost in the JSON result is the list-price equivalent, not what is billed
(subscription usage), so we record it as `equivalent_cost` and bill 0. Meant
for local development and evals without API credits; production uses the API.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from loom.llm.base import LLMMessage, LLMProvider


class ClaudeCodeNotFoundError(RuntimeError):
    pass


# CLI aliases map to canonical ids so list-price cost estimates still resolve.
MODEL_ALIASES = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-5", "opus": "claude-opus-5"}


class ClaudeCodeProvider(LLMProvider):
    name = "claude_code"

    def __init__(self, model: str = "haiku", timeout_s: int = 180, binary: str = "claude") -> None:
        self.cli_model = model
        self.model = MODEL_ALIASES.get(model, model)
        self.timeout_s = timeout_s
        self.binary = shutil.which(binary) or binary
        # Fail at construction, not on the first LLM call deep inside the graph.
        if shutil.which(self.binary) is None:
            raise ClaudeCodeNotFoundError(
                "`claude` CLI not found; install Claude Code or use the API provider"
            )

    def _raw_complete(
        self, system: str, messages: list[LLMMessage], max_tokens: int, temperature: float
    ) -> tuple[str, int, int]:
        prompt = flatten_messages(messages)
        cmd = [
            self.binary,
            "--print",
            "--output-format",
            "json",
            "--model",
            self.cli_model,
            "--system-prompt",
            system,
            "--tools",
            "",
            "--max-turns",
            "1",
            "--no-session-persistence",
        ]
        # Drop the API key so the CLI authenticates with the claude.ai login instead.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout_s, env=env
        )
        payload = parse_result(proc.stdout)
        # The CLI exits 0 even on auth/API errors; the JSON says so.
        if payload.get("is_error"):
            raise RuntimeError(f"claude -p failed: {payload.get('result', proc.stderr)[:300]}")
        usage = payload.get("usage", {})
        in_tok = int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0))
        return str(payload.get("result", "")), in_tok, out_tok


def flatten_messages(messages: list[LLMMessage]) -> str:
    """Turn a multi-turn exchange into one prompt (the CLI takes a single user prompt)."""
    # A single user message is passed through untouched to keep prompts identical to the API path.
    if len(messages) == 1:
        return messages[0].content
    parts = []
    for m in messages:
        label = "Your previous reply" if m.role == "assistant" else "User"
        parts.append(f"[{label}]\n{m.content}")
    return "\n\n".join(parts)


def parse_result(stdout: str) -> dict:
    """The CLI may print warnings before the JSON object; take the last JSON object."""
    start = stdout.find("{")
    if start == -1:
        raise RuntimeError(f"no JSON in claude output: {stdout[:200]!r}")
    return json.loads(stdout[start:])
