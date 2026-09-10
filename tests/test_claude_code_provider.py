"""ClaudeCodeProvider shells out to `claude -p`; tests stub the subprocess."""

from __future__ import annotations

import json
import subprocess

import pytest

from loom.llm.base import LLMMessage
from loom.llm.claude_code_provider import ClaudeCodeProvider, flatten_messages, parse_result


def _fake_run(result: dict):
    def run(cmd, input, capture_output, text, timeout, env):
        run.calls.append((cmd, input, env))
        return subprocess.CompletedProcess(
            cmd, 0, stdout="warning line\n" + json.dumps(result), stderr=""
        )

    run.calls = []  # type: ignore[attr-defined]
    return run


# A normal reply is parsed from the JSON, tokens are counted, and the API key is stripped from env.
def test_complete_parses_json_and_strips_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")
    run = _fake_run(
        {"result": "hi", "is_error": False, "usage": {"input_tokens": 5, "output_tokens": 2}}
    )
    monkeypatch.setattr(subprocess, "run", run)
    p = ClaudeCodeProvider(model="haiku")
    r = p.complete("sys", [LLMMessage(role="user", content="q")])
    assert r.text == "hi" and r.usage.input_tokens == 5 and r.usage.output_tokens == 2
    cmd, prompt, env = run.calls[0]
    assert "--system-prompt" in cmd and prompt == "q" and "ANTHROPIC_API_KEY" not in env


# The CLI exits 0 on API errors, so is_error must become an exception.
def test_is_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda b: "/usr/bin/claude")
    monkeypatch.setattr(subprocess, "run", _fake_run({"result": "401 bad", "is_error": True}))
    with pytest.raises(RuntimeError, match="401"):
        ClaudeCodeProvider().complete("s", [LLMMessage(role="user", content="q")])


# Multi-turn corrections are flattened into one labelled prompt; single turns pass through.
def test_flatten_messages() -> None:
    assert flatten_messages([LLMMessage(role="user", content="q")]) == "q"
    txt = flatten_messages(
        [
            LLMMessage(role="user", content="q"),
            LLMMessage(role="assistant", content="bad"),
            LLMMessage(role="user", content="fix"),
        ]
    )
    assert "[Your previous reply]\nbad" in txt and txt.endswith("fix")


# Warnings printed before the JSON are skipped; no JSON at all is an error.
def test_parse_result() -> None:
    assert parse_result('noise\n{"result": "x"}')["result"] == "x"
    with pytest.raises(RuntimeError):
        parse_result("nothing")
