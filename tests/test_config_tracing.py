"""Settings come from the environment; the tracer is a no-op without Langfuse keys."""

from __future__ import annotations

import pytest

from loom.config import SandboxBackend, Settings, VectorBackend
from loom.llm.base import LLMUsage
from loom.tracing import get_tracer


# LOOM_-prefixed env vars override defaults, and enum values are validated.
def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOOM_VECTOR_BACKEND", "pgvector")
    monkeypatch.setenv("LOOM_SANDBOX_BACKEND", "docker")
    monkeypatch.setenv("LOOM_EXECUTOR_MAX_RETRIES", "5")
    s = Settings()
    assert s.vector_backend == VectorBackend.PGVECTOR
    assert s.sandbox_backend == SandboxBackend.DOCKER
    assert s.executor_max_retries == 5


# A bad enum value fails at startup instead of deep inside a run.
def test_settings_rejects_bad_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOOM_SANDBOX_BACKEND", "carrier-pigeon")
    with pytest.raises(ValueError):
        Settings()


# Without keys every tracer call is a no-op that never raises.
def test_tracer_noop_without_keys() -> None:
    t = get_tracer(Settings(langfuse_public_key=None, langfuse_secret_key=None))
    assert not t.enabled
    t.start_trace("q", input={"question": "x"})
    with t.span("planner"):
        t.generation("planner", "v1", LLMUsage(model="m"))
        t.event("retry", attempt=1)
    t.end_trace(output={"ok": True})


# With a client, spans nest under the root and generations carry cost + prompt version.
def test_tracer_records_with_stub_client() -> None:
    from loom.tracing.langfuse_tracer import Tracer

    log: list[tuple[str, dict]] = []

    class Obs:
        def __init__(self, kind: str, kw: dict) -> None:
            log.append((kind, kw))

        def start_observation(self, **kw):
            return Obs(kw["as_type"], kw)

        def update(self, **kw) -> None:
            pass

        def end(self) -> None:
            pass

        def create_event(self, **kw) -> None:
            log.append(("event", kw))

    class Client(Obs):
        def __init__(self) -> None:
            pass

        def flush(self) -> None:
            log.append(("flush", {}))

    t = Tracer(Client())
    t.start_trace("q")
    with t.span("planner"):
        t.generation("planner", "v3", LLMUsage(model="m", cost_usd=0.01, input_tokens=5))
    t.end_trace()
    kinds = [k for k, _ in log]
    assert kinds == ["agent", "span", "generation", "flush"]
    gen = log[2][1]
    assert gen["version"] == "v3" and gen["cost_details"] == {"total": 0.01}
