"""CLI tests: `loom ask` exit codes and output, `loom ingest` wiring."""

from __future__ import annotations

import pytest

from loom import cli
from loom.agents.schemas import CheckStatus, Memo, VerificationReport
from loom.agents.state import LoomState
from loom.config import Settings


def _state(overall: CheckStatus) -> LoomState:
    return LoomState(
        question="q",
        dataset="nfip",
        thread_id="cli1",
        memo=Memo(title="M", headline="H", findings=["f"], recommendation="r"),
        verification=VerificationReport(overall=overall, checks=[]),
    )


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> dict:
    import sys
    import types

    import loom.api.app as api_mod

    calls: dict = {}

    def fake_run(question, dataset, settings=None, injection=None, thread_id=None, **kw):
        calls["args"] = (question, dataset, injection)
        return calls["state"]

    # The supervisor module is stubbed so this test never runs the real graph.
    monkeypatch.setitem(
        sys.modules, "loom.agents.supervisor", types.SimpleNamespace(run_question=fake_run)
    )
    monkeypatch.setattr(api_mod, "render_memo", lambda state: f"# {state.memo.title}")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return calls


# `loom ask` prints the memo, writes memo.md, and exits 0 when verification passes.
def test_ask_pass_exit_zero(
    patched: dict, settings: Settings, capsys: pytest.CaptureFixture
) -> None:
    patched["state"] = _state(CheckStatus.PASS)
    code = cli.main(["ask", "How much?", "--dataset", "nfip"])
    assert code == 0
    assert patched["args"] == ("How much?", "nfip", None)
    assert "# M" in capsys.readouterr().out
    assert (settings.output_dir / "cli1" / "memo.md").read_text() == "# M"


# `loom ask` exits 2 when the verifier failed, so scripts can tell "unverified" apart.
def test_ask_fail_exit_two(patched: dict) -> None:
    patched["state"] = _state(CheckStatus.FAIL)
    assert cli.main(["ask", "q", "--dataset", "nfip", "--injection", "scale"]) == 2
    assert patched["args"][2] == "scale"


# `loom ingest --dataset all` calls ingest once per dataset with the limit.
def test_ingest_wiring(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    import sys
    import types

    calls: list = []
    fake_mod = types.SimpleNamespace(ingest=lambda ds, s, limit=None: calls.append((ds, limit)))
    monkeypatch.setitem(sys.modules, "loom.data.ingest", fake_mod)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["ingest", "--dataset", "all", "--limit", "5"]) == 0
    assert calls == [("ttc", 5), ("nfip", 5)]
