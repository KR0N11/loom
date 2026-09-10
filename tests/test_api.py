"""API tests: stubbed runner, persisted state, listing, health and bad dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from loom.agents.schemas import (
    CheckStatus,
    ExecutionLanguage,
    Memo,
    QueryRecord,
    VerificationCheck,
    VerificationReport,
)
from loom.agents.state import LoomState
from loom.api.app import app, get_app_settings, get_runner
from loom.config import Settings


def make_state(question: str, dataset: str, **kwargs) -> LoomState:
    """A minimal but valid finished state, as the supervisor would return."""
    memo = Memo(
        title="Test memo",
        headline="Total paid was 100.",
        findings=["Finding one"],
        recommendation="Do the thing.",
        appendix_queries=[
            QueryRecord(
                attempt=1, language=ExecutionLanguage.SQL, code="SELECT 1", ok=True, row_count=1
            )
        ],
    )
    report = VerificationReport(
        overall=CheckStatus.PASS,
        checks=[VerificationCheck(name="reconcile", status=CheckStatus.PASS, detail="matched")],
    )
    return LoomState(
        question=question,
        dataset=dataset,
        thread_id=kwargs.get("thread_id", "t1"),
        memo=memo,
        verification=report,
    )


@pytest.fixture
def client(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Stub memo_to_markdown so the API test does not depend on the narrator package.
    import loom.api.app as api_mod

    monkeypatch.setattr(api_mod, "render_memo", lambda state: f"# {state.memo.title}")
    app.dependency_overrides[get_runner] = lambda: make_state
    app.dependency_overrides[get_app_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


# /ask returns the memo markdown and writes state.json under outputs/<thread_id>/.
def test_ask_persists_and_returns_markdown(client: TestClient, settings: Settings) -> None:
    resp = client.post("/ask", json={"question": "How much?", "dataset": "nfip"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["memo_markdown"] == "# Test memo"
    assert body["verification"]["overall"] == "pass"
    state_file = Path(settings.output_dir) / body["thread_id"] / "state.json"
    assert state_file.exists()
    assert json.loads(state_file.read_text())["question"] == "How much?"


# /memos lists thread ids that have a persisted state, and /memos/{id} returns it.
def test_list_and_get_memos(client: TestClient) -> None:
    assert client.get("/memos").json() == []
    thread_id = client.post("/ask", json={"question": "q", "dataset": "ttc"}).json()["thread_id"]
    assert client.get("/memos").json() == [thread_id]
    assert client.get(f"/memos/{thread_id}").json()["dataset"] == "ttc"
    assert client.get("/memos/nope").status_code == 404


# /health reports datasets and whether the DuckDB file exists.
def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["datasets"]) == {"ttc", "nfip"}
    assert body["duckdb_exists"] is False


# An unknown dataset is rejected before any pipeline work starts.
def test_unknown_dataset_is_400(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "q", "dataset": "nope"})
    assert resp.status_code == 400
    assert client.get("/memos").json() == []


# /datasets lists adapters with descriptions.
def test_datasets(client: TestClient) -> None:
    names = {d["name"] for d in client.get("/datasets").json()}
    assert names == {"ttc", "nfip"}
