"""FastAPI application.

Approach: one endpoint runs the supervisor graph, persists the full typed state
to outputs/<thread_id>/state.json for audit, and returns the memo plus the
verification report. The runner is injected (FastAPI dependency) so tests can
swap in a stub instead of the real LangGraph pipeline.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException

from loom.agents.state import LoomState
from loom.api.models import AskRequest, AskResponse, DatasetInfo, HealthResponse
from loom.config import Settings, get_settings
from loom.data.adapters import list_adapters

Runner = Callable[..., LoomState]

app = FastAPI(title="Loom", version="0.1.0")


def get_runner() -> Runner:
    """Default runner is the real supervisor graph; imported lazily to keep startup fast."""
    from loom.agents.supervisor import run_question

    return run_question


def get_app_settings() -> Settings:
    return get_settings()


# Annotated dependencies keep the route signatures free of call-in-default warnings.
RunnerDep = Annotated[Runner, Depends(get_runner)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]


def render_memo(state: LoomState) -> str:
    """Markdown for the memo, or a plain failure note when the pipeline produced none."""
    # A failed run has no memo; the caller still deserves a readable explanation.
    if state.memo is None:
        return "# Analysis failed\n\n" + "\n".join(f"- {e}" for e in state.errors)
    from loom.agents.narrator import memo_to_markdown

    return memo_to_markdown(state.memo)


def persist_state(state: LoomState, settings: Settings) -> Path:
    """Write the full state next to the memo charts so every answer is auditable later."""
    out_dir = Path(settings.output_dir) / state.thread_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "state.json"
    path.write_text(state.model_dump_json(indent=2))
    return path


@app.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        datasets=sorted(list_adapters()),
        duckdb_exists=Path(settings.duckdb_path).exists(),
    )


@app.get("/datasets", response_model=list[DatasetInfo])
def datasets() -> list[DatasetInfo]:
    return [
        DatasetInfo(name=name, description=cls.description)
        for name, cls in sorted(list_adapters().items())
    ]


@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    runner: RunnerDep,
    settings: SettingsDep,
) -> AskResponse:
    # Reject unknown datasets here so the caller gets a 400, not a DuckDB error deep inside.
    if body.dataset not in list_adapters():
        raise HTTPException(status_code=400, detail=f"unknown dataset {body.dataset!r}")
    thread_id = uuid.uuid4().hex[:12]
    state = runner(
        body.question,
        body.dataset,
        settings=settings,
        injection=body.injection,
        thread_id=thread_id,
    )
    persist_state(state, settings)
    return AskResponse(
        thread_id=state.thread_id,
        memo=state.memo,
        verification=state.verification,
        usage=state.usage,
        errors=state.errors,
        memo_markdown=render_memo(state),
    )


@app.get("/memos")
def list_memos(settings: SettingsDep) -> list[str]:
    root = Path(settings.output_dir)
    # No answers yet means no folder yet; that is an empty list, not an error.
    if not root.exists():
        return []
    return sorted(p.parent.name for p in root.glob("*/state.json"))


@app.get("/memos/{thread_id}")
def get_memo(thread_id: str, settings: SettingsDep) -> dict[str, Any]:
    path = Path(settings.output_dir) / thread_id / "state.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="memo not found")
    data: dict[str, Any] = json.loads(path.read_text())
    return data
