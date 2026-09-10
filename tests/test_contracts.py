"""Contract tests: agent output schemas are strict and their shape is pinned by snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from loom.agents import schemas as s

AGENT_OUTPUTS = [
    s.PlannerOutput,
    s.ExecutorOutput,
    s.KeyNumbersOutput,
    s.VerifierOutput,
    s.NarratorOutput,
    s.AnalysisPlan,
    s.SemanticContext,
    s.ExecutionResult,
    s.VerificationReport,
    s.Memo,
]

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


# Every agent output rejects fields we do not handle, so an LLM cannot smuggle extras in.
@pytest.mark.parametrize("model", AGENT_OUTPUTS)
def test_extra_fields_rejected(model):
    with pytest.raises(ValidationError):
        model.model_validate({"unexpected_field": 1})


# Required fields are enforced: a plan with no steps is invalid.
def test_required_fields():
    with pytest.raises(ValidationError):
        s.PlannerOutput.model_validate({"analysis_type": "trend", "steps": []})
    with pytest.raises(ValidationError):
        s.ExecutorOutput.model_validate({"language": "sql"})


# Enums reject unknown values so a fifth analysis type cannot appear silently.
def test_enum_values():
    with pytest.raises(ValidationError):
        s.PlannerOutput.model_validate(
            {"analysis_type": "forecast", "steps": [{"step_id": 1, "description": "x"}]}
        )
    with pytest.raises(ValidationError):
        s.ExecutorOutput.model_validate({"language": "r", "code": "x"})
    out = s.PlannerOutput.model_validate(
        {"analysis_type": "trend", "steps": [{"step_id": 1, "description": "x"}]}
    )
    assert out.analysis_type is s.AnalysisType.TREND


# JSON schema snapshot: a prompt/schema change that alters the contract shows up as a diff.
@pytest.mark.parametrize("model", AGENT_OUTPUTS)
def test_schema_snapshot(model):
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    path = SNAPSHOT_DIR / f"{model.__name__}.json"
    current = json.dumps(model.model_json_schema(), indent=2, sort_keys=True)
    if not path.exists():
        path.write_text(current)
    assert json.loads(path.read_text()) == json.loads(current), f"schema drift in {model.__name__}"
