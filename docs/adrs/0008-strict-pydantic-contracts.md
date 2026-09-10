# ADR 0008: Strict Pydantic contracts per agent with snapshot tests

Status: Accepted. Date: 2026-09-10.

## Context

An LLM can add, rename or omit fields at any time. Downstream agents and the
UI must not break silently when that happens.

## Decision

Every agent output (`AnalysisPlan`, `SemanticContext`, `ExecutionResult`,
`VerificationReport`, `Memo`) and every model-facing partial schema
(`PlannerOutput`, `ExecutorOutput`, `KeyNumbersOutput`, `VerifierOutput`,
`NarratorOutput`) is a Pydantic model with `extra="forbid"`. State enums
(`AnalysisType`, `ExecutionLanguage`, `CheckStatus`) replace bare strings.
Contract tests assert that extra fields are rejected and that the JSON schema
of each model matches a committed snapshot.

## Consequences

- A prompt or schema change that alters a contract fails CI until the
  snapshot is deliberately updated.
- Model replies with unexpected fields trigger one corrective retry, then a
  typed `LLMJSONError`.
- Slightly more boilerplate per agent, which is the point.
