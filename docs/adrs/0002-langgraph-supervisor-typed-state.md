# ADR 0002: LangGraph supervisor with typed Pydantic state and checkpointing

Status: Accepted. Date: 2026-09-10.

## Context

Five agents run in sequence with one conditional branch (skip the verifier
when the executor produced nothing). A regulated audience wants to see the
state after every step and resume or replay a run.

## Decision

Model the pipeline as a LangGraph `StateGraph` over `LoomState`, a Pydantic
model with one slot per agent output (`plan`, `semantic`, `executions`,
`verification`, `memo`) plus usage, errors and an eval-only `injection`
field. The supervisor owns routing; each worker agent exposes
`run(state) -> dict` and fills only its own slot. State is checkpointed after
every node with the SQLite saver locally (a Postgres saver is the production
swap).

## Consequences

- Every intermediate result is inspectable and typed; a node cannot write a
  field the schema does not know.
- Runs can be resumed by `thread_id` after a crash or a rate-limit error.
- The graph is deliberately simple (linear plus one branch). Dynamic
  re-planning loops are a future extension, not a current feature.
