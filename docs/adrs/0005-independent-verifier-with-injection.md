# ADR 0005: Independent verifier with an injection mode

Status: Accepted. Date: 2026-09-10.

## Context

A confident, wrong number in a memo is the worst outcome for a regulated
analytics team. Asking the same model to "double check" its own query is not
independent. And a verifier that is never tested against known-bad output has
an unknown catch rate.

## Decision

The verifier re-derives every key number via an alternate SQL path (different
aggregation route or a different table through an allowed join), runs it in
the sandbox, and compares within a 1% relative tolerance. It adds
deterministic checks that need no LLM: units, sample size, time-range
coverage, nulls, duplicate rows and truncation. Any FAIL makes the memo
headline start with `UNVERIFIED:` (enforced in code, not in the prompt).

For measurement, `LoomState.injection` selects one of six corruptions applied
to executor output before verification: `scale`, `sign`, `off_by_year`,
`drop_rows`, `swap_unit`, `zero`. The eval harness reports catch rate per mode
and the false-alarm rate on uninjected runs.

## Consequences

- Verifier quality is a number, and CI fails if it drops at all (ADR 0007).
- Each question costs roughly one extra LLM call and one extra query.
- Some corruptions (`swap_unit`) are only catchable by the unit check, which
  depends on metric units being curated in the semantic layer.
