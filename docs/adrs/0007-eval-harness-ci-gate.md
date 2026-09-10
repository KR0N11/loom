# ADR 0007: Eval harness as a CI regression gate

Status: Accepted. Date: 2026-09-10.

## Context

Prompt edits, model upgrades and retrieval changes move accuracy in ways unit
tests cannot see. The project needs a benchmark early and a rule that blocks
regressions.

## Decision

Maintain `benchmarks/questions.yaml` (150 to 200 questions, both datasets,
difficulty tags, gold SQL and gold answers, `reviewed` flag). The runner
records per-question execution accuracy, numeric match, retrieval precision
and recall, verifier outcome, zero-correction, cost and latency, and runs the
injection modes on a sample. CI runs a stratified sample when an API key
secret is present and compares against `benchmarks/baseline_results.json`
from `main`: fail if execution accuracy drops by more than 5 points or if
verifier catch rate drops at all.

## Consequences

- Quality is tracked per difficulty, dataset and analysis type in a markdown
  report attached to every CI run.
- Benchmark runs cost money; the CI sample is small and the full run is manual.
- Unreviewed generated questions can be wrong; the report states the reviewed
  count so readers can weigh the numbers.
