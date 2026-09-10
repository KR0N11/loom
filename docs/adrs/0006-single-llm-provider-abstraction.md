# ADR 0006: One LLM provider abstraction

Status: Accepted. Date: 2026-09-10.

## Context

Agents need structured JSON output, cost accounting and vendor portability
(Anthropic API for development, Amazon Bedrock inside an AWS account).

## Decision

`LLMProvider` exposes `complete` and `complete_json(schema)`. JSON extraction,
one corrective retry on validation failure, latency timing and cost estimation
live in the base class; `AnthropicProvider` and `BedrockProvider` implement
only the raw call. A `FakeProvider` with scripted replies serves tests.
`LOOM_LLM_PROVIDER` selects the vendor.

## Consequences

- Prompts and retry behaviour are identical across vendors.
- Every call carries tokens, cost and latency into tracing and the eval report.
- Vendor-specific features (tool use, extended thinking) are not exposed; the
  agents rely on plain JSON contracts.
