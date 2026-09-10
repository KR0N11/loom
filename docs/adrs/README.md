# Architecture Decision Records

| # | Decision |
|---|---|
| [0001](0001-duckdb-analytical-engine.md) | DuckDB as the analytical engine |
| [0002](0002-langgraph-supervisor-typed-state.md) | LangGraph supervisor with typed Pydantic state and checkpointing |
| [0003](0003-semantic-layer-pgvector-hybrid-search.md) | Semantic layer in pgvector with hybrid search and a cross-encoder reranker |
| [0004](0004-sandboxed-execution-strategy.md) | Sandboxed execution with swappable Docker / subprocess / Lambda backends |
| [0005](0005-independent-verifier-with-injection.md) | Independent verifier with an injection mode to measure catch rate |
| [0006](0006-single-llm-provider-abstraction.md) | One LLM provider abstraction: Anthropic default, Bedrock by config |
| [0007](0007-eval-harness-ci-gate.md) | Eval harness as a CI regression gate |
| [0008](0008-strict-pydantic-contracts.md) | Strict Pydantic contracts per agent with snapshot tests |

Format: Context, Decision, Consequences. Status is Accepted unless stated.
