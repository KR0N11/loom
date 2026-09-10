"""defog/sqlcoder-7b-2 baseline: same runner as Qwen, different prompt format."""

from __future__ import annotations

from loom.eval.baselines.small_sql import QwenSQLBaseline

SQLCODER_MODEL = "defog/sqlcoder-7b-2"


class SQLCoderBaseline(QwenSQLBaseline):
    name = "sqlcoder-7b-2"

    def __init__(self, settings, sandbox, generate=None, max_new_tokens: int = 256) -> None:  # type: ignore[no-untyped-def]
        super().__init__(settings, sandbox, SQLCODER_MODEL, generate, max_new_tokens)

    def build_prompt(self, question: str, context_text: str) -> str:
        # sqlcoder was trained on this task/instructions/schema/answer layout.
        return (
            "### Task\n"
            f"Generate a SQL query to answer [QUESTION]{question}[/QUESTION]\n\n"
            "### Instructions\n"
            "- Use DuckDB syntax and only the tables, columns and joins listed.\n"
            "- Use the metric definitions exactly as written.\n\n"
            "### Database Schema\n"
            f"{context_text}\n\n"
            "### Answer\n"
            f"Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]\n"
            "[SQL]\n"
        )
