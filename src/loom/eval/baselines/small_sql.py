"""Small open-weight SQL model baseline (Qwen coder by default).

Approach: prompt the model with the same semantic context the executor sees,
extract one SQL statement, run it through the sandbox and score it exactly like
the LLM pipeline. `generate` is injectable so tests never load weights.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from loom.config import Settings
from loom.eval.benchmark import BenchmarkQuestion
from loom.eval.metrics import execution_accuracy, numeric_cells, numeric_match
from loom.eval.runner import BaselineRow
from loom.sandbox.base import SandboxExecutor, SandboxRequest

_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply; prefer a fenced block, else the first SELECT/WITH."""
    fenced = _SQL_FENCE.search(text)
    if fenced:
        return fenced.group(1).strip().rstrip(";")
    match = re.search(r"(?is)\b(select|with)\b.*", text)
    return (match.group(0) if match else text).strip().rstrip(";")


class QwenSQLBaseline:
    name = "qwen-coder"

    def __init__(
        self,
        settings: Settings,
        sandbox: SandboxExecutor,
        model_name: str | None = None,
        generate: Callable[[str], str] | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.settings = settings
        self.sandbox = sandbox
        self.model_name = model_name or settings.small_sql_model
        self._generate = generate
        self.max_new_tokens = max_new_tokens
        self._pipe: Any = None

    def build_prompt(self, question: str, context_text: str) -> str:
        return (
            "You are a DuckDB SQL expert. Using only the tables, columns, joins and metric "
            "definitions below, write one SQL query that answers the question. Reply with the "
            "SQL inside a ```sql fence and nothing else.\n\n"
            f"{context_text}\n\nQuestion: {question}\n"
        )

    def _load(self) -> Callable[[str], str]:
        # Weights load on first use only; tests inject `generate` and never reach here.
        if self._generate is not None:
            return self._generate
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without the ml extra
            raise RuntimeError(
                "transformers/torch are not installed; install with `uv sync --extra ml`"
            ) from exc
        tokenizer: Any = AutoTokenizer.from_pretrained(self.model_name)
        model: Any = AutoModelForCausalLM.from_pretrained(self.model_name)

        def generate(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            return tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

        self._generate = generate
        return generate

    def generate_sql(self, question: str, context_text: str) -> str:
        return extract_sql(self._load()(self.build_prompt(question, context_text)))

    def answer(self, question: BenchmarkQuestion, context_text: str) -> BaselineRow:
        row = BaselineRow(
            baseline=self.name,
            id=question.id,
            difficulty=question.difficulty.value,
            dataset=question.dataset,
        )
        start = time.perf_counter()
        try:
            sql = self.generate_sql(question.question, context_text)
            result = self.sandbox.run(
                SandboxRequest(
                    language="sql",
                    code=sql,
                    duckdb_path=str(self.settings.duckdb_path),
                    row_limit=self.settings.sandbox_row_limit,
                    timeout_s=self.settings.sandbox_timeout_s,
                )
            )
            row.exec_ok = result.ok
            if result.ok:
                row.exec_acc = execution_accuracy(result.rows, question.gold_rows)
                row.numeric_match, _ = numeric_match(
                    numeric_cells(result.rows), question.gold_answer
                )
            else:
                row.error = result.error
        except Exception as exc:  # noqa: BLE001 - a baseline failure is a data point
            row.error = f"{type(exc).__name__}: {exc}"[:300]
        row.latency_s = time.perf_counter() - start
        return row
