"""Executor worker agent.

Approach: for each plan step the LLM writes one SQL (or pandas) snippet from
the semantic context, the sandbox runs it, and on failure the error is fed back
for up to N retries. Every attempt is recorded as a QueryRecord for the audit
appendix. A second small LLM call extracts the headline numbers the verifier
will re-derive. Pattern: worker agent; sandbox via Strategy (SandboxExecutor).
"""

from __future__ import annotations

import json

from loom.agents.prompts import prompt
from loom.agents.schemas import (
    ExecutionLanguage,
    ExecutionResult,
    ExecutorOutput,
    KeyNumbersOutput,
    PlanStep,
    QueryRecord,
)
from loom.agents.state import LoomState
from loom.agents.strategies import get_strategy
from loom.config import Settings, get_settings
from loom.llm.base import LLMMessage, LLMProvider, LLMUsage
from loom.sandbox.base import SandboxExecutor, SandboxRequest, SandboxResult
from loom.tracing import Tracer

MAX_ROWS_FOR_SUMMARY = 30


class ExecutorAgent:
    def __init__(
        self,
        provider: LLMProvider,
        executor: SandboxExecutor,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.provider = provider
        self.executor = executor
        self.settings = settings or get_settings()
        self.tracer = tracer or Tracer(None)

    def run(self, state: LoomState) -> dict:
        # The graph guarantees plan and semantic are set before this node runs.
        assert state.plan is not None and state.semantic is not None
        usage = state.usage
        errors = list(state.errors)
        executions: list[ExecutionResult] = []
        for step in state.plan.steps:
            result, step_usage = self._run_step(state, step, executions)
            usage = usage + step_usage
            # A step that never succeeds blocks later steps that depend on it; stop here.
            if result is None:
                errors.append(
                    f"step {step.step_id} failed after {self.settings.executor_max_retries} retries"
                )
                break
            executions.append(result)
        return {"executions": executions, "usage": usage, "errors": errors}

    def _run_step(
        self, state: LoomState, step: PlanStep, previous: list[ExecutionResult]
    ) -> tuple[ExecutionResult | None, LLMUsage]:
        assert state.plan is not None and state.semantic is not None
        strategy = get_strategy(state.plan.analysis_type)
        version, text = prompt("executor")
        system = text.format(
            strategy_guidance=strategy.executor_guidance(),
            context_text=state.semantic.context_text,
        )
        prior = "\n".join(f"Step {p.step_id} result: {p.summary}" for p in previous)
        user = (
            f"Question: {state.question}\nAnalysis type: {state.plan.analysis_type}\n"
            f"Step {step.step_id}: {step.description}\nTables suggested: {step.needs_tables}\n"
            f"{prior}"
        )
        messages = [LLMMessage(role="user", content=user)]
        total = LLMUsage(model=self.provider.model)
        queries: list[QueryRecord] = []
        out, u = self.provider.complete_json(system, messages, ExecutorOutput)
        total = total + u
        self.tracer.generation("executor", version, u, input=user, output=out)
        attempt = 1
        # Retry loop: feed the sandbox error back so the model can repair its own code.
        while True:
            sandbox_result = self._execute(state, out)
            queries.append(
                QueryRecord(
                    attempt=attempt,
                    language=out.language,
                    code=out.code,
                    ok=sandbox_result.ok,
                    error=sandbox_result.error,
                    row_count=sandbox_result.row_count,
                    elapsed_s=sandbox_result.elapsed_s,
                )
            )
            if sandbox_result.ok:
                break
            if attempt > self.settings.executor_max_retries:
                return None, total
            self.tracer.event("executor_retry", attempt=attempt, error=sandbox_result.error)
            fix_version, fix_text = prompt("executor_fix")
            fix_system = fix_text.format(error=sandbox_result.error, code=out.code)
            out, u = self.provider.complete_json(fix_system, messages, ExecutorOutput)
            total = total + u
            self.tracer.generation("executor_fix", fix_version, u, output=out)
            attempt += 1
        keys, u = self._key_numbers(state, step, sandbox_result)
        total = total + u
        result = ExecutionResult(
            step_id=step.step_id,
            language=out.language,
            code=out.code,
            columns=sandbox_result.columns,
            rows=sandbox_result.rows,
            row_count=sandbox_result.row_count,
            key_numbers=keys.key_numbers,
            queries=queries,
            summary=keys.summary,
        )
        return result, total

    def _execute(self, state: LoomState, out: ExecutorOutput) -> SandboxResult:
        assert state.semantic is not None
        req = SandboxRequest(
            language=out.language.value,
            code=out.code,
            duckdb_path=str(self.settings.duckdb_path),
            row_limit=self.settings.sandbox_row_limit,
            timeout_s=self.settings.sandbox_timeout_s,
            memory_mb=self.settings.sandbox_memory_mb,
            tables=state.semantic.tables if out.language == ExecutionLanguage.PANDAS else [],
        )
        with self.tracer.span("sandbox", input=out.code, kind="tool"):
            return self.executor.run(req)

    def _key_numbers(
        self, state: LoomState, step: PlanStep, res: SandboxResult
    ) -> tuple[KeyNumbersOutput, LLMUsage]:
        version, text = prompt("key_numbers")
        sample = {"columns": res.columns, "rows": res.rows[:MAX_ROWS_FOR_SUMMARY]}
        user = (
            f"Question: {state.question}\nStep: {step.description}\n"
            f"Total rows: {res.row_count}\nResult sample (JSON): {json.dumps(sample, default=str)}"
        )
        out, u = self.provider.complete_json(
            system=text, messages=[LLMMessage(role="user", content=user)], schema=KeyNumbersOutput
        )
        self.tracer.generation("key_numbers", version, u, input=user, output=out)
        return out, u
