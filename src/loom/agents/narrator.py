"""Narrator worker agent.

Approach: the LLM drafts the memo from the plan, the execution results and the
verification report; code then enforces the rules a prompt cannot guarantee:
a failed verification forces an "UNVERIFIED:" headline and the flags become
caveats, and every executed query (executor and verifier) is appended for
audit. Charts are rendered from real result columns only.
"""

from __future__ import annotations

import json
from pathlib import Path

from loom.agents.prompts import prompt
from loom.agents.schemas import (
    ChartSpec,
    CheckStatus,
    ExecutionResult,
    Memo,
    NarratorOutput,
    QueryRecord,
)
from loom.agents.state import LoomState
from loom.charts import ChartColumnError, render_chart
from loom.config import Settings, get_settings
from loom.llm.base import LLMMessage, LLMProvider
from loom.tracing import Tracer

UNVERIFIED_PREFIX = "UNVERIFIED:"
MAX_ROWS_IN_PROMPT = 20


class NarratorAgent:
    def __init__(
        self, provider: LLMProvider, settings: Settings | None = None, tracer: Tracer | None = None
    ) -> None:
        self.provider = provider
        self.settings = settings or get_settings()
        self.tracer = tracer or Tracer(None)

    def run(self, state: LoomState) -> dict:
        usage = state.usage
        # The failed-analysis path: no numbers, so no LLM call, just an honest memo.
        if not state.executions:
            return {"memo": _failure_memo(state), "usage": usage}
        version, text = prompt("narrator")
        user = self._user_prompt(state)
        out, u = self.provider.complete_json(
            text, [LLMMessage(role="user", content=user)], NarratorOutput
        )
        usage = usage + u
        self.tracer.generation("narrator", version, u, input=user, output=out)
        memo = self._finalize(state, out)
        return {"memo": memo, "usage": usage}

    def _user_prompt(self, state: LoomState) -> str:
        assert state.plan is not None
        results = [
            {
                "step_id": ex.step_id,
                "summary": ex.summary,
                "key_numbers": [k.model_dump() for k in ex.key_numbers],
                "columns": ex.columns,
                "rows": ex.rows[:MAX_ROWS_IN_PROMPT],
                "row_count": ex.row_count,
            }
            for ex in state.executions
        ]
        verification = (
            state.verification.model_dump(exclude={"queries"}) if state.verification else {}
        )
        return (
            f"Question: {state.question}\nPlan: {state.plan.model_dump_json()}\n"
            f"Results: {json.dumps(results, default=str)}\n"
            f"Verification: {json.dumps(verification, default=str)}"
        )

    def _finalize(self, state: LoomState, out: NarratorOutput) -> Memo:
        verification = state.verification
        headline = out.headline
        caveats = list(out.caveats)
        summary = "no verification run"
        if verification is not None:
            summary = f"{verification.overall.value.upper()}: " + "; ".join(
                verification.flags or ["all checks passed"]
            )
            # A failed reconciliation must be impossible to miss, whatever the model wrote.
            if verification.overall == CheckStatus.FAIL and not headline.startswith(
                UNVERIFIED_PREFIX
            ):
                headline = f"{UNVERIFIED_PREFIX} {headline}"
            for flag in verification.flags:
                if flag not in caveats:
                    caveats.append(flag)
        charts = self._render_charts(state, out.charts)
        appendix = [q for ex in state.executions for q in ex.queries]
        if verification is not None:
            appendix.extend(verification.queries)
        return Memo(
            title=out.title,
            headline=headline,
            findings=out.findings,
            recommendation=out.recommendation,
            caveats=caveats,
            charts=charts,
            appendix_queries=appendix,
            verification_summary=summary,
        )

    def _render_charts(self, state: LoomState, specs: list[ChartSpec]) -> list[ChartSpec]:
        out_dir = Path(self.settings.output_dir) / state.thread_id
        rendered: list[ChartSpec] = []
        for i, spec in enumerate(specs):
            ex = _result_with_columns(state.executions, spec)
            # A chart whose columns exist nowhere would plot nothing; drop it rather than guess.
            if ex is None:
                continue
            try:
                path = render_chart(spec, ex, out_dir, stem=f"chart_{i + 1}")
            except (ChartColumnError, ValueError, TypeError):
                continue
            rendered.append(spec.model_copy(update={"path": str(path)}))
        return rendered


def _result_with_columns(
    executions: list[ExecutionResult], spec: ChartSpec
) -> ExecutionResult | None:
    for ex in executions:
        if spec.x in ex.columns and spec.y in ex.columns and ex.rows:
            return ex
    return None


def _failure_memo(state: LoomState) -> Memo:
    return Memo(
        title="Analysis could not be completed",
        headline=f"{UNVERIFIED_PREFIX} no query succeeded for: {state.question}",
        findings=[],
        recommendation="Rephrase the question or check the dataset; no numbers were produced.",
        caveats=list(state.errors),
        charts=[],
        appendix_queries=[],
        verification_summary="not run",
    )


def memo_to_markdown(memo: Memo) -> str:
    """Render the memo plus the audit appendix as markdown."""
    lines = [f"# {memo.title}", "", f"**{memo.headline}**", "", "## Findings"]
    lines += [f"- {f}" for f in memo.findings] or ["- (none)"]
    lines += ["", "## Recommendation", "", memo.recommendation, "", "## Caveats"]
    lines += [f"- {c}" for c in memo.caveats] or ["- (none)"]
    lines += ["", "## Verification", "", memo.verification_summary]
    if memo.charts:
        lines += ["", "## Charts", ""]
        # memo.md is written next to the chart files, so link by file name only.
        lines += [f"![{c.title}]({Path(c.path).name})" for c in memo.charts if c.path]
    lines += ["", "## Audit appendix", "", "Every query executed, in order:", ""]
    for i, q in enumerate(memo.appendix_queries, 1):
        lines.append(_query_md(i, q))
    return "\n".join(lines) + "\n"


def _query_md(i: int, q: QueryRecord) -> str:
    status = "ok" if q.ok else f"error: {q.error}"
    return (
        f"### Query {i} (attempt {q.attempt}, {q.language.value}, {status}, {q.row_count} rows, "
        f"{q.elapsed_s:.2f}s)\n\n```{q.language.value}\n{q.code}\n```\n"
    )
