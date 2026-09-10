"""Verifier worker agent.

Approach: trust nothing the executor said. For every headline number the LLM
writes an ALTERNATE query via a different aggregation path, the sandbox runs
it, and the two values must agree within tolerance. Around that sit
deterministic checks (units, sample size, time range, nulls, duplicates,
truncation) that need no LLM. In eval injection mode the executor output is
corrupted first so the catch rate can be measured. Pattern: worker agent.
"""

from __future__ import annotations

import re

from loom.agents.injection import inject
from loom.agents.prompts import prompt
from loom.agents.schemas import (
    AnalysisType,
    CheckStatus,
    ExecutionLanguage,
    ExecutionResult,
    KeyNumber,
    QueryRecord,
    VerificationCheck,
    VerificationReport,
    VerifierOutput,
)
from loom.agents.state import LoomState
from loom.agents.strategies import get_strategy
from loom.config import Settings, get_settings
from loom.llm.base import LLMMessage, LLMProvider, LLMUsage
from loom.sandbox.base import SandboxExecutor, SandboxRequest
from loom.tracing import Tracer

REL_TOLERANCE = 0.01
ABS_FLOOR = 1e-9
SMALL_SAMPLE = 30
_RATE_WORDS = ("avg", "mean", "rate", "share", "ratio", "median", "pct", "percent")
_DATE_WORDS = re.compile(r"\b(date|year|month|day|between|date_trunc|interval)\b", re.IGNORECASE)


def _close(a: float, b: float) -> bool:
    """Within 1% relative, with an absolute floor so 0 vs 1e-12 still passes."""
    return abs(a - b) <= max(ABS_FLOOR, REL_TOLERANCE * max(abs(a), abs(b)))


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _to_float(value: object) -> float | None:
    """Accept ints, floats and numeric strings (a sandbox may serialise decimals as text)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


class VerifierAgent:
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
        assert state.plan is not None and state.semantic is not None
        executions = state.executions
        # Injection (eval only) happens here so the narrator downstream sees the corrupted numbers.
        if state.injection:
            self.tracer.event("injection", mode=state.injection)
            executions = inject(executions, state.injection)
        usage = state.usage
        checks: list[VerificationCheck] = []
        reconciled: list[KeyNumber] = []
        queries: list[QueryRecord] = []
        wanted = set(get_strategy(state.plan.analysis_type).verifier_checks())
        for ex in executions:
            alt_checks, alt_keys, alt_queries, alt_nulls, u = self._reconcile(state, ex)
            usage = usage + u
            checks.extend(alt_checks)
            reconciled.extend(alt_keys)
            queries.extend(alt_queries)
            checks.extend(self._deterministic_checks(state, ex, wanted, alt_nulls))
        flags = [f"{c.name}: {c.detail}" for c in checks if c.status != CheckStatus.PASS]
        report = VerificationReport(
            overall=_overall(checks),
            checks=checks,
            reconciled_numbers=reconciled,
            flags=flags,
            queries=queries,
        )
        return {"verification": report, "executions": executions, "usage": usage}

    def _reconcile(
        self, state: LoomState, ex: ExecutionResult
    ) -> tuple[list[VerificationCheck], list[KeyNumber], list[QueryRecord], bool, LLMUsage]:
        assert state.semantic is not None
        checks: list[VerificationCheck] = []
        # Nothing to reconcile when the executor produced no headline numbers.
        if not ex.key_numbers:
            checks.append(
                VerificationCheck(
                    name=f"reconcile_step_{ex.step_id}",
                    status=CheckStatus.SKIPPED,
                    detail="no key numbers to re-derive",
                )
            )
            return checks, [], [], False, LLMUsage(model=self.provider.model)
        version, text = prompt("verifier")
        system = text.format(context_text=state.semantic.context_text)
        keys_desc = "\n".join(
            f"- {k.name} = {k.value} {k.unit} ({k.derivation})" for k in ex.key_numbers
        )
        user = (
            f"Question: {state.question}\nOriginal {ex.language} code:\n{ex.code}\n"
            f"Key numbers to re-derive:\n{keys_desc}"
        )
        out, u = self.provider.complete_json(
            system, [LLMMessage(role="user", content=user)], VerifierOutput
        )
        self.tracer.generation("verifier", version, u, input=user, output=out)
        req = SandboxRequest(
            language="sql",
            code=out.alternate_sql,
            duckdb_path=str(self.settings.duckdb_path),
            row_limit=self.settings.sandbox_row_limit,
            timeout_s=self.settings.sandbox_timeout_s,
            memory_mb=self.settings.sandbox_memory_mb,
        )
        with self.tracer.span("sandbox_verify", input=out.alternate_sql, kind="tool"):
            res = self.executor.run(req)
        record = QueryRecord(
            attempt=1,
            language=ExecutionLanguage.SQL,
            code=out.alternate_sql,
            ok=res.ok,
            error=res.error,
            row_count=res.row_count,
            elapsed_s=res.elapsed_s,
        )
        # A failed alternate query is a warning, not proof of error: we simply could not confirm.
        if not res.ok or not res.rows:
            checks.append(
                VerificationCheck(
                    name=f"reconcile_step_{ex.step_id}",
                    status=CheckStatus.WARN,
                    detail=f"alternate query failed: {res.error or 'no rows'}",
                )
            )
            return checks, [], [record], False, u
        alt = dict(zip([_norm(c) for c in res.columns], res.rows[0], strict=False))
        has_nulls = any(v is None for v in res.rows[0])
        reconciled: list[KeyNumber] = []
        for k in ex.key_numbers:
            observed = _to_float(alt.get(_norm(k.name)))
            # The alternate query must name every key number; a missing one cannot be confirmed.
            if observed is None:
                checks.append(
                    VerificationCheck(
                        name=f"reconcile:{k.name}",
                        status=CheckStatus.WARN,
                        detail="alternate query did not return this number",
                        expected=k.value,
                    )
                )
                continue
            ok = _close(k.value, observed)
            checks.append(
                VerificationCheck(
                    name=f"reconcile:{k.name}",
                    status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                    detail=(
                        "re-derived value agrees within 1%"
                        if ok
                        else f"executor said {k.value}, alternate path says {observed}"
                    ),
                    expected=k.value,
                    observed=observed,
                )
            )
            reconciled.append(k.model_copy(update={"value": observed}))
        return checks, reconciled, [record], has_nulls, u

    def _deterministic_checks(
        self, state: LoomState, ex: ExecutionResult, wanted: set[str], alt_nulls: bool
    ) -> list[VerificationCheck]:
        assert state.plan is not None and state.semantic is not None
        checks: list[VerificationCheck] = []
        sid = ex.step_id
        if "sample_size" in wanted:
            if ex.row_count == 0:
                checks.append(
                    _check(f"sample_size_{sid}", CheckStatus.FAIL, "query returned zero rows")
                )
            elif ex.row_count < SMALL_SAMPLE and any(
                w in k.name.lower() for k in ex.key_numbers for w in _RATE_WORDS
            ):
                checks.append(
                    _check(
                        f"sample_size_{sid}",
                        CheckStatus.WARN,
                        f"average/rate computed over only {ex.row_count} rows",
                    )
                )
            else:
                checks.append(
                    _check(f"sample_size_{sid}", CheckStatus.PASS, f"{ex.row_count} rows")
                )
        if "duplicate_check" in wanted:
            seen = {tuple(map(str, r)) for r in ex.rows}
            # Duplicate full rows usually mean a join fanned out; totals built on them are wrong.
            if len(seen) < len(ex.rows):
                checks.append(
                    _check(
                        f"duplicate_check_{sid}",
                        CheckStatus.FAIL,
                        f"{len(ex.rows) - len(seen)} duplicate rows in result",
                    )
                )
            else:
                checks.append(
                    _check(f"duplicate_check_{sid}", CheckStatus.PASS, "no duplicate rows")
                )
        if "row_truncation" in wanted:
            truncated = ex.row_count >= self.settings.sandbox_row_limit
            checks.append(
                _check(
                    f"row_truncation_{sid}",
                    CheckStatus.WARN if truncated else CheckStatus.PASS,
                    "result hit the row limit; aggregates may be partial"
                    if truncated
                    else "under row limit",
                )
            )
        if "unit_check" in wanted:
            known = _known_units(state)
            for k in ex.key_numbers:
                unit = k.unit.strip().lower()
                status = CheckStatus.PASS if (not unit or unit in known) else CheckStatus.WARN
                detail = (
                    "unit matches semantic layer"
                    if status == CheckStatus.PASS
                    else f"unit {k.unit!r} not defined in semantic layer"
                )
                checks.append(_check(f"unit_check:{k.name}", status, detail))
        if "null_check" in wanted:
            has_null = alt_nulls or any(c is None for r in ex.rows for c in r)
            checks.append(
                _check(
                    f"null_check_{sid}",
                    CheckStatus.WARN if has_null else CheckStatus.PASS,
                    "nulls present in result; aggregates may silently exclude rows"
                    if has_null
                    else "no nulls in result",
                )
            )
        if "time_range" in wanted and state.plan.analysis_type in (
            AnalysisType.TREND,
            AnalysisType.COMPARISON,
            AnalysisType.DRIVER,
        ):
            has_date = bool(_DATE_WORDS.search(ex.code))
            checks.append(
                _check(
                    f"time_range_{sid}",
                    CheckStatus.PASS if has_date else CheckStatus.WARN,
                    "query references a date/time column"
                    if has_date
                    else "no explicit time filter or grain in query",
                )
            )
        return checks


def _known_units(state: LoomState) -> set[str]:
    assert state.semantic is not None
    units: set[str] = {"count", "rows", "claims", "delays", "records", "%", "percent", "ratio"}
    # Units named anywhere in the retrieved docs count as known (columns and metrics carry them).
    for doc in state.semantic.docs:
        for m in re.finditer(r"Unit: ([^.\n]+)", doc.text):
            units.add(m.group(1).strip().lower())
    return units


def _check(name: str, status: CheckStatus, detail: str) -> VerificationCheck:
    return VerificationCheck(name=name, status=status, detail=detail)


def _overall(checks: list[VerificationCheck]) -> CheckStatus:
    statuses = {c.status for c in checks}
    if CheckStatus.FAIL in statuses:
        return CheckStatus.FAIL
    if CheckStatus.WARN in statuses:
        return CheckStatus.WARN
    return CheckStatus.PASS
