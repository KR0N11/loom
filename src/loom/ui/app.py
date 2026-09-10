"""Streamlit front end.

Approach: the UI is a pure client. It calls the FastAPI backend when it is
reachable and falls back to running the graph in-process otherwise, so the
demo works with a single `loom ui` command. All rendering is driven by the
same typed response models the API returns.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st

API_URL = os.environ.get("LOOM_API_URL", "http://localhost:8000")

EXAMPLES: dict[str, list[str]] = {
    "ttc": [
        "How many subway delays were there in 2024?",
        "What is the average subway delay in minutes by line?",
        "How did total bus delay minutes trend month by month in 2024?",
        "Compare streetcar delays on weekdays versus weekends.",
        "Which delay codes drove the increase in subway delay minutes from 2023 to 2024?",
    ],
    "nfip": [
        "What was the total amount paid on claims in Florida in 2022?",
        "What is the average claim severity by state since 2018?",
        "How did the number of claims trend by year of loss since 2015?",
        "Compare building loss ratio in Special Flood Hazard Areas versus outside them.",
        "Which flood zones contributed most to the increase in total paid from 2016 to 2017?",
        "What share of claims come from primary residences in Texas?",
    ],
}

STATUS_COLOR = {"pass": "#2e7d32", "warn": "#ef6c00", "fail": "#c62828", "skipped": "#757575"}


def api_alive() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def ask(question: str, dataset: str) -> dict[str, Any]:
    """Prefer the API; fall back to in-process so the UI works without a server."""
    if api_alive():
        resp = requests.post(
            f"{API_URL}/ask", json={"question": question, "dataset": dataset}, timeout=900
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data
    from loom.agents.supervisor import run_question
    from loom.api.app import persist_state, render_memo
    from loom.config import get_settings

    settings = get_settings()
    state = run_question(question, dataset, settings=settings)
    persist_state(state, settings)
    return {
        "thread_id": state.thread_id,
        "memo": state.memo.model_dump() if state.memo else None,
        "verification": state.verification.model_dump() if state.verification else None,
        "usage": state.usage.model_dump(),
        "errors": state.errors,
        "memo_markdown": render_memo(state),
    }


def list_memos() -> list[str]:
    if api_alive():
        try:
            data: list[str] = requests.get(f"{API_URL}/memos", timeout=5).json()
            return data
        except requests.RequestException:
            return []
    from loom.config import get_settings

    root = Path(get_settings().output_dir)
    return sorted(p.parent.name for p in root.glob("*/state.json")) if root.exists() else []


def render_verification(report: dict[str, Any] | None) -> None:
    if not report:
        st.info("No verification report.")
        return
    color = STATUS_COLOR.get(report["overall"], "#000")
    st.markdown(
        f"**Overall:** <span style='color:{color};font-weight:700'>{report['overall'].upper()}</span>",
        unsafe_allow_html=True,
    )
    rows = "".join(
        f"<tr><td>{c['name']}</td>"
        f"<td style='color:{STATUS_COLOR.get(c['status'], '#000')};font-weight:600'>{c['status']}</td>"
        f"<td>{c['detail']}</td><td>{c.get('expected') if c.get('expected') is not None else ''}</td>"
        f"<td>{c.get('observed') if c.get('observed') is not None else ''}</td></tr>"
        for c in report["checks"]
    )
    st.markdown(
        "<table><tr><th>Check</th><th>Status</th><th>Detail</th><th>Expected</th>"
        f"<th>Observed</th></tr>{rows}</table>",
        unsafe_allow_html=True,
    )
    # Flags are the human-readable reasons a memo is marked unverified.
    for flag in report.get("flags", []):
        st.warning(flag)


def render_charts(memo: dict[str, Any] | None) -> None:
    if not memo:
        return
    for chart in memo.get("charts", []):
        path = chart.get("path")
        if not path or not Path(path).exists():
            continue
        if path.endswith(".png"):
            st.image(path, caption=chart.get("title", ""))
        html_twin = Path(path).with_suffix(".html")
        if html_twin.exists():
            st.components.v1.html(html_twin.read_text(), height=450)


def render_appendix(memo: dict[str, Any] | None) -> None:
    if not memo:
        return
    with st.expander("Audit appendix: every query executed"):
        for q in memo.get("appendix_queries", []):
            status = "ok" if q["ok"] else f"error: {q.get('error')}"
            st.markdown(
                f"**Attempt {q['attempt']}** · {q['language']} · {status} · {q['row_count']} rows"
            )
            st.code(q["code"], language="sql" if q["language"] == "sql" else "python")


def render_answer(data: dict[str, Any]) -> None:
    usage = data.get("usage", {})
    st.caption(
        f"thread {data['thread_id']} · cost ${usage.get('cost_usd', 0):.4f} · "
        f"latency {usage.get('latency_s', 0):.1f}s"
    )
    for err in data.get("errors", []):
        st.error(err)
    # The UI renders charts and the appendix itself, so drop the markdown copies.
    body = data["memo_markdown"].split("## Audit appendix")[0]
    body = "\n".join(line for line in body.splitlines() if not line.startswith("!["))
    st.markdown(body)
    render_charts(data.get("memo"))
    st.subheader("Verification")
    render_verification(data.get("verification"))
    render_appendix(data.get("memo"))


def main() -> None:
    st.set_page_config(page_title="Loom", layout="wide")
    st.title("Loom: ask a question, get a verified memo")

    with st.sidebar:
        dataset = st.selectbox("Dataset", list(EXAMPLES), index=1)
        st.markdown("**Example questions**")
        for example in EXAMPLES[dataset]:
            if st.button(example, key=example):
                st.session_state["question"] = example
        st.markdown("---")
        st.markdown("**Past memos**")
        for thread_id in list_memos()[-15:][::-1]:
            if st.button(thread_id, key=f"memo-{thread_id}"):
                st.session_state["show_memo"] = thread_id

    question = st.text_area("Question", value=st.session_state.get("question", ""), height=80)
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Planning, executing, verifying, writing..."):
            try:
                st.session_state["answer"] = ask(question.strip(), dataset)
                st.session_state.pop("show_memo", None)
            except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
                st.error(f"Request failed: {exc}")

    # A past memo selected from the sidebar takes priority over the last live answer.
    if "show_memo" in st.session_state:
        thread_id = st.session_state["show_memo"]
        resp = requests.get(f"{API_URL}/memos/{thread_id}", timeout=10) if api_alive() else None
        if resp is not None and resp.status_code == 200:
            state = resp.json()
            from loom.agents.narrator import memo_to_markdown
            from loom.agents.schemas import Memo

            memo = state.get("memo")
            render_answer(
                {
                    "thread_id": thread_id,
                    "memo": memo,
                    "verification": state.get("verification"),
                    "usage": state.get("usage", {}),
                    "errors": state.get("errors", []),
                    "memo_markdown": memo_to_markdown(Memo.model_validate(memo))
                    if memo
                    else "# Analysis failed",
                }
            )
        else:
            st.info("Memo not available from the API.")
    elif "answer" in st.session_state:
        render_answer(st.session_state["answer"])


if __name__ == "__main__":
    main()
