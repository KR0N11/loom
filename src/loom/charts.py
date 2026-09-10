"""Chart rendering for memos.

Approach: the narrator proposes a ChartSpec (kind, x, y column names); this
module draws it from an ExecutionResult's rows with matplotlib (PNG, for the
markdown memo) and plotly (HTML twin, for the Streamlit UI). Nothing here
knows about LLMs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from loom.agents.schemas import ChartSpec, ExecutionResult  # noqa: E402


class ChartColumnError(ValueError):
    """The spec names a column the result does not have."""


def _column(ex: ExecutionResult, name: str) -> list:
    # The narrator may hallucinate a column; failing here keeps the memo honest.
    if name not in ex.columns:
        raise ChartColumnError(f"column {name!r} not in result columns {ex.columns}")
    idx = ex.columns.index(name)
    return [r[idx] for r in ex.rows]


def render_chart(spec: ChartSpec, ex: ExecutionResult, out_dir: Path, stem: str = "chart") -> Path:
    """Write <out_dir>/<stem>.png (and .html via plotly); return the PNG path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = [str(v) for v in _column(ex, spec.x)]
    ys = [float(v) if v is not None else 0.0 for v in _column(ex, spec.y)]
    fig, ax = plt.subplots(figsize=(8, 4))
    # Only two chart kinds are supported on purpose; more kinds means more ways to mislead.
    if spec.kind == "line":
        ax.plot(xs, ys, marker="o")
    else:
        ax.bar(xs, ys)
    ax.set_title(spec.title)
    ax.set_xlabel(spec.x)
    ax.set_ylabel(spec.y)
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    _render_plotly(spec, xs, ys, out_dir / f"{stem}.html")
    return png


def _render_plotly(spec: ChartSpec, xs: list[str], ys: list[float], path: Path) -> None:
    import plotly.graph_objects as go

    trace = (
        go.Scatter(x=xs, y=ys, mode="lines+markers") if spec.kind == "line" else go.Bar(x=xs, y=ys)
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(title=spec.title, xaxis_title=spec.x, yaxis_title=spec.y)
    fig.write_html(str(path), include_plotlyjs="cdn")
