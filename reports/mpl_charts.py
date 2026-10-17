"""Static report charts with matplotlib (no browser needed, unlike Plotly image export).

Each function returns PNG bytes. Styling mirrors the web dashboard: same palette,
values printed on bars, light grid, no chart junk.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from core.palette import FAIL_COLOR, MUTED, ORANGE_RAMP, ORDINAL_BLUE, RISK_COLORS, STATUS  # noqa: E402
from reports.scripts import configure_matplotlib  # noqa: E402

DPI = 200
configure_matplotlib()  # Indian-script names in chart labels draw as glyphs, not boxes
TEXT = "#2b2b2b"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8c6c0")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def fail_rate_by_subject(by_subject: pd.DataFrame) -> bytes:
    data = by_subject.dropna(subset=["fail_rate"]).sort_values("fail_rate")
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * max(len(data), 2) + 0.8))
    bars = ax.barh(data["subject"].astype(str), data["fail_rate"], color=FAIL_COLOR, height=0.6)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in data["fail_rate"]], padding=3, fontsize=8, color=TEXT)
    ax.set_xlabel("Fail rate (%)", fontsize=8)
    ax.set_xlim(0, max(float(data["fail_rate"].max()) * 1.2, 5) if len(data) else 5)
    ax.grid(axis="x", color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    return _png(fig)


def fail_heatmap(matrix: pd.DataFrame) -> bytes:
    z = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 0.55 * len(matrix.index) + 1.3))
    cmap = LinearSegmentedColormap.from_list("orange", ORANGE_RAMP)
    vmax = max(40.0, float(np.nanmax(z)) if np.isfinite(z).any() else 40.0)
    image = ax.imshow(np.nan_to_num(z, nan=0.0), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [str(c) for c in matrix.columns], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)), [f"Section {s}" for s in matrix.index], fontsize=8)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            value = z[i, j]
            label = "–" if np.isnan(value) else f"{value:.1f}%"
            color = "white" if not np.isnan(value) and value > vmax * 0.55 else TEXT
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.ax.tick_params(labelsize=7)
    bar.set_label("Fail rate (%)", fontsize=7)
    return _png(fig)


def grade_distribution(grades: pd.DataFrame) -> bytes:
    passing = ["C", "B", "B+", "A", "A+", "O"]
    ramp = dict(zip(passing, ORDINAL_BLUE, strict=True))
    colors = [STATUS["critical"] if g == "F" else ramp.get(g, MUTED) for g in grades["grade"]]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bars = ax.bar(grades["grade"], grades["count"], color=colors, width=0.65)
    ax.bar_label(
        bars,
        labels=[f"{c} ({s:.0f}%)" for c, s in zip(grades["count"], grades["share"], strict=True)],
        padding=2,
        fontsize=7,
        color=TEXT,
    )
    ax.set_ylabel("Records", fontsize=8)
    ax.set_ylim(0, max(float(grades["count"].max()) * 1.2, 1))
    ax.grid(axis="y", color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    return _png(fig)


def co_attainment(co_table: pd.DataFrame, settings) -> bytes:
    level_colors = {0: STATUS["critical"], 1: ORDINAL_BLUE[0], 2: ORDINAL_BLUE[2], 3: ORDINAL_BLUE[5]}
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    colors = [level_colors[int(lv)] for lv in co_table["level"]]
    bars = ax.bar(co_table["co"].astype(str), co_table["pct_attained"], color=colors, width=0.6)
    ax.bar_label(
        bars,
        labels=[f"{p:.0f}% · L{int(lv)}" for p, lv in zip(co_table["pct_attained"], co_table["level"], strict=True)],
        padding=2,
        fontsize=7,
        color=TEXT,
    )
    target = settings.threshold_for_level(settings.target_level)
    # The target goes in a legend under the chart so it never collides with bar labels.
    ax.axhline(
        target,
        color=STATUS["critical"],
        linestyle="--",
        linewidth=1,
        label=f"Target: Level {settings.target_level} ({target:g}% of students)",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7, frameon=False)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Students attaining (%)", fontsize=8)
    ax.grid(axis="y", color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    return _png(fig)


def risk_levels(counts: pd.Series) -> bytes:
    order = ["High", "Medium", "Low"]
    counts = counts.reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    bars = ax.bar([f"{lv}" for lv in order], counts.values, color=[RISK_COLORS[lv] for lv in order], width=0.6)
    ax.bar_label(bars, padding=2, fontsize=8, color=TEXT)
    ax.set_ylabel("Students", fontsize=8)
    ax.set_ylim(0, max(float(counts.max()) * 1.2, 1))
    _style(ax)
    return _png(fig)
