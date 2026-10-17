"""Plotly figure builders. Pure functions: data in, figure out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ui.theme import (
    DECLINED,
    FAIL_COLOR,
    FAIL_DIM,
    IMPROVED,
    MUTED,
    ORANGE_RAMP,
    ORDINAL_BLUE,
    RISK_COLORS,
    SCORE_COLOR,
    STABLE,
    STATUS,
    style_figure,
)


def _pct(value: float) -> str:
    return "–" if pd.isna(value) else f"{value:.1f}%"


def _ramp_scale(ramp: list[str]) -> list[list]:
    return [[i / (len(ramp) - 1), color] for i, color in enumerate(ramp)]


def fail_rate_bar(summary: pd.DataFrame, category: str, horizontal: bool = False, highlight: str | None = None):
    """Fail rate per category; the drilled-into category stays full strength."""
    data = summary.dropna(subset=["fail_rate"]).sort_values("fail_rate", ascending=horizontal)
    # Highlight with colours, not a per-bar opacity array: opacity arrays combined with text
    # labels stop Plotly from emitting click-selection events (see DEBUG_LOG #6).
    colors = [FAIL_COLOR if highlight in (None, c) else FAIL_DIM for c in data[category]]
    custom = np.stack([data["fails"], data["students"], data["avg_score"].round(1)], axis=-1) if len(data) else None
    common = {"marker": {"color": colors}, "customdata": custom}
    hover = (
        f"<b>%{{{'y' if horizontal else 'x'}}}</b><br>Fail rate %{{{'x' if horizontal else 'y'}:.1f}}%"
        "<br>Fails %{customdata[0]} · Students %{customdata[1]}<br>Avg score %{customdata[2]}%<extra></extra>"
    )
    labels = [_pct(v) for v in data["fail_rate"]]
    if horizontal:
        trace = go.Bar(y=data[category], x=data["fail_rate"], orientation="h", text=labels, **common)
    else:
        trace = go.Bar(x=data[category], y=data["fail_rate"], text=labels, **common)
    fig = go.Figure(trace)
    fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate=hover)
    top = float(data["fail_rate"].max()) if len(data) else 10.0
    axis = {"title": "Fail rate (%)", "ticksuffix": "%", "range": [0, max(top * 1.22, 5.0)]}
    if horizontal:
        fig.update_xaxes(**axis, showgrid=True)
        fig.update_yaxes(title=None, type="category")
    else:
        fig.update_yaxes(**axis)
        fig.update_xaxes(title=None, type="category")
    return style_figure(fig, height=max(260, 46 * len(data) + 60) if horizontal else 300, legend=False)


def fail_heatmap(matrix: pd.DataFrame):
    """Section x subject fail-rate heatmap with values printed in each cell."""
    z = matrix.to_numpy(dtype=float)
    text = [[_pct(v) for v in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(matrix.columns),
            y=[f"Section {s}" for s in matrix.index],
            text=text,
            texttemplate="%{text}",
            colorscale=_ramp_scale(ORANGE_RAMP),
            zmin=0,
            zmax=max(40.0, float(np.nanmax(z)) if np.isfinite(z).any() else 40.0),
            colorbar={"title": "Fail %", "ticksuffix": "%", "thickness": 12},
            hoverinfo="skip",
            xgap=2,
            ygap=2,
        )
    )
    # Plotly heatmaps emit no selection events in Streamlit, so an invisible marker sits on
    # every cell to catch clicks (and carry the tooltip). See DEBUG_LOG #7.
    cells = [
        (y, x, v) for y, row in zip(matrix.index, z, strict=True) for x, v in zip(matrix.columns, row, strict=True)
    ]
    fig.add_trace(
        go.Scatter(
            x=[c[1] for c in cells],
            y=[f"Section {c[0]}" for c in cells],
            mode="markers",
            marker={"size": 44, "opacity": 0, "symbol": "square"},
            customdata=[_pct(c[2]) for c in cells],
            hovertemplate="<b>%{y} · %{x}</b><br>Fail rate %{customdata}<br><i>Click to drill down</i><extra></extra>",
            showlegend=False,
        )
    )
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category", side="bottom")
    return style_figure(fig, height=90 + 56 * len(matrix.index), legend=False)


def score_histogram(df: pd.DataFrame, pass_pct: float):
    fig = px.histogram(df, x="score_pct", nbins=20, range_x=[0, 100], color_discrete_sequence=[SCORE_COLOR])
    fig.update_traces(
        marker_line_width=2, marker_line_color="rgba(0,0,0,0)", hovertemplate="%{x}%: %{y} records<extra></extra>"
    )
    fig.add_vline(
        x=pass_pct, line_dash="dash", line_color=STATUS["critical"], line_width=2,
        annotation_text=f"Pass mark {pass_pct:g}%", annotation_position="top left",
        annotation_font_color=STATUS["critical"],
    )  # fmt: skip
    fig.update_xaxes(title="Score (%)", ticksuffix="%")
    fig.update_yaxes(title="Records", showgrid=True)
    return style_figure(fig, height=300, legend=False)


def grade_bar(dist: pd.DataFrame):
    """Grade bands; passing grades on an ordinal blue ramp, F in the fail status colour."""
    passing = ["C", "B", "B+", "A", "A+", "O"]
    ramp = dict(zip(passing, ORDINAL_BLUE, strict=True))
    colors = [STATUS["critical"] if g == "F" else ramp[g] for g in dist["grade"]]
    fig = go.Figure(
        go.Bar(
            x=dist["grade"],
            y=dist["count"],
            marker_color=colors,
            text=[f"{c} ({s:.0f}%)" for c, s in zip(dist["count"], dist["share"], strict=True)],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Grade %{x}: %{y} records<extra></extra>",
        )
    )
    fig.update_xaxes(title=None, type="category")
    fig.update_yaxes(title="Records", showgrid=True, range=[0, max(float(dist["count"].max()) * 1.18, 1.0)])
    return style_figure(fig, height=300, legend=False)


def attendance_scatter(df: pd.DataFrame, threshold: float, pass_pct: float, colors: dict[str, str]):
    data = df.dropna(subset=["attendance_pct", "score_pct"])
    fig = px.scatter(
        data,
        x="attendance_pct",
        y="score_pct",
        color="section",
        color_discrete_map=colors,
        category_orders={"section": sorted(colors)},
        custom_data=["student_name", "roll_no", "subject"],
        opacity=0.65,
    )
    fig.update_traces(
        marker={"size": 8, "line": {"width": 1, "color": "rgba(255,255,255,0.8)"}},
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>%{customdata[2]}"
        "<br>Attendance %{x:.0f}% · Score %{y:.1f}%<extra></extra>",
    )
    if len(data) >= 3 and data["attendance_pct"].nunique() > 1:
        slope, intercept = np.polyfit(data["attendance_pct"], data["score_pct"], 1)
        xs = np.array([data["attendance_pct"].min(), data["attendance_pct"].max()])
        fig.add_trace(
            go.Scatter(
                x=xs, y=slope * xs + intercept, mode="lines", name="Trend",
                line={"color": MUTED, "width": 2, "dash": "dash"}, hoverinfo="skip",
            )
        )  # fmt: skip
    fig.add_vline(x=threshold, line_dash="dot", line_color=MUTED, annotation_text=f"{threshold:g}% attendance")
    fig.add_hline(y=pass_pct, line_dash="dot", line_color=STATUS["critical"], annotation_text="Pass mark")
    fig.update_xaxes(title="Attendance (%)", ticksuffix="%")
    fig.update_yaxes(title="Score (%)", ticksuffix="%", showgrid=True)
    fig = style_figure(fig, height=380, legend=True)
    fig.update_layout(legend_title_text="Section")
    return fig


def attendance_band_bar(bands: pd.DataFrame):
    data = bands.dropna(subset=["avg_score"])
    fig = go.Figure(
        go.Bar(
            x=data["band"].astype(str),
            y=data["avg_score"],
            marker_color=SCORE_COLOR,
            text=[_pct(v) for v in data["avg_score"]],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([data["records"], data["fail_rate"].fillna(0).round(1)], axis=-1),
            hovertemplate="<b>%{x}</b><br>Average score %{y:.1f}%<br>Records %{customdata[0]}"
            " · Fail rate %{customdata[1]}%<extra></extra>",
        )
    )
    fig.update_xaxes(title="Attendance band", type="category")
    fig.update_yaxes(title="Average score (%)", ticksuffix="%", range=[0, 105], showgrid=True)
    return style_figure(fig, height=380, legend=False)


def mst_trend_bar(summary: pd.DataFrame, by: str = "subject"):
    """100% stacked bars: share of students improving / stable / declining."""
    data = summary.copy()
    totals = data[["Improved", "Stable", "Declined"]].sum(axis=1).replace(0, np.nan)
    fig = go.Figure()
    for trend, color in (("Declined", DECLINED), ("Stable", STABLE), ("Improved", IMPROVED)):
        share = data[trend] / totals * 100
        fig.add_trace(
            go.Bar(
                y=data[by], x=share, name=trend, orientation="h", marker_color=color,
                text=[f"{v:.0f}%" if v >= 7 else "" for v in share.fillna(0)], textposition="inside",
                customdata=data[trend], insidetextanchor="middle",
                hovertemplate=f"<b>%{{y}}</b><br>{trend}: %{{customdata}} (%{{x:.1f}}%)<extra></extra>",
            )
        )  # fmt: skip
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_xaxes(title="Share of students (%)", ticksuffix="%", range=[0, 100])
    fig.update_yaxes(title=None, type="category")
    return style_figure(fig, height=max(240, 44 * len(data) + 80), legend=True)


def _closed(values: pd.Series) -> list:
    """Repeat the first point so a radar outline closes."""
    return list(values) + [values.iloc[0]] if len(values) else []


def profile_radar(subjects: pd.DataFrame, student_name: str):
    data = subjects.dropna(subset=["score_pct"])
    theta = _closed(data["subject"])
    close = _closed
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=close(data["class_avg"]), theta=theta, name="Class average", line={"color": MUTED, "dash": "dash"},
            hovertemplate="%{theta}: %{r:.1f}%<extra>Class average</extra>",
        )
    )  # fmt: skip
    fig.add_trace(
        go.Scatterpolar(
            r=close(data["score_pct"]), theta=theta, name=student_name, fill="toself",
            line={"color": SCORE_COLOR, "width": 2}, fillcolor="rgba(42,120,214,0.18)",
            hovertemplate="%{theta}: %{r:.1f}%<extra>" + student_name + "</extra>",
        )
    )  # fmt: skip
    fig.update_layout(
        polar={"radialaxis": {"range": [0, 100], "ticksuffix": "%", "angle": 90, "tickfont": {"size": 10}}}
    )
    fig = style_figure(fig, height=380, legend=True)
    # Wide side margins so long subject names ("Computer Networks") are not clipped.
    fig.update_layout(margin={"l": 130, "r": 130, "t": 48, "b": 36})
    return fig


def profile_components(subjects: pd.DataFrame):
    """MST-1 -> MST-2 -> End term per subject, in time order on an ordinal ramp."""
    steps = [("mst1_pct", "MST-1", ORDINAL_BLUE[0]), ("mst2_pct", "MST-2", ORDINAL_BLUE[2]),
             ("end_term_pct", "End term", ORDINAL_BLUE[5])]  # fmt: skip
    fig = go.Figure()
    for column, label, color in steps:
        if column in subjects:
            fig.add_trace(
                go.Bar(
                    x=subjects["subject"], y=subjects[column], name=label, marker_color=color,
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<extra></extra>",
                )
            )  # fmt: skip
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title="Score (%)", ticksuffix="%", range=[0, 105], showgrid=True)
    fig.update_xaxes(title=None, type="category")
    return style_figure(fig, height=340, legend=True)


def profile_attendance(subjects: pd.DataFrame, threshold: float):
    data = subjects.dropna(subset=["attendance_pct"])
    below = data["attendance_pct"] < threshold
    fig = go.Figure(
        go.Bar(
            x=data["subject"],
            y=data["attendance_pct"],
            marker_color=[STATUS["critical"] if b else SCORE_COLOR for b in below],
            # Labels sit inside the bars so they never collide with the requirement line.
            text=[f"{v:.0f}%" + (" ⚠" if b else "") for v, b in zip(data["attendance_pct"], below, strict=True)],
            textposition="inside",
            insidetextanchor="end",
            textfont={"color": "white"},
            hovertemplate="<b>%{x}</b><br>Attendance %{y:.0f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=threshold, line_dash="dash", line_color=MUTED, annotation_text=f"Required {threshold:g}%",
        annotation_position="top left", annotation_yshift=2,
    )  # fmt: skip
    fig.update_yaxes(title="Attendance (%)", ticksuffix="%", range=[0, 110], showgrid=True)
    fig.update_xaxes(title=None, type="category")
    return style_figure(fig, height=300, legend=False)


def co_attainment_bar(co_table: pd.DataFrame, settings):
    level_colors = {0: STATUS["critical"], 1: ORDINAL_BLUE[0], 2: ORDINAL_BLUE[2], 3: ORDINAL_BLUE[5]}
    fig = go.Figure(
        go.Bar(
            x=co_table["co"],
            y=co_table["pct_attained"],
            marker_color=[level_colors[int(lv)] for lv in co_table["level"]],
            text=[f"{p:.0f}% · Level {lv}" for p, lv in zip(co_table["pct_attained"], co_table["level"], strict=True)],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([co_table["students_attained"], co_table["students_assessed"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}% of students attained"
            "<br>%{customdata[0]} of %{customdata[1]} students<extra></extra>",
        )
    )
    for level in (1, 2, 3):
        y = settings.threshold_for_level(level)
        is_target = level == settings.target_level
        fig.add_hline(
            y=y,
            line_dash="dash" if is_target else "dot",
            line_width=2 if is_target else 1,
            line_color=STATUS["critical"] if is_target else MUTED,
            annotation_text=f"Level {level} ({y:g}%)" + (" · target" if is_target else ""),
            annotation_position="right",
            annotation_font_size=11,
        )
    fig.update_yaxes(title="Students attaining the CO (%)", ticksuffix="%", range=[0, 110], showgrid=True)
    fig.update_xaxes(title=None, type="category")
    fig = style_figure(fig, height=360, legend=False)
    fig.update_layout(margin={"l": 8, "r": 130, "t": 12, "b": 8})
    return fig


def risk_distribution(counts: pd.Series):
    order = ["High", "Medium", "Low"]
    counts = counts.reindex(order, fill_value=0)
    fig = go.Figure(
        go.Bar(
            x=[f"{lv} risk" for lv in order],
            y=counts.values,
            marker_color=[RISK_COLORS[lv] for lv in order],
            text=counts.values,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y} students<extra></extra>",
        )
    )
    fig.update_yaxes(title="Students", showgrid=True)
    fig.update_xaxes(title=None, type="category")
    return style_figure(fig, height=280, legend=False)


def feature_importance_bar(importance: pd.DataFrame):
    data = importance.sort_values("importance")
    fig = go.Figure(
        go.Bar(
            y=data["label"], x=data["importance"], orientation="h", marker_color=SCORE_COLOR,
            text=[f"{v:.2f}" for v in data["importance"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )  # fmt: skip
    fig.update_xaxes(title="Relative importance", showgrid=True)
    fig.update_yaxes(title=None, type="category")
    return style_figure(fig, height=60 + 42 * len(data), legend=False)


def confusion_heatmap(matrix: pd.DataFrame):
    z = matrix.to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=z, x=list(matrix.columns), y=list(matrix.index), text=z, texttemplate="%{text}",
            colorscale=[[0, "#cde2fb"], [1, "#184f95"]], showscale=False, xgap=2, ygap=2,
            hovertemplate="Actual %{y} · Predicted %{x}: %{z}<extra></extra>",
        )
    )  # fmt: skip
    fig.update_xaxes(title="Predicted", type="category", side="bottom")
    fig.update_yaxes(title="Actual", type="category", autorange="reversed")
    return style_figure(fig, height=260, legend=False)


def auto_chart(table: pd.DataFrame):
    """Best-effort chart for an AI result table: one category + one number -> bar."""
    if table is None or table.empty or len(table) > 40:
        return None
    numeric = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    categorical = [c for c in table.columns if c not in numeric]
    if len(categorical) != 1 or not numeric:
        return None
    x, y = categorical[0], numeric[0]
    fig = go.Figure(
        go.Bar(
            x=table[x].astype(str), y=table[y], marker_color=SCORE_COLOR,
            text=[f"{v:.1f}" if isinstance(v, float) else str(v) for v in table[y]],
            textposition="outside", cliponaxis=False,
            hovertemplate=f"<b>%{{x}}</b><br>{y}: %{{y}}<extra></extra>",
        )
    )  # fmt: skip
    fig.update_xaxes(title=x, type="category")
    fig.update_yaxes(title=y, showgrid=True)
    return style_figure(fig, height=320, legend=False)


def reason_bar(reasons: pd.DataFrame):
    """How many flagged students show each warning sign (horizontal, most common on top)."""
    data = reasons.sort_values("students")
    fig = go.Figure(
        go.Bar(
            y=data["reason"], x=data["students"], orientation="h", marker_color=FAIL_COLOR,
            text=data["students"], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x} students<extra></extra>",
        )
    )  # fmt: skip
    fig.update_xaxes(title="Students", showgrid=True, range=[0, max(float(data["students"].max()) * 1.15, 1.0)])
    fig.update_yaxes(title=None, type="category")
    return style_figure(fig, height=60 + 40 * len(data), legend=False)
