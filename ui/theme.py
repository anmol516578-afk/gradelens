"""Visual system: palette, chart styling and the small amount of custom CSS.

Colors come from a validated, colour-blind-checked reference palette:

* categorical hues are assigned in a fixed order and follow the entity
  (Section A is always blue, even when filters hide other sections);
* magnitudes use a single-hue ramp, light -> dark;
* improvement vs decline is diverging (blue <-> red, grey midpoint);
* status colours (pass/fail, risk) are reserved and always shipped with a
  text label or icon, never colour alone.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SCORE_COLOR = SERIES[0]  # anything measuring marks / scores
FAIL_COLOR = SERIES[1]  # anything measuring fail rates
FAIL_DIM = "#f6c3ad"  # same hue, de-emphasised (bars outside the drill-down)
OTHER_COLOR = "#a8a69f"

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORDINAL_BLUE = ["#86b6ef", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # steps 250-700 (>= 2:1 on light)
ORANGE_RAMP = ["#fde8de", "#f9c3a9", "#f39a74", "#eb6834", "#c4501f", "#933a14", "#62260c"]

IMPROVED, STABLE, DECLINED = "#2a78d6", "#b5b3ab", "#e34948"

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
RESULT_COLORS = {"Pass": STATUS["good"], "Fail": STATUS["critical"], "Incomplete": OTHER_COLOR}
RISK_COLORS = {"High": STATUS["critical"], "Medium": STATUS["warning"], "Low": STATUS["good"]}
RISK_ICONS = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}

MUTED = "#898781"
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def section_colors(all_sections: list[str]) -> dict[str, str]:
    """Stable colour per section, derived from the full (unfiltered) list."""
    ordered = sorted(all_sections)
    return {s: (SERIES[i] if i < len(SERIES) else OTHER_COLOR) for i, s in enumerate(ordered)}


def risk_label(level: str) -> str:
    return f"{RISK_ICONS.get(level, '')} {level}".strip()


def style_figure(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 12, "t": 28 if legend else 12, "b": 8},
        font={"family": FONT, "size": 13},
        hoverlabel={"font": {"family": FONT, "size": 13}},
        barcornerradius=4,
        bargap=0.28,
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0, "title": None},
        title=None,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(zeroline=False)
    return fig


PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
.gl-hero {padding: 1.6rem 1.8rem; border-radius: 14px;
  border: 1px solid rgba(127,127,127,0.18);
  background: linear-gradient(135deg, rgba(42,120,214,0.10), rgba(27,175,122,0.06));}
.gl-hero h1 {margin: 0 0 .35rem 0; font-size: 2.1rem;}
.gl-hero p {margin: 0; opacity: .85; font-size: 1.05rem;}
.gl-feature {padding: .9rem 1rem; border-radius: 12px; height: 100%;
  border: 1px solid rgba(127,127,127,0.18);}
.gl-feature b {display:block; margin-bottom:.25rem;}
.gl-chip {display:inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .82rem;
  border: 1px solid rgba(127,127,127,0.30); margin-right: .35rem;}
.gl-muted {opacity: .72; font-size: .9rem;}
h4 {margin-bottom: .1rem !important;}
@media (max-width: 640px) {
  .block-container {padding-left: 1rem; padding-right: 1rem;}
  .gl-hero h1 {font-size: 1.6rem;}
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
