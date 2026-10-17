"""Colour palette shared by the web UI (Plotly) and the reports (matplotlib, Excel).

Validated, colour-blind-checked reference palette:

* categorical hues are assigned in a fixed order and follow the entity;
* magnitudes use a single-hue ramp, light -> dark;
* improvement vs decline is diverging (blue <-> red, grey midpoint);
* status colours (pass/fail, risk) are reserved and always shipped with a
  text label or icon, never colour alone.
"""

from __future__ import annotations

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
