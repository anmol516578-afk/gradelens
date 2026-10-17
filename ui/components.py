"""Small shared UI building blocks."""

from __future__ import annotations

import html
import math
import re

import pandas as pd
import streamlit as st

from ui import state

TAB_DASHBOARD = "📊 Dashboard"
TAB_AI = "🤖 Ask AI"
TAB_RISK = "⚠️ At-Risk"
TAB_CO = "🎯 CO Attainment"
TAB_PROFILE = "👤 Student Profile"
TAB_REPORTS = "📄 Reports"
TAB_QUALITY = "🧹 Data Quality"
TABS = [TAB_DASHBOARD, TAB_AI, TAB_RISK, TAB_CO, TAB_PROFILE, TAB_REPORTS, TAB_QUALITY]
TAB_KEY = "main_tab"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:.{digits}f}%"


def fmt_num(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{int(value):,}"


def chips(items: list[str]) -> None:
    """Pill labels. Values come from uploaded sheets, so they are HTML-escaped (see DEBUG_LOG #38)."""
    html_chips = " ".join(f'<span class="gl-chip">{html.escape(str(item))}</span>' for item in items)
    st.markdown(html_chips, unsafe_allow_html=True)


def heading(title: str, caption: str | None = None) -> None:
    st.markdown(f"#### {title}")
    if caption:
        st.caption(caption)


def open_student(roll_no: str) -> None:
    """Select a student and switch to the profile tab (call from callbacks)."""
    st.session_state["selected_student"] = roll_no
    st.session_state["profile_student"] = roll_no
    st.session_state[TAB_KEY] = TAB_PROFILE


def _load_sample_from_hero() -> None:
    state.load_sample("long")


def empty_state() -> None:
    st.markdown(
        """
        <div class="gl-hero">
          <h1>🎓 GradeLens</h1>
          <p>Upload a marks sheet and get an instant dashboard, an AI analyst you can ask questions in
          plain English, an at-risk student list, CO attainment and one-click reports for your HOD.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    features = [
        ("📊 Instant dashboard", "Fail rates by section and subject, grade bands, attendance impact, MST trends."),
        ("🤖 Ask in plain English", "“Which section has the highest fail rate in Python?” — with charts."),
        ("⚠️ Early warnings", "At-risk students with reasons, CO attainment, PDF & Excel reports."),
    ]
    for col, (title, text) in zip((c1, c2, c3), features, strict=True):
        col.markdown(f'<div class="gl-feature"><b>{title}</b>{text}</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1, 2])
    left.button(
        "▶ Load sample data", type="primary", key="load_sample_hero", on_click=_load_sample_from_hero, width="stretch"
    )
    right.markdown(
        '<span class="gl-muted">…or upload your own CSV / Excel file from the sidebar. '
        "Messy sheets are fine: absent markers, title rows and odd column names are handled.</span>",
        unsafe_allow_html=True,
    )
    with st.expander("What columns does my sheet need?"):
        st.markdown(
            "At minimum a **roll number or student name** and **some marks** (MST-1, MST-2, assignment, "
            "end term or total). Optional: section, semester, subject, attendance, result.\n\n"
            "**Long format** — one row per student per subject:\n\n"
            "`Roll No | Student Name | Section | Subject | Attendance (%) | MST 1 | MST 2 | End Term | Total`\n\n"
            "**Wide format** — one row per student, one column per subject and component:\n\n"
            "`Roll No | Name | Section | Python - MST1 | Python - MST2 | DBMS - MST1 | ...`"
        )


def kpi_row(current: dict, baseline: dict | None) -> None:
    """KPI cards; deltas compare the filtered view with the whole class."""
    cols = st.columns(5)

    def delta(key: str, digits: int = 1, suffix: str = " pts") -> str | None:
        if baseline is None:
            return None
        a, b = current.get(key), baseline.get(key)
        if a is None or b is None or any(isinstance(v, float) and math.isnan(v) for v in (a, b)):
            return None
        diff = a - b
        return None if abs(diff) < 10 ** (-digits) else f"{diff:+.{digits}f}{suffix} vs all"

    cols[0].metric("Students", fmt_num(current["students"]), border=True, help=f"{current['records']:,} records")
    cols[1].metric("Pass rate", fmt_pct(current["pass_rate"]), delta=delta("pass_rate"), border=True)
    cols[2].metric("Average score", fmt_pct(current["avg_score"]), delta=delta("avg_score"), border=True)
    at_risk = current.get("at_risk")
    cols[3].metric(
        "At-risk students",
        fmt_num(at_risk) if at_risk is not None else "–",
        border=True,
        help="High or medium risk. See the At-Risk tab for reasons.",
    )
    cols[4].metric(
        "Absentees",
        fmt_num(current["absentees"]),
        border=True,
        help="Students marked absent in at least one MST, assignment or end-term exam.",
    )


FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
PLAIN_NUMBER = re.compile(r"[-+]?\d+(\.\d+)?%?")


def _looks_like_formula(value) -> bool:
    return isinstance(value, str) and value.startswith(FORMULA_PREFIXES) and not PLAIN_NUMBER.fullmatch(value.strip())


def csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet formulas in text cells (CSV injection, OWASP).

    Excel runs a CSV cell that starts with = + - @ as a formula. Text cells that do get a leading
    apostrophe, which Excel hides; numbers are untouched.
    """
    out = df.copy()
    for column in out.columns:
        if out[column].dtype == object or pd.api.types.is_string_dtype(out[column]):
            values = out[column]
            risky = values.map(_looks_like_formula)
            out.loc[risky, column] = "'" + values[risky].astype(str)
    return out


def download_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV", key: str | None = None) -> None:
    data = csv_safe(df).to_csv(index=False).encode("utf-8-sig")  # BOM: Excel then reads names like "Zoë" correctly
    st.download_button(label, data, file_name=filename, mime="text/csv", key=key, icon="⬇️")


def score_column(label: str = "Score", help_text: str | None = None):
    return st.column_config.ProgressColumn(label, format="%.1f%%", min_value=0, max_value=100, help=help_text)
