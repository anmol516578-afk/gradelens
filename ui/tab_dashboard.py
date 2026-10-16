"""📊 Dashboard tab: where students fail, score spread, attendance, MST trends."""

from __future__ import annotations

from functools import partial

import pandas as pd
import streamlit as st

from core import analytics as an
from ui import charts, state
from ui import components as C
from ui.context import AppContext
from ui.theme import PLOTLY_CONFIG, risk_label

NONCE_KEY = "chart_nonce"


def _nonce() -> int:
    return st.session_state.setdefault(NONCE_KEY, 0)


def _point_value(chart_key: str, axis: str) -> str | None:
    event = st.session_state.get(chart_key)
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return None
    if not points:
        return None
    value = points[0].get(axis)
    return None if value is None else str(value)


def _on_subject_click(chart_key: str) -> None:
    value = _point_value(chart_key, "y")
    st.session_state["drill_subject"] = value or state.ALL


def _on_section_click(chart_key: str) -> None:
    value = _point_value(chart_key, "x")
    st.session_state["drill_section"] = value or state.ALL


def _on_heatmap_click(chart_key: str) -> None:
    section = _point_value(chart_key, "y")
    subject = _point_value(chart_key, "x")
    if section and subject:
        st.session_state["drill_section"] = section.removeprefix("Section ")
        st.session_state["drill_subject"] = subject
    else:
        st.session_state["drill_section"] = state.ALL
        st.session_state["drill_subject"] = state.ALL


def _clear_drill() -> None:
    st.session_state["drill_section"] = state.ALL
    st.session_state["drill_subject"] = state.ALL
    st.session_state[NONCE_KEY] = _nonce() + 1


def _on_student_select(table_key: str, rolls_key: str) -> None:
    event = st.session_state.get(table_key)
    rows = event["selection"]["rows"] if event else []
    rolls = st.session_state.get(rolls_key, [])
    if rows and rows[0] < len(rolls):
        C.open_student(rolls[rows[0]])


def _drill_values(df: pd.DataFrame) -> tuple[str | None, str | None]:
    sections = [state.ALL, *sorted(df["section"].unique())]
    subjects = [state.ALL, *sorted(df["subject"].unique())]
    if st.session_state.get("drill_section") not in sections:
        st.session_state["drill_section"] = state.ALL
    if st.session_state.get("drill_subject") not in subjects:
        st.session_state["drill_subject"] = state.ALL
    section = st.session_state["drill_section"]
    subject = st.session_state["drill_subject"]
    return (None if section == state.ALL else section), (None if subject == state.ALL else subject)


def render(ctx: AppContext) -> None:
    df = ctx.filtered
    if df.empty:
        st.info("No records match the current filters. Try clearing them in the sidebar.", icon="🔍")
        return
    drill_section, drill_subject = _drill_values(df)
    nonce = _nonce()

    C.heading("Where are students failing?", "Click a bar or a heatmap cell to drill into that group below.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        key = f"chart_fail_subject_{nonce}"
        fig = charts.fail_rate_bar(
            an.group_summary(df, ["subject"]), "subject", horizontal=True, highlight=drill_subject
        )
        st.markdown("**Fail rate by subject**")
        st.plotly_chart(
            fig, key=key, on_select=partial(_on_subject_click, key), selection_mode="points",
            config=PLOTLY_CONFIG, alt="Bar chart of fail rate by subject",
        )  # fmt: skip
    with right:
        key = f"chart_fail_section_{nonce}"
        fig = charts.fail_rate_bar(an.group_summary(df, ["section"]), "section", highlight=drill_section)
        st.markdown("**Fail rate by section**")
        st.plotly_chart(
            fig, key=key, on_select=partial(_on_section_click, key), selection_mode="points",
            config=PLOTLY_CONFIG, alt="Bar chart of fail rate by section",
        )  # fmt: skip

    matrix = an.fail_rate_matrix(df)
    if not matrix.empty and matrix.size > 1:
        key = f"chart_heatmap_{nonce}"
        st.markdown("**Fail rate: section × subject**")
        st.plotly_chart(
            charts.fail_heatmap(matrix), key=key, on_select=partial(_on_heatmap_click, key),
            selection_mode="points", config=PLOTLY_CONFIG, alt="Heatmap of fail rate by section and subject",
        )  # fmt: skip

    st.divider()
    C.heading("How are scores distributed?")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Score distribution**")
        st.plotly_chart(charts.score_histogram(df, ctx.config.pass_pct), config=PLOTLY_CONFIG, key="chart_hist")
    with right:
        st.markdown("**Grade bands**")
        st.plotly_chart(
            charts.grade_bar(an.grade_distribution(df, ctx.config.pass_pct)), config=PLOTLY_CONFIG, key="chart_grades"
        )

    _attendance_section(ctx, df)
    _mst_section(ctx, df)

    st.divider()
    C.heading(
        "Subject difficulty", "Difficulty index = average of (100 − average score) and fail rate. Higher is harder."
    )
    difficulty = an.subject_difficulty(df)
    st.dataframe(
        difficulty[["rank", "subject", "difficulty_index", "avg_score", "fail_rate", "students", "avg_attendance"]],
        hide_index=True,
        width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "subject": "Subject",
            "difficulty_index": st.column_config.ProgressColumn(
                "Difficulty", min_value=0, max_value=100, format="%.1f"
            ),
            "avg_score": st.column_config.NumberColumn("Avg score", format="%.1f%%"),
            "fail_rate": st.column_config.NumberColumn("Fail rate", format="%.1f%%"),
            "students": "Students",
            "avg_attendance": st.column_config.NumberColumn("Avg attendance", format="%.1f%%"),
        },
    )

    st.divider()
    _student_section(ctx, df, drill_section, drill_subject, nonce)


def _attendance_section(ctx: AppContext, df: pd.DataFrame) -> None:
    if "attendance_pct" not in df or df["attendance_pct"].notna().sum() < 3:
        return
    st.divider()
    info = an.attendance_performance(df, ctx.config.attendance_threshold)
    below, above = info["below"], info["above"]
    caption = f"Correlation between attendance and score: r = {info['r']:.2f} across {info['n']:,} records."
    if below["records"] and above["records"]:
        caption += (
            f" Below {ctx.config.attendance_threshold:g}% attendance the fail rate is "
            f"{C.fmt_pct(below['fail_rate'])} vs {C.fmt_pct(above['fail_rate'])} above it."
        )
    C.heading("Does attendance matter?", caption)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Attendance vs score**")
        st.plotly_chart(
            charts.attendance_scatter(df, ctx.config.attendance_threshold, ctx.config.pass_pct, ctx.section_colors),
            config=PLOTLY_CONFIG,
            key="chart_att_scatter",
        )
    with right:
        st.markdown("**Average score by attendance band**")
        if not info["bands"].empty:
            st.plotly_chart(charts.attendance_band_bar(info["bands"]), config=PLOTLY_CONFIG, key="chart_att_bands")


def _mst_section(ctx: AppContext, df: pd.DataFrame) -> None:
    changes = an.mst_change(df, ctx.config.mst_change_tolerance)
    if changes.empty:
        return
    st.divider()
    C.heading(
        "MST-1 → MST-2: who is slipping?",
        f"A change of more than {ctx.config.mst_change_tolerance:g} percentage points counts as improved or declined.",
    )
    left, right = st.columns([3, 1], gap="large")
    with left:
        st.plotly_chart(
            charts.mst_trend_bar(an.mst_trend_summary(df, ctx.config.mst_change_tolerance)),
            config=PLOTLY_CONFIG,
            key="chart_mst",
        )
    with right:
        counts = changes["trend"].value_counts()
        st.metric("Declined", int(counts.get("Declined", 0)), border=True, help="Student-subject pairs")
        st.metric("Improved", int(counts.get("Improved", 0)), border=True)
        biggest = changes.nsmallest(1, "change")
        if not biggest.empty and biggest["change"].iloc[0] < 0:
            row = biggest.iloc[0]
            st.caption(f"Biggest drop: **{row['student_name']}** in {row['subject']} ({row['change']:.0f} pts).")


def _student_section(ctx: AppContext, df: pd.DataFrame, section: str | None, subject: str | None, nonce: int) -> None:
    cohort = an.apply_filters(df, sections=[section] if section else None, subjects=[subject] if subject else None)
    label = " × ".join(x for x in (f"Section {section}" if section else None, subject) if x) or "everyone in view"
    C.heading(f"Students — {label}", "Select a row to open the student's profile.")

    c1, c2, c3, c4 = st.columns([2, 2, 3, 1.4], vertical_alignment="bottom")
    c1.selectbox("Section", [state.ALL, *sorted(df["section"].unique())], key="drill_section")
    c2.selectbox("Subject", [state.ALL, *sorted(df["subject"].unique())], key="drill_subject")
    query = c3.text_input("Search name or roll no", key="student_search", placeholder="e.g. Sharma or 2023BCA0")
    c4.button("Clear", key="clear_drill", on_click=_clear_drill, width="stretch", disabled=not (section or subject))

    table = an.student_summary(cohort)
    if ctx.risk is not None:
        levels = ctx.risk.students.set_index("roll_no")["risk_level"]
        table["risk"] = table["roll_no"].map(levels).map(lambda lv: risk_label(lv) if isinstance(lv, str) else "")
    if query:
        q = query.strip().lower()
        table = table[
            table["student_name"].str.lower().str.contains(q, regex=False)
            | table["roll_no"].str.lower().str.contains(q, regex=False)
        ]
    table = table.sort_values("avg_score").reset_index(drop=True)
    rolls_key = "_student_table_rolls"
    st.session_state[rolls_key] = table["roll_no"].tolist()
    table_key = f"student_table_{nonce}"
    columns = ["roll_no", "student_name", "section", "avg_score", "min_score", "fails", "avg_attendance"]
    columns += ["weakest_subject"] + (["risk"] if "risk" in table else [])
    st.dataframe(
        table[columns],
        hide_index=True,
        width="stretch",
        height=min(420, 38 + 35 * max(len(table), 1)),
        key=table_key,
        on_select=partial(_on_student_select, table_key, rolls_key),
        selection_mode="single-row",
        column_config={
            "roll_no": "Roll No",
            "student_name": "Name",
            "section": "Section",
            "avg_score": C.score_column("Avg score"),
            "min_score": st.column_config.NumberColumn("Lowest", format="%.1f%%"),
            "fails": st.column_config.NumberColumn("Fails", help="Subjects failed"),
            "avg_attendance": st.column_config.NumberColumn("Attendance", format="%.0f%%"),
            "weakest_subject": "Weakest subject",
            "risk": "Risk",
        },
    )
    st.caption(f"{len(table):,} student(s). Sorted by average score, lowest first.")

    top, bottom = an.top_bottom(cohort, n=5)
    left, right = st.columns(2, gap="large")
    small_cols = {
        "roll_no": "Roll No",
        "student_name": "Name",
        "section": "Sec",
        "avg_score": st.column_config.NumberColumn("Avg score", format="%.1f%%"),
    }
    with left:
        st.markdown("**🏆 Top performers**")
        st.dataframe(top[list(small_cols)], hide_index=True, width="stretch", column_config=small_cols)
    with right:
        st.markdown("**🤝 Need the most support**")
        st.dataframe(bottom[list(small_cols)], hide_index=True, width="stretch", column_config=small_cols)
