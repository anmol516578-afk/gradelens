"""🎯 CO Attainment tab."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from core.co_attainment import (
    COError,
    COSettings,
    compute_co_attainment,
    default_mapping,
    overall_attainment,
)
from core.data_loader import LoadError, read_table
from ui import charts, state
from ui import components as C
from ui.context import AppContext
from ui.theme import PLOTLY_CONFIG

CO_OPTIONS = [f"CO{i}" for i in range(1, 9)]
SOURCE_SAMPLE = "Sample: Python end-term (question-wise)"
SOURCE_UPLOAD = "Upload a question-wise marks sheet"


def _load_source() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    """Returns (raw question-wise sheet, initial mapping or None, signature)."""
    options = [SOURCE_UPLOAD]
    if st.session_state.get("source") == "sample":
        options.insert(0, SOURCE_SAMPLE)
    choice = st.radio("Question-wise marks", options, key="co_source", horizontal=True)
    if choice == SOURCE_SAMPLE:
        raw, mapping = state.cached_sample_co()
        return raw, mapping, "sample"
    uploaded = st.file_uploader(
        "Question-wise marks (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        key="co_uploader",
        help="One row per student with columns Q1, Q2, ... (optionally 'Q1 (10)' to give max marks). "
        "Roll No and Name columns are picked up automatically.",
    )
    if uploaded is None:
        return None, None, None
    data = uploaded.getvalue()
    try:
        raw = read_table(data, uploaded.name)
    except LoadError as exc:
        st.error(str(exc))
        return None, None, None
    return raw, None, hashlib.md5(data, usedforsecurity=False).hexdigest()


def render(ctx: AppContext) -> None:
    C.heading(
        "Course Outcome attainment",
        "Map each exam question to a Course Outcome, then see what share of students attained each CO.",
    )
    raw, initial_mapping, sig = _load_source()
    if raw is None:
        st.info(
            "Upload a question-wise marks sheet (columns Q1, Q2, …) to calculate CO attainment. "
            "Tip: load the sample data to try it with a ready-made example.",
            icon="🎯",
        )
        return
    try:
        marks = state.cached_question_marks(raw)
    except COError as exc:
        st.error(str(exc))
        return

    if st.session_state.get("co_upload_sig") != sig:
        st.session_state["co_upload_sig"] = sig
        st.session_state["co_mapping_df"] = (
            initial_mapping.copy() if initial_mapping is not None else default_mapping(raw, marks)
        )

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("**1 · Question → CO mapping**")
        mapping = st.data_editor(
            st.session_state["co_mapping_df"],
            key=f"co_editor_{sig}",
            hide_index=True,
            width="stretch",
            disabled=["question"],
            column_config={
                "question": "Question",
                "co": st.column_config.SelectboxColumn("Course Outcome", options=CO_OPTIONS, required=True),
                "max_marks": st.column_config.NumberColumn("Max marks", min_value=0.5, step=0.5, format="%.1f"),
            },
        )
    with right:
        st.markdown("**2 · Attainment rules**")
        student_threshold = st.slider(
            "A student attains a CO at",
            40.0, 90.0, 60.0, 5.0, format="%.0f%%", key="co_student_threshold",
            help="Minimum % of the CO's marks a student must score.",
        )  # fmt: skip
        target = st.segmented_control("Target level", [1, 2, 3], default=2, key="co_target", required=True)
        with st.expander("Level thresholds (% of students attaining)"):
            l3 = st.number_input("Level 3 at ≥", 0.0, 100.0, 70.0, 5.0, key="co_l3")
            l2 = st.number_input("Level 2 at ≥", 0.0, 100.0, 60.0, 5.0, key="co_l2")
            l1 = st.number_input("Level 1 at ≥", 0.0, 100.0, 50.0, 5.0, key="co_l1")
        st.info(
            "These defaults are a common OBE method. Use the thresholds in your college's OBE / NBA policy.",
            icon="📘",
        )

    if not (l3 >= l2 >= l1):
        st.error("Level thresholds must satisfy Level 3 ≥ Level 2 ≥ Level 1.")
        return
    settings = COSettings(student_threshold, l3, l2, l1, int(target or 2))
    try:
        co_table, students, warnings = compute_co_attainment(marks, mapping, settings)
    except COError as exc:
        st.error(str(exc))
        return
    st.session_state["co_result"] = {"table": co_table, "settings": settings}
    for warning in warnings:
        st.caption(f"ℹ️ {warning}")

    st.divider()
    met = int(co_table["meets_target"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Average attainment level", f"{overall_attainment(co_table):.2f} / 3", border=True)
    m2.metric(f"COs at target (Level {settings.target_level}+)", f"{met} of {len(co_table)}", border=True)
    m3.metric("Students assessed", int(co_table["students_assessed"].max()), border=True)

    st.plotly_chart(charts.co_attainment_bar(co_table, settings), config=PLOTLY_CONFIG, key="chart_co")
    display = co_table.assign(status=co_table["meets_target"].map({True: "✅ Meets target", False: "⚠️ Below target"}))[
        ["co", "questions", "max_marks", "students_attained", "students_assessed", "pct_attained", "level", "status"]
    ]
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "co": "CO",
            "questions": "Questions",
            "max_marks": st.column_config.NumberColumn("Max marks", format="%.0f"),
            "students_attained": "Attained",
            "students_assessed": "Assessed",
            "pct_attained": st.column_config.ProgressColumn("% attained", min_value=0, max_value=100, format="%.1f%%"),
            "level": st.column_config.NumberColumn("Level", help="0-3"),
            "status": "Status",
        },
    )
    with st.expander("Student-wise CO scores"):
        st.dataframe(students, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        C.download_csv(co_table, "co_attainment.csv", "Download CO summary", key="dl_co")
    with c2:
        C.download_csv(students, "co_student_scores.csv", "Download student-wise scores", key="dl_co_students")
