"""👤 Student Profile tab."""

from __future__ import annotations

import streamlit as st

from core import analytics as an
from ui import charts
from ui import components as C
from ui.context import AppContext
from ui.theme import PLOTLY_CONFIG, risk_label

PROFILE_KEY = "profile_student"


def _step(rolls: list[str], offset: int) -> None:
    current = st.session_state.get(PROFILE_KEY)
    idx = rolls.index(current) if current in rolls else 0
    st.session_state[PROFILE_KEY] = rolls[(idx + offset) % len(rolls)]
    st.session_state["selected_student"] = st.session_state[PROFILE_KEY]


def render(ctx: AppContext) -> None:
    df = ctx.semester_scope
    if df.empty:
        st.info("No students to show.", icon="👤")
        return
    roster = df.drop_duplicates("roll_no")[["roll_no", "student_name", "section"]].sort_values("roll_no")
    rolls = roster["roll_no"].tolist()
    labels = {r.roll_no: f"{r.roll_no} · {r.student_name} (Sec {r.section})" for r in roster.itertuples()}
    if st.session_state.get(PROFILE_KEY) not in labels:
        selected = st.session_state.get("selected_student")
        st.session_state[PROFILE_KEY] = selected if selected in labels else rolls[0]

    c1, c2, c3 = st.columns([6, 1, 1], vertical_alignment="bottom")
    c1.selectbox(
        "Find a student", rolls, format_func=labels.get, key=PROFILE_KEY, help="Type a name or roll number to search."
    )
    c2.button("◀ Prev", key="profile_prev", on_click=_step, args=(rolls, -1), width="stretch")
    c3.button("Next ▶", key="profile_next", on_click=_step, args=(rolls, 1), width="stretch")
    roll = st.session_state[PROFILE_KEY]
    profile = an.student_profile(df, roll)
    summary, subjects = profile["summary"], profile["subjects"]

    risk_row = None
    if ctx.risk is not None:
        match = ctx.risk.students[ctx.risk.students["roll_no"] == roll]
        risk_row = match.iloc[0] if not match.empty else None

    st.markdown(f"### {profile['student_name']}")
    chips = [f"Roll {roll}", f"Section {profile['section']}"]
    semester = df.loc[df["roll_no"] == roll, "semester"].iloc[0]
    chips.append(f"Semester {semester}")
    if risk_row is not None:
        chips.append(f"{risk_label(risk_row['risk_level'])} risk")
    st.markdown(" ".join(f'<span class="gl-chip">{c}</span>' for c in chips), unsafe_allow_html=True)
    st.write("")

    m1, m2, m3, m4 = st.columns(4)
    diff = summary["avg_score"] - profile["class_avg_score"]
    m1.metric("Average score", C.fmt_pct(summary["avg_score"]), delta=f"{diff:+.1f} pts vs class", border=True)
    m2.metric("Subjects failed", f"{summary['fails']} of {summary['subjects']}", border=True)
    attendance = summary["avg_attendance"]
    att_delta = None
    if attendance == attendance:  # not NaN
        att_delta = f"{attendance - ctx.config.attendance_threshold:+.0f} pts vs required"
    m3.metric("Attendance", C.fmt_pct(attendance, 0), delta=att_delta, border=True)
    m4.metric(
        "Lowest score",
        C.fmt_pct(summary["min_score"]),
        delta=summary["weakest_subject"] or None,
        delta_color="off",
        delta_arrow="off",
        border=True,
        help="The student's weakest subject.",
    )

    if risk_row is not None and risk_row["reasons"]:
        with st.container(border=True):
            st.markdown(f"**Why {risk_label(risk_row['risk_level'])} risk?**")
            for reason in risk_row["reasons"]:
                st.markdown(f"- {reason}")
            st.caption("Decision support only — talk to the student before drawing conclusions.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Scores vs class average**")
        st.plotly_chart(
            charts.profile_radar(subjects, profile["student_name"]), config=PLOTLY_CONFIG, key="chart_profile_radar"
        )
    with right:
        st.markdown("**MST-1 → MST-2 → End term**")
        st.plotly_chart(charts.profile_components(subjects), config=PLOTLY_CONFIG, key="chart_profile_components")
    if "attendance_pct" in subjects and subjects["attendance_pct"].notna().any():
        st.markdown("**Attendance by subject**")
        st.plotly_chart(
            charts.profile_attendance(subjects, ctx.config.attendance_threshold),
            config=PLOTLY_CONFIG,
            key="chart_profile_attendance",
        )

    st.markdown("**Marks**")
    cols = [
        c
        for c in ("subject", "attendance_pct", "assignment", "mst1", "mst2", "end_term", "total", "score_pct",
                  "class_avg", "vs_class", "result")
        if c in subjects
    ]  # fmt: skip
    st.dataframe(
        subjects[cols],
        hide_index=True,
        width="stretch",
        column_config={
            "subject": "Subject",
            "attendance_pct": st.column_config.NumberColumn("Attendance", format="%.0f%%"),
            "assignment": st.column_config.NumberColumn("Assignment", format="%.1f"),
            "mst1": st.column_config.NumberColumn("MST-1", format="%.1f"),
            "mst2": st.column_config.NumberColumn("MST-2", format="%.1f"),
            "end_term": st.column_config.NumberColumn("End term", format="%.1f"),
            "total": st.column_config.NumberColumn("Total", format="%.1f"),
            "score_pct": C.score_column("Score"),
            "class_avg": st.column_config.NumberColumn("Class avg", format="%.1f%%"),
            "vs_class": st.column_config.NumberColumn("vs class", format="%+.1f"),
            "result": "Result",
        },
    )
    absent = [
        label
        for comp, label in (
            ("mst1", "MST-1"),
            ("mst2", "MST-2"),
            ("end_term", "End term"),
            ("assignment", "Assignment"),
        )
        if f"{comp}_absent" in subjects and subjects[f"{comp}_absent"].any()
    ]
    if absent:
        st.caption("Marked absent in: " + ", ".join(absent) + ". Blank cells above are absences, not zeros.")
