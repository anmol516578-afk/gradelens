"""Sidebar: data source, global filters and the marks scheme."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from core.config import ROLE_LABELS, AnalysisConfig
from core.data_loader import SUPPORTED_EXTENSIONS, LoadError
from ui import state

MAX_UPLOAD_MB = 50


@dataclass(frozen=True)
class Filters:
    semester: str | None
    sections: tuple[str, ...]
    subjects: tuple[str, ...]

    @property
    def active(self) -> bool:
        return bool(self.semester or self.sections or self.subjects)

    def describe(self) -> str:
        parts = []
        if self.semester:
            parts.append(f"Semester {self.semester}")
        if self.sections:
            parts.append("Section " + ", ".join(self.sections))
        if self.subjects:
            parts.append(", ".join(self.subjects))
        return " · ".join(parts) if parts else "All students"


def _on_upload() -> None:
    uploaded = st.session_state.get("uploader")
    if uploaded is not None:
        state.load_upload(uploaded.getvalue(), uploaded.name)


def _on_load_sample() -> None:
    state.load_sample(st.session_state.get("sample_choice", "long"))
    st.toast("Sample data loaded — 180 students, 5 subjects.", icon="✅")


def render_source() -> None:
    with st.sidebar:
        st.markdown("## 🎓 GradeLens")
        st.caption("AI student performance analyst")
        st.markdown("#### 1 · Data")
        st.file_uploader(
            "Upload a marks sheet",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            key="uploader",
            on_change=_on_upload,
            help="CSV or Excel. Long format (one row per student per subject) or wide format "
            "(one row per student). Column names are detected automatically.",
        )
        with st.expander("…or try the sample data", expanded=st.session_state.get("source") is None):
            st.selectbox(
                "Sample layout",
                options=list(state.SAMPLE_CHOICES),
                format_func=lambda k: state.SAMPLE_CHOICES[k],
                key="sample_choice",
            )
            st.button("Load sample data", key="load_sample", on_click=_on_load_sample, width="stretch", type="primary")

        try:
            sheets = state.sheet_options()
        except LoadError:
            sheets = []  # the main area reports the error when it tries to read the file
        if len(sheets) > 1:
            if st.session_state.get("sheet") not in sheets:
                st.session_state["sheet"] = state.default_sheet(sheets)
            st.selectbox("Sheet", sheets, key="sheet", help="This workbook has several sheets.")


def _reset_filters() -> None:
    for key in ("f_semester", "f_sections", "f_subjects", "drill_section", "drill_subject"):
        st.session_state.pop(key, None)


def render_filters(df_all: pd.DataFrame) -> Filters:
    with st.sidebar:
        st.markdown("#### 2 · Filters")
        semesters = sorted(df_all["semester"].dropna().unique())
        semester = None
        if len(semesters) > 1:
            choice = st.selectbox("Semester", [state.ALL, *semesters], key="f_semester")
            semester = None if choice == state.ALL else choice
        sections = st.multiselect(
            "Sections", sorted(df_all["section"].unique()), key="f_sections", placeholder="All sections"
        )
        subjects = st.multiselect(
            "Subjects", sorted(df_all["subject"].unique()), key="f_subjects", placeholder="All subjects"
        )
        st.slider(
            "Pass mark",
            min_value=30.0,
            max_value=60.0,
            step=1.0,
            key=state.CONFIG_KEYS["pass_pct"],
            format="%.0f%%",
            help="Minimum overall percentage needed to pass a subject.",
        )
        st.slider(
            "Attendance requirement",
            min_value=50.0,
            max_value=95.0,
            step=5.0,
            key=state.CONFIG_KEYS["attendance_threshold"],
            format="%.0f%%",
            help="Used for at-risk flags and attendance charts.",
        )
        st.button("Reset filters", key="reset_filters", on_click=_reset_filters, width="stretch")
    return Filters(semester, tuple(sections), tuple(subjects))


def render_marks_scheme(effective: AnalysisConfig | None = None) -> None:
    """``effective`` is the scheme in use for this sheet (only the components it contains)."""
    config = effective or state.current_config()
    with st.sidebar:
        with st.expander("Marks scheme", expanded=False):
            overrides = st.session_state.get("detected_overrides") or {}
            if overrides:
                detected = ", ".join(f"{ROLE_LABELS[k]} out of {v:g}" for k, v in overrides.items())
                st.info(f"Detected from your sheet: {detected}.", icon="🔎")
            cols = st.columns(2)
            for i, component in enumerate(("assignment", "mst1", "mst2", "end_term")):
                cols[i % 2].number_input(
                    f"{ROLE_LABELS[component]} max",
                    min_value=1.0,
                    max_value=1000.0,
                    step=1.0,
                    format="%.0f",
                    key=state.CONFIG_KEYS[f"max_{component}"],
                )
            st.number_input(
                "Internal total max (0 = auto)",
                min_value=0.0,
                max_value=1000.0,
                step=1.0,
                format="%.0f",
                key=state.CONFIG_KEYS["max_internal_total"],
            )
            st.number_input(
                "Total max (0 = auto)",
                min_value=0.0,
                max_value=2000.0,
                step=1.0,
                format="%.0f",
                key=state.CONFIG_KEYS["max_total"],
            )
            st.caption(f"In use: internal out of **{config.internal_max:g}**, total out of **{config.total_max:g}**.")


def render_footer() -> None:
    with st.sidebar:
        st.divider()
        st.caption("🔒 Files stay in your session and are never stored on the server.")
