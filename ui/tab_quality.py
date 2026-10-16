"""🧹 Data Quality tab: what was fixed, what needs attention, and the column mapping."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.cleaning import INFO, DataQualityReport
from core.column_mapping import mapping_from_tuple
from core.config import ALL_ROLES, IGNORE, ROLE_LABELS, AnalysisConfig
from ui import components as C
from ui import state
from ui.context import AppContext

LABEL_TO_ROLE = {label: role for role, label in ROLE_LABELS.items()}
FORMAT_LABELS = {
    "long": "Long (one row per student per subject)",
    "wide": "Wide (one row per student, columns per subject)",
    "wide-totals": "Wide (one total column per subject)",
}


def _examples(values: pd.Series) -> str:
    vals = values.astype(str).str.strip()
    vals = vals[vals != ""].unique()[:3]
    return ", ".join(vals)


def mapping_editor(raw: pd.DataFrame, compact: bool = False) -> None:
    """Editable table: which canonical field each uploaded column feeds."""
    entries = mapping_from_tuple(st.session_state["mapping"])
    table = pd.DataFrame(
        {
            "Column in your file": [e.column for e in entries],
            "Maps to": [ROLE_LABELS[e.role] for e in entries],
            "Subject (wide sheets)": [e.subject for e in entries],
            "Example values": [_examples(raw[e.column]) if e.column in raw else "" for e in entries],
        }
    )
    sig = abs(hash(st.session_state.get("source_sig"))) % 10_000_000
    edited = st.data_editor(
        table,
        key=f"mapping_editor_{sig}",
        hide_index=True,
        width="stretch",
        disabled=["Column in your file", "Example values"],
        column_config={
            "Maps to": st.column_config.SelectboxColumn(
                options=[ROLE_LABELS[r] for r in (*ALL_ROLES, IGNORE)], required=True
            ),
            "Subject (wide sheets)": st.column_config.TextColumn(
                help="Only for wide sheets: which subject this column belongs to. Leave blank otherwise."
            ),
        },
    )
    notes = st.session_state.get("mapping_notes") or []
    for note in notes:
        st.caption(f"ℹ️ {note}")
    c1, c2, _ = st.columns([1.3, 1.9, 2])
    if c1.button("Apply mapping", type="primary", key=f"apply_mapping_{sig}", width="stretch"):
        hints = {e.column: e.max_hint for e in entries}
        new_entries = []
        for row in edited.itertuples(index=False):
            column, label, subject = row[0], row[1], row[2]
            entry = next(e for e in entries if e.column == column)
            entry.role = LABEL_TO_ROLE.get(label, IGNORE)
            entry.subject = (subject or "").strip() if isinstance(subject, str) else ""
            entry.max_hint = hints.get(column)
            new_entries.append(entry)
        state.apply_mapping(new_entries)
        st.toast("Mapping applied — data reprocessed.", icon="✅")
        st.rerun()
    if c2.button("Reset to auto-detected", key=f"reset_mapping_{sig}", width="stretch"):
        st.session_state["mapping"] = st.session_state["mapping_auto"]
        st.rerun()
    if not compact:
        st.caption(
            "Tip: set a column to '— Ignore —' to leave it out. For wide sheets, give each subject column "
            "its subject name and its component (MST-1, End Term, ...)."
        )


def _issue_block(issues, title: str, icon: str, expanded: bool) -> None:
    if not issues:
        return
    with st.expander(f"{icon} {title} ({len(issues)})", expanded=expanded):
        for issue in issues:
            column = f" · {ROLE_LABELS.get(issue.column, issue.column)}" if issue.column else ""
            st.markdown(f"**{issue.category}{column}** — {issue.message}")
            if issue.examples:
                st.dataframe(pd.DataFrame(issue.examples), hide_index=True, width="stretch")


def _scheme_table(config: AnalysisConfig) -> pd.DataFrame:
    rows = [(ROLE_LABELS[c], config.max_marks(c)) for c in ("assignment", "mst1", "mst2", "end_term")]
    rows += [("Internal total", config.internal_max), ("Total", config.total_max)]
    rows += [("Pass mark", f"{config.pass_pct:g}%"), ("Attendance requirement", f"{config.attendance_threshold:g}%")]
    return pd.DataFrame(rows, columns=["Setting", "Value"]).astype(str)


def render(ctx: AppContext) -> None:
    report: DataQualityReport = ctx.prepared.report
    cols = st.columns(5)
    cols[0].metric("Rows loaded", f"{report.rows_loaded:,}", border=True)
    cols[1].metric("Rows after cleaning", f"{report.rows_clean:,}", border=True)
    cols[2].metric("Students", f"{report.students:,}", border=True)
    cols[3].metric("Auto-fixes", len(report.fixed), border=True)
    cols[4].metric("Needs attention", len(report.attention), border=True)
    st.caption(f"Layout detected: **{FORMAT_LABELS.get(report.data_format, report.data_format)}**")

    if not report.issues:
        st.success("No data problems found. 🎉", icon="✅")
    _issue_block(report.attention, "Needs your attention", "⚠️", expanded=True)
    _issue_block(report.fixed, "Fixed automatically", "✅", expanded=False)
    _issue_block(report.by_severity(INFO), "Notes", "ℹ️", expanded=False)

    st.divider()
    left, right = st.columns([2, 1], gap="large")
    with left:
        C.heading("Column mapping", "How each column in your file was understood. Change it if something is wrong.")
        mapping_editor(ctx.raw)
    with right:
        C.heading("Marks scheme in use", "Change it under 'Marks scheme' in the sidebar.")
        st.dataframe(_scheme_table(ctx.config), hide_index=True, width="stretch")

    st.divider()
    C.heading("Cleaned data", f"{len(ctx.prepared.clean):,} rows. Absences are blank values with an _absent flag.")
    st.dataframe(ctx.prepared.clean.head(500), width="stretch", height=320)
    C.download_csv(ctx.prepared.clean, "gradelens_cleaned_data.csv", "Download cleaned data", key="dl_clean")
