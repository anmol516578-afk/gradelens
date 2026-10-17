"""📄 Reports tab: one-click PDF and Excel reports for the HOD."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time

import streamlit as st

from agent import llm as llm_mod
from agent.llm import LLMError
from core.co_attainment import COError, COSettings, compute_co_attainment
from reports.data import ReportMeta, build_report_data, template_summary
from reports.data import llm_summary as write_llm_summary
from reports.excel_report import build_excel
from reports.pdf_report import build_pdf
from ui import ai_settings, state
from ui import components as C
from ui.context import AppContext

log = logging.getLogger("gradelens.reports")

SUMMARY_KEY = "report_summary"
AUTO_KEY = "report_summary_auto"
OUTPUT_KEY = "report_output"
META_KEYS = {
    "title": "meta_title",
    "college": "meta_college",
    "department": "meta_department",
    "semester": "meta_semester",
    "faculty": "meta_faculty",
}


def co_for_report(ctx: AppContext) -> dict | None:
    """CO results from the CO tab, or the sample's question-wise marks with default settings."""
    if st.session_state.get("co_result"):
        return st.session_state["co_result"]
    if st.session_state.get("source") == "sample":
        raw, mapping = state.cached_sample_co()
        try:
            marks = state.cached_question_marks(raw)
            settings = COSettings()
            table, _, _ = compute_co_attainment(marks, mapping, settings)
        except COError:
            return None
        return {"table": table, "settings": settings}
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "report"


def _meta(ctx: AppContext, include_names: bool) -> ReportMeta:
    values = {field: str(st.session_state.get(key) or "").strip() for field, key in META_KEYS.items()}
    return ReportMeta(
        title=values["title"] or "Student Performance Report",
        college=values["college"],
        department=values["department"],
        semester=values["semester"],
        faculty=values["faculty"],
        scope=ctx.filters.describe(),
        include_names=include_names,
    )


def _signature(ctx: AppContext, meta: ReportMeta, summary: str, include_co: bool) -> str:
    payload = {
        "filters": ctx.filters.describe(),
        "config": ctx.config.to_key(),
        "rows": len(ctx.filtered),
        "source": str(st.session_state.get("source_sig")),
        "meta": [meta.title, meta.college, meta.department, meta.semester, meta.faculty, meta.include_names],
        "summary": summary,
        "co": include_co,
        "risk": ctx.risk.method if ctx.risk is not None else None,
    }
    return hashlib.md5(json.dumps(payload, default=str).encode(), usedforsecurity=False).hexdigest()


def _write_with_ai(ctx: AppContext, include_co: bool) -> None:
    data = build_report_data(
        ctx.filtered,
        ctx.config,
        _meta(ctx, True),
        ctx.risk,
        co_for_report(ctx) if include_co else None,
        ctx.prepared.report,
    )
    try:
        text = write_llm_summary(llm_mod.build_client(ai_settings.current()), data)
    except (LLMError, ValueError) as exc:
        st.session_state["report_ai_error"] = str(exc)
        return
    st.session_state[SUMMARY_KEY] = text
    st.session_state["report_ai_error"] = None


def _reset_summary() -> None:
    st.session_state[SUMMARY_KEY] = st.session_state.get(AUTO_KEY, "")


def render(ctx: AppContext) -> None:
    C.heading(
        "One-click reports for your HOD",
        f"Reports cover the data in view ({ctx.filters.describe()}). The PDF is for reading and sharing; "
        "the Excel workbook is for working with the numbers.",
    )
    if ctx.filtered.empty:
        st.info("No records match the current filters.", icon="🔍")
        return

    co_available = co_for_report(ctx) is not None
    semesters = ", ".join(sorted(ctx.filtered["semester"].astype(str).unique()))
    st.session_state.setdefault(META_KEYS["semester"], semesters)
    st.session_state.setdefault(META_KEYS["title"], "Student Performance Report")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**1 · Cover page**")
        st.text_input("Report title", key=META_KEYS["title"])
        c1, c2 = st.columns(2)
        c1.text_input("College / university", key=META_KEYS["college"], placeholder="e.g. Rayat Bahra University")
        c2.text_input("Department", key=META_KEYS["department"], placeholder="e.g. Computer Applications")
        c1.text_input("Semester", key=META_KEYS["semester"])
        c2.text_input("Faculty / prepared by", key=META_KEYS["faculty"], placeholder="Your name")
    with right:
        st.markdown("**2 · Contents**")
        include_names = st.toggle(
            "Include student names",
            value=True,
            key="report_names",
            help="Turn off for a copy you share outside the department: names and roll numbers "
            "are left out and the at-risk list becomes counts per section.",
        )
        include_co = st.toggle(
            "Include CO attainment",
            value=co_available,
            key="report_co",
            disabled=not co_available,
            help=None if co_available else "Open 🎯 CO Attainment and add question-wise marks first.",
        )
        include_co = include_co and co_available
        items = [
            "Key figures and executive summary",
            "Results by section and subject",
            "Fail-rate charts and heatmap",
            "Grade distribution",
            "At-risk students with reasons" + ("" if include_names else " (counts only)"),
            "Method and data-quality notes",
        ]
        if include_co:
            items.insert(5, "CO attainment table and chart")
        st.markdown("\n".join(f"- {item}" for item in items))
        st.caption("Excel adds sheets for every at-risk student and the full cleaned data.")

    meta = _meta(ctx, include_names)
    co_result = co_for_report(ctx) if include_co else None
    data = build_report_data(ctx.filtered, ctx.config, meta, ctx.risk, co_result, ctx.prepared.report)
    automatic = template_summary(data)
    # Keep the summary in step with the data unless the user has edited it.
    if st.session_state.get(SUMMARY_KEY) in (None, st.session_state.get(AUTO_KEY)):
        st.session_state[SUMMARY_KEY] = automatic
    st.session_state[AUTO_KEY] = automatic

    st.markdown("**3 · Executive summary**")
    st.text_area(
        "Executive summary",
        key=SUMMARY_KEY,
        height=150,
        label_visibility="collapsed",
        help="Written from the numbers. Edit freely, or let the AI write it.",
    )
    b1, b2, _ = st.columns([1.3, 1.3, 3])
    ai_ready = ai_settings.current().configured
    b1.button(
        "✨ Write with AI",
        key="report_ai",
        on_click=_write_with_ai,
        args=(ctx, include_co),
        disabled=not ai_ready,
        width="stretch",
        help="Only aggregate figures are sent — never student names."
        if ai_ready
        else "Set up a model in ⚙️ AI settings to use this.",
    )
    b2.button(
        "↺ Use automatic text",
        key="report_reset",
        on_click=_reset_summary,
        width="stretch",
        disabled=st.session_state[SUMMARY_KEY] == automatic,
    )
    if st.session_state.get("report_ai_error"):
        st.error(st.session_state["report_ai_error"], icon="🤖")

    summary = st.session_state[SUMMARY_KEY].strip() or automatic
    signature = _signature(ctx, meta, summary, include_co)
    output = st.session_state.get(OUTPUT_KEY)
    st.write("")
    if st.button("📄 Generate reports", type="primary", key="report_generate"):
        with st.spinner("Building the PDF and Excel reports…"):
            start = time.monotonic()
            try:
                pdf = build_pdf(data, summary)
                xlsx = build_excel(data, summary)
            except Exception as exc:  # noqa: BLE001 - show the problem, keep the app running
                log.exception("Report generation failed")
                st.error(f"Couldn't build the reports: {type(exc).__name__}: {exc}", icon="📄")
                return
            output = {"signature": signature, "pdf": pdf, "xlsx": xlsx, "seconds": time.monotonic() - start}
            st.session_state[OUTPUT_KEY] = output
        st.toast("Reports ready to download", icon="✅")

    if output and output["signature"] == signature:
        name = _slug(f"{meta.title} {meta.semester} {ctx.filters.describe()}")
        d1, d2, _ = st.columns([1.3, 1.3, 3])
        d1.download_button(
            "⬇️ PDF report",
            output["pdf"],
            file_name=f"{name}.pdf",
            mime="application/pdf",
            key="report_dl_pdf",
            width="stretch",
        )
        d2.download_button(
            "⬇️ Excel workbook",
            output["xlsx"],
            file_name=f"{name}.xlsx",
            key="report_dl_xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
        st.caption(
            f"Built in {output['seconds']:.1f}s · PDF {len(output['pdf']) / 1024:,.0f} KB · "
            f"Excel {len(output['xlsx']) / 1024:,.0f} KB"
        )
    elif output:
        st.info("Settings or data changed since the last build — press **Generate reports** again.", icon="🔄")
