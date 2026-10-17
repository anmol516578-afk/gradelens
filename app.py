"""GradeLens — AI Student Performance Analyst.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")  # optional local settings; real environment variables win

from core import analytics as an  # noqa: E402
from core.column_mapping import MappingError  # noqa: E402
from core.data_loader import LoadError  # noqa: E402
from core.derive import assessment_stage  # noqa: E402
from ui import (  # noqa: E402
    ai_settings,
    sidebar,
    state,
    tab_ai,
    tab_co,
    tab_dashboard,
    tab_profile,
    tab_quality,
    tab_reports,
    tab_risk,
    theme,
)
from ui import components as C  # noqa: E402
from ui.context import AppContext  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
log = logging.getLogger("gradelens")

st.set_page_config(page_title="GradeLens", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")


RENDERERS = {
    C.TAB_DASHBOARD: tab_dashboard.render,
    C.TAB_AI: tab_ai.render,
    C.TAB_RISK: tab_risk.render,
    C.TAB_CO: tab_co.render,
    C.TAB_PROFILE: tab_profile.render,
    C.TAB_REPORTS: tab_reports.render,
    C.TAB_QUALITY: tab_quality.render,
}


def _sidebar_tail() -> None:
    on_ai_tab = st.session_state.get(C.TAB_KEY) == C.TAB_AI
    ai_settings.render(expanded=on_ai_tab and not ai_settings.current().configured)
    sidebar.render_footer()


def _header(ctx: AppContext) -> None:
    stage = assessment_stage(ctx.derived_all)
    stage_label = "Final results" if stage == "final" else "Mid-semester (no end-term marks yet)"
    st.markdown(f"## {st.session_state['source_name']}")
    chips = [
        stage_label,
        f"{ctx.prepared.report.subjects} subjects",
        f"{ctx.derived_all['section'].nunique()} sections",
        ctx.filters.describe(),
    ]
    st.markdown(" ".join(f'<span class="gl-chip">{c}</span>' for c in chips), unsafe_allow_html=True)
    st.write("")


def _on_new_source(raw) -> None:
    state.reset_for_new_source()
    overrides = state.detect_for_new_source(raw)
    st.session_state["source_sig"] = state.source_signature()
    log.info(
        "Loaded %s: %d rows, format=%s", st.session_state["source_name"], len(raw), st.session_state["data_format"]
    )
    if overrides:
        st.toast("Marks scheme adjusted to match your sheet — see 'Marks scheme' in the sidebar.", icon="🔎")


def main() -> None:
    theme.inject_css()
    state.init_state()
    sidebar.render_source()

    try:
        raw = state.current_raw()
    except LoadError as exc:
        st.error(f"Couldn't read that file. {exc}", icon="📄")
        _sidebar_tail()
        return
    if raw is None:
        C.empty_state()
        _sidebar_tail()
        return

    if state.source_signature() != st.session_state.get("source_sig"):
        _on_new_source(raw)

    config = state.current_config()
    try:
        prepared = state.cached_prepare(raw, st.session_state["mapping"], config.to_key())
    except MappingError as exc:
        st.error(str(exc), icon="🧭")
        C.heading("Fix the column mapping", "Tell GradeLens which column holds what, then press Apply.")
        tab_quality.mapping_editor(raw, compact=True)
        _sidebar_tail()
        return

    derived_all = state.cached_derive(prepared.clean, config.to_key())
    filters = sidebar.render_filters(derived_all)
    sidebar.render_marks_scheme()
    _sidebar_tail()

    filtered = an.apply_filters(derived_all, [filters.semester] if filters.semester else None, filters.sections,
                                filters.subjects)  # fmt: skip
    ctx = AppContext(
        prepared=prepared,
        raw=raw,
        config=config,
        derived_all=derived_all,
        filtered=filtered,
        filters=filters,
        section_colors=theme.section_colors(list(derived_all["section"].unique())),
    )

    ctx.risk = tab_risk.compute_risk(filtered, derived_all, config)

    _header(ctx)
    current = an.kpis(filtered, at_risk_count=ctx.risk.at_risk_count)
    baseline = an.kpis(derived_all) if filters.active else None
    C.kpi_row(current, baseline)
    st.write("")

    tabs = st.tabs(C.TABS, key=C.TAB_KEY, on_change="rerun")
    for label, tab in zip(C.TABS, tabs, strict=True):
        if tab.open is False:
            continue
        with tab:
            RENDERERS[label](ctx)


main()
