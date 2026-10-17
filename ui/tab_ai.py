"""🤖 Ask AI tab: chat with an analyst agent that writes and runs code on the marks."""

from __future__ import annotations

import time

import pandas as pd
import plotly.io as pio
import streamlit as st

from agent import llm as llm_mod
from agent.analyst import ANSWERED, AgentResult, AnalystAgent
from agent.llm import LLMError
from agent.masking import Pseudonymizer
from core import analytics as an
from ui import ai_settings, charts
from ui import components as C
from ui.context import AppContext
from ui.theme import FONT, PLOTLY_CONFIG, SERIES

PENDING = "ai_pending"
CHAT, HISTORY = "chat", "agent_history"
MAX_CHAT_MESSAGES = 40


@st.cache_data(show_spinner=False, max_entries=4)
def cached_pseudonymizer(df_all: pd.DataFrame) -> Pseudonymizer:
    return Pseudonymizer.from_frame(df_all)


# --- Suggested questions ---------------------------------------------------------------


def suggest_questions(df: pd.DataFrame, config) -> list[str]:
    """Example questions that fit the columns and values in the uploaded sheet."""
    questions: list[str] = []
    subjects = an.subject_difficulty(df)["subject"].tolist() if df["result"].isin(["Pass", "Fail"]).any() else []
    hardest = subjects[0] if subjects else None
    if df["section"].nunique() > 1 and hardest:
        questions.append(f"Which section has the highest fail rate in {hardest}?")
    if {"mst1", "mst2"} <= set(df.columns) and df["mst2"].notna().any():
        questions.append("Which students passed MST-1 but failed MST-2?")
    if "attendance_pct" in df and df["attendance_pct"].notna().any():
        questions.append(
            f"Compare average scores of students below and above {config.attendance_threshold:g}% attendance."
        )
    questions.append("Who are the 10 weakest students overall, and in which subjects are they weakest?")
    if df["subject"].nunique() > 1 and {"mst1_pct", "mst2_pct"} <= set(df.columns):
        questions.append("Which subject has the biggest drop from MST-1 to MST-2 on average?")
    absent_cols = [c for c in ("mst1_absent", "mst2_absent", "end_term_absent") if c in df and df[c].any()]
    if absent_cols:
        questions.append("Which subject had the most absentees, and in which exam?")
    return questions[:5]


# --- State helpers -------------------------------------------------------------------


def _on_submit() -> None:
    text = (st.session_state.get("ai_input") or "").strip()
    if text:
        st.session_state[PENDING] = text


def _on_chip() -> None:
    choice = st.session_state.get("ai_chip")
    if choice:
        st.session_state[PENDING] = choice
    st.session_state["ai_chip"] = None


def _clear_chat() -> None:
    st.session_state[CHAT] = []
    st.session_state[HISTORY] = []
    st.session_state.pop(PENDING, None)


# --- Rendering -------------------------------------------------------------------------


def _figure_from_json(figure_json: str):
    fig = pio.from_json(figure_json)
    fig.update_layout(margin={"l": 8, "r": 12, "t": 48, "b": 8}, height=380)
    return fig


def render_answer(message: dict, index: int) -> None:
    status = message.get("status", ANSWERED)
    if status == ANSWERED:
        st.markdown(message["answer"])
    else:
        st.warning(message["answer"], icon="⚠️")

    table: pd.DataFrame | None = message.get("table")
    figure_json = message.get("figure_json")
    if figure_json:
        st.plotly_chart(_figure_from_json(figure_json), config=PLOTLY_CONFIG, key=f"ai_fig_{index}")
    elif table is not None:
        auto = charts.auto_chart(table)
        if auto is not None:
            st.plotly_chart(auto, config=PLOTLY_CONFIG, key=f"ai_auto_{index}")

    if message.get("assumptions"):
        st.caption("**Assumptions:** " + " · ".join(message["assumptions"]))

    steps = message.get("steps") or []
    code_steps = [s for s in steps if s.get("code")]
    if code_steps:
        with st.expander(f"Show code ({len(code_steps)} step{'s' if len(code_steps) != 1 else ''})"):
            for step in code_steps:
                icon = "✅" if step["ok"] else "⚠️"
                st.markdown(f"{icon} **Step {step['number']}** · {step.get('purpose') or step['tool']}")
                st.code(step["code"], language="sql" if step["tool"] == "run_sql" else "python")
                if step.get("error"):
                    st.caption(f"Error: {step['error']} — the agent rewrote the code.")
    if table is not None and not table.empty:
        shown = len(table)
        total = message.get("total_rows") or shown
        label = f"Show data ({total:,} row{'s' if total != 1 else ''})"
        with st.expander(label):
            st.dataframe(table, hide_index=True, width="stretch")
            if total > shown:
                st.caption(f"Showing the first {shown:,} of {total:,} rows.")
            C.download_csv(table, "gradelens_ai_result.csv", "Download result", key=f"ai_dl_{index}")
    meta = []
    if message.get("elapsed") is not None:
        meta.append(f"⏱ {message['elapsed']:.1f}s")
    if steps:
        meta.append(f"{len(steps)} step{'s' if len(steps) != 1 else ''}")
    if meta:
        st.caption(" · ".join(meta))


def _setup_card() -> None:
    with st.container(border=True):
        st.markdown("**Connect an AI model to start asking questions**")
        st.markdown(
            "The dashboard, at-risk list, CO attainment and reports all work without AI. "
            "To chat with your data, pick one of these:\n\n"
            "1. **Free cloud key** — get a key from Google Gemini or Groq, open **⚙️ AI settings** in the sidebar, "
            "choose the provider and paste it.\n"
            "2. **Fully offline** — install [Ollama](https://ollama.com), run `ollama pull qwen3:8b`, "
            "and choose *Ollama (local, free)*.\n"
            "3. **For a deployed app** — set `LLM_API_KEY` (and optionally `LLM_PROVIDER`, `LLM_MODEL`) as "
            "environment variables or Streamlit secrets. See the README."
        )


def _status_handler(status):
    def on_event(kind: str, data: dict) -> None:
        if kind == "thinking":
            status.update(label="🧭 Planning the analysis…" if data["first"] else "🤔 Reading the results…")
        elif kind == "tool_start":
            purpose = data.get("purpose") or data["tool"].replace("_", " ")
            if data.get("retrying"):
                status.update(label="🔧 Fixing an error and retrying…")
            else:
                status.update(label=f"✍️ Writing code: {purpose}")
            status.markdown(f"**Step {data['step']}** · {purpose}")
            if data.get("code"):
                status.code(data["code"], language="sql" if data["tool"] == "run_sql" else "python")
            status.update(label=f"▶️ Running: {purpose}")
        elif kind == "tool_ok":
            status.caption(f"✅ {data['summary']}")
        elif kind == "tool_error":
            status.caption(f"⚠️ {data['error']}")
        elif kind == "answer":
            status.update(label="✍️ Writing the answer…")

    return on_event


def _to_message(result: AgentResult, pseudo: Pseudonymizer, elapsed: float) -> dict:
    return {
        "role": "assistant",
        "status": result.status,
        "answer": pseudo.unmask_text(result.answer),
        "assumptions": [pseudo.unmask_text(a) for a in result.assumptions],
        "steps": [
            {"number": s.number, "tool": s.tool, "purpose": s.purpose, "code": s.code, "ok": s.ok, "error": s.error}
            for s in result.steps
        ],
        "table": pseudo.unmask_table(result.table),
        "total_rows": result.total_rows,
        "figure_json": pseudo.unmask_figure_json(result.figure_json),
        "elapsed": elapsed,
    }


def _run_agent(ctx: AppContext, question: str, pseudo: Pseudonymizer) -> dict:
    settings = ai_settings.current()
    masked_question = pseudo.mask_text(question)
    with st.status("🧭 Planning the analysis…", expanded=True) as status:
        start = time.monotonic()
        try:
            model = llm_mod.build_client(settings)
            agent = AnalystAgent(
                model,
                pseudo.mask_frame(ctx.filtered),
                ctx.config,
                scope=ctx.filters.describe(),
                figure_style={"colorway": SERIES, "font_family": FONT},
            )
            result = agent.run(masked_question, st.session_state[HISTORY], on_event=_status_handler(status))
        except LLMError as exc:
            status.update(label="AI service error", state="error", expanded=False)
            return {"role": "assistant", "status": "error", "answer": str(exc), "steps": [], "elapsed": None}
        elapsed = time.monotonic() - start
        done = result.status == ANSWERED
        label = f"Done in {elapsed:.1f}s · {len(result.steps)} step{'s' if len(result.steps) != 1 else ''}"
        status.update(label=label if done else "Stopped early", state="complete" if done else "error", expanded=False)
    if result.status == ANSWERED:
        st.session_state[HISTORY] = [*st.session_state[HISTORY], result.history_entry()][-8:]
    return _to_message(result, pseudo, elapsed)


def render(ctx: AppContext) -> None:
    st.session_state.setdefault(CHAT, [])
    st.session_state.setdefault(HISTORY, [])
    settings = ai_settings.current()

    head, clear = st.columns([5, 1], vertical_alignment="bottom")
    with head:
        C.heading(
            "Ask your data anything",
            f"The AI analyst writes and runs code on the data in view ({ctx.filters.describe()}), then explains "
            "the result. 🔒 Student names are masked before AI processing.",
        )
    clear.button(
        "Clear chat", key="ai_clear", on_click=_clear_chat, width="stretch", disabled=not st.session_state[CHAT]
    )

    if ctx.filtered.empty:
        st.info("No records match the current filters, so there is nothing to analyse.", icon="🔍")
        return
    if not settings.configured:
        _setup_card()
    else:
        st.caption(f"Model: {settings.describe()}")

    suggestions = suggest_questions(ctx.filtered, ctx.config)
    st.pills(
        "Try asking",
        suggestions,
        key="ai_chip",
        on_change=_on_chip,
        selection_mode="single",
        disabled=not settings.configured,
    )

    pseudo = cached_pseudonymizer(ctx.derived_all)
    for index, message in enumerate(st.session_state[CHAT]):
        with st.chat_message(message["role"], avatar="🧑‍🏫" if message["role"] == "user" else "🤖"):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer(message, index)

    question = st.session_state.pop(PENDING, None)
    if question and settings.configured:
        st.session_state[CHAT].append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍🏫"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🤖"):
            message = _run_agent(ctx, question, pseudo)
            st.session_state[CHAT].append(message)
            st.session_state[CHAT] = st.session_state[CHAT][-MAX_CHAT_MESSAGES:]
            render_answer(message, len(st.session_state[CHAT]) - 1)
        if message.get("status") == ANSWERED:
            st.toast("Answer ready", icon="✅")

    st.chat_input(
        "Ask about these marks… e.g. “Average MST-2 score by section in DBMS”"
        if settings.configured
        else "Connect an AI model in ⚙️ AI settings to start",
        key="ai_input",
        on_submit=_on_submit,
        disabled=not settings.configured,
    )
