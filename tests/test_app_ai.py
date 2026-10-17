"""Ask AI tab, end to end through Streamlit (AppTest) with a scripted MockLLM."""

from __future__ import annotations

import pytest

from agent import llm as llm_mod
from agent.llm import MockLLM, tool_reply
from tests.test_app_smoke import loaded, open_tab
from ui import components as C

LLM_VARS = ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")

WEAKEST = (
    "result = df.groupby('student_id', as_index=False)['score_pct'].mean().nsmallest(5, 'score_pct')\n"
    "fig = px.bar(result, x='student_id', y='score_pct', title='Weakest students')"
)


@pytest.fixture()
def no_llm_env(monkeypatch):
    for var in LLM_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def mock_llm(monkeypatch, no_llm_env):
    """Configure a key and make build_client return a fresh scripted mock for every question."""
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    created: list[MockLLM] = []

    def factory(settings):
        mock = MockLLM(
            [
                tool_reply("run_python", code="result = df['nope']", purpose="First try"),
                tool_reply("run_python", code=WEAKEST, purpose="Weakest students"),
                lambda messages, tools: tool_reply(
                    "final_answer",
                    answer=f"The weakest student is {_first_id(messages)}.",
                    assumptions=["Averaged across subjects"],
                ),
            ]
        )
        created.append(mock)
        return mock

    monkeypatch.setattr(llm_mod, "build_client", factory)
    return created


def _first_id(messages) -> str:
    """Pull the first student_id out of the last tool result, like a model would."""
    import re

    text = messages[-1]["content"]
    return re.search(r"S\d{3}", text).group(0)


def ai_tab():
    return open_tab(loaded(), C.TAB_AI)


def test_without_key_shows_setup_message(no_llm_env):
    at = ai_tab()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Connect an AI model" in text
    assert "Student names are masked before AI processing" in " ".join(c.value for c in at.caption)
    assert at.chat_input(key="ai_input").disabled
    # The rest of the app still works without a key.
    for label in (C.TAB_DASHBOARD, C.TAB_RISK, C.TAB_REPORTS):
        open_tab(at, label)


def test_question_runs_agent_and_shows_real_names(mock_llm):
    at = ai_tab()
    assert not at.chat_input(key="ai_input").disabled
    at.chat_input(key="ai_input").set_value("Who is the weakest student?").run()
    assert not at.exception, [e.value for e in at.exception]
    chat = at.session_state["chat"]
    assert [m["role"] for m in chat] == ["user", "assistant"]
    answer = chat[1]
    assert answer["status"] == "answered"
    # The model only saw an ID; the user sees a real name and roll number.
    assert "The weakest student is " in answer["answer"]
    assert "(2023BCA" in answer["answer"]
    assert [s["ok"] for s in answer["steps"]] == [False, True]
    assert "student_name" in answer["table"].columns
    assert answer["figure_json"] and "S0" not in answer["figure_json"].split('"x"')[1][:200]
    assert any(e.type == "code" or getattr(e, "language", None) for e in at.main)  # code shown
    # Nothing identifying went to the model.
    payload = mock_llm[0].payload_text()
    names = set(answer["table"]["student_name"])
    assert names and not any(n in payload for n in names)
    assert at.session_state["agent_history"], "follow-up context is kept"


def test_followup_and_clear_chat(mock_llm):
    at = ai_tab()
    at.chat_input(key="ai_input").set_value("Who is the weakest student?").run()
    at.chat_input(key="ai_input").set_value("And in section B?").run()
    assert not at.exception
    assert len(at.session_state["chat"]) == 4
    second_payload = mock_llm[1].calls[0]["messages"]
    assert [m["role"] for m in second_payload][:3] == ["system", "user", "assistant"]
    at.button(key="ai_clear").click().run()
    assert at.session_state["chat"] == [] and at.session_state["agent_history"] == []


def test_suggested_question_chip(mock_llm):
    at = ai_tab()
    chips = at.button_group(key="ai_chip")
    assert len(chips.options) >= 3
    assert any("MST-1" in str(o) for o in chips.options)
    at.session_state["ai_pending"] = chips.options[0]
    at.run()
    assert not at.exception
    assert at.session_state["chat"][0]["content"] == chips.options[0]


def test_names_typed_in_question_are_masked(mock_llm):
    at = ai_tab()
    at.chat_input(key="ai_input").set_value("How is Sai Chauhan doing?").run()
    assert not at.exception
    payload = mock_llm[0].payload_text()
    assert "Sai Chauhan" not in payload and "How is S0" in payload
    assert at.session_state["chat"][0]["content"] == "How is Sai Chauhan doing?"  # user sees what they typed


def test_sidebar_key_overrides_and_test_connection(no_llm_env, monkeypatch):
    monkeypatch.setattr(llm_mod, "build_client", lambda s: MockLLM([llm_mod.text_reply("ready")]))
    at = ai_tab()
    at.selectbox(key="ai_provider").set_value("groq").run()
    assert at.text_input(key="ai_model").value == llm_mod.PROVIDERS["groq"].default_model
    at.text_input(key="ai_api_key").input("gsk-typed").run()
    assert not at.chat_input(key="ai_input").disabled
    at.button(key="ai_test").click().run()
    assert any("Connected to Groq" in s.value for s in at.success)
