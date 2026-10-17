"""Reports tab through AppTest: generate, download, summary editing, AI summary."""

from __future__ import annotations

from agent import llm as llm_mod
from agent.llm import MockLLM, text_reply
from tests.test_app_ai import LLM_VARS
from tests.test_app_smoke import loaded, open_tab
from ui import components as C


def reports_tab(monkeypatch):
    for var in LLM_VARS:
        monkeypatch.delenv(var, raising=False)
    return open_tab(loaded(), C.TAB_REPORTS)


def test_generate_offers_pdf_and_excel_downloads(monkeypatch):
    at = reports_tab(monkeypatch)
    assert not at.error
    summary = at.text_area(key="report_summary").value
    assert "180 students across 5 subjects" in summary
    assert at.toggle(key="report_co").value  # the sample includes question-wise marks for CO attainment
    at.text_input(key="meta_college").input("Rayat Bahra University").run()
    at.button(key="report_generate").click().run()
    assert not at.exception, [e.value for e in at.exception]
    output = at.session_state["report_output"]
    assert output["pdf"].startswith(b"%PDF") and output["xlsx"][:2] == b"PK"
    assert any("Built in" in c.value for c in at.caption)


def test_changes_require_regeneration(monkeypatch):
    at = reports_tab(monkeypatch)
    at.button(key="report_generate").click().run()
    at.toggle(key="report_names").set_value(False).run()
    assert any("press **Generate reports** again" in i.value for i in at.info)


def test_edited_summary_is_kept_and_can_be_reset(monkeypatch):
    at = reports_tab(monkeypatch)
    automatic = at.text_area(key="report_summary").value
    at.text_area(key="report_summary").input("My own words.").run()
    at.multiselect(key="f_sections").set_value(["B"]).run()  # data changes, edited text stays
    open_tab(at, C.TAB_REPORTS)
    assert at.text_area(key="report_summary").value == "My own words."
    at.button(key="report_reset").click().run()
    new_auto = at.text_area(key="report_summary").value
    assert new_auto != "My own words." and new_auto != automatic  # automatic text now describes Section B
    assert "Section B" in new_auto or "60 students" in new_auto


def test_write_with_ai(monkeypatch):
    for var in LLM_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    mocks: list[MockLLM] = []

    def factory(settings):
        mocks.append(MockLLM([text_reply("AI-written summary for the HOD.")]))
        return mocks[-1]

    monkeypatch.setattr(llm_mod, "build_client", factory)
    at = open_tab(loaded(), C.TAB_REPORTS)
    at.button(key="report_ai").click().run()
    assert not at.exception
    assert at.text_area(key="report_summary").value == "AI-written summary for the HOD."
    assert "Sai Chauhan" not in mocks[0].payload_text()


def test_ai_button_disabled_without_key(monkeypatch):
    at = reports_tab(monkeypatch)
    assert at.button(key="report_ai").disabled
