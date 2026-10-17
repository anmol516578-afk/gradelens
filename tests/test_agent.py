"""Agent loop with a scripted MockLLM: success, self-correction, limits, privacy."""

from __future__ import annotations

import pandas as pd
import pytest

from agent import analyst
from agent.analyst import ANSWERED, FAILED, GAVE_UP, MAX_STEPS_REACHED, AnalystAgent, history_messages
from agent.llm import LLMError, LLMReply, MockLLM, ToolCall, text_reply, tool_reply
from agent.masking import Pseudonymizer
from agent.sandbox import ExecutionResult
from agent.tools import Toolbox

FAIL_BY_SECTION = (
    "graded = df[df['result'].isin(['Pass', 'Fail'])]\n"
    "result = (graded.assign(f=graded['result'].eq('Fail'))"
    ".groupby('section')['f'].mean().mul(100).round(1).reset_index(name='fail_rate'))\n"
    "fig = px.bar(result, x='section', y='fail_rate', title='Fail rate by section')"
)


@pytest.fixture(scope="module")
def pseudo(derived) -> Pseudonymizer:
    return Pseudonymizer.from_frame(derived)


@pytest.fixture(scope="module")
def masked(derived, pseudo) -> pd.DataFrame:
    return pseudo.mask_frame(derived)


def make_agent(llm, masked, config, **kwargs) -> AnalystAgent:
    return AnalystAgent(llm, masked, config, scope="All students", **kwargs)


def test_success_path_runs_code_and_answers(masked, config):
    llm = MockLLM(
        [
            tool_reply("run_python", code=FAIL_BY_SECTION, purpose="Fail rate by section"),
            tool_reply("final_answer", answer="Section B has the highest fail rate.", assumptions=["Ignored blanks"]),
        ]
    )
    events: list[str] = []
    result = make_agent(llm, masked, config).run("Which section fails most?", on_event=lambda k, d: events.append(k))
    assert result.status == ANSWERED
    assert result.answer == "Section B has the highest fail rate."
    assert result.assumptions == ["Ignored blanks"]
    assert len(result.steps) == 1 and result.steps[0].ok
    assert result.steps[0].purpose == "Fail rate by section"
    rates = result.table.set_index("section")["fail_rate"]
    assert rates.idxmax() == "B"  # section B is the deliberately weaker one
    assert result.figure_json and "Fail rate by section" in result.figure_json
    assert events[0] == "thinking" and "tool_start" in events and events[-1] == "answer"
    # The model saw the computed numbers before answering.
    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool" and "fail_rate" in tool_message["content"]


def test_error_then_fix(masked, config):
    llm = MockLLM(
        [
            tool_reply("run_python", code="result = df.groupby('sec')['score_pct'].mean()", purpose="Average"),
            tool_reply("run_python", code="result = df.groupby('section')['score_pct'].mean()", purpose="Average"),
            tool_reply("final_answer", answer="Averages computed."),
        ]
    )
    events: list[tuple[str, dict]] = []
    result = make_agent(llm, masked, config).run("Average by section?", on_event=lambda k, d: events.append((k, d)))
    assert result.status == ANSWERED
    assert [s.ok for s in result.steps] == [False, True]
    assert "KeyError" in result.steps[0].error
    # The error went back to the model, and the retry was flagged so the UI can say "Fixing error".
    second_call_messages = llm.calls[1]["messages"]
    assert second_call_messages[-1]["role"] == "tool" and "KeyError" in second_call_messages[-1]["content"]
    starts = [d for k, d in events if k == "tool_start"]
    assert starts[0]["retrying"] is False and starts[1]["retrying"] is True
    assert len(result.table) == 3


def test_max_steps_exhaustion(masked, config):
    llm = MockLLM([tool_reply("run_python", code="result = len(df)", purpose="Count")], repeat_last=True)
    result = make_agent(llm, masked, config).run("Loop forever")
    assert result.status == MAX_STEPS_REACHED
    assert result.llm_calls == analyst.MAX_STEPS == 6
    assert "ran out of steps" in result.answer
    assert result.scalar == len(masked)  # the last computed value is still shown
    last_messages = llm.calls[-1]["messages"]
    assert any(m["role"] == "user" and "final_answer now" in m["content"] for m in last_messages)


def test_gives_up_after_repeated_errors(masked, config):
    llm = MockLLM([tool_reply("run_python", code="result = df['nope']")], repeat_last=True)
    result = make_agent(llm, masked, config).run("Broken")
    assert result.status == GAVE_UP
    assert len(result.steps) == analyst.MAX_ERROR_RETRIES + 1  # first try + 3 retries
    assert "KeyError" in result.answer


def test_no_real_name_or_roll_number_reaches_the_llm(derived, pseudo, masked, config):
    name = derived["student_name"].iloc[0]
    roll = derived["roll_no"].iloc[0]
    student_id = pseudo.roll_to_id[roll]
    code = (
        "result = df.groupby('student_id')['score_pct'].mean().nsmallest(10).reset_index()\n"
        "fig = px.bar(result, x='student_id', y='score_pct')"
    )
    llm = MockLLM(
        [
            tool_reply("get_column_values", column="section"),
            tool_reply("run_python", code=code, purpose="Weakest students"),
            tool_reply("run_sql", query="SELECT student_id, score_pct FROM marks ORDER BY score_pct LIMIT 30"),
            tool_reply("final_answer", answer=f"{student_id} needs support."),
        ]
    )
    question = pseudo.mask_text(f"How is {name} ({roll}) doing compared with the weakest students?")
    history = [{"question": pseudo.mask_text(f"Tell me about {name}"), "answer": f"{student_id} is fine.", "code": ""}]
    result = make_agent(llm, masked, config).run(question, history=history)
    assert result.status == ANSWERED
    payload = llm.payload_text()
    leaked = [n for n in pseudo.real_names() if n in payload]
    assert not leaked, leaked[:5]
    assert not [r for r in derived["roll_no"].unique() if r in payload]
    assert student_id in payload  # the pseudonym is what the model works with
    # ...and the app can turn the answer back into the real name for display.
    assert name in pseudo.unmask_text(result.answer)


def test_plain_text_answer_is_accepted(masked, config):
    result = make_agent(MockLLM([text_reply("There are 180 students.")]), masked, config).run("How many?")
    assert result.status == ANSWERED and result.answer == "There are 180 students."


def test_empty_reply_gets_a_nudge(masked, config):
    llm = MockLLM([LLMReply(content=None), tool_reply("final_answer", answer="Done")])
    result = make_agent(llm, masked, config).run("Q")
    assert result.status == ANSWERED
    assert "Please use a tool" in llm.calls[1]["messages"][-1]["content"]


def test_llm_error_is_reported(masked, config):
    def boom(messages, tools):
        raise LLMError("OpenAI rejected the API key.")

    result = make_agent(MockLLM([boom]), masked, config).run("Q")
    assert result.status == FAILED and "rejected the API key" in result.answer


def test_invalid_arguments_and_unknown_tool_go_back_to_model(masked, config):
    bad = LLMReply(tool_calls=[ToolCall("c1", "run_python", {}, "{oops", parse_error="Could not parse")])
    llm = MockLLM([bad, tool_reply("delete_everything"), tool_reply("final_answer", answer="ok")])
    result = make_agent(llm, masked, config).run("Q")
    assert result.status == ANSWERED
    assert [s.ok for s in result.steps] == [False, False]
    assert "Unknown tool" in llm.calls[2]["messages"][-1]["content"]


def test_column_values_tool(masked):
    outcome = Toolbox(masked).run(ToolCall("c", "get_column_values", {"column": "subject"}))
    assert outcome.ok and "Python" in outcome.llm_text
    missing = Toolbox(masked).run(ToolCall("c", "get_column_values", {"column": "student_name"}))
    assert not missing.ok and "does not exist" in missing.llm_text


def test_tool_output_to_llm_is_truncated(masked):
    outcome = Toolbox(masked).run(ToolCall("c", "run_python", {"code": "result = df"}))
    assert outcome.ok
    assert f"{len(masked)} row(s)" in outcome.llm_text and "First 30 rows shown" in outcome.llm_text
    assert outcome.llm_text.count("\n") < 40


def test_followup_history_is_sent(masked, config):
    llm = MockLLM([tool_reply("final_answer", answer="Section B only: 12%.")])
    history = [{"question": "Fail rate by section?", "answer": "A 8%, B 18%", "code": "result = 1"}]
    make_agent(llm, masked, config).run("Now only section B", history=history)
    roles = [m["role"] for m in llm.calls[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "result = 1" in llm.calls[0]["messages"][2]["content"]
    assert len(history_messages([history[0]] * 10)) == 2 * analyst.HISTORY_TURNS


def test_system_prompt_has_dictionary_scheme_and_masked_samples(masked, config, pseudo):
    agent = make_agent(MockLLM([]), masked, config)
    prompt = agent.system_prompt
    for text in ("`student_id`", "`mst1`", "Out of 15", "Pass mark: 40%", "Sections: A, B, C", "First 5 rows"):
        assert text in prompt, text
    assert not any(name in prompt for name in pseudo.real_names())


def test_fake_executor_can_be_injected(masked, config):
    calls = []

    def fake(mode, code, df, timeout):
        calls.append(mode)
        return ExecutionResult(ok=True, kind="scalar", scalar=42)

    llm = MockLLM([tool_reply("run_sql", query="SELECT 42"), tool_reply("final_answer", answer="42")])
    result = make_agent(llm, masked, config, toolbox=Toolbox(masked, executor=fake)).run("Q")
    assert calls == ["sql"] and result.scalar == 42 and result.has_scalar


# --- Realistic model habits (DEBUG_LOG #42) ---------------------------------------------


def test_fenced_code_and_habitual_imports_still_run(masked):
    code = "```python\nimport pandas as pd\nimport numpy as np\nresult = round(df['score_pct'].mean(), 1)\n```"
    outcome = Toolbox(masked).run(ToolCall("c", "run_python", {"code": code}))
    assert outcome.ok, outcome.error
    assert "Imports are not needed" in outcome.llm_text
    assert outcome.code == "result = round(df['score_pct'].mean(), 1)"


def test_other_imports_are_still_blocked_inside_fences(masked):
    outcome = Toolbox(masked).run(ToolCall("c", "run_python", {"code": "```python\nimport os\nresult = 1\n```"}))
    assert not outcome.ok and "import" in outcome.error


def test_query_error_suggests_boolean_indexing(masked):
    outcome = Toolbox(masked).run(ToolCall("c", "run_python", {"code": "result = df.query('score_pct < 40')"}))
    assert not outcome.ok and "boolean indexing" in outcome.llm_text


def test_argument_aliases(masked):
    assert Toolbox(masked).run(ToolCall("c", "run_python", {"python": "result = len(df)"})).ok
    assert Toolbox(masked).run(ToolCall("c", "run_sql", {"sql": "SELECT COUNT(*) AS n FROM marks"})).ok


def test_code_written_as_text_is_run_before_anything_succeeds(masked, config):
    llm = MockLLM(
        [
            text_reply("Let me compute it:\n```python\nresult = df.groupby('section')['score_pct'].mean()\n```"),
            tool_reply("final_answer", answer="Averages computed."),
        ]
    )
    result = make_agent(llm, masked, config).run("Average by section?")
    assert result.status == ANSWERED and len(result.steps) == 1 and result.steps[0].ok
    assert len(result.table) == 3
    assert "I ran the code in your message" in llm.calls[1]["messages"][-1]["content"]


def test_final_text_with_example_code_is_not_executed(masked, config):
    llm = MockLLM(
        [
            tool_reply("run_python", code="result = len(df)"),
            text_reply("There are 900 rows. You could check with:\n```python\nlen(df)\n```"),
        ]
    )
    result = make_agent(llm, masked, config).run("How many rows?")
    assert result.status == ANSWERED and len(result.steps) == 1
    assert result.answer.startswith("There are 900 rows.")


def test_final_answer_without_answer_falls_back_to_reply_text(masked, config):
    reply = tool_reply("final_answer", content="Section B is weakest.", assumptions=["Used score_pct"])
    result = make_agent(MockLLM([reply]), masked, config).run("Weakest section?")
    assert result.answer == "Section B is weakest." and result.assumptions == ["Used score_pct"]


def test_tool_call_messages_use_null_content():
    message = tool_reply("run_python", code="result = 1").to_message()
    assert message["content"] is None and message["tool_calls"]
    assert text_reply("hi").to_message()["content"] == "hi"
