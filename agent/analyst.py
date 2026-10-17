"""ReAct-style analyst agent: plan -> write code -> run -> observe -> (fix) -> answer.

The model decides each step with tool calls. Code runs in the sandbox on the
masked data frame; results go back to the model truncated. The loop stops at
``final_answer``, after ``max_steps`` model calls, or after the model fails
to fix an error ``max_error_retries`` times in a row.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from agent import prompts
from agent.llm import ChatModel, LLMError, ToolCall
from agent.tools import FINAL_ANSWER, RUN_PYTHON, RUN_SQL, TOOL_SPECS, Toolbox, ToolOutcome, extract_code
from core.config import AnalysisConfig

log = logging.getLogger("gradelens.agent")

MAX_STEPS = 6
MAX_ERROR_RETRIES = 3
HISTORY_TURNS = 4

ANSWERED, MAX_STEPS_REACHED, GAVE_UP, FAILED = "answered", "max_steps", "gave_up", "error"

EventHandler = Callable[[str, dict], None]


@dataclass
class AgentStep:
    number: int
    tool: str
    purpose: str
    code: str
    ok: bool
    error: str | None = None
    summary: str = ""


@dataclass
class AgentResult:
    question: str
    answer: str
    status: str
    assumptions: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    table: pd.DataFrame | None = None
    total_rows: int = 0
    scalar: object = None
    has_scalar: bool = False
    figure_json: str | None = None
    llm_calls: int = 0
    error: str | None = None

    @property
    def code_steps(self) -> list[AgentStep]:
        return [s for s in self.steps if s.code]

    @property
    def step_log(self) -> list[str]:
        lines = []
        for step in self.steps:
            mark = "✅" if step.ok else "⚠️"
            what = step.purpose or step.tool
            lines.append(f"{mark} Step {step.number}: {what}" + (f" — {step.error}" if step.error else ""))
        return lines

    def history_entry(self) -> dict:
        """Compact, masked record of this turn for follow-up questions."""
        last_code = next((s.code for s in reversed(self.steps) if s.ok and s.code), "")
        return {"question": self.question, "answer": self.answer, "code": last_code}


def history_messages(history: list[dict] | None, turns: int = HISTORY_TURNS) -> list[dict]:
    messages: list[dict] = []
    for entry in (history or [])[-turns:]:
        messages.append({"role": "user", "content": entry["question"]})
        content = entry["answer"]
        if entry.get("code"):
            content += f"\n\n(Code used for this answer:\n{entry['code']}\n)"
        messages.append({"role": "assistant", "content": content})
    return messages


def _first_line(text: str, limit: int = 160) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


class AnalystAgent:
    def __init__(
        self,
        llm: ChatModel,
        df: pd.DataFrame,
        config: AnalysisConfig,
        scope: str = "All students",
        max_steps: int = MAX_STEPS,
        max_error_retries: int = MAX_ERROR_RETRIES,
        toolbox: Toolbox | None = None,
        figure_style: dict | None = None,
    ) -> None:
        self.llm = llm
        self.df = df
        self.config = config
        self.scope = scope
        self.max_steps = max_steps
        self.max_error_retries = max_error_retries
        self.toolbox = toolbox or Toolbox(df, figure_style=figure_style)
        self.system_prompt = prompts.build_system_prompt(df, config, scope)

    def run(
        self, question: str, history: list[dict] | None = None, on_event: EventHandler | None = None
    ) -> AgentResult:
        emit = on_event or (lambda kind, data: None)
        messages = [{"role": "system", "content": self.system_prompt}, *history_messages(history)]
        messages.append({"role": "user", "content": question})
        result = AgentResult(question=question, answer="", status=FAILED)
        consecutive_errors = 0
        latest_output: ToolOutcome | None = None

        for step_number in range(1, self.max_steps + 1):
            emit("thinking", {"step": step_number, "first": step_number == 1})
            if step_number == self.max_steps:
                messages.append({"role": "user", "content": prompts.last_step_nudge()})
            try:
                reply = self.llm.complete(messages, TOOL_SPECS)
            except LLMError as exc:
                result.error = str(exc)
                result.answer = f"The AI service returned an error: {exc}"
                emit("failed", {"error": str(exc)})
                return self._attach_output(result, latest_output)
            result.llm_calls += 1
            messages.append(reply.to_message())

            if not reply.tool_calls:
                code = extract_code(reply.content or "")
                if code and latest_output is None:
                    # Weaker (often local) models write the code as text instead of calling run_python.
                    # Run it for them, but only before anything has succeeded, so a final answer that
                    # merely shows example code is not executed (DEBUG_LOG #42).
                    mode = RUN_SQL if code.lstrip().upper().startswith(("SELECT", "WITH")) else RUN_PYTHON
                    key = "query" if mode == RUN_SQL else "code"
                    call = ToolCall(f"text_{step_number}", mode, {key: code, "purpose": "Code from the reply"})
                    outcome = self._run_tool(call, step_number, result, emit)
                    messages.append(
                        {
                            "role": "user",
                            "content": "I ran the code in your message. "
                            + outcome.llm_text
                            + "\nUse the tools from now on.",
                        }
                    )
                    if outcome.ok:
                        consecutive_errors = 0
                        latest_output = outcome if outcome.has_output else latest_output
                    else:
                        consecutive_errors += 1
                    continue
                if reply.content:
                    # Some models answer in plain text instead of calling final_answer.
                    result.answer, result.status = reply.content, ANSWERED
                    emit("answer", {"answer": result.answer})
                    return self._attach_output(result, latest_output)
                messages.append({"role": "user", "content": prompts.nudge_for_tool()})
                continue

            final_call = None
            for call in reply.tool_calls:
                if call.name == FINAL_ANSWER and not call.parse_error:
                    final_call = call
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": "Answer delivered."})
                    continue
                outcome = self._run_tool(call, step_number, result, emit)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": outcome.llm_text})
                if outcome.ok:
                    consecutive_errors = 0
                    if outcome.has_output:
                        latest_output = outcome
                else:
                    consecutive_errors += 1

            if final_call is not None:
                args = final_call.arguments
                answer = str(args.get("answer") or args.get("text") or "").strip() or (reply.content or "").strip()
                result.answer = answer or "(The model gave an empty answer.)"
                raw_assumptions = args.get("assumptions") or []
                if isinstance(raw_assumptions, str):
                    raw_assumptions = [raw_assumptions]
                result.assumptions = [str(a).strip() for a in raw_assumptions if str(a).strip()]
                result.status = ANSWERED
                emit("answer", {"answer": result.answer})
                return self._attach_output(result, latest_output)

            if consecutive_errors > self.max_error_retries:
                last_error = next((s.error for s in reversed(result.steps) if s.error), "unknown error")
                result.status = GAVE_UP
                result.answer = (
                    f"I couldn't get the analysis to run after {self.max_error_retries} attempts to fix it. "
                    f"Last error: {last_error}. Try rephrasing the question or naming the exact column or subject."
                )
                emit("gave_up", {"error": last_error})
                return self._attach_output(result, latest_output)

        result.status = MAX_STEPS_REACHED
        result.answer = (
            f"I ran out of steps ({self.max_steps}) before finishing."
            + (" The last result I computed is shown below." if latest_output else "")
            + " Try a more specific question."
        )
        emit("max_steps", {})
        return self._attach_output(result, latest_output)

    def _run_tool(self, call, step_number: int, result: AgentResult, emit: EventHandler) -> ToolOutcome:
        args = call.arguments
        code = str(args.get("code") or args.get("query") or args.get("sql") or "")
        purpose = str(args.get("purpose") or "").strip()
        retrying = bool(result.steps) and not result.steps[-1].ok
        emit(
            "tool_start",
            {"step": step_number, "tool": call.name, "purpose": purpose, "code": code, "retrying": retrying},
        )
        outcome = self.toolbox.run(call)
        summary = _first_line(outcome.llm_text)
        result.steps.append(
            AgentStep(
                len(result.steps) + 1,
                call.name,
                outcome.purpose or purpose,
                outcome.code or code,
                outcome.ok,
                outcome.error,
                summary,
            )
        )
        if outcome.ok:
            emit("tool_ok", {"step": step_number, "tool": call.name, "summary": summary})
        else:
            log.info("Agent step %d failed: %s", step_number, outcome.error)
            emit("tool_error", {"step": step_number, "tool": call.name, "error": outcome.error})
        return outcome

    @staticmethod
    def _attach_output(result: AgentResult, outcome: ToolOutcome | None) -> AgentResult:
        if outcome is not None:
            result.table = outcome.table
            result.total_rows = outcome.total_rows
            result.scalar = outcome.scalar
            result.has_scalar = outcome.has_scalar
            result.figure_json = outcome.figure_json
        return result
