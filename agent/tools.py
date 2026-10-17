"""The analyst's tools: JSON schemas for the model and a dispatcher that runs them.

Every tool works on the *masked* data frame only. Output sent back to the
model is truncated to the shape plus at most ``LLM_MAX_ROWS`` rows.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from agent import sandbox
from agent.llm import ToolCall

LLM_MAX_ROWS = 30
LLM_MAX_CHARS = 6000
MAX_DISTINCT = 50

RUN_PYTHON, RUN_SQL, COLUMN_VALUES, FINAL_ANSWER = "run_python", "run_sql", "get_column_values", "final_answer"

FENCED_BLOCK = re.compile(r"```[ \t]*(?:python|py|sql|duckdb)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
# Imports of modules the sandbox already provides; models add these out of habit (DEBUG_LOG #42).
PROVIDED_IMPORT = re.compile(
    r"^\s*(import pandas as pd|import numpy as np|import math|import plotly\.express as px"
    r"|import plotly\.graph_objects as go|from plotly import (express as px|graph_objects as go))\s*(#.*)?$"
)


def extract_code(text: str) -> str | None:
    """The first fenced code block in a message, if any."""
    match = FENCED_BLOCK.search(text or "")
    return match.group(1).strip() if match else None


def normalize_code(code: str) -> tuple[str, list[str]]:
    """Strip markdown fences and imports of already-provided modules. Returns (code, notes for the model)."""
    notes = []
    fenced = extract_code(code) if "```" in code else None
    if fenced is not None:
        code = fenced
    kept = []
    for line in code.splitlines():
        if PROVIDED_IMPORT.match(line):
            notes.append("Imports are not needed: pd, np, px, go and math are already available.")
            continue
        kept.append(line)
    return "\n".join(kept).strip(), list(dict.fromkeys(notes))


_PURPOSE = {
    "type": "string",
    "description": "A short, plain-English description of what this step does (shown to the user), "
    "e.g. 'Fail rate by section for Python'.",
}

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": RUN_PYTHON,
            "description": (
                "Run pandas/numpy code on the DataFrame `df` (one row per student per subject). "
                "pd, np, px (plotly.express) and go (plotly.graph_objects) are already imported; do not import "
                "anything. Assign the answer to `result` (DataFrame, Series or scalar). Optionally assign a Plotly "
                "figure to `fig`. No file, network or eval access. 10-second time limit."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to run."}, "purpose": _PURPOSE},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": RUN_SQL,
            "description": (
                "Run one read-only DuckDB SELECT (or WITH ... SELECT) query on the table `marks` (same data as `df`). "
                "Good for grouping and filtering. The result table is returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A single SELECT query."},
                    "purpose": _PURPOSE,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": COLUMN_VALUES,
            "description": "List the distinct values of a column with counts (use it to get exact spellings).",
            "parameters": {
                "type": "object",
                "properties": {"column": {"type": "string", "description": "Column name."}},
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": FINAL_ANSWER,
            "description": (
                "Give the final answer to the user. Call this once you have computed everything you need. "
                "Lead with the direct answer and the key numbers taken from your tool results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The answer in Markdown: 1-5 short sentences or bullets. Refer to students "
                        "by student_id (the app shows their real names).",
                    },
                    "assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Assumptions or interpretation choices you made (may be empty).",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


@dataclass
class ToolOutcome:
    tool: str
    ok: bool
    llm_text: str
    purpose: str = ""
    code: str = ""
    error: str | None = None
    table: pd.DataFrame | None = None
    total_rows: int = 0
    scalar: object = None
    has_scalar: bool = False
    figure_json: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_output(self) -> bool:
        return self.table is not None or self.has_scalar or self.figure_json is not None


def _truncate(text: str, limit: int = LLM_MAX_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... (output truncated)"


def table_preview(table: pd.DataFrame, total_rows: int | None = None, max_rows: int = LLM_MAX_ROWS) -> str:
    total = len(table) if total_rows is None else total_rows
    shown = table.head(max_rows)
    with pd.option_context("display.max_columns", 50, "display.width", 200, "display.max_colwidth", 60):
        body = shown.to_string(index=False, float_format=lambda v: f"{v:.2f}")
    header = f"Result: table with {total} row(s) x {table.shape[1]} column(s)."
    if total > max_rows:
        header += f" First {max_rows} rows shown."
    return f"{header}\n{body}"


def describe_execution(result: sandbox.ExecutionResult) -> str:
    """What the model sees after its code ran."""
    parts = []
    if result.table is not None:
        parts.append(table_preview(result.table, result.total_rows))
    elif result.kind in ("scalar", "json", "text"):
        parts.append(f"Result: {result.scalar!r}")
    else:
        parts.append(
            "The code ran but `result` was not set (or was None). Assign the answer to `result` so it can be shown."
        )
    if result.figure_json:
        parts.append("A Plotly figure `fig` was created and will be shown to the user.")
    if result.stdout.strip():
        parts.append("Printed output:\n" + result.stdout.strip()[:1500])
    parts.extend(result.notes)
    return _truncate("\n\n".join(parts))


class Toolbox:
    """Runs tool calls against the masked frame."""

    def __init__(
        self,
        df: pd.DataFrame,
        timeout: int = sandbox.DEFAULT_TIMEOUT,
        executor: Callable[..., sandbox.ExecutionResult] = sandbox.execute,
        figure_style: dict | None = None,
    ) -> None:
        self.df = df
        self.timeout = timeout
        self._execute = executor
        self.figure_style = figure_style

    def run(self, call: ToolCall) -> ToolOutcome:
        if call.parse_error:
            return ToolOutcome(call.name, False, call.parse_error, error=call.parse_error)
        args = call.arguments
        purpose = str(args.get("purpose") or "").strip()
        if call.name == RUN_PYTHON:
            code = args.get("code") or args.get("python") or args.get("source") or ""
            return self._run_code("python", str(code), purpose)
        if call.name == RUN_SQL:
            return self._run_code("sql", str(args.get("query") or args.get("sql") or args.get("code") or ""), purpose)
        if call.name == COLUMN_VALUES:
            return self._column_values(str(args.get("column") or ""))
        message = f"Unknown tool '{call.name}'. Use run_python, run_sql, get_column_values or final_answer."
        return ToolOutcome(call.name, False, message, error=message)

    def _run_code(self, mode: str, code: str, purpose: str) -> ToolOutcome:
        tool = RUN_SQL if mode == "sql" else RUN_PYTHON
        code, notes = normalize_code(code)
        extra = {"figure_style": self.figure_style} if self.figure_style else {}
        result = self._execute(mode, code, self.df, timeout=self.timeout, **extra)
        result.notes = [*notes, *result.notes]
        if not result.ok:
            error = result.error or "Unknown error"
            text = f"Error: {error}\nFix the code and try again (check column names and dtypes)."
            if "'.query'" in error or "'.eval'" in error:
                text += " Use boolean indexing, e.g. df[df['score_pct'] < 40], instead of query()/eval()."
            return ToolOutcome(tool, False, _truncate(text), purpose, code, error=error)
        has_scalar = result.kind in ("scalar", "json", "text")
        return ToolOutcome(
            tool,
            True,
            describe_execution(result),
            purpose,
            code,
            table=result.table,
            total_rows=result.total_rows,
            scalar=result.scalar,
            has_scalar=has_scalar,
            figure_json=result.figure_json,
            notes=list(result.notes),
        )

    def _column_values(self, column: str) -> ToolOutcome:
        if column not in self.df.columns:
            message = f"Column '{column}' does not exist. Available columns: {', '.join(self.df.columns)}."
            return ToolOutcome(COLUMN_VALUES, False, message, f"Values of {column}", error=message)
        counts = self.df[column].value_counts(dropna=False)
        lines = [f"{'(blank)' if pd.isna(v) else v}: {n}" for v, n in counts.head(MAX_DISTINCT).items()]
        more = f"\n... and {len(counts) - MAX_DISTINCT} more distinct values" if len(counts) > MAX_DISTINCT else ""
        text = f"Distinct values of '{column}' ({len(counts)} total), with row counts:\n" + "\n".join(lines) + more
        return ToolOutcome(COLUMN_VALUES, True, _truncate(text), f"Checking values of {column}")
