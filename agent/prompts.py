"""System prompt and data dictionary for the analyst agent."""

from __future__ import annotations

import pandas as pd

from core.config import ROLE_DESCRIPTIONS, AnalysisConfig

SAMPLE_ROWS = 5

EXTRA_DESCRIPTIONS = {
    "assignment_pct": "Assignment marks as % of its maximum.",
    "mst1_pct": "MST-1 marks as % of its maximum.",
    "mst2_pct": "MST-2 marks as % of its maximum.",
    "end_term_pct": "End-term marks as % of its maximum.",
    "internal_pct": "Internal total as % of the internal maximum.",
    "assignment_absent": "True if the student was absent / did not submit (the mark is blank, not zero).",
    "mst1_absent": "True if absent in MST-1 (mst1 is blank, not zero).",
    "mst2_absent": "True if absent in MST-2 (mst2 is blank, not zero).",
    "end_term_absent": "True if absent in the end-term exam (counts as Fail).",
}


def _max_note(column: str, config: AnalysisConfig) -> str:
    if column in ("assignment", "mst1", "mst2", "end_term", "internal_total", "total"):
        return f" Out of {config.max_marks(column):g}."
    return ""


def data_dictionary(df: pd.DataFrame, config: AnalysisConfig) -> str:
    lines = []
    for column in df.columns:
        description = ROLE_DESCRIPTIONS.get(column) or EXTRA_DESCRIPTIONS.get(column, "")
        dtype = "number" if pd.api.types.is_numeric_dtype(df[column]) and df[column].dtype != bool else "text"
        if df[column].dtype == bool:
            dtype = "true/false"
        blanks = int(df[column].isna().sum())
        blank_note = f" {blanks} blank." if blanks else ""
        lines.append(f"- `{column}` ({dtype}): {description}{_max_note(column, config)}{blank_note}")
    return "\n".join(lines)


def _values(df: pd.DataFrame, column: str, limit: int = 30) -> str:
    if column not in df:
        return "(not in this sheet)"
    values = sorted(str(v) for v in df[column].dropna().unique())
    shown = ", ".join(values[:limit])
    return shown + (f", ... ({len(values)} total)" if len(values) > limit else "")


def build_system_prompt(df: pd.DataFrame, config: AnalysisConfig, scope: str = "All students") -> str:
    students = df["student_id"].nunique() if "student_id" in df else len(df)
    sample = df.head(SAMPLE_ROWS)
    with pd.option_context("display.max_columns", 50, "display.width", 250):
        sample_text = sample.to_string(index=False, float_format=lambda v: f"{v:.1f}")
    return f"""You are GradeLens Analyst, a careful data analyst helping college faculty and HODs understand \
student marks. You answer questions by running code on the data, never by guessing.

## The data
A pandas DataFrame `df` (also queryable in SQL as table `marks`). One row = one student in one subject.
{len(df)} rows, {students} students. Data in view: {scope}.

### Columns
{data_dictionary(df, config)}

### Values
- Sections: {_values(df, "section")}
- Subjects: {_values(df, "subject")}
- Semesters: {_values(df, "semester")}
- Results: {_values(df, "result")}

### Marks scheme
Assignment /{config.max_assignment:g}, MST-1 /{config.max_mst1:g}, MST-2 /{config.max_mst2:g}, \
End term /{config.max_end_term:g}; internal total /{config.internal_max:g}; total /{config.total_max:g}.
Pass mark: {config.pass_pct:g}% (of score_pct). Attendance requirement: {config.attendance_threshold:g}%.

### First {len(sample)} rows
{sample_text}

## Tools
- run_python: pandas/numpy code on `df`; set `result`, optionally `fig` (Plotly). pd, np, px, go are pre-imported.
- run_sql: one DuckDB SELECT on table `marks`.
- get_column_values: distinct values of a column (exact spellings).
- final_answer: your answer to the user, plus any assumptions.

## Rules
1. Never guess numbers. Every number in your answer must come from a tool result in this conversation.
2. Fail rate = rows with result == 'Fail' / rows with result in ('Pass', 'Fail') x 100. Ignore 'Incomplete' rows.
3. "Passed/failed" a single test (MST-1, MST-2, assignment, end term) means that test's *_pct column is at least / \
below the pass mark ({config.pass_pct:g}%). Absent students (blank marks, *_absent True) did not pass that test; \
say how you treated them.
4. Blank marks are missing or absent, never zero. Use dropna or the *_absent flags deliberately.
5. Students are identified only by `student_id` (S001, ...). You never see names; never invent them. When listing \
students, include student_id and the relevant numbers; the app shows the real names.
6. Match text values exactly (use get_column_values if unsure). Subject and section names are case-sensitive.
7. When a chart helps (comparisons, distributions, trends), build one simple Plotly figure in `fig` with a clear \
title and axis labels. Skip charts for single numbers.
8. If the data cannot answer the question (e.g. a needed column does not exist), say so plainly instead of guessing.
9. Keep code short. Do not import anything. No file, network or eval access.
10. Finish by calling final_answer. Lead with the direct answer and key numbers (1-5 short sentences or bullets). \
Round percentages to 1 decimal. Put interpretation choices in `assumptions`.
"""


def last_step_nudge() -> str:
    return "You have used almost all of your steps. Call final_answer now with what you have computed so far."


def nudge_for_tool() -> str:
    return "Please use a tool: run code to compute the answer, or call final_answer if you already have it."
