"""Excel report: Summary, At-Risk, CO Attainment and Cleaned Data sheets.

Header styling, frozen header rows, auto-filters, number formats and
conditional formatting (colour scales for scores and fail rates, red/amber
fills for risk levels, red text for Fail) so the workbook is useful as-is.

The workbook is written in openpyxl's *write-only* (streaming) mode: rows go
straight to disk-backed XML instead of living in memory as Cell objects. For
5,000 students x 8 subjects that is ~4 MB of extra memory instead of ~260 MB,
and about twice as fast (DEBUG_LOG #53). Column widths are therefore decided
before a sheet's rows are written, and sheets are written top to bottom.
"""

from __future__ import annotations

import io
import math
import re
import textwrap

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule, DataBarRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from core.palette import SCORE_COLOR
from reports.data import ReportData

HEADER_FILL = PatternFill("solid", fgColor=SCORE_COLOR.lstrip("#").upper())
HEADER_FONT = Font(bold=True, color="FFFFFF")
HEADER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)
HEADER_BORDER = Border(bottom=Side(style="thin", color="D9D7D0"))
TITLE_FONT = Font(bold=True, size=15)
SUBTITLE_FONT = Font(size=10, color="6B6963")
SECTION_FONT = Font(bold=True, size=12)
LABEL_FONT = Font(bold=True)
HIGH_FILL = PatternFill("solid", fgColor="F8D4D4")
MEDIUM_FILL = PatternFill("solid", fgColor="FDEBC2")
FAIL_FONT = Font(color="C0392B", bold=True)
PCT = "0.0"
GREEN_TO_RED = ("F8696B", "FFEB84", "63BE7B")  # red (low) -> yellow -> green (high)
SUMMARY_LINE = 110  # characters per executive-summary line (text overflows across the empty cells to its right)

CLEAN_COLUMNS = {
    "roll_no": "Roll No",
    "student_name": "Name",
    "section": "Section",
    "semester": "Semester",
    "subject": "Subject",
    "attendance_pct": "Attendance %",
    "assignment": "Assignment",
    "mst1": "MST-1",
    "mst2": "MST-2",
    "end_term": "End Term",
    "internal_total": "Internal Total",
    "total": "Total",
    "score_pct": "Score %",
    "result": "Result",
    "result_reported": "Result (as in sheet)",
    "assignment_absent": "Absent: Assignment",
    "mst1_absent": "Absent: MST-1",
    "mst2_absent": "Absent: MST-2",
    "end_term_absent": "Absent: End Term",
}


def _clean_value(value):
    """Plain Python values for openpyxl: NaN/NA -> blank, numpy scalars -> Python."""
    if isinstance(value, (list, tuple, dict)):
        return "; ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
    if value is None or pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _cell(ws, value, font=None, fill=None, number_format=None, alignment=None, border=None) -> WriteOnlyCell:
    cell = WriteOnlyCell(ws, value=_clean_value(value))
    if cell.data_type == "f":
        # openpyxl treats any text starting with "=" as a formula. A student named
        # '=HYPERLINK("http://…")' must stay text in the HOD's workbook (DEBUG_LOG #39).
        cell.data_type = "s"
    if font is not None:
        cell.font = font
    if fill is not None:
        cell.fill = fill
    if number_format is not None:
        cell.number_format = number_format
    if alignment is not None:
        cell.alignment = alignment
    if border is not None:
        cell.border = border
    return cell


def _header_cells(ws, columns) -> list[WriteOnlyCell]:
    return [_cell(ws, str(c), font=HEADER_FONT, fill=HEADER_FILL, alignment=HEADER_ALIGN, border=HEADER_BORDER)
            for c in columns]  # fmt: skip


class _Sheet:
    """Builds one write-only sheet top to bottom, keeping track of the current row."""

    def __init__(self, wb: Workbook, title: str) -> None:
        self.ws = wb.create_sheet(title)
        self.rows: list[list[WriteOnlyCell]] = []
        self.overflow_rows: set[int] = set()  # long text that spills right; ignored when sizing columns

    @property
    def next_row(self) -> int:
        return len(self.rows) + 1

    def line(self, *values, font=None, overflow: bool = False) -> None:
        if overflow:
            self.overflow_rows.add(len(self.rows))
        self.rows.append([_cell(self.ws, v, font=font if i == 0 else None) for i, v in enumerate(values)])

    def blank(self) -> None:
        self.rows.append([])

    def table(self, frame: pd.DataFrame, formats: dict[str, str] | None = None) -> tuple[int, int, dict[str, str]]:
        """Header + rows; returns (first_data_row, last_row, {column name: letter})."""
        formats = formats or {}
        letters = {column: get_column_letter(j) for j, column in enumerate(frame.columns, start=1)}
        self.rows.append(_header_cells(self.ws, frame.columns))
        first = self.next_row
        column_formats = [formats.get(c) for c in frame.columns]
        for row in frame.itertuples(index=False):
            self.rows.append([_cell(self.ws, v, number_format=f) for v, f in zip(row, column_formats, strict=True)])
        return first, self.next_row - 1, letters

    def autosize(self, minimum: int = 8, maximum: int = 60, overrides: dict[str, float] | None = None) -> None:
        widths: dict[int, int] = {}
        for i, row in enumerate(self.rows):
            if i in self.overflow_rows:
                continue
            for j, cell in enumerate(row, start=1):
                if cell.value is not None:
                    length = 7 if isinstance(cell.value, float) else len(str(cell.value))
                    widths[j] = max(widths.get(j, 0), length)
        for j, width in widths.items():
            self.ws.column_dimensions[get_column_letter(j)].width = max(minimum, min(maximum, width + 2))
        for letter, width in (overrides or {}).items():
            self.ws.column_dimensions[letter].width = width

    def flush(self) -> None:
        for row in self.rows:
            self.ws.append(row)
        self.rows = []


def _summary_lines(summary: str) -> list[str]:
    """Executive summary as short lines (write-only sheets cannot merge cells for one wrapped block)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary.strip()) if s.strip()]
    lines: list[str] = []
    for sentence in sentences:
        lines += textwrap.wrap(sentence, SUMMARY_LINE) or [sentence]
    return lines


def _summary_sheet(wb: Workbook, data: ReportData, summary: str) -> None:
    sheet = _Sheet(wb, "Summary")
    ws = sheet.ws
    meta = data.meta
    sheet.line(meta.title, font=TITLE_FONT, overflow=True)
    sheet.line(" · ".join(x for x in (meta.college, meta.department) if x) or "GradeLens report", font=SUBTITLE_FONT,
               overflow=True)  # fmt: skip
    sheet.blank()
    details = [
        ("Semester", meta.semester),
        ("Faculty", meta.faculty),
        ("Scope", meta.scope),
        ("Stage", "Final results" if data.stage == "final" else "Mid-semester"),
        ("Generated", meta.generated),
    ]
    for label, value in details:
        sheet.line(label, value or "–", font=LABEL_FONT)
    sheet.blank()
    sheet.line("Executive summary", font=SECTION_FONT)
    for text in _summary_lines(summary):
        sheet.line(text, overflow=True)
    sheet.blank()

    k = data.kpis
    sheet.line("Key figures", font=SECTION_FONT)
    kpis = pd.DataFrame(
        [
            ["Students", k["students"]],
            ["Records (student × subject)", k["records"]],
            ["Pass rate %", round(k["pass_rate"], 1) if not math.isnan(k["pass_rate"]) else None],
            ["Average score %", round(k["avg_score"], 1) if not math.isnan(k["avg_score"]) else None],
            ["Failed records", k["fails"]],
            ["At-risk students (high + medium)", k.get("at_risk")],
            ["Absentees", k["absentees"]],
        ],
        columns=["Measure", "Value"],
    )
    sheet.table(kpis)
    sheet.blank()

    sheet.line("By section", font=SECTION_FONT)
    sections = data.by_section.rename(
        columns={
            "section": "Section",
            "students": "Students",
            "records": "Records",
            "avg_score": "Avg score %",
            "pass_rate": "Pass rate %",
            "fail_rate": "Fail rate %",
            "fails": "Fails",
            "avg_attendance": "Attendance %",
        }
    )
    first, last, letters = sheet.table(sections, {c: PCT for c in sections.columns if "%" in c})
    if last >= first:
        ws.conditional_formatting.add(
            f"{letters['Fail rate %']}{first}:{letters['Fail rate %']}{last}",
            ColorScaleRule(start_type="min", start_color="FFFFFF", end_type="max", end_color="F39A74"),
        )
    sheet.blank()

    sheet.line("By subject (hardest first)", font=SECTION_FONT)
    subjects = data.by_subject[
        ["rank", "subject", "difficulty_index", "avg_score", "fail_rate", "fails", "students", "avg_attendance"]
    ].rename(
        columns={
            "rank": "Rank",
            "subject": "Subject",
            "difficulty_index": "Difficulty",
            "avg_score": "Avg score %",
            "fail_rate": "Fail rate %",
            "fails": "Fails",
            "students": "Students",
            "avg_attendance": "Attendance %",
        }
    )
    first, last, letters = sheet.table(subjects, {"Difficulty": PCT, **{c: PCT for c in subjects.columns if "%" in c}})
    if last >= first:
        ws.conditional_formatting.add(
            f"{letters['Difficulty']}{first}:{letters['Difficulty']}{last}",
            DataBarRule(start_type="num", start_value=0, end_type="num", end_value=100, color="EB6834"),
        )

    if not data.matrix.empty:
        sheet.blank()
        sheet.line("Fail rate %: section × subject", font=SECTION_FONT)
        matrix = data.matrix.round(1).reset_index().rename(columns={"section": "Section"})
        first, last, _ = sheet.table(matrix, {c: PCT for c in matrix.columns if c != "Section"})
        start, end = get_column_letter(2), get_column_letter(len(matrix.columns))
        if last >= first:
            ws.conditional_formatting.add(
                f"{start}{first}:{end}{last}",
                ColorScaleRule(
                    start_type="num", start_value=0, start_color="FFFFFF", end_type="max", end_color="EB6834"
                ),
            )
    sheet.autosize(overrides={"A": 34})
    sheet.flush()


def _risk_sheet(wb: Workbook, data: ReportData) -> None:
    sheet = _Sheet(wb, "At-Risk")
    ws = sheet.ws
    if data.risk_students is None:
        sheet.line("Risk scores were not included in this report.")
        sheet.flush()
        return
    students = data.risk_students
    frame = pd.DataFrame(
        {
            "Roll No": students["roll_no"],
            "Name": students["student_name"] if data.meta.include_names else "(omitted)",
            "Section": students["section"],
            "Risk level": students["risk_level"],
            "Risk score": students["risk_score"].astype(int),
            "Model fail %": (students["ml_prob"] * 100).round(1),
            "Avg score %": students["avg_score"].round(1),
            "Attendance %": students["avg_attendance"].round(1),
            "Subjects failed": students["fails"],
            "Reasons": students["reasons"].map(lambda rs: "; ".join(rs) if rs else ""),
        }
    )
    if not data.meta.include_names:
        frame["Roll No"] = "(omitted)"
    if frame["Model fail %"].isna().all():
        frame = frame.drop(columns=["Model fail %"])
    sheet.line("Students by risk level — decision support, not a final judgement", font=SECTION_FONT, overflow=True)
    sheet.blank()
    first, last, letters = sheet.table(frame, {"Avg score %": PCT, "Attendance %": PCT, "Model fail %": PCT})
    ws.freeze_panes = "A4"
    ws.auto_filter.ref = f"A3:{get_column_letter(len(frame.columns))}{max(last, 3)}"
    if last >= first:
        level = letters["Risk level"]
        full = f"A{first}:{get_column_letter(len(frame.columns))}{last}"
        ws.conditional_formatting.add(full, FormulaRule(formula=[f'${level}{first}="High"'], fill=HIGH_FILL))
        ws.conditional_formatting.add(full, FormulaRule(formula=[f'${level}{first}="Medium"'], fill=MEDIUM_FILL))
        score = letters["Risk score"]
        ws.conditional_formatting.add(
            f"{score}{first}:{score}{last}",
            DataBarRule(start_type="num", start_value=0, end_type="num", end_value=100, color="D03B3B"),
        )
    sheet.autosize(overrides={letters["Reasons"]: 90})
    sheet.flush()


def _co_sheet(wb: Workbook, data: ReportData) -> None:
    sheet = _Sheet(wb, "CO Attainment")
    ws = sheet.ws
    if data.co_table is None or data.co_table.empty:
        sheet.line("No CO attainment data. Open the 🎯 CO Attainment tab (question-wise marks) to include it.")
        sheet.flush()
        return
    s = data.co_settings
    sheet.line("Course Outcome attainment", font=SECTION_FONT)
    sheet.line(
        f"A student attains a CO at ≥ {s.student_threshold:g}% of its marks. Level 3 ≥ {s.level3:g}% of "
        f"students, Level 2 ≥ {s.level2:g}%, Level 1 ≥ {s.level1:g}%. Target: Level {s.target_level}.",
        font=SUBTITLE_FONT,
        overflow=True,
    )
    sheet.blank()
    co = data.co_table.assign(status=data.co_table["meets_target"].map({True: "Meets target", False: "Below target"}))
    frame = co[
        ["co", "questions", "max_marks", "students_attained", "students_assessed", "pct_attained", "level", "status"]
    ].rename(
        columns={
            "co": "CO",
            "questions": "Questions",
            "max_marks": "Max marks",
            "students_attained": "Attained",
            "students_assessed": "Assessed",
            "pct_attained": "% attained",
            "level": "Level",
            "status": "Status",
        }
    )
    first, last, letters = sheet.table(frame, {"% attained": PCT})
    if last >= first:
        col = letters["% attained"]
        ws.conditional_formatting.add(
            f"{col}{first}:{col}{last}",
            ColorScaleRule(
                start_type="num",
                start_value=0,
                start_color=GREEN_TO_RED[0],
                mid_type="num",
                mid_value=s.threshold_for_level(s.target_level),
                mid_color=GREEN_TO_RED[1],
                end_type="num",
                end_value=100,
                end_color=GREEN_TO_RED[2],
            ),
        )
        status = letters["Status"]
        ws.conditional_formatting.add(
            f"{status}{first}:{status}{last}", CellIsRule(operator="equal", formula=['"Below target"'], font=FAIL_FONT)
        )
    sheet.autosize()
    sheet.flush()


def _data_sheet(wb: Workbook, data: ReportData) -> None:
    """Thousands of rows: streamed straight into the file, rounded rather than styled cell by cell."""
    ws = wb.create_sheet("Cleaned Data")
    columns = [c for c in CLEAN_COLUMNS if c in data.clean.columns]
    frame = data.clean[columns].rename(columns=CLEAN_COLUMNS)
    if not data.meta.include_names:
        frame = frame.drop(columns=[c for c in ("Name",) if c in frame])
        frame["Roll No"] = "(omitted)"
    floats = [c for c in frame.columns if pd.api.types.is_float_dtype(frame[c])]
    frame = frame.assign(**{c: frame[c].round(2) for c in floats})
    letters = {column: get_column_letter(j) for j, column in enumerate(frame.columns, start=1)}
    for column, letter in letters.items():  # widths first: write-only sheets emit them before any row
        ws.column_dimensions[letter].width = 24 if column == "Name" else max(9, min(28, len(column) + 3))
    ws.freeze_panes = "A2"
    ws.append(_header_cells(ws, frame.columns))
    text_columns = [j for j, c in enumerate(frame.columns) if not pd.api.types.is_numeric_dtype(frame[c])]
    # Convert once per column (NaN -> blank, numpy -> Python), not once per cell.
    plain = frame.astype(object).where(frame.notna(), None)
    for row in plain.itertuples(index=False, name=None):
        values = list(row)
        for j in text_columns:
            value = values[j]
            if isinstance(value, str) and value.startswith("="):
                values[j] = _cell(ws, value)  # stored as text, never as a formula (DEBUG_LOG #39)
        ws.append(values)
    first, last = 2, 1 + len(frame)
    ws.auto_filter.ref = f"A1:{get_column_letter(len(frame.columns))}{max(last, 1)}"
    if last >= first:
        if "Score %" in letters:
            col = letters["Score %"]
            ws.conditional_formatting.add(
                f"{col}{first}:{col}{last}",
                ColorScaleRule(
                    start_type="num",
                    start_value=0,
                    start_color=GREEN_TO_RED[0],
                    mid_type="num",
                    mid_value=data.config.pass_pct + 10,
                    mid_color=GREEN_TO_RED[1],
                    end_type="num",
                    end_value=100,
                    end_color=GREEN_TO_RED[2],
                ),
            )
        if "Result" in letters:
            col = letters["Result"]
            ws.conditional_formatting.add(
                f"{col}{first}:{col}{last}", CellIsRule(operator="equal", formula=['"Fail"'], font=FAIL_FONT)
            )


def build_excel(data: ReportData, summary: str) -> bytes:
    wb = Workbook(write_only=True)
    _summary_sheet(wb, data, summary)
    _risk_sheet(wb, data)
    _co_sheet(wb, data)
    _data_sheet(wb, data)
    wb.properties.title = data.meta.title
    wb.properties.creator = "GradeLens"
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
