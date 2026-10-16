"""Course Outcome (CO) attainment, OBE / NBA style.

Default method (every threshold is configurable):

1. A student *attains* a CO when they score at least 60% of the maximum
   marks of the questions mapped to that CO.
2. The CO's attainment level is 3 if at least 70% of assessed students
   attain it, 2 if at least 60%, 1 if at least 50%, otherwise 0.

Unattempted questions score 0. A student with no marks at all for the
exam (absent) is excluded rather than counted as failing every CO.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.cleaning import ABSENT_MARKERS, MISSING_MARKERS
from core.column_mapping import extract_max_hint, match_role
from core.config import NICE_MAX_MARKS

QUESTION_PATTERN = re.compile(r"^\s*q(?:uestion)?\s*[-_.]?\s*(\d+)\s*([a-z])?\s*$", re.I)


class COError(ValueError):
    """Friendly error for invalid CO inputs."""


@dataclass(frozen=True)
class COSettings:
    student_threshold: float = 60.0
    level3: float = 70.0
    level2: float = 60.0
    level1: float = 50.0
    target_level: int = 2

    def level_for(self, pct_attained: float) -> int:
        if pd.isna(pct_attained):
            return 0
        if pct_attained >= self.level3:
            return 3
        if pct_attained >= self.level2:
            return 2
        if pct_attained >= self.level1:
            return 1
        return 0

    def threshold_for_level(self, level: int) -> float:
        return {3: self.level3, 2: self.level2, 1: self.level1}.get(level, 0.0)


def _strip_hint(name: str) -> str:
    return re.sub(r"[\(\[].*?[\)\]]|/\s*\d+\s*$", "", str(name)).strip()


def detect_question_columns(columns) -> list[str]:
    return [c for c in columns if QUESTION_PATTERN.match(_strip_hint(c))]


def _nice_ceiling(value: float) -> float:
    for nice in NICE_MAX_MARKS:
        if nice >= value:
            return float(nice)
    return float(math.ceil(value))


def parse_question_marks(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw question-wise sheet -> roll_no, student_name, section + numeric Q columns."""
    questions = detect_question_columns(raw.columns)
    if not questions:
        raise COError("No question columns found. Name them Q1, Q2, ... (e.g. 'Q1', 'Q 2', 'Question 3').")
    out = pd.DataFrame(index=raw.index)
    for column in raw.columns:
        if column in questions:
            continue
        role, score = match_role(column)
        if role in ("roll_no", "student_name", "section") and role not in out and score >= 80:
            out[role] = raw[column].astype(str).str.strip()
    if "roll_no" not in out:
        out["roll_no"] = [f"STU{i + 1:04d}" for i in range(len(raw))]
    if "student_name" not in out:
        out["student_name"] = out["roll_no"]
    for column in questions:
        text = raw[column].astype(str).str.strip().str.lower()
        blank = text.isin(ABSENT_MARKERS | MISSING_MARKERS)
        out[_strip_hint(column)] = pd.to_numeric(text.where(~blank), errors="coerce")
    return out.reset_index(drop=True)


def question_columns(marks: pd.DataFrame) -> list[str]:
    return [c for c in marks.columns if QUESTION_PATTERN.match(c)]


def default_mapping(raw: pd.DataFrame, marks: pd.DataFrame) -> pd.DataFrame:
    """Starting point for the editable mapping table: every question -> CO1."""
    rows = []
    hints = {_strip_hint(c): extract_max_hint(c) for c in detect_question_columns(raw.columns)}
    for question in question_columns(marks):
        observed = marks[question].max()
        max_marks = hints.get(question) or (_nice_ceiling(observed) if pd.notna(observed) and observed > 0 else 10.0)
        rows.append({"question": question, "co": "CO1", "max_marks": float(max_marks)})
    return pd.DataFrame(rows, columns=["question", "co", "max_marks"])


def validate_mapping(marks: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if mapping.empty:
        raise COError("The question-to-CO mapping is empty.")
    mapping = mapping.copy()
    mapping["question"] = mapping["question"].astype(str).str.strip()
    mapping["co"] = mapping["co"].astype(str).str.strip().str.upper()
    mapping["max_marks"] = pd.to_numeric(mapping["max_marks"], errors="coerce")
    unknown = sorted(set(mapping["question"]) - set(marks.columns))
    if unknown:
        raise COError(f"Mapping refers to question(s) not in the marks sheet: {', '.join(unknown)}")
    missing_co = mapping["co"].isna() | mapping["co"].isin(["", "NAN", "NONE"])
    if missing_co.any():
        raise COError("Every question needs a CO (e.g. CO1).")
    if mapping["max_marks"].isna().any() or (mapping["max_marks"] <= 0).any():
        raise COError("Max marks must be a positive number for every question.")
    if mapping["question"].duplicated().any():
        raise COError("Each question can map to only one CO.")
    return mapping


def compute_co_attainment(
    marks: pd.DataFrame, mapping: pd.DataFrame, settings: COSettings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Returns (co_table, student_table, warnings)."""
    settings = settings or COSettings()
    mapping = validate_mapping(marks, mapping)
    warnings: list[str] = []
    questions = list(mapping["question"])
    scores = marks[questions].astype(float).copy()

    max_by_q = mapping.set_index("question")["max_marks"]
    over = scores.gt(max_by_q, axis=1) | scores.lt(0)
    if over.any().any():
        warnings.append(f"{int(over.sum().sum())} question mark(s) were outside 0-max and were ignored.")
        scores = scores.mask(over)

    absent_exam = scores.isna().all(axis=1)
    if absent_exam.any():
        warnings.append(f"{int(absent_exam.sum())} student(s) had no marks at all and were excluded.")

    co_rows, student_cols = [], {}
    for co, group in mapping.groupby("co", sort=True):
        qs = list(group["question"])
        max_marks = float(group["max_marks"].sum())
        obtained = scores.loc[~absent_exam, qs].fillna(0).sum(axis=1)
        pct = obtained / max_marks * 100
        attained = pct >= settings.student_threshold
        pct_attained = float(attained.mean() * 100) if len(attained) else float("nan")
        student_cols[co] = pct
        co_rows.append(
            {
                "co": co,
                "questions": ", ".join(qs),
                "max_marks": max_marks,
                "students_assessed": int(len(attained)),
                "students_attained": int(attained.sum()),
                "pct_attained": round(pct_attained, 2),
                "level": settings.level_for(pct_attained),
            }
        )
    co_table = pd.DataFrame(co_rows)
    co_table["meets_target"] = co_table["level"] >= settings.target_level

    ids = marks.loc[~absent_exam, [c for c in ("roll_no", "student_name", "section") if c in marks]]
    student_table = pd.concat([ids, pd.DataFrame(student_cols)], axis=1).reset_index(drop=True)
    return co_table, student_table, warnings


def overall_attainment(co_table: pd.DataFrame) -> float:
    """Average attainment level across COs (0-3)."""
    return float(np.round(co_table["level"].mean(), 2)) if not co_table.empty else float("nan")
