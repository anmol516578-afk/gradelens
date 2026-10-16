"""Guess what each uploaded column means and reshape the sheet to long format.

Three layouts are supported:

* **long**  – one row per student per subject, with a Subject column;
* **wide**  – one row per student, columns like "Python - MST1";
* **wide-totals** – one row per student, one numeric column per subject.

Column names are matched against alias lists with rapidfuzz. Digits must
agree exactly ("MST 1" never maps to MST-2) and headers such as
"MST-1 (24)" also yield a max-marks hint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz

from core.config import IGNORE, NUMERIC_ROLES

ALIASES: dict[str, list[str]] = {
    "roll_no": [
        "roll no",
        "roll number",
        "rollno",
        "roll",
        "enrollment no",
        "enrolment no",
        "enrollment number",
        "enrollment",
        "registration no",
        "reg no",
        "registration number",
        "university roll no",
        "uid",
        "student id",
        "admission no",
        "urn",
        "crn",
        "class roll no",
        "roll num",
    ],
    "student_name": [
        "name",
        "student name",
        "name of student",
        "student",
        "full name",
        "candidate name",
        "name of the student",
        "students name",
        "student full name",
    ],
    "section": ["section", "sec", "class section", "group", "division", "div"],
    "semester": ["semester", "sem", "semester no", "sem no"],
    "subject": [
        "subject",
        "course",
        "subject name",
        "course name",
        "paper",
        "subject code",
        "course code",
        "course title",
        "subject title",
        "paper name",
    ],
    "attendance_pct": [
        "attendance",
        "attendance pct",
        "attendance percentage",
        "attendance percent",
        "att",
        "att pct",
        "percentage attendance",
        "attd",
        "attn",
        "attendance per",
    ],
    "assignment": ["assignment", "assignments", "assignment marks", "asgn", "assign", "home assignment"],
    "mst1": [
        "mst 1",
        "mid sem 1",
        "midsem 1",
        "mid semester 1",
        "mid term 1",
        "midterm 1",
        "sessional 1",
        "st 1",
        "cat 1",
        "internal test 1",
        "test 1",
        "mse 1",
        "minor 1",
        "mid sem test 1",
        "mst 1 marks",
    ],
    "mst2": [
        "mst 2",
        "mid sem 2",
        "midsem 2",
        "mid semester 2",
        "mid term 2",
        "midterm 2",
        "sessional 2",
        "st 2",
        "cat 2",
        "internal test 2",
        "test 2",
        "mse 2",
        "minor 2",
        "mid sem test 2",
        "mst 2 marks",
    ],
    "end_term": [
        "end term",
        "endterm",
        "end sem",
        "endsem",
        "end semester",
        "end semester exam",
        "end term exam",
        "final exam",
        "final",
        "external",
        "external marks",
        "ete",
        "ese",
        "university exam",
        "theory external",
        "major",
        "end sem marks",
    ],
    "internal_total": [
        "internal total",
        "internal",
        "internals",
        "internal marks",
        "total internal",
        "ia",
        "cia",
        "internal assessment",
        "sessional total",
    ],
    "total": [
        "total",
        "total marks",
        "grand total",
        "overall total",
        "marks obtained",
        "total obtained",
        "aggregate",
        "total score",
        "marks",
    ],
    "result": ["result", "status", "pass fail", "outcome", "result status", "pass or fail", "remarks"],
}

# Roles that may appear once per subject in a wide sheet.
COMPONENT_ROLES = ("attendance_pct", "assignment", "mst1", "mst2", "end_term", "internal_total", "total")
GENERIC_WORDS = {"marks", "score", "pct", "percentage", "max", "out of", "obtained", "total", "no", "number"}
ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}

PASS_WORDS = {"pass", "p", "passed", "cleared", "qualified", "pass "}
FAIL_WORDS = {"fail", "f", "failed", "reappear", "re appear", "re-appear", "er", "detained", "not cleared", "rl"}

FUZZY_THRESHOLD = 75.0
SPLIT_THRESHOLD = 90.0

_MAX_HINT_PATTERNS = (
    re.compile(r"[\(\[]\s*(?:max\.?|mm|out\s*of|max\s*marks)?\s*:?\s*(\d+(?:\.\d+)?)\s*(?:marks)?\s*[\)\]]", re.I),
    re.compile(r"(?:out\s*of|/)\s*(\d+(?:\.\d+)?)\s*$", re.I),
)


class MappingError(ValueError):
    """The sheet cannot be reshaped with the current mapping."""


MIN_HINT, MAX_HINT = 5.0, 1000.0


def extract_max_hint(name: str) -> float | None:
    for pattern in _MAX_HINT_PATTERNS:
        match = pattern.search(str(name))
        if match:
            value = float(match.group(1))
            if MIN_HINT <= value <= MAX_HINT:
                return value
    return None


def strip_max_hint(name: str) -> str:
    """Remove '(24)', '/30', 'out of 60' from a header (only plausible max marks)."""
    text = str(name)

    def _drop(match: re.Match) -> str:
        return " " if MIN_HINT <= float(match.group(1)) <= MAX_HINT else match.group(0)

    for pattern in _MAX_HINT_PATTERNS:
        text = pattern.sub(_drop, text)
    return text.strip()


def normalize_name(name: str) -> str:
    text = strip_max_hint(name).lower()
    text = text.replace("%", " pct ").replace("&", " and ")
    text = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(ROMAN.get(tok, tok) for tok in text.split())


_ALIAS_TABLE: list[tuple[str, str, frozenset[str]]] = []
_EXACT: dict[str, str] = {}
for _role, _aliases in ALIASES.items():
    for _alias in _aliases:
        _norm = normalize_name(_alias)
        _ALIAS_TABLE.append((_norm, _role, frozenset(re.findall(r"\d+", _norm))))
        _EXACT.setdefault(_norm.replace(" ", ""), _role)


def match_role(name: str) -> tuple[str | None, float]:
    """Best canonical role for a header and a 0-100 confidence."""
    norm = normalize_name(name)
    if not norm:
        return None, 0.0
    compact = norm.replace(" ", "")
    if compact in _EXACT:
        return _EXACT[compact], 100.0
    digits = frozenset(re.findall(r"\d+", norm))
    best_role, best_score = None, 0.0
    for alias_norm, role, alias_digits in _ALIAS_TABLE:
        if alias_digits != digits:
            continue
        score = 0.6 * fuzz.token_set_ratio(norm, alias_norm) + 0.4 * fuzz.token_sort_ratio(norm, alias_norm)
        if score > best_score:
            best_role, best_score = role, score
    if best_score >= FUZZY_THRESHOLD:
        return best_role, round(best_score, 1)
    return None, round(best_score, 1)


def header_score(cells: list[str]) -> float:
    """How header-like a row of cells is (used to skip title rows)."""
    score = 0.0
    for cell in cells:
        if not cell:
            continue
        role, confidence = match_role(cell)
        if role and confidence >= 80:
            score += 1.0
        elif not re.fullmatch(r"[\d.\s%-]+", cell):
            score += 0.1
    return score


def split_subject_component(name: str) -> tuple[str, str, float] | None:
    """Split 'Python - MST1' style headers into (subject, role, confidence)."""
    text = strip_max_hint(name)
    candidates: list[tuple[str, str]] = []
    for sep in (" - ", " – ", " — ", "_", ":", "|", "/", "-", "(", ")", "."):
        if sep not in text:
            continue
        parts = [p.strip() for p in text.split(sep) if p.strip()]
        if len(parts) >= 2:
            candidates.append((sep.join(parts[:-1]), parts[-1]))
            candidates.append((sep.join(parts[1:]), parts[0]))
    tokens = text.split()
    for k in range(1, len(tokens)):
        head, tail = " ".join(tokens[:k]), " ".join(tokens[k:])
        candidates.append((head, tail))
        candidates.append((tail, head))

    best: tuple[str, str, float] | None = None
    for subject, component in candidates:
        role, score = match_role(component)
        if role not in COMPONENT_ROLES or score < SPLIT_THRESHOLD:
            continue
        subject = subject.strip(" -_:|/().")
        subject_role, subject_score = match_role(subject)
        if subject_role is not None and subject_score >= 80:
            continue
        if not re.search(r"[A-Za-z]{2,}", subject) or normalize_name(subject) in GENERIC_WORDS:
            continue
        if best is None or score > best[2] or (score == best[2] and len(subject) > len(best[0])):
            best = (subject, role, score)
    return best


def _non_empty(values: pd.Series) -> pd.Series:
    values = values.astype(str).str.strip()
    return values[(values != "") & (values.str.lower() != "nan")]


def looks_like_result(values: pd.Series) -> bool:
    vals = _non_empty(values).str.lower()
    if vals.empty:
        return False
    return vals.isin(PASS_WORDS | FAIL_WORDS).mean() >= 0.8


def looks_numeric(values: pd.Series) -> bool:
    vals = _non_empty(values)
    if vals.empty:
        return False
    absent = vals.str.lower().isin({"ab", "abs", "absent", "-", "a"})
    numeric = pd.to_numeric(vals.str.rstrip("%"), errors="coerce").notna()
    return (numeric | absent).mean() >= 0.8 and numeric.mean() >= 0.5


def looks_like_section(values: pd.Series) -> bool:
    vals = _non_empty(values)
    if vals.empty or looks_numeric(values):
        return False
    return vals.nunique() <= 12 and vals.str.len().max() <= 12 and vals.str.contains(r"[A-Za-z]").all()


@dataclass
class MappingEntry:
    column: str
    role: str = IGNORE
    subject: str = ""
    confidence: float = 0.0
    max_hint: float | None = None

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "role": self.role,
            "subject": self.subject,
            "confidence": self.confidence,
            "max_hint": self.max_hint,
        }


@dataclass
class MappingGuess:
    entries: list[MappingEntry]
    data_format: str
    notes: list[str] = field(default_factory=list)

    def as_tuple(self) -> tuple:
        return mapping_to_tuple(self.entries)


def mapping_to_tuple(entries: list[MappingEntry]) -> tuple:
    return tuple((e.column, e.role, e.subject, e.max_hint) for e in entries)


def mapping_from_tuple(items: tuple) -> list[MappingEntry]:
    return [MappingEntry(column=c, role=r, subject=s, max_hint=h, confidence=100.0) for c, r, s, h in items]


def _resolve_duplicate_roles(entries: list[MappingEntry], notes: list[str]) -> None:
    seen: dict[tuple[str, str], MappingEntry] = {}
    for entry in sorted(entries, key=lambda e: -e.confidence):
        if entry.role == IGNORE:
            continue
        key = (entry.role, entry.subject)
        if key in seen:
            notes.append(
                f"'{entry.column}' also looked like {entry.role}; kept '{seen[key].column}' and ignored this one."
            )
            entry.role = IGNORE
        else:
            seen[key] = entry


def guess_mapping(df: pd.DataFrame) -> MappingGuess:
    """Best-guess mapping for every column of a raw sheet."""
    notes: list[str] = []
    entries: list[MappingEntry] = []
    splits: dict[str, tuple[str, str, float]] = {}
    for column in df.columns:
        role, score = match_role(column)
        entry = MappingEntry(column=column, role=role or IGNORE, confidence=score, max_hint=extract_max_hint(column))
        if score < 100:
            split = split_subject_component(column)
            if split:
                splits[column] = split
        entries.append(entry)

    has_subject = any(e.role == "subject" and e.confidence >= 80 for e in entries)
    split_subjects = {s[0].lower() for s in splits.values()}
    data_format = "long"
    if not has_subject and len(split_subjects) >= 2:
        data_format = "wide"
        for entry in entries:
            if entry.column in splits:
                subject, role, score = splits[entry.column]
                entry.role, entry.subject, entry.confidence = role, subject, score

    # Value-based heuristics for columns the names did not explain.
    mapped_roles = {e.role for e in entries if not e.subject}
    for entry in entries:
        values = df[entry.column]
        if entry.role == "result" and not looks_like_result(values):
            notes.append(f"'{entry.column}' does not contain Pass/Fail values, so it was ignored.")
            entry.role, entry.confidence = IGNORE, 0.0
        if entry.role != IGNORE:
            continue
        if "result" not in mapped_roles and looks_like_result(values):
            entry.role, entry.confidence = "result", 70.0
            mapped_roles.add("result")
        elif "section" not in mapped_roles and looks_like_section(values) and data_format != "wide":
            entry.role, entry.confidence = "section", 60.0
            mapped_roles.add("section")

    if data_format == "long" and not has_subject:
        component_mapped = any(e.role in COMPONENT_ROLES and e.role != "attendance_pct" for e in entries)
        numeric_unmapped = [e for e in entries if e.role == IGNORE and looks_numeric(df[e.column])]
        if not component_mapped and len(numeric_unmapped) >= 2:
            data_format = "wide-totals"
            for entry in numeric_unmapped:
                entry.role, entry.subject, entry.confidence = "total", entry.column.strip(), 60.0
            notes.append("Each numeric column was treated as one subject's total marks.")

    _resolve_duplicate_roles(entries, notes)
    return MappingGuess(entries=entries, data_format=data_format, notes=notes)


def detect_format(entries: list[MappingEntry]) -> str:
    active = [e for e in entries if e.role != IGNORE]
    if any(e.role == "subject" and not e.subject for e in active):
        return "long"
    subject_level = [e for e in active if e.subject]
    if subject_level:
        if all(e.role == "total" for e in subject_level):
            return "wide-totals"
        return "wide"
    return "long"


def normalize_to_long(df: pd.DataFrame, entries: list[MappingEntry]) -> tuple[pd.DataFrame, str]:
    """Reshape the raw sheet into canonical long format (values stay as text)."""
    entries = [MappingEntry(**{**e.to_dict()}) for e in entries]
    missing = [e.column for e in entries if e.role != IGNORE and e.column not in df.columns]
    if missing:
        raise MappingError(f"Mapped column(s) not found in the sheet: {', '.join(missing)}")
    notes: list[str] = []
    _resolve_duplicate_roles(entries, notes)
    data_format = detect_format(entries)

    student_level = [e for e in entries if e.role != IGNORE and not e.subject]
    subject_level = [e for e in entries if e.role != IGNORE and e.subject]

    base = pd.DataFrame({e.role: df[e.column].astype(str) for e in student_level}, index=df.index)

    if data_format in ("wide", "wide-totals"):
        base = base.drop(columns=["subject"], errors="ignore")
        frames = []
        subjects: dict[str, list[MappingEntry]] = {}
        for entry in subject_level:
            subjects.setdefault(entry.subject.strip(), []).append(entry)
        for subject, group in subjects.items():
            part = base.copy()
            for entry in group:
                part[entry.role] = df[entry.column].astype(str)
            part["subject"] = subject
            frames.append(part)
        long_df = pd.concat(frames, ignore_index=True)
    else:
        long_df = base.reset_index(drop=True)
        if "subject" not in long_df:
            long_df["subject"] = "General"

    if "roll_no" not in long_df and "student_name" not in long_df:
        raise MappingError(
            "Couldn't find a Roll No or Student Name column. Open the Data Quality tab and map one of your columns."
        )
    if not any(role in long_df for role in NUMERIC_ROLES):
        raise MappingError(
            "Couldn't find any marks columns (MST, assignment, end term, total...). "
            "Open the Data Quality tab and map your marks columns."
        )
    return long_df, data_format


def max_hints(entries: list[MappingEntry]) -> dict[str, float]:
    """Max marks read from headers like 'MST-1 (24)', most common per role."""
    by_role: dict[str, list[float]] = {}
    for entry in entries:
        if entry.role in COMPONENT_ROLES and entry.role != "attendance_pct" and entry.max_hint:
            by_role.setdefault(entry.role, []).append(entry.max_hint)
    return {role: max(set(vals), key=vals.count) for role, vals in by_role.items()}
