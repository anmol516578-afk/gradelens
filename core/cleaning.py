"""Turn canonical-but-raw text data into a clean, typed DataFrame.

Every change is recorded in a :class:`DataQualityReport` so teachers can see
what was fixed automatically and what still needs a human decision.

Key rule: absent markers ("AB", "Absent", "-") become an *absent flag* and a
missing value, never a silent zero.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.column_mapping import FAIL_WORDS, PASS_WORDS
from core.config import INTERNAL_COMPONENTS, MARK_COMPONENTS, NICE_MAX_MARKS, ROLE_LABELS, AnalysisConfig

ABSENT_MARKERS = {"ab", "ab.", "abs", "absent", "absnt", "-", "a"}
MISSING_MARKERS = {"", "na", "n/a", "n.a.", "nan", "none", "null", "nil", "--", "nat"}
TEXT_ROLES = ("roll_no", "student_name", "section", "semester", "subject")
MAX_EXAMPLES = 8

FIXED, ATTENTION, INFO = "fixed", "attention", "info"


@dataclass
class Issue:
    category: str
    severity: str
    count: int
    message: str
    column: str | None = None
    examples: list[dict] = field(default_factory=list)


@dataclass
class DataQualityReport:
    rows_loaded: int
    rows_clean: int = 0
    students: int = 0
    subjects: int = 0
    data_format: str = "long"
    issues: list[Issue] = field(default_factory=list)

    def add(self, category: str, severity: str, count: int, message: str, column=None, examples=None) -> None:
        if count > 0:
            self.issues.append(Issue(category, severity, int(count), message, column, (examples or [])[:MAX_EXAMPLES]))

    def by_severity(self, severity: str) -> list[Issue]:
        return [i for i in self.issues if i.severity == severity]

    @property
    def fixed(self) -> list[Issue]:
        return self.by_severity(FIXED)

    @property
    def attention(self) -> list[Issue]:
        return self.by_severity(ATTENTION)

    def to_frame(self) -> pd.DataFrame:
        order = {ATTENTION: 0, FIXED: 1, INFO: 2}
        rows = [
            {
                "Severity": i.severity,
                "Category": i.category,
                "Column": ROLE_LABELS.get(i.column, i.column or ""),
                "Count": i.count,
                "Details": i.message,
            }
            for i in sorted(self.issues, key=lambda i: order.get(i.severity, 3))
        ]
        return pd.DataFrame(rows, columns=["Severity", "Category", "Column", "Count", "Details"])


# --- Small normalizers --------------------------------------------------------


def normalize_section(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in MISSING_MARKERS:
        return ""
    # Drop a "Section"/"Sec." prefix and stray punctuation at the edges, but keep the label's own
    # punctuation: "AI&ML" and "CSE-A" are real section names (DEBUG_LOG #41).
    stripped = re.sub(r"(?i)\b(section|sec)\b\.?", " ", text)
    stripped = re.sub(r"\s+", " ", _strip_edge_punctuation(stripped))
    return (stripped or text).upper()


def _strip_edge_punctuation(text: str) -> str:
    """Trim anything that is not a letter, vowel sign or digit (in any script) from both ends."""

    def keep(ch: str) -> bool:
        return unicodedata.category(ch)[0] in "LMN"

    start, end = 0, len(text)
    while start < end and not keep(text[start]):
        start += 1
    while end > start and not keep(text[end - 1]):
        end -= 1
    return text[start:end]


def normalize_semester(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in MISSING_MARKERS:
        return ""
    digits = re.findall(r"\d+", text)
    if digits:
        return str(int(digits[0]))
    return text.upper()


def normalize_roll(value: object) -> str:
    text = str(value).strip()
    if text.lower() in MISSING_MARKERS:
        return ""
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]
    return text


def collapse_spaces(value: object) -> str:
    text = " ".join(str(value).split())
    return "" if text.lower() in MISSING_MARKERS else text


def _parse_numeric(series: pd.Series, allow_absent: bool) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (values, absent_mask, unparseable_mask)."""
    text = series.astype(str).str.strip()
    low = text.str.lower()
    markers = ABSENT_MARKERS if allow_absent else ABSENT_MARKERS - {"a"}
    absent = low.isin(markers)
    missing = low.isin(MISSING_MARKERS)
    cleaned = text.str.replace("%", "", regex=False).str.strip()
    values = pd.to_numeric(cleaned.where(~(absent | missing)), errors="coerce").astype(float)
    unparseable = values.isna() & ~absent & ~missing
    return values, absent, unparseable


def _example_rows(df: pd.DataFrame, mask: pd.Series, column: str, raw: pd.Series | None = None) -> list[dict]:
    rows = []
    for idx in df.index[mask][:MAX_EXAMPLES]:
        row = {
            "Roll No": df.at[idx, "roll_no"] if "roll_no" in df else "",
            "Name": df.at[idx, "student_name"] if "student_name" in df else "",
            "Subject": df.at[idx, "subject"] if "subject" in df else "",
            "Column": ROLE_LABELS.get(column, column),
        }
        if raw is not None:
            row["Value"] = raw.at[idx]
        rows.append(row)
    return rows


# --- Max-marks inference --------------------------------------------------------


def _nice_ceiling(value: float) -> float:
    for nice in NICE_MAX_MARKS:
        if nice >= value:
            return float(nice)
    return float(math.ceil(value / 10.0) * 10)


def infer_max_marks(long_raw: pd.DataFrame, config: AnalysisConfig, hints: dict[str, float]) -> dict[str, float]:
    """Max marks the data implies when the configured ones are clearly wrong.

    Header hints like 'MST-1 (24)' win. Otherwise, if more than 2% of values
    exceed the configured maximum, the scheme is assumed to differ and the
    99.5th percentile is rounded up to a "nice" maximum. Isolated typos
    (a single 51 out of 15) do not trigger this - they are flagged instead.
    """
    overrides: dict[str, float] = {}
    current = config
    for component in (*MARK_COMPONENTS, "internal_total", "total"):
        if component in hints:
            overrides[component] = float(hints[component])
        elif component in long_raw:
            values, _, _ = _parse_numeric(long_raw[component], allow_absent=True)
            values = values.dropna()
            limit = current.max_marks(component)
            if not values.empty and (values > limit + 1e-9).mean() > 0.02:
                overrides[component] = _nice_ceiling(float(values.quantile(0.995)))
        if component in overrides:
            current = current.with_max_marks({component: overrides[component]})
    return {k: v for k, v in overrides.items() if not math.isclose(v, config.max_marks(k))}


# --- Main entry point ---------------------------------------------------------------


def clean(
    long_raw: pd.DataFrame, config: AnalysisConfig, data_format: str = "long"
) -> tuple[pd.DataFrame, DataQualityReport]:
    report = DataQualityReport(rows_loaded=len(long_raw), data_format=data_format)
    df = long_raw.copy()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": "", "None": ""})

    # 1. Empty rows and exact duplicates.
    empty = (df == "").all(axis=1)
    report.add("Empty rows", FIXED, empty.sum(), f"Removed {int(empty.sum())} completely empty row(s).")
    df = df[~empty]
    dup_mask = df.duplicated(keep="first")
    report.add(
        "Duplicate rows",
        FIXED,
        dup_mask.sum(),
        f"Removed {int(dup_mask.sum())} exact duplicate row(s) (kept the first copy).",
        examples=_example_rows(df, dup_mask, "roll_no"),
    )
    df = df[~dup_mask].reset_index(drop=True)

    # 2. Identity columns.
    if "roll_no" in df:
        df["roll_no"] = df["roll_no"].map(normalize_roll)
    if "student_name" in df:
        df["student_name"] = df["student_name"].map(collapse_spaces)
    if "roll_no" not in df:
        names = df["student_name"].where(df["student_name"] != "", "Unknown")
        codes = {name: f"STU{i + 1:04d}" for i, name in enumerate(pd.unique(names))}
        df["roll_no"] = names.map(codes)
        report.add("Roll numbers", FIXED, len(codes), "No roll number column: generated IDs from student names.")
    no_roll = df["roll_no"] == ""
    if no_roll.any() and "student_name" in df:
        has_name = no_roll & (df["student_name"] != "")
        df.loc[has_name, "roll_no"] = "NOROLL-" + df.loc[has_name, "student_name"].str.replace(" ", "-")
        report.add(
            "Missing roll numbers",
            ATTENTION,
            has_name.sum(),
            "Rows without a roll number were given a temporary ID from the student name.",
            "roll_no",
            _example_rows(df, has_name, "roll_no"),
        )
    unusable = df["roll_no"] == ""
    report.add("Unidentified rows", ATTENTION, unusable.sum(), "Dropped rows with neither roll number nor name.")
    df = df[~unusable].reset_index(drop=True)
    if "student_name" not in df:
        df["student_name"] = df["roll_no"]
    df["student_name"] = df["student_name"].where(df["student_name"] != "", "Unknown (" + df["roll_no"] + ")")

    # Consistent name per roll number.
    name_counts = df.groupby("roll_no")["student_name"].nunique()
    conflicting = name_counts[name_counts > 1].index
    if len(conflicting):
        canonical = df.groupby("roll_no")["student_name"].agg(lambda s: s.value_counts().index[0])
        df["student_name"] = df["roll_no"].map(canonical)
        report.add(
            "Name spelling",
            FIXED,
            len(conflicting),
            "Some roll numbers had different name spellings; used the most common spelling.",
            "student_name",
        )

    # 3. Section / semester / subject labels.
    if "section" in df:
        original = df["section"].copy()
        df["section"] = df["section"].map(normalize_section)
        changed = (original != df["section"]) & (df["section"] != "")
        variants = (
            pd.DataFrame({"From": original[changed], "To": df["section"][changed]}).drop_duplicates().to_dict("records")
        )
        report.add(
            "Section labels",
            FIXED,
            changed.sum(),
            f"Standardized {int(changed.sum())} section label(s), e.g. "
            + ", ".join(f"'{v['From']}' -> {v['To']}" for v in variants[:4]),
            "section",
            variants,
        )
        missing_section = df["section"] == ""
        df.loc[missing_section, "section"] = "Unassigned"
        report.add(
            "Missing section",
            ATTENTION,
            missing_section.sum(),
            "Rows without a section were put in 'Unassigned'.",
            "section",
        )
        sections_per_roll = df.groupby("roll_no")["section"].nunique()
        multi = sections_per_roll[sections_per_roll > 1].index
        if len(multi):
            canonical = df.groupby("roll_no")["section"].agg(lambda s: s.value_counts().index[0])
            df["section"] = df["roll_no"].map(canonical)
            report.add(
                "Section conflicts",
                ATTENTION,
                len(multi),
                "Students listed in more than one section; used the most common.",
                "section",
            )
    else:
        df["section"] = "All"
    if "semester" in df:
        df["semester"] = df["semester"].map(normalize_semester).replace("", "Unknown")
    else:
        df["semester"] = "All"

    df["subject"] = df["subject"].map(collapse_spaces).replace("", "Unknown subject")
    lowered = df["subject"].str.lower()
    canonical_subject = df.groupby(lowered)["subject"].agg(lambda s: s.value_counts().index[0])
    merged = lowered.map(canonical_subject)
    variants_merged = (merged != df["subject"]).sum()
    report.add(
        "Subject names", FIXED, variants_merged, "Merged subject names that differed only in capitalization.", "subject"
    )
    df["subject"] = merged

    # 4. Numeric columns.
    for col in [c for c in ("attendance_pct", *MARK_COMPONENTS, "internal_total", "total") if c in df]:
        raw = df[col].copy()
        is_mark = col in MARK_COMPONENTS
        values, absent, unparseable = _parse_numeric(raw, allow_absent=is_mark or col in ("internal_total", "total"))
        report.add(
            "Unreadable values",
            ATTENTION,
            unparseable.sum(),
            f"{ROLE_LABELS[col]}: {int(unparseable.sum())} value(s) were not numbers and were left blank.",
            col,
            _example_rows(df, unparseable, col, raw),
        )
        if is_mark:
            df[f"{col}_absent"] = absent
            report.add(
                "Absent markers",
                FIXED,
                absent.sum(),
                f"{ROLE_LABELS[col]}: {int(absent.sum())} 'AB'/'Absent'/'-' marker(s) recorded as absent (not zero).",
                col,
            )
            class_wide = class_wide_absence(df.assign(**{f"{col}_absent": absent}), col)
            if class_wide.any():
                subjects = sorted(df.loc[class_wide, "subject"].unique())
                report.add(
                    "Class-wide absence",
                    ATTENTION,
                    class_wide.sum(),
                    f"{ROLE_LABELS[col]}: almost everyone is marked absent in {', '.join(subjects)}. If this test "
                    "has not been held yet, leave the cells blank instead of 'AB' (absences count as 0 marks).",
                    col,
                )
        if col == "attendance_pct":
            valid = values.dropna()
            if len(valid) >= 5 and valid.max() <= 1.0:
                values = values * 100
                report.add(
                    "Attendance scale",
                    FIXED,
                    len(valid),
                    "Attendance looked like fractions (0-1); converted to %.",
                    col,
                )
            upper = 100.0
        else:
            upper = config.max_marks(col)
        out_of_range = (values < 0) | (values > upper + 1e-9)
        report.add(
            "Out-of-range marks",
            ATTENTION,
            out_of_range.sum(),
            f"{ROLE_LABELS[col]}: {int(out_of_range.sum())} value(s) outside 0-{upper:g} were left blank. "
            "Check for typos or adjust the marks scheme.",
            col,
            _example_rows(df, out_of_range, col, raw),
        )
        values = values.mask(out_of_range)
        absent_or_flagged = absent if is_mark else pd.Series(False, index=df.index)
        missing = values.isna() & ~absent_or_flagged & ~out_of_range & ~unparseable
        if col not in ("internal_total", "total"):
            report.add(
                "Blank values",
                ATTENTION,
                missing.sum(),
                f"{ROLE_LABELS[col]}: {int(missing.sum())} blank cell(s). Analyses skip blanks.",
                col,
                _example_rows(df, missing, col),
            )
        df[col] = values

    # 5. Same student + subject listed twice with different values.
    conflict = df.duplicated(["roll_no", "subject"], keep="first")
    report.add(
        "Conflicting duplicates",
        ATTENTION,
        conflict.sum(),
        "Same student and subject appeared more than once with different marks; kept the first row.",
        examples=_example_rows(df, conflict, "subject"),
    )
    df = df[~conflict].reset_index(drop=True)

    # 6. Reported result.
    if "result" in df:
        low = df["result"].str.lower().str.strip()
        df["result_reported"] = np.select([low.isin(PASS_WORDS), low.isin(FAIL_WORDS)], ["Pass", "Fail"], default=None)
        df["result_reported"] = df["result_reported"].astype(object).where(df["result_reported"].notna(), None)
        df = df.drop(columns=["result"])

    # 7. Totals: fill gaps and check consistency.
    _check_totals(df, report)

    report.rows_clean = len(df)
    report.students = int(df["roll_no"].nunique())
    report.subjects = int(df["subject"].nunique())
    return df, report


CLASS_WIDE_SHARE, CLASS_WIDE_MIN_ROWS = 0.8, 5


def class_wide_absence(df: pd.DataFrame, component: str) -> pd.Series:
    """Rows in a subject where at least 80% of the class is absent for this component.

    That pattern usually means the test was not held or marks were not entered, not that
    each student skipped it, so it is reported once instead of as individual warnings.
    """
    flag = f"{component}_absent"
    if flag not in df or "subject" not in df:
        return pd.Series(False, index=df.index)
    grouped = df.groupby("subject")[flag]
    share = grouped.transform("mean")
    size = grouped.transform("size")
    return (share >= CLASS_WIDE_SHARE) & (size >= CLASS_WIDE_MIN_ROWS)


def _component_sum(df: pd.DataFrame, components: tuple[str, ...]) -> tuple[pd.Series, pd.Series]:
    """Sum with absences as 0; also returns rows where every component is known."""
    present = [c for c in components if c in df]
    filled = pd.DataFrame(
        {c: df[c].where(~df.get(f"{c}_absent", pd.Series(False, index=df.index)), 0.0) for c in present}
    )
    complete = filled.notna().all(axis=1) if present else pd.Series(False, index=df.index)
    return filled.sum(axis=1, min_count=1), complete


def _check_totals(df: pd.DataFrame, report: DataQualityReport) -> None:
    targets = (("internal_total", INTERNAL_COMPONENTS), ("total", ("internal_total", "end_term")))
    for total_col, components in targets:
        if total_col not in df or not any(c in df for c in components):
            continue
        if total_col == "total" and not {"internal_total", "end_term"} <= set(df.columns):
            continue
        computed, complete = _component_sum(df, components)
        gap = df[total_col].isna() & complete
        df.loc[gap, total_col] = computed[gap]
        report.add(
            "Totals filled",
            FIXED,
            gap.sum(),
            f"{ROLE_LABELS[total_col]}: filled {int(gap.sum())} blank total(s) from their components.",
            total_col,
        )
        comparable = df[total_col].notna() & complete
        mismatch = comparable & ((df[total_col] - computed).abs() > 0.51)
        if comparable.sum() and mismatch.sum() / comparable.sum() > 0.5:
            report.add(
                "Totals formula",
                INFO,
                mismatch.sum(),
                f"{ROLE_LABELS[total_col]} follows a different formula than the sum of components; "
                "the sheet's own totals were kept.",
                total_col,
            )
        else:
            report.add(
                "Totals mismatch",
                ATTENTION,
                mismatch.sum(),
                f"{ROLE_LABELS[total_col]} does not equal the sum of its components in {int(mismatch.sum())} row(s); "
                "the sheet's values were kept.",
                total_col,
                _example_rows(df, mismatch, total_col),
            )
