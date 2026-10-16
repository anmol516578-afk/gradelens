"""Synthetic, reproducible marks data for demos and tests.

The data is deliberately realistic *and* deliberately messy:

* attendance correlates with marks, section B is weaker, Discrete Maths is
  the hard subject, ~10% of student-subject records fail, and some students
  slide from MST-1 to MST-2;
* the long-format file contains blanks, "AB"/"Absent"/"-" markers,
  inconsistent section labels, an exact duplicate row and an out-of-range
  mark so the cleaning code always has something to do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SUBJECT_DIFFICULTY = {
    "Python": 0.30,
    "DBMS": 0.10,
    "Data Structures": -0.10,
    "Computer Networks": 0.05,
    "Discrete Maths": -0.85,
}
SECTION_EFFECT = {"A": 0.15, "B": -0.45, "C": 0.05}
STUDENTS_PER_SECTION = 60
SEMESTER = 5
MAX = {"assignment": 10, "mst1": 15, "mst2": 15, "end_term": 60}

CO_SUBJECT = "Python"
CO_MAPPING = [
    ("Q1", "CO1", 10),
    ("Q2", "CO1", 10),
    ("Q3", "CO2", 10),
    ("Q4", "CO3", 10),
    ("Q5", "CO3", 10),
    ("Q6", "CO4", 10),
]
CO_DIFFICULTY = {"CO1": 12.0, "CO2": 0.0, "CO3": -16.0, "CO4": -5.0}

FIRST_NAMES = [
    "Aarav",
    "Vivaan",
    "Aditya",
    "Vihaan",
    "Arjun",
    "Sai",
    "Reyansh",
    "Ayaan",
    "Krishna",
    "Ishaan",
    "Shaurya",
    "Atharv",
    "Kabir",
    "Rohan",
    "Karan",
    "Harsh",
    "Yash",
    "Dev",
    "Manav",
    "Pranav",
    "Gurpreet",
    "Harjot",
    "Jaskaran",
    "Simranjeet",
    "Navdeep",
    "Rahul",
    "Aman",
    "Nikhil",
    "Varun",
    "Siddharth",
    "Ananya",
    "Diya",
    "Aadhya",
    "Saanvi",
    "Pari",
    "Anika",
    "Navya",
    "Myra",
    "Sara",
    "Ira",
    "Kavya",
    "Riya",
    "Meera",
    "Isha",
    "Tanvi",
    "Sneha",
    "Pooja",
    "Neha",
    "Priya",
    "Shruti",
    "Harleen",
    "Jasleen",
    "Manpreet",
    "Simran",
    "Gurleen",
    "Fatima",
    "Zoya",
    "Ayesha",
    "Lakshmi",
    "Divya",
]
SURNAMES = [
    "Sharma",
    "Verma",
    "Singh",
    "Kaur",
    "Gupta",
    "Patel",
    "Reddy",
    "Nair",
    "Iyer",
    "Das",
    "Banerjee",
    "Khan",
    "Joshi",
    "Mehta",
    "Kulkarni",
    "Rao",
    "Pillai",
    "Chopra",
    "Malhotra",
    "Bhatia",
    "Sandhu",
    "Gill",
    "Dhillon",
    "Thakur",
    "Rana",
    "Jain",
    "Agarwal",
    "Mishra",
    "Pandey",
    "Yadav",
    "Chauhan",
    "Saini",
    "Bansal",
    "Mittal",
    "Kapoor",
    "Arora",
    "Sethi",
    "Grewal",
    "Bajwa",
    "Menon",
]

LONG_COLUMNS = {
    "roll_no": "Roll No",
    "student_name": "Student Name",
    "section": "Section",
    "semester": "Semester",
    "subject": "Subject",
    "attendance_pct": "Attendance (%)",
    "assignment": "Assignment",
    "mst1": "MST 1",
    "mst2": "MST 2",
    "end_term": "End Term",
    "internal_total": "Internal Total",
    "total": "Total",
    "result": "Result",
}


@dataclass
class SampleBundle:
    clean_long: pd.DataFrame  # ground truth, numeric, absences as NaN
    messy_long: pd.DataFrame  # what a real sheet looks like (all strings)
    messy_wide: pd.DataFrame
    co_marks: pd.DataFrame
    co_mapping: pd.DataFrame


def _unique_names(rng: np.random.Generator, n: int) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < n:
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(SURNAMES)}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _clip_round(values: np.ndarray, max_marks: float) -> np.ndarray:
    return np.clip(np.round(values), 0, max_marks)


def generate_clean(seed: int = 42, students_per_section: int = STUDENTS_PER_SECTION) -> pd.DataFrame:
    """Ground-truth long-format data with numeric marks (absent = NaN + flag)."""
    rng = np.random.default_rng(seed)
    sections = list(SECTION_EFFECT)
    n_students = students_per_section * len(sections)
    names = _unique_names(rng, n_students)
    ability = rng.normal(0, 1, n_students)
    diligence = rng.normal(0, 1, n_students)

    rows = []
    for i in range(n_students):
        section = sections[i // students_per_section]
        roll = f"2023BCA{i + 1:03d}"
        for subject, difficulty in SUBJECT_DIFFICULTY.items():
            attendance = float(np.clip(round(80 + 7 * diligence[i] + 3 * ability[i] + rng.normal(0, 6)), 30, 100))
            skill = (
                0.85 * ability[i]
                + difficulty
                + SECTION_EFFECT[section]
                + 0.25 * (attendance - 80) / 10
                + rng.normal(0, 0.45)
            )
            base_pct = float(np.clip(66 + 15 * skill, 3, 98))
            assignment = base_pct / 100 * 10 + 1.0 + 0.8 * diligence[i] + rng.normal(0, 0.8)
            mst1_pct = float(np.clip(base_pct + rng.normal(0, 9), 0, 100))
            declines = rng.random() < 0.15
            if declines:
                mst2_pct = float(np.clip(mst1_pct - rng.uniform(15, 30), 0, 100))
            else:
                mst2_pct = float(np.clip(mst1_pct + rng.normal(1, 8), 0, 100))
            end_pct = float(np.clip(base_pct + 0.35 * (mst2_pct - mst1_pct) + rng.normal(0, 9), 0, 100))
            rows.append(
                {
                    "roll_no": roll,
                    "student_name": names[i],
                    "section": section,
                    "semester": SEMESTER,
                    "subject": subject,
                    "attendance_pct": attendance,
                    "assignment": float(_clip_round(np.array([assignment]), MAX["assignment"])[0]),
                    "mst1": float(_clip_round(np.array([mst1_pct / 100 * MAX["mst1"]]), MAX["mst1"])[0]),
                    "mst2": float(_clip_round(np.array([mst2_pct / 100 * MAX["mst2"]]), MAX["mst2"])[0]),
                    "end_term": float(_clip_round(np.array([end_pct / 100 * MAX["end_term"]]), MAX["end_term"])[0]),
                }
            )
    df = pd.DataFrame(rows)

    # A handful of genuine absences (absent marks count as 0 towards totals).
    for component, rate in (("mst1", 0.010), ("mst2", 0.006), ("end_term", 0.006)):
        absent = rng.random(len(df)) < rate
        df[f"{component}_absent"] = absent
        df.loc[absent, component] = np.nan

    internal = df[["assignment", "mst1", "mst2"]].fillna(0).sum(axis=1)
    df["internal_total"] = internal
    df["total"] = internal + df["end_term"].fillna(0)
    df["result"] = np.where((df["total"] >= 40) & ~df["end_term_absent"], "Pass", "Fail")
    return df


def _fmt(value: float) -> str:
    if pd.isna(value):
        return ""
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def make_messy_long(clean: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Render clean data the way a real department spreadsheet looks."""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=clean.index)
    for role, header in LONG_COLUMNS.items():
        out[header] = clean[role].map(_fmt) if clean[role].dtype.kind in "fi" else clean[role].astype(str)

    absent_words = {"mst1": "AB", "mst2": "Absent", "end_term": "-"}
    for component, word in absent_words.items():
        out.loc[clean[f"{component}_absent"], LONG_COLUMNS[component]] = word

    n = len(out)
    # Data-entry gaps (not absences).
    for col, rate in (("Attendance (%)", 0.015), ("Assignment", 0.015)):
        blanks = rng.random(n) < rate
        out.loc[blanks, col] = ""

    # Inconsistent section labels.
    variants = {"A": ["a", "Sec A", " A ", "Section-A"], "B": ["b", "Sec. B", "Section B"], "C": ["c", "SEC C"]}
    for section, labels in variants.items():
        idx = out.index[(clean["section"] == section) & (rng.random(n) < 0.08)]
        out.loc[idx, "Section"] = rng.choice(labels, size=len(idx))

    # One out-of-range typo (15 entered as 51) on a row that has an MST-2 mark.
    candidates = out.index[clean["mst2"].notna()]
    out.loc[candidates[17], "MST 2"] = "51"

    # One exact duplicate row (copy-paste accident), inserted mid-sheet.
    dup = out.loc[[out.index[40]]]
    out = pd.concat([out.iloc[:300], dup, out.iloc[300:]], ignore_index=True)
    return out


def make_messy_wide(clean: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """One row per student; '<Subject> - <Component>' columns."""
    rng = np.random.default_rng(seed)
    students = clean.drop_duplicates("roll_no")[["roll_no", "student_name", "section", "semester"]]
    wide = students.rename(
        columns={"roll_no": "Roll No", "student_name": "Name", "section": "Section", "semester": "Semester"}
    ).reset_index(drop=True)
    wide["Semester"] = wide["Semester"].astype(str)
    components = {
        "attendance_pct": "Attendance",
        "assignment": "Assignment",
        "mst1": "MST1",
        "mst2": "MST2",
        "end_term": "End Term",
        "total": "Total",
    }
    for subject in SUBJECT_DIFFICULTY:
        block = clean[clean["subject"] == subject].set_index("roll_no")
        for role, label in components.items():
            values = block[role].reindex(wide["Roll No"]).map(_fmt).to_numpy()
            if role in ("mst1", "mst2", "end_term"):
                absent = block[f"{role}_absent"].reindex(wide["Roll No"]).fillna(False).to_numpy(dtype=bool)
                values = np.where(absent, "AB", values)
            wide[f"{subject} - {label}"] = values
    messy = rng.random(len(wide)) < 0.06
    wide.loc[messy, "Section"] = wide.loc[messy, "Section"].str.lower()
    return wide


def make_co_data(clean: pd.DataFrame, seed: int = 23) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Question-wise end-term marks for one subject plus the question->CO map."""
    rng = np.random.default_rng(seed)
    block = clean[clean["subject"] == CO_SUBJECT].reset_index(drop=True)
    end_pct = block["end_term"] / MAX["end_term"] * 100
    co_marks = block[["roll_no", "student_name", "section"]].rename(
        columns={"roll_no": "Roll No", "student_name": "Student Name", "section": "Section"}
    )
    for question, co, max_marks in CO_MAPPING:
        pct = np.clip(end_pct + CO_DIFFICULTY[co] + rng.normal(0, 12, len(block)), 0, 100)
        marks = np.clip(np.round(pct / 100 * max_marks), 0, max_marks)
        co_marks[question] = [("AB" if np.isnan(m) else str(int(m))) for m in marks]
    mapping = pd.DataFrame(CO_MAPPING, columns=["question", "co", "max_marks"])
    return co_marks, mapping


def generate_all(seed: int = 42, students_per_section: int = STUDENTS_PER_SECTION) -> SampleBundle:
    clean = generate_clean(seed, students_per_section)
    co_marks, co_mapping = make_co_data(clean)
    return SampleBundle(
        clean_long=clean,
        messy_long=make_messy_long(clean),
        messy_wide=make_messy_wide(clean),
        co_marks=co_marks,
        co_mapping=co_mapping,
    )


SAMPLE_FILES = {
    "long": "marks_long.csv",
    "wide": "marks_wide.csv",
    "xlsx": "marks_sample.xlsx",
    "co_marks": "co_question_marks.csv",
    "co_mapping": "co_mapping.csv",
}


def write_sample_files(directory: Path, seed: int = 42) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    bundle = generate_all(seed)
    paths = {key: directory / name for key, name in SAMPLE_FILES.items()}
    bundle.messy_long.to_csv(paths["long"], index=False)
    bundle.messy_wide.to_csv(paths["wide"], index=False)
    bundle.co_marks.to_csv(paths["co_marks"], index=False)
    bundle.co_mapping.to_csv(paths["co_mapping"], index=False)
    readme = pd.DataFrame(
        {
            "About this workbook": [
                "Synthetic marks for a BCA 5th-semester class (3 sections, 5 subjects).",
                "Sheet 'Marks' is long format (one row per student per subject).",
                "Sheet 'Wide' is wide format (one row per student, columns per subject).",
                "All names are randomly generated; no real students are included.",
            ]
        }
    )
    with pd.ExcelWriter(paths["xlsx"], engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="Read Me", index=False)
        bundle.messy_long.to_excel(writer, sheet_name="Marks", index=False)
        bundle.messy_wide.to_excel(writer, sheet_name="Wide", index=False)
    return paths
