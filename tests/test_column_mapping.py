import pandas as pd
import pytest

from core.column_mapping import (
    MappingEntry,
    MappingError,
    detect_format,
    extract_max_hint,
    guess_mapping,
    mapping_from_tuple,
    mapping_to_tuple,
    match_role,
    max_hints,
    normalize_name,
    normalize_to_long,
    split_subject_component,
)
from core.config import IGNORE


@pytest.mark.parametrize(
    "header, role",
    [
        ("Roll No.", "roll_no"),
        ("Enroll. No.", "roll_no"),
        ("Enrollment Number", "roll_no"),
        ("Name of the Student", "student_name"),
        ("Sec", "section"),
        ("MST-I", "mst1"),
        ("MST-II", "mst2"),
        ("Mid Sem 1 (24)", "mst1"),
        ("Mid Sem-2 (24)", "mst2"),
        ("Sessional-1", "mst1"),
        ("End Sem Marks", "end_term"),
        ("ETE (60)", "end_term"),
        ("Total Marks (100)", "total"),
        ("Grand Total", "total"),
        ("Attendance %", "attendance_pct"),
        ("Att.", "attendance_pct"),
        ("Internal Assessment", "internal_total"),
        ("Total Internal Marks", "internal_total"),
        ("Result", "result"),
        ("Course Code", "subject"),
    ],
)
def test_messy_headers_map_to_roles(header, role):
    assert match_role(header)[0] == role


def test_digits_never_cross_between_tests():
    assert match_role("MST 1")[0] == "mst1"
    assert match_role("MST 2")[0] == "mst2"
    assert match_role("Mid Term 2")[0] == "mst2"
    assert match_role("Test 3")[0] is None  # no alias has a 3


def test_unrelated_header_is_unmapped():
    role, score = match_role("Hostel Block")
    assert role is None and score < 75


def test_normalize_name_and_hints():
    assert normalize_name("MST-II (24)") == "mst 2"
    assert normalize_name("Attendance (%)") == "attendance pct"
    assert extract_max_hint("MST-1 (24)") == 24
    assert extract_max_hint("End Term / 60") == 60
    assert extract_max_hint("MST (2)") is None  # '2' is the test number, not max marks
    assert extract_max_hint("Python") is None


@pytest.mark.parametrize(
    "header, expected",
    [
        ("Python - MST1", ("Python", "mst1")),
        ("Data Structures - End Term", ("Data Structures", "end_term")),
        ("MST1_DBMS", ("DBMS", "mst1")),
        ("Python Attendance", ("Python", "attendance_pct")),
        ("Computer Networks: Total", ("Computer Networks", "total")),
    ],
)
def test_wide_headers_split(header, expected):
    split = split_subject_component(header)
    assert split is not None and split[:2] == expected


@pytest.mark.parametrize("header", ["Internal Total", "Student Name", "Attendance (%)", "Total Marks", "End Term"])
def test_long_headers_do_not_split(header):
    assert match_role(header)[1] == 100  # exact match wins; guess_mapping never splits these


def test_guess_long_sample(bundle):
    guess = guess_mapping(bundle.messy_long)
    assert guess.data_format == "long"
    roles = {e.column: e.role for e in guess.entries}
    assert roles["MST 1"] == "mst1" and roles["Attendance (%)"] == "attendance_pct"
    assert all(e.role != IGNORE for e in guess.entries)


def test_guess_wide_sample(bundle):
    guess = guess_mapping(bundle.messy_wide)
    assert guess.data_format == "wide"
    subjects = {e.subject for e in guess.entries if e.subject}
    assert subjects == {"Python", "DBMS", "Data Structures", "Computer Networks", "Discrete Maths"}
    long_df, fmt = normalize_to_long(bundle.messy_wide, guess.entries)
    assert fmt == "wide" and len(long_df) == 900
    assert {"roll_no", "subject", "mst1", "end_term"} <= set(long_df.columns)


def test_wide_totals_layout():
    raw = pd.DataFrame({"Roll No": ["1", "2"], "Name": ["A B", "C D"], "Python": ["45", "AB"], "DBMS": ["60", "70"]})
    guess = guess_mapping(raw)
    assert guess.data_format == "wide-totals"
    long_df, fmt = normalize_to_long(raw, guess.entries)
    assert fmt == "wide-totals"
    assert sorted(long_df["subject"].unique()) == ["DBMS", "Python"]
    assert list(long_df.loc[long_df["subject"] == "Python", "total"]) == ["45", "AB"]


def test_value_heuristics_for_result_and_section():
    raw = pd.DataFrame(
        {
            "Roll": ["1", "2", "3"],
            "Cls": ["A", "B", "A"],
            "Outcome?": ["Pass", "Fail", "pass"],
            "Total": ["50", "30", "70"],
            "Remarks": ["good", "needs work", "ok"],
        }
    )
    guess = guess_mapping(raw)
    roles = {e.column: e.role for e in guess.entries}
    assert roles["Cls"] == "section"
    assert roles["Outcome?"] == "result"
    assert roles["Remarks"] == IGNORE  # free text, not pass/fail


def test_duplicate_roles_keep_best_match():
    raw = pd.DataFrame({"Roll No": ["1"], "Total": ["50"], "Total Marks": ["50"], "Subject": ["X"]})
    guess = guess_mapping(raw)
    totals = [e for e in guess.entries if e.role == "total"]
    assert len(totals) == 1 and totals[0].column == "Total"
    assert any("Total Marks" in note for note in guess.notes)


def test_single_subject_sheet_gets_general_subject():
    raw = pd.DataFrame({"Roll No": ["1", "2"], "MST 1": ["10", "12"], "MST 2": ["11", "9"]})
    long_df, fmt = normalize_to_long(raw, guess_mapping(raw).entries)
    assert fmt == "long" and set(long_df["subject"]) == {"General"}


def test_missing_identity_or_marks_raise_friendly_errors():
    with pytest.raises(MappingError, match="Roll No or Student Name"):
        normalize_to_long(pd.DataFrame({"MST 1": ["1"]}), [MappingEntry("MST 1", "mst1")])
    with pytest.raises(MappingError, match="marks columns"):
        normalize_to_long(pd.DataFrame({"Roll No": ["1"]}), [MappingEntry("Roll No", "roll_no")])
    with pytest.raises(MappingError, match="not found"):
        normalize_to_long(pd.DataFrame({"Roll No": ["1"]}), [MappingEntry("Gone", "mst1")])


def test_mapping_tuple_roundtrip_and_hints():
    entries = [MappingEntry("MST-1 (24)", "mst1", max_hint=24.0), MappingEntry("Roll", "roll_no")]
    restored = mapping_from_tuple(mapping_to_tuple(entries))
    assert [(e.column, e.role, e.max_hint) for e in restored] == [
        ("MST-1 (24)", "mst1", 24.0),
        ("Roll", "roll_no", None),
    ]
    assert max_hints(entries) == {"mst1": 24.0}
    assert detect_format(entries) == "long"


def _sheet(columns: dict[str, list]) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_wide_sheet_with_subject_names_in_indian_scripts():
    """ "गणित - MST 1" normalised to "mst 1" alone and was taken as a plain MST-1 column."""
    marks = [10, 12, 8]
    columns = {"Roll No": ["R1", "R2", "R3"], "Name": ["A1", "B1", "C1"]}
    for subject in ("गणित", "ਪੰਜਾਬੀ", "Python"):
        columns[f"{subject} - MST 1"] = marks
        columns[f"{subject} - End Term"] = [40, 50, 30]
    guess = guess_mapping(_sheet(columns))
    subjects = {(e.subject, e.role) for e in guess.entries if e.subject}
    assert ("गणित", "mst1") in subjects and ("ਪੰਜਾਬੀ", "end_term") in subjects and ("Python", "mst1") in subjects
    assert len(subjects) == 6


def test_bilingual_headers_stay_one_subject():
    """Headers like "MST 1 / मध्य सत्र 1" split into a 'subject' too, but no exam repeats: not wide."""
    columns = {
        "Roll No / अनुक्रमांक": ["R1", "R2", "R3"],
        "Name / नाम": ["A1", "B1", "C1"],
        "MST 1 / मध्य सत्र 1": [10, 12, 8],
        "MST 2 / मध्य सत्र 2": [11, 9, 13],
        "End Term / अंतिम परीक्षा": [40, 50, 30],
    }
    roles = {e.column: (e.role, e.subject) for e in guess_mapping(_sheet(columns)).entries}
    assert roles["MST 1 / मध्य सत्र 1"] == ("mst1", "")
    assert roles["MST 2 / मध्य सत्र 2"] == ("mst2", "")
    assert roles["End Term / अंतिम परीक्षा"] == ("end_term", "")
    assert roles["Roll No / अनुक्रमांक"] == ("roll_no", "")


def test_section_values_in_indian_scripts_are_recognised():
    """pandas 3 runs .str regexes with RE2, where [^\\W\\d_] is ASCII-only; letters are counted in Python."""
    columns = {
        "Roll No": [f"R{i}" for i in range(6)],
        "Python MST1": [10, 12, 8, 9, 11, 7],
        "Group": ["क", "ख", "क", "ख", "क", "ख"],
    }
    roles = {e.column: e.role for e in guess_mapping(_sheet(columns)).entries}
    assert roles["Group"] == "section"
