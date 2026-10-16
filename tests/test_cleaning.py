import numpy as np
import pandas as pd
import pytest

from core.cleaning import ATTENTION, FIXED, INFO, clean, infer_max_marks, normalize_section, normalize_semester
from core.config import AnalysisConfig
from core.derive import add_derived, assessment_stage


def raw(rows):
    return pd.DataFrame(rows).astype(str)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("a", "A"),
        ("Sec A", "A"),
        (" A ", "A"),
        ("Section-A", "A"),
        ("Sec. B", "B"),
        ("SEC C", "C"),
        ("B1", "B1"),
        ("", ""),
    ],
)
def test_section_normalization(value, expected):
    assert normalize_section(value) == expected


@pytest.mark.parametrize("value, expected", [("5", "5"), ("5.0", "5"), ("Sem 5", "5"), ("5th", "5"), ("V", "V")])
def test_semester_normalization(value, expected):
    assert normalize_semester(value) == expected


def test_absent_is_flagged_not_zero(config):
    df, report = clean(
        raw(
            [
                {"roll_no": "1", "subject": "Py", "mst1": "AB", "mst2": "10"},
                {"roll_no": "2", "subject": "Py", "mst1": "Absent", "mst2": "-"},
                {"roll_no": "3", "subject": "Py", "mst1": "12", "mst2": ""},
            ]
        ),
        config,
    )
    assert df["mst1_absent"].tolist() == [True, True, False]
    assert df["mst2_absent"].tolist() == [False, True, False]
    assert np.isnan(df.loc[0, "mst1"]) and df.loc[2, "mst1"] == 12
    assert np.isnan(df.loc[2, "mst2"])  # blank is missing, not absent
    categories = {(i.category, i.column): i for i in report.issues}
    assert categories[("Absent markers", "mst1")].count == 2
    assert categories[("Blank values", "mst2")].severity == ATTENTION


def test_exact_and_conflicting_duplicates(config):
    rows = [
        {"roll_no": "1", "subject": "Py", "total": "50"},
        {"roll_no": "1", "subject": "Py", "total": "50"},
        {"roll_no": "1", "subject": "Py", "total": "70"},
        {"roll_no": "2", "subject": "Py", "total": "40"},
    ]
    df, report = clean(raw(rows), config)
    assert len(df) == 2
    assert df.loc[df["roll_no"] == "1", "total"].item() == 50
    by_cat = {i.category: i for i in report.issues}
    assert by_cat["Duplicate rows"].count == 1 and by_cat["Duplicate rows"].severity == FIXED
    assert by_cat["Conflicting duplicates"].count == 1 and by_cat["Conflicting duplicates"].severity == ATTENTION


def test_out_of_range_marks_are_blanked_and_flagged(config):
    df, report = clean(
        raw([{"roll_no": "1", "subject": "Py", "mst2": "51"}, {"roll_no": "2", "subject": "Py", "mst2": "-3"}]), config
    )
    assert df["mst2"].isna().all()
    issue = next(i for i in report.issues if i.category == "Out-of-range marks")
    assert issue.count == 2 and issue.examples[0]["Value"] == "51"


def test_attendance_fraction_and_percent_sign(config):
    rows = [
        {"roll_no": str(i), "subject": "Py", "attendance_pct": v}
        for i, v in enumerate(["0.8", "0.75", "0.9", "1", "0.6"])
    ]
    df, report = clean(raw(rows), config)
    assert df["attendance_pct"].tolist() == pytest.approx([80, 75, 90, 100, 60])
    assert any(i.category == "Attendance scale" for i in report.issues)
    df2, _ = clean(raw([{"roll_no": "1", "subject": "Py", "attendance_pct": "82%"}]), config)
    assert df2.loc[0, "attendance_pct"] == 82


def test_unreadable_values_reported(config):
    df, report = clean(raw([{"roll_no": "1", "subject": "Py", "mst1": "twelve"}]), config)
    assert np.isnan(df.loc[0, "mst1"])
    assert any(i.category == "Unreadable values" and i.count == 1 for i in report.issues)


def test_totals_filled_and_mismatch_flagged(config):
    rows = [
        {"roll_no": "1", "subject": "Py", "assignment": "8", "mst1": "10", "mst2": "12", "internal_total": ""},
        {"roll_no": "2", "subject": "Py", "assignment": "8", "mst1": "10", "mst2": "12", "internal_total": "35"},
        {"roll_no": "3", "subject": "Py", "assignment": "5", "mst1": "5", "mst2": "5", "internal_total": "15"},
        {"roll_no": "4", "subject": "Py", "assignment": "5", "mst1": "AB", "mst2": "5", "internal_total": "10"},
    ]
    df, report = clean(raw(rows), config)
    assert df.loc[0, "internal_total"] == 30
    by_cat = {i.category: i for i in report.issues}
    assert by_cat["Totals filled"].count == 1
    assert by_cat["Totals mismatch"].count == 1  # only row 2; absent counts as zero for row 4


def test_totals_with_different_formula_become_info(config):
    rows = [
        {"roll_no": str(i), "subject": "Py", "assignment": "5", "mst1": "5", "mst2": "5", "internal_total": "20"}
        for i in range(4)
    ]
    _, report = clean(raw(rows), config)
    assert any(i.category == "Totals formula" and i.severity == INFO for i in report.issues)


def test_missing_ids_names_and_sections(config):
    rows = [
        {"roll_no": "", "student_name": "Asha Rao", "subject": "Py", "total": "50"},
        {"roll_no": "", "student_name": "", "subject": "Py", "total": "60"},
        {"roll_no": "7", "student_name": "", "subject": "Py", "total": "70", "section": ""},
    ]
    rows[0]["section"] = "A"
    rows[1]["section"] = "A"
    df, report = clean(raw(rows), config)
    assert len(df) == 2
    assert df.loc[0, "roll_no"] == "NOROLL-Asha-Rao"
    assert df.loc[1, "student_name"] == "Unknown (7)" and df.loc[1, "section"] == "Unassigned"
    cats = {i.category for i in report.attention}
    assert {"Missing roll numbers", "Unidentified rows", "Missing section"} <= cats


def test_name_and_section_consistency_per_roll(config):
    rows = [
        {"roll_no": "1", "student_name": "Asha Rao", "section": "A", "subject": "Py", "total": "50"},
        {"roll_no": "1", "student_name": "Asha  Rao", "section": "A", "subject": "DB", "total": "50"},
        {"roll_no": "1", "student_name": "Asha Raoo", "section": "B", "subject": "CN", "total": "50"},
    ]
    df, report = clean(raw(rows), config)
    assert set(df["student_name"]) == {"Asha Rao"}
    assert set(df["section"]) == {"A"}
    assert {"Name spelling", "Section conflicts"} <= {i.category for i in report.issues}


def test_names_only_sheet_gets_generated_ids(config):
    df, report = clean(raw([{"student_name": "Asha Rao", "subject": "Py", "total": "50"}]), config)
    assert df.loc[0, "roll_no"] == "STU0001"
    assert any(i.category == "Roll numbers" for i in report.fixed)


def test_subject_case_variants_merge(config):
    rows = [
        {"roll_no": "1", "subject": "Python", "total": "50"},
        {"roll_no": "2", "subject": "Python", "total": "50"},
        {"roll_no": "3", "subject": "python", "total": "50"},
    ]
    df, _ = clean(raw(rows), config)
    assert set(df["subject"]) == {"Python"}


def test_reported_result_normalized(config):
    rows = [
        {"roll_no": "1", "subject": "Py", "total": "50", "result": "PASS"},
        {"roll_no": "2", "subject": "Py", "total": "20", "result": "Re-appear"},
        {"roll_no": "3", "subject": "Py", "total": "20", "result": "??"},
    ]
    df, _ = clean(raw(rows), config)
    assert df["result_reported"].tolist() == ["Pass", "Fail", None]


def test_infer_max_marks_from_data_and_hints(config):
    rng = np.random.default_rng(0)
    long_raw = pd.DataFrame({"mst1": rng.integers(5, 29, 200).astype(str), "mst2": ["10"] * 199 + ["51"]})
    overrides = infer_max_marks(long_raw, config, hints={})
    assert overrides == {"mst1": 30.0}  # MST out of 30; a single typo in MST-2 does not change its max
    assert infer_max_marks(long_raw, config, hints={"mst1": 24.0})["mst1"] == 24.0


def test_full_sample_matches_ground_truth(bundle, prepared_long):
    prepared, tuned = prepared_long
    derived = add_derived(prepared.clean, tuned)
    truth = bundle.clean_long.set_index(["roll_no", "subject"])
    merged = derived.set_index(["roll_no", "subject"]).join(truth[["result", "total"]], rsuffix="_truth")
    assert len(merged) == 900
    assert (merged["result"] == merged["result_truth"]).all()
    assert ((merged["total"] - merged["total_truth"]).abs() <= 0.5).all()
    assert assessment_stage(derived) == "final"
    report = prepared.report
    assert report.rows_loaded == 901 and report.rows_clean == 900 and report.students == 180
    cats = {i.category for i in report.issues}
    assert {"Duplicate rows", "Section labels", "Absent markers", "Out-of-range marks", "Blank values"} <= cats


def test_derived_columns_and_pass_mark(config):
    df, _ = clean(
        raw(
            [
                {"roll_no": "1", "subject": "Py", "assignment": "10", "mst1": "15", "mst2": "15", "end_term": "60"},
                {"roll_no": "2", "subject": "Py", "assignment": "4", "mst1": "6", "mst2": "6", "end_term": "-"},
                {"roll_no": "3", "subject": "Py", "assignment": "4", "mst1": "6", "mst2": "6", "end_term": "30"},
            ]
        ),
        config,
    )
    derived = add_derived(df, config)
    assert derived["internal_total"].tolist() == [40, 16, 16]
    assert derived["total"].tolist() == [100, 16, 46]
    assert derived["score_pct"].tolist() == [100, 16, 46]
    assert derived["result"].tolist() == ["Pass", "Fail", "Pass"]
    strict = add_derived(df, AnalysisConfig(pass_pct=50))
    assert strict["result"].tolist() == ["Pass", "Fail", "Fail"]


def test_mid_semester_data_uses_internals(config):
    df, _ = clean(raw([{"roll_no": "1", "subject": "Py", "assignment": "8", "mst1": "12", "mst2": "12"}]), config)
    derived = add_derived(df, config)
    assert derived.loc[0, "score_pct"] == 80.0
    assert assessment_stage(derived) == "mid-semester"
    empty = add_derived(clean(raw([{"roll_no": "1", "subject": "Py", "attendance_pct": "80"}]), config)[0], config)
    assert empty.loc[0, "result"] == "Incomplete"
