"""CO attainment checked against a hand-worked example.

Questions: Q1 (CO1, max 10), Q2 (CO1, max 10), Q3 (CO2, max 5). Threshold 60%.

    S1: 8, 7, 4   -> CO1 15/20 = 75% (attained)  CO2 4/5 = 80% (attained)
    S2: 5, 6, 2   -> CO1 11/20 = 55%             CO2 2/5 = 40%
    S3: 9, -, 3   -> CO1  9/20 = 45% (blank = 0) CO2 3/5 = 60% (attained, boundary)
    S4: 6, 6, 5   -> CO1 12/20 = 60% (attained)  CO2 5/5 = 100% (attained)
    S5: AB AB AB  -> absent for the whole exam: excluded

CO1: 2 of 4 students = 50% -> level 1.   CO2: 3 of 4 = 75% -> level 3.
"""

import pandas as pd
import pytest

from core.co_attainment import (
    COError,
    COSettings,
    compute_co_attainment,
    default_mapping,
    detect_question_columns,
    overall_attainment,
    parse_question_marks,
)


@pytest.fixture
def raw():
    return pd.DataFrame(
        {
            "Roll No": ["S1", "S2", "S3", "S4", "S5"],
            "Student Name": ["Asha", "Ravi", "Meena", "Karan", "Dev"],
            "Q1 (10)": ["8", "5", "9", "6", "AB"],
            "Q2 (10)": ["7", "6", "", "6", "AB"],
            "Q3 (5)": ["4", "2", "3", "5", "AB"],
        }
    )


@pytest.fixture
def mapping():
    return pd.DataFrame({"question": ["Q1", "Q2", "Q3"], "co": ["CO1", "CO1", "CO2"], "max_marks": [10, 10, 5]})


def test_hand_worked_example(raw, mapping):
    marks = parse_question_marks(raw)
    co_table, students, warnings = compute_co_attainment(marks, mapping, COSettings())
    table = co_table.set_index("co")
    assert table.loc["CO1", "students_assessed"] == 4
    assert table.loc["CO1", "students_attained"] == 2
    assert table.loc["CO1", "pct_attained"] == 50.0
    assert table.loc["CO1", "level"] == 1
    assert table.loc["CO2", "students_attained"] == 3
    assert table.loc["CO2", "pct_attained"] == 75.0
    assert table.loc["CO2", "level"] == 3
    assert table.loc["CO1", "max_marks"] == 20 and table.loc["CO1", "questions"] == "Q1, Q2"
    assert table["meets_target"].tolist() == [False, True]
    assert students["roll_no"].tolist() == ["S1", "S2", "S3", "S4"]
    assert students.set_index("roll_no").loc["S3", "CO1"] == pytest.approx(45.0)
    assert any("excluded" in w for w in warnings)
    assert overall_attainment(co_table) == 2.0


def test_thresholds_are_configurable(raw, mapping):
    marks = parse_question_marks(raw)
    strict = COSettings(student_threshold=75, level3=90, level2=80, level1=40, target_level=3)
    table = compute_co_attainment(marks, mapping, strict)[0].set_index("co")
    assert table.loc["CO1", "students_attained"] == 1  # only S1 reaches 75%
    assert table.loc["CO1", "level"] == 0  # 25% < 40%
    assert table.loc["CO2", "students_attained"] == 2  # S1 80%, S4 100%
    assert table.loc["CO2", "level"] == 1


@pytest.mark.parametrize("pct, level", [(70, 3), (69.99, 2), (60, 2), (50, 1), (49.9, 0), (float("nan"), 0)])
def test_level_boundaries(pct, level):
    assert COSettings().level_for(pct) == level


def test_question_detection_and_default_mapping(raw):
    assert detect_question_columns(["Roll", "Q1", "q 2", "Question 3", "Q4a", "Quiz", "Total"]) == [
        "Q1",
        "q 2",
        "Question 3",
        "Q4a",
    ]
    marks = parse_question_marks(raw)
    default = default_mapping(raw, marks)
    assert default["question"].tolist() == ["Q1", "Q2", "Q3"]
    assert default["max_marks"].tolist() == [10, 10, 5]  # read from the '(10)' / '(5)' hints
    assert set(default["co"]) == {"CO1"}


def test_out_of_range_question_marks_are_ignored(mapping):
    raw = pd.DataFrame({"Roll No": ["S1", "S2"], "Q1": ["15", "8"], "Q2": ["5", "8"], "Q3": ["4", "4"]})
    co_table, _, warnings = compute_co_attainment(parse_question_marks(raw), mapping)
    assert any("outside" in w for w in warnings)
    # S1's Q1 (15 > 10) is ignored -> CO1 = 5/20 = 25%; S2 = 16/20 = 80%
    assert co_table.set_index("co").loc["CO1", "students_attained"] == 1


def test_invalid_inputs_raise(raw, mapping):
    marks = parse_question_marks(raw)
    with pytest.raises(COError, match="not in the marks sheet"):
        compute_co_attainment(
            marks, pd.concat([mapping, pd.DataFrame([{"question": "Q9", "co": "CO1", "max_marks": 5}])])
        )
    bad = mapping.copy()
    bad.loc[0, "max_marks"] = 0
    with pytest.raises(COError, match="positive"):
        compute_co_attainment(marks, bad)
    with pytest.raises(COError, match="empty"):
        compute_co_attainment(marks, mapping.iloc[0:0])
    with pytest.raises(COError, match="No question columns"):
        parse_question_marks(pd.DataFrame({"Roll No": ["1"], "Total": ["5"]}))
    no_co = mapping.copy()
    no_co.loc[1, "co"] = ""
    with pytest.raises(COError, match="needs a CO"):
        compute_co_attainment(marks, no_co)
    no_co.loc[1, "co"] = None
    with pytest.raises(COError, match="needs a CO"):
        compute_co_attainment(marks, no_co)
    dup = pd.concat([mapping, mapping.iloc[[0]]])
    with pytest.raises(COError, match="only one CO"):
        compute_co_attainment(marks, dup)


def test_sample_co_data(bundle):
    marks = parse_question_marks(bundle.co_marks)
    co_table, students, _ = compute_co_attainment(marks, bundle.co_mapping)
    levels = co_table.set_index("co")["level"]
    assert levels["CO1"] > levels["CO3"]  # CO3 was generated as the hard outcome
    assert len(students) >= 175
