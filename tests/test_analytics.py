"""Analytics checked against a small fixture whose numbers were worked out by hand.

Fixture (score %, result, attendance %, MST-1 %, MST-2 %):

    R1 Asha  A  Python 80 Pass 90 60 80 | DBMS 30 Fail 50 50 30
    R2 Ravi  A  Python 50 Pass 70 70 72 | DBMS 45 Pass 80 40 40
    R3 Meena B  Python 35 Fail 60 30 20 | DBMS 90 Pass 95 90 100
    R4 Karan B  Python -- Incomplete 85 (absent MST-1) | DBMS 60 Pass 100 -- --
"""

import math

import numpy as np
import pandas as pd
import pytest

from core import analytics as an
from tests.conftest import make_frame

NAN = float("nan")


@pytest.fixture
def frame():
    rows = [
        ("R1", "Asha", "A", "Python", 80, "Pass", 90, 60, 80),
        ("R1", "Asha", "A", "DBMS", 30, "Fail", 50, 50, 30),
        ("R2", "Ravi", "A", "Python", 50, "Pass", 70, 70, 72),
        ("R2", "Ravi", "A", "DBMS", 45, "Pass", 80, 40, 40),
        ("R3", "Meena", "B", "Python", 35, "Fail", 60, 30, 20),
        ("R3", "Meena", "B", "DBMS", 90, "Pass", 95, 90, 100),
        ("R4", "Karan", "B", "Python", NAN, "Incomplete", 85, NAN, NAN),
        ("R4", "Karan", "B", "DBMS", 60, "Pass", 100, NAN, NAN),
    ]
    df = make_frame(
        [
            dict(
                roll_no=r, student_name=n, section=s, semester="5", subject=sub, score_pct=sc, result=res,
                attendance_pct=att, mst1_pct=m1, mst2_pct=m2, mst1=0.0, assignment=0.0, mst2=0.0, end_term=0.0,
            )
            for r, n, s, sub, sc, res, att, m1, m2 in rows
        ]
    )  # fmt: skip
    df.loc[6, "mst1_absent"] = True
    return df


def test_kpis(frame):
    k = an.kpis(frame, at_risk_count=3)
    assert k["students"] == 4 and k["records"] == 8 and k["subjects"] == 2
    assert k["fails"] == 2
    assert k["fail_rate"] == pytest.approx(200 / 7)
    assert k["pass_rate"] == pytest.approx(500 / 7)
    assert k["avg_score"] == pytest.approx(390 / 7)
    assert k["avg_attendance"] == pytest.approx(78.75)
    assert k["absentees"] == 1 and k["at_risk"] == 3


def test_group_summary_by_section(frame):
    out = an.group_summary(frame, ["section"]).set_index("section")
    assert out.loc["A", "students"] == 2 and out.loc["A", "records"] == 4
    assert out.loc["A", "avg_score"] == pytest.approx(51.25)
    assert out.loc["A", "fail_rate"] == pytest.approx(25.0)
    assert out.loc["A", "pass_rate"] == pytest.approx(75.0)
    assert out.loc["A", "avg_attendance"] == pytest.approx(72.5)
    assert out.loc["B", "avg_score"] == pytest.approx(185 / 3)
    assert out.loc["B", "fail_rate"] == pytest.approx(100 / 3)  # Incomplete row excluded
    assert out.loc["B", "avg_attendance"] == pytest.approx(85.0)


def test_fail_rate_matrix(frame):
    matrix = an.fail_rate_matrix(frame)
    assert matrix.loc["A", "Python"] == 0 and matrix.loc["A", "DBMS"] == 50
    assert matrix.loc["B", "Python"] == 100 and matrix.loc["B", "DBMS"] == 0


@pytest.mark.parametrize(
    "score, pass_pct, band",
    [(95, 40, "O"), (80, 40, "A+"), (70, 40, "A"), (60, 40, "B+"), (50, 40, "B"), (45, 40, "C"), (39.9, 40, "F"),
     (45, 50, "F"), (55, 50, "B")],
)  # fmt: skip
def test_grade_band(score, pass_pct, band):
    assert an.grade_band(score, pass_pct) == band


def test_grade_distribution(frame):
    dist = an.grade_distribution(frame, 40).set_index("grade")["count"].to_dict()
    assert dist == {"O": 1, "A+": 1, "A": 0, "B+": 1, "B": 1, "C": 1, "F": 2}


def test_mst_change(frame):
    changes = an.mst_change(frame, tolerance=5)
    assert len(changes) == 6  # Karan has no MST percentages
    trend = changes.set_index(["roll_no", "subject"])["trend"]
    assert trend[("R1", "Python")] == "Improved"
    assert trend[("R1", "DBMS")] == "Declined"
    assert trend[("R2", "Python")] == "Stable"  # +2 within tolerance
    assert trend[("R3", "Python")] == "Declined"
    summary = an.mst_trend_summary(frame, 5, by=None).iloc[0].to_dict()
    assert summary == {"Improved": 2, "Stable": 2, "Declined": 2}
    by_subject = an.mst_trend_summary(frame, 5).set_index("subject")
    assert by_subject.loc["DBMS"].to_dict() == {"Improved": 1, "Stable": 1, "Declined": 1}


def test_attendance_performance(frame):
    result = an.attendance_performance(frame, threshold=75)
    paired = frame.dropna(subset=["score_pct"])
    assert result["n"] == 7
    assert result["r"] == pytest.approx(np.corrcoef(paired["attendance_pct"], paired["score_pct"])[0, 1])
    assert result["below"]["records"] == 3
    assert result["below"]["avg_score"] == pytest.approx(115 / 3)
    assert result["below"]["fail_rate"] == pytest.approx(200 / 3)
    assert result["above"]["avg_score"] == pytest.approx(68.75)
    assert result["above"]["fail_rate"] == 0
    bands = result["bands"].set_index("band")
    assert bands.loc["<60%", "records"] == 1  # 50% attendance
    assert bands.loc["85%+", "records"] == 3


def test_student_summary_and_top_bottom(frame):
    summary = an.student_summary(frame).set_index("roll_no")
    assert summary.loc["R1", "avg_score"] == 55 and summary.loc["R1", "min_score"] == 30
    assert summary.loc["R1", "fails"] == 1 and summary.loc["R1", "weakest_subject"] == "DBMS"
    assert summary.loc["R4", "avg_score"] == 60 and summary.loc["R4", "subjects"] == 2
    top, bottom = an.top_bottom(frame, n=2)
    assert top["roll_no"].tolist() == ["R3", "R4"]
    assert bottom["roll_no"].tolist() == ["R2", "R1"]


def test_subject_difficulty(frame):
    diff = an.subject_difficulty(frame).set_index("subject")
    assert diff.loc["Python", "difficulty_index"] == pytest.approx(39.2)
    assert diff.loc["DBMS", "difficulty_index"] == pytest.approx(34.4)
    assert diff.loc["Python", "rank"] == 1


def test_student_profile(frame):
    profile = an.student_profile(frame, "R1")
    subjects = profile["subjects"].set_index("subject")
    assert profile["student_name"] == "Asha" and profile["section"] == "A"
    assert subjects.loc["Python", "class_avg"] == pytest.approx(55)
    assert subjects.loc["Python", "vs_class"] == pytest.approx(25)
    assert subjects.loc["DBMS", "section_avg"] == pytest.approx(37.5)
    with pytest.raises(KeyError):
        an.student_profile(frame, "nope")


def test_filters(frame):
    out = an.apply_filters(frame, sections=["B"], subjects=["DBMS"])
    assert out["roll_no"].tolist() == ["R3", "R4"]
    assert len(an.apply_filters(frame)) == 8
    assert an.apply_filters(frame, semesters=["9"]).empty


def test_empty_inputs_are_safe(frame):
    empty = frame.iloc[0:0]
    assert an.group_summary(empty, ["section"]).empty
    assert an.fail_rate_matrix(empty).empty
    assert an.student_summary(empty).empty
    assert an.subject_difficulty(empty).empty
    assert math.isnan(an.kpis(empty)["avg_score"])
    no_mst = frame.drop(columns=["mst1_pct", "mst2_pct"])
    assert an.mst_change(no_mst).empty and an.mst_trend_summary(no_mst).empty
    assert an.attendance_performance(frame.drop(columns=["attendance_pct"]))["n"] == 0


def test_on_full_sample(derived):
    k = an.kpis(derived)
    assert k["students"] == 180 and k["records"] == 900
    assert 5 < k["fail_rate"] < 15
    diff = an.subject_difficulty(derived)
    assert diff.iloc[0]["subject"] == "Discrete Maths"
    sections = an.group_summary(derived, ["section"]).set_index("section")
    assert sections["fail_rate"].idxmax() == "B"
    assert an.attendance_performance(derived)["r"] > 0.3
    assert isinstance(an.fail_rate_matrix(derived), pd.DataFrame)
