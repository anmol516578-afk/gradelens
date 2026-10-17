from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from core.config import AnalysisConfig
from core.derive import add_derived
from ml import risk_model as rm
from tests.conftest import make_frame

CONFIG = AnalysisConfig()


def _row(roll, subject, *, att=85.0, a=8.0, m1=12.0, m2=12.0, end=40.0, **extra) -> dict:
    return {"roll_no": roll, "student_name": f"Student {roll}", "section": "A", "semester": "5", "subject": subject,
            "attendance_pct": att, "assignment": a, "mst1": m1, "mst2": m2, "end_term": end, **extra}  # fmt: skip


def derive(rows: list[dict]) -> pd.DataFrame:
    frame = make_frame(rows)
    for column in [c for c in frame.columns if c.endswith("_absent")]:
        frame[column] = frame[column].fillna(False).astype(bool)  # cleaned data always has boolean flags
    return add_derived(frame, CONFIG)


@pytest.fixture(scope="module")
def students() -> pd.DataFrame:
    rows = [
        # R1: healthy in both subjects -> 0 points
        _row("R1", "Python"), _row("R1", "DBMS"),
        # R2: fails DBMS (8+5+5+10 = 28%) and attendance 70% -> 30 + 20 = 50 (High)
        _row("R2", "Python", att=70), _row("R2", "DBMS", att=70, a=8, m1=5, m2=5, end=10),
        # R3: mid-semester style: internals below 40% in two subjects, no end term -> 2 x 20 = 40 (Medium)
        _row("R3", "Python", a=3, m1=5, m2=6, end=np.nan), _row("R3", "DBMS", a=4, m1=4, m2=5, end=np.nan),
        # R4: MST-2 fell 33 points in Python (15 -> 10 is -33 pts) and absent in DBMS MST-1 -> 10 + 10 = 20 (Low)
        _row("R4", "Python", m1=15, m2=10), _row("R4", "DBMS", m1=np.nan, mst1_absent=True),
        # R5: small MST dip (12 -> 10 = -13 pts) only: below the 20-point threshold -> no MST reason
        _row("R5", "Python", m1=12, m2=10), _row("R5", "DBMS"),
    ]  # fmt: skip
    return derive(rows)


def scores(df) -> pd.DataFrame:
    return rm.rule_scores(df, CONFIG).set_index("roll_no")


def test_healthy_student_has_no_points(students):
    s = scores(students).loc["R1"]
    assert s["rule_score"] == 0 and s["rule_level"] == rm.LOW and s["rule_reasons"] == []


def test_fail_plus_low_attendance_is_high(students):
    s = scores(students).loc["R2"]
    assert s["rule_score"] == 50 and s["rule_level"] == rm.HIGH
    assert s["rule_reasons"][0] == "Failed DBMS (28%)"  # most serious reason first
    assert s["rule_reasons"][1] == "Attendance 70% — below the 75% requirement"


def test_low_internals_mid_semester_is_medium(students):
    s = scores(students).loc["R3"]
    assert s["rule_score"] == 40 and s["rule_level"] == rm.MEDIUM
    assert s["rule_reasons"] == ["Internal marks below 40% in Python (35%), DBMS (32%)"]


def test_sharp_mst_drop_and_absence(students):
    s = scores(students).loc["R4"]
    assert s["rule_score"] == 20 and s["rule_level"] == rm.LOW
    assert "MST-2 fell sharply vs MST-1 in Python (-33 pts)" in s["rule_reasons"]
    assert "Absent in MST-1 (DBMS)" in s["rule_reasons"]


def test_small_dip_is_not_a_reason(students):
    """Regression: an empty filtered frame used to pick up NaN rows ("nan (-7 pts)")."""
    s = scores(students).loc["R5"]
    assert s["rule_reasons"] == [] and s["rule_score"] == 0


def test_borderline_and_capped_points():
    # R6: 3 fails (60, capped) + very low attendance (30) + sharp MST drops (20, capped) = 110 -> capped at 100
    rows = [_row("R6", subj, att=50, a=2, m1=12, m2=2, end=5) for subj in ("A", "B", "C")]
    rows += [_row("R7", subj, a=6, m1=8, m2=8, end=22) for subj in ("A", "B")]  # 44% in both: borderline
    s = scores(derive(rows))
    assert s.loc["R6", "rule_score"] == 100
    assert s.loc["R7", "rule_reasons"] == ["Just above the pass mark (40–50%) in 2 subjects: A, B"]
    assert s.loc["R7", "rule_score"] == rm.POINTS["borderline_multiple"]


def test_rules_are_fast_on_large_sheets(derived):
    big = pd.concat([derived.assign(roll_no=derived["roll_no"] + f"_{i}") for i in range(6)], ignore_index=True)
    start = time.perf_counter()
    result = rm.rule_scores(big, CONFIG)
    assert len(result) == 1080
    assert time.perf_counter() - start < 2.0


# --- ML -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model(derived) -> rm.RiskModel:
    return rm.train(derived)


def test_trains_and_compares_two_models(model, derived):
    report = model.report
    assert set(report.metrics["model"]) == {"Logistic regression", "Random forest"}
    assert report.chosen in set(report.metrics["model"])
    best = report.metrics.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]["model"]
    assert report.chosen == best
    assert report.chosen_metrics["roc_auc"] > 0.85 and report.chosen_metrics["f1"] > 0.5
    assert report.confusion.to_numpy().sum() == report.n_rows == 900
    assert report.n_fails == int((derived["result"] == "Fail").sum())
    assert report.folds == 5
    assert report.importance["importance"].sum() == pytest.approx(1.0)
    assert set(report.features) == set(rm.FEATURES)  # end term is never a feature (that would leak the answer)


def test_assess_with_model_only_scores_subjects_not_yet_failed(model, derived):
    result = rm.assess(derived, CONFIG, model)
    assert result.method == "rules+ml" and result.model_report is model.report
    fails = derived[derived["result"] == "Fail"]
    failed = set(zip(fails["roll_no"], fails["subject"], strict=True))
    scored = result.students.dropna(subset=["ml_prob"])
    assert not any((r, s) in failed for r, s in zip(scored["roll_no"], scored["ml_subject"], strict=True))
    # Level is the higher of the rule level and the model level.
    for row in result.students.itertuples():
        expected = min(row.rule_level, rm.level_for_probability(row.ml_prob), key=rm.LEVELS.index)
        assert row.risk_level == expected
    assert result.at_risk_count == int(result.students["risk_level"].isin([rm.HIGH, rm.MEDIUM]).sum())
    assert any(any(r.startswith("Model:") for r in rs) for rs in result.students["reasons"])


def test_out_of_fold_probabilities_survive_filtering(model, derived):
    """Probabilities are keyed by (roll_no, subject), so a filtered or reordered view gets the same values."""
    full = rm.assess(derived, CONFIG, model).students.set_index("roll_no")["ml_prob"]
    section_b = derived[derived["section"] == "B"].iloc[::-1]
    part = rm.assess(section_b, CONFIG, model).students.set_index("roll_no")["ml_prob"]
    pd.testing.assert_series_equal(part.sort_index(), full.loc[part.index].sort_index(), check_names=False)


def test_tiny_data_falls_back_to_rules(derived):
    tiny = derived[derived["roll_no"].isin(derived["roll_no"].unique()[:10])]
    ok, reason = rm.can_train(tiny)
    assert not ok and "too little data" in reason
    with pytest.raises(ValueError, match="too little data"):
        rm.train(tiny)
    result = rm.assess(tiny, CONFIG, None, fallback_reason=reason)
    assert result.method == "rules" and result.fallback_reason == reason
    assert result.students["ml_prob"].isna().all()
    assert len(result.students) == 10


def test_mid_semester_sheet_cannot_train_but_saved_model_scores_it(model, derived, tmp_path):
    mid = derived.drop(columns=["end_term", "end_term_pct", "end_term_absent", "total"])
    mid = add_derived(mid, CONFIG)
    ok, reason = rm.can_train(mid)
    assert not ok and "no final" in reason
    path = rm.save_model(model, tmp_path / "m.joblib")
    loaded = rm.load_model(path)
    assert loaded.report.source == "saved" and loaded.features == model.features
    result = rm.assess(mid, CONFIG, loaded)
    assert result.method == "rules+ml"
    assert result.students["ml_prob"].notna().all()
    np.testing.assert_allclose(loaded.predict_proba(mid), model.predict_proba(mid))


def test_bad_model_files_are_ignored(tmp_path):
    assert rm.load_model(tmp_path / "missing.joblib") is None
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a model")
    assert rm.load_model(corrupt) is None
    import joblib

    joblib.dump({"version": 999}, tmp_path / "old.joblib")
    assert rm.load_model(tmp_path / "old.joblib") is None


def test_model_dir_is_configurable(monkeypatch, tmp_path):
    monkeypatch.setenv("GRADELENS_MODEL_DIR", str(tmp_path))
    assert rm.model_dir() == tmp_path


def test_explain_names_weak_indicators(model):
    weak = pd.Series({"attendance_pct": 50.0, "assignment_pct": 20.0, "mst1_pct": 10.0, "mst2_pct": 5.0})
    reasons = model.explain(weak)
    assert 1 <= len(reasons) <= 2 and all("vs" in r and "average" in r for r in reasons)
    strong = pd.Series({"attendance_pct": 95.0, "assignment_pct": 95.0, "mst1_pct": 95.0, "mst2_pct": 95.0})
    assert model.explain(strong) == []
