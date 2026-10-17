"""At-risk students: transparent rule-based scores plus an optional ML model.

Rules (always available) turn early-warning signals into points with a
readable reason for each one. The ML model (when the sheet has final results)
learns to predict *failing a subject* from early indicators only —
attendance, assignment, MST-1 and MST-2 — so it can be reused on next
semester's mid-semester sheet before anyone has failed.

Everything here is decision support, not a verdict about a student.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.cleaning import class_wide_absence
from core.config import AnalysisConfig

log = logging.getLogger("gradelens.risk")

HIGH, MEDIUM, LOW = "High", "Medium", "Low"
LEVELS = (HIGH, MEDIUM, LOW)
HIGH_AT, MEDIUM_AT = 50, 25
ML_HIGH_AT, ML_MEDIUM_AT = 0.75, 0.5  # fail probability thresholds (the model is trained class-balanced)

FEATURES = ("attendance_pct", "assignment_pct", "mst1_pct", "mst2_pct")
FEATURE_LABELS = {"attendance_pct": "Attendance", "assignment_pct": "Assignment", "mst1_pct": "MST-1",
                  "mst2_pct": "MST-2"}  # fmt: skip
ABSENT_FLAGS = {"assignment_pct": "assignment_absent", "mst1_pct": "mst1_absent", "mst2_pct": "mst2_absent"}
MIN_ROWS, MIN_FAILS, MAX_FOLDS = 100, 10, 5
MODEL_VERSION = 1
MODEL_FILENAME = "risk_model.joblib"

# Points for each rule (shown in the UI under "How the score works"). Calibrated so that one
# signal on its own is Low, a failed subject or two signals is Medium, and a fail plus another
# signal (or several signals) is High. See DECISIONS.md.
POINTS = {
    "failed_subject": 30,
    "failed_cap": 60,
    "internal_below_pass": 20,
    "internal_cap": 40,
    "attendance_low": 20,
    "attendance_very_low": 30,  # 15+ points below the requirement
    "attendance_some_subjects": 5,
    "mst_drop": 10,  # MST-2 at least MST_DROP_AT points below MST-1
    "mst_drop_cap": 20,
    "borderline_multiple": 10,  # 2+ subjects within 10 points of the pass mark
    "absent": 10,
}
MST_DROP_AT = 20.0  # percentage points (3 marks out of 15); smaller dips are too common to mean much


def model_dir() -> Path:
    return Path(os.environ.get("GRADELENS_MODEL_DIR") or Path(__file__).resolve().parents[1] / "models")


def level_for_score(score: float) -> str:
    return HIGH if score >= HIGH_AT else MEDIUM if score >= MEDIUM_AT else LOW


def level_for_probability(prob: float | None) -> str:
    if prob is None or (isinstance(prob, float) and math.isnan(prob)):
        return LOW
    return HIGH if prob >= ML_HIGH_AT else MEDIUM if prob >= ML_MEDIUM_AT else LOW


def _join(items: list[str], limit: int = 3) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


# --- Rules -------------------------------------------------------------------------


def final_fails(df: pd.DataFrame) -> pd.Series:
    """Rows that have actually been failed: a Fail result backed by final marks.

    Mid-semester rows (no end-term or total yet) get a *projected* result from internals; those
    are reported as "internal marks below pass", not as a fail that has already happened.
    """
    has_final = pd.Series(False, index=df.index)
    if "total" in df:
        has_final |= df["total"].notna()
    if "end_term_absent" in df:
        has_final |= df["end_term_absent"].astype(bool)
    return df["result"].eq("Fail") & has_final


def _row_flags(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Per student-subject signals, computed in one vectorised pass."""
    pass_pct = config.pass_pct
    index = df.index
    false = pd.Series(False, index=index)
    failed = final_fails(df)
    internal = df["internal_pct"] if "internal_pct" in df else pd.Series(np.nan, index=index)
    if {"mst1_pct", "mst2_pct"} <= set(df.columns):
        change = df["mst2_pct"] - df["mst1_pct"]
    else:
        change = pd.Series(np.nan, index=index)
    attendance = df["attendance_pct"] if "attendance_pct" in df else pd.Series(np.nan, index=index)
    flags = pd.DataFrame(
        {
            "roll_no": df["roll_no"],
            "subject": df["subject"].astype(str),
            "score_pct": df["score_pct"],
            "internal_pct": internal,
            "attendance_pct": attendance,
            "change": change,
            "failed": failed,
            "end_absent": df["end_term_absent"].astype(bool) if "end_term_absent" in df else false,
            "low_internal": (internal < pass_pct) & ~failed,
            "drop": change <= -MST_DROP_AT,
            "borderline": df["score_pct"].between(pass_pct, pass_pct + 10, inclusive="left"),
            "low_attendance": attendance < config.attendance_threshold,
        },
        index=index,
    )
    for component in ("mst1", "mst2", "assignment"):
        flag = f"{component}_absent"
        individual = ~class_wide_absence(df, component)  # a whole class "absent" is a data issue, not a warning sign
        flags[f"absent_{component}"] = (df[flag].astype(bool) & individual) if flag in df else false
    return flags


def _reasons_and_points(rows: list[dict], attendance: float, config: AnalysisConfig) -> tuple[int, list[str]]:
    """Readable reasons for one student, most serious first (rows are plain dicts: fast)."""
    points, reasons = 0, []
    pass_pct, threshold = config.pass_pct, config.attendance_threshold

    failed = [r for r in rows if r["failed"]]
    if failed:
        points += min(POINTS["failed_cap"], POINTS["failed_subject"] * len(failed))
        detail = []
        for r in failed:
            how = "absent in end term" if r["end_absent"] else f"{r['score_pct']:.0f}%"
            detail.append(f"{r['subject']} ({how})")
        reasons.append(f"Failed {_join(detail)}")

    low_internal = [r for r in rows if r["low_internal"]]
    if low_internal:
        points += min(POINTS["internal_cap"], POINTS["internal_below_pass"] * len(low_internal))
        detail = [f"{r['subject']} ({r['internal_pct']:.0f}%)" for r in low_internal]
        reasons.append(f"Internal marks below {pass_pct:g}% in {_join(detail)}")

    if not math.isnan(attendance):
        if attendance < threshold:
            points += POINTS["attendance_very_low" if attendance < threshold - 15 else "attendance_low"]
            reasons.append(f"Attendance {attendance:.0f}% — below the {threshold:g}% requirement")
        else:
            low = [r["subject"] for r in rows if r["low_attendance"]]
            if low:
                points += POINTS["attendance_some_subjects"]
                reasons.append(f"Attendance below {threshold:g}% in {_join(low)}")

    drops = sorted((r for r in rows if r["drop"]), key=lambda r: r["change"])
    if drops:
        points += min(POINTS["mst_drop_cap"], POINTS["mst_drop"] * len(drops))
        detail = [f"{r['subject']} ({r['change']:+.0f} pts)" for r in drops]
        reasons.append(f"MST-2 fell sharply vs MST-1 in {_join(detail)}")

    borderline = [r["subject"] for r in rows if r["borderline"]]
    if len(borderline) >= 2:
        points += POINTS["borderline_multiple"]
        reasons.append(
            f"Just above the pass mark ({pass_pct:g}–{pass_pct + 10:g}%) in {len(borderline)} subjects: "
            f"{_join(borderline)}"
        )

    absences = [f"{label} ({r['subject']})" for r in rows
                for component, label in (("mst1", "MST-1"), ("mst2", "MST-2"), ("assignment", "assignment"))
                if r[f"absent_{component}"]]  # fmt: skip
    if absences:
        points += POINTS["absent"]
        reasons.append(f"Absent in {_join(absences)}")

    return min(100, points), reasons


def rule_scores(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """One row per student: rule score (0-100), level and reasons."""
    columns = ["roll_no", "student_name", "section", "rule_score", "rule_level", "rule_reasons",
               "avg_score", "avg_attendance", "subjects", "fails"]  # fmt: skip
    if df.empty:
        return pd.DataFrame(columns=columns)
    flags = _row_flags(df, config)
    summary = (
        df.assign(_fail=flags["failed"])
        .groupby("roll_no", sort=False)
        .agg(
            student_name=("student_name", "first"),
            section=("section", "first"),
            avg_score=("score_pct", "mean"),
            subjects=("subject", "nunique"),
            fails=("_fail", "sum"),
        )
    )
    summary["avg_attendance"] = flags.groupby("roll_no", sort=False)["attendance_pct"].mean()
    rows_by_student: dict[str, list[dict]] = {}
    for record in flags.to_dict("records"):
        rows_by_student.setdefault(record["roll_no"], []).append(record)
    scores, reasons = [], []
    for roll, attendance in summary["avg_attendance"].items():
        score, why = _reasons_and_points(rows_by_student[roll], float(attendance), config)
        scores.append(score)
        reasons.append(why)
    summary["rule_score"] = scores
    summary["rule_level"] = [level_for_score(s) for s in scores]
    summary["rule_reasons"] = reasons
    summary["fails"] = summary["fails"].astype(int)
    summary["subjects"] = summary["subjects"].astype(int)
    return summary.reset_index()[columns]


# --- ML ----------------------------------------------------------------------------


def available_features(df: pd.DataFrame) -> list[str]:
    return [f for f in FEATURES if f in df and df[f].notna().mean() >= 0.5]


def build_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Feature matrix (one row per student-subject). Absent counts as 0%; other blanks are imputed."""
    X = pd.DataFrame(index=df.index)
    for feature in features:
        values = df[feature].astype(float) if feature in df else pd.Series(np.nan, index=df.index)
        flag = ABSENT_FLAGS.get(feature)
        if flag and flag in df:
            values = values.where(~df[flag].astype(bool), 0.0)
        X[feature] = values
    return X


def _candidates(random_state: int) -> dict[str, Pipeline]:
    return {
        "Logistic regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
            ]
        ),
        "Random forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),  # fmt: skip
            ]
        ),
    }


@dataclass
class ModelReport:
    chosen: str
    metrics: pd.DataFrame  # one row per candidate model
    confusion: pd.DataFrame
    importance: pd.DataFrame
    features: list[str]
    n_rows: int
    n_fails: int
    folds: int
    trained_at: str
    source: str = "trained"  # "trained" on this data | "saved" model loaded from disk

    @property
    def chosen_metrics(self) -> dict:
        return self.metrics.set_index("model").loc[self.chosen].to_dict()


@dataclass
class RiskModel:
    pipeline: Pipeline
    features: list[str]
    report: ModelReport
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    oof_proba: pd.Series | None = field(default=None, repr=False)  # out-of-fold, keyed by (roll_no, subject)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        X = build_features(df, self.features)
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=df.index)

    def explain(self, row: pd.Series) -> list[str]:
        """Early indicators where this row is clearly worse than the training average."""
        weights = self.report.importance.set_index("feature")["importance"]
        scored = []
        for feature in self.features:
            value = row.get(feature)
            mean, std = self.feature_means[feature], self.feature_stds[feature] or 1.0
            if value is None or pd.isna(value):
                continue
            deficit = (mean - float(value)) / std
            if deficit > 0.5:
                scored.append((deficit * float(weights.get(feature, 0.0)), feature, float(value), mean))
        scored.sort(reverse=True)
        return [f"{FEATURE_LABELS[f]} {v:.0f}% vs {m:.0f}% average" for _, f, v, m in scored[:2]]


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    graded = df[df["result"].isin(["Pass", "Fail"])]
    if "end_term" in graded:
        has_final = graded["end_term"].notna() | graded.get("end_term_absent", False)
        graded = graded[has_final]
    else:
        graded = graded.iloc[0:0]
    return graded


def can_train(df: pd.DataFrame) -> tuple[bool, str]:
    rows = training_rows(df)
    features = available_features(df)
    if "end_term" not in df or rows.empty:
        return False, "the sheet has no final (end-term) results to learn from yet"
    if len(features) < 2:
        return False, "fewer than two early indicators (attendance, assignment, MST-1, MST-2) are present"
    fails = int((rows["result"] == "Fail").sum())
    if len(rows) < MIN_ROWS or fails < MIN_FAILS:
        return False, (
            f"there is too little data to train reliably ({len(rows)} graded records and {fails} fails; "
            f"the model needs at least {MIN_ROWS} records and {MIN_FAILS} fails)"
        )
    return True, ""


METRICS = ("f1", "roc_auc", "precision", "recall", "accuracy")


def _fold_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    predicted = (proba >= 0.5).astype(int)
    return {
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan"),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "accuracy": accuracy_score(y_true, predicted),
    }


def row_keys(df: pd.DataFrame) -> pd.MultiIndex:
    """(roll_no, subject) identifies a student-subject regardless of row order or filtering."""
    return pd.MultiIndex.from_arrays([df["roll_no"].astype(str), df["subject"].astype(str)])


def train(df: pd.DataFrame, random_state: int = 42) -> RiskModel:
    """Compare logistic regression and random forest with stratified CV; refit the winner on all rows."""
    ok, reason = can_train(df)
    if not ok:
        raise ValueError(f"Can't train the risk model: {reason}.")
    rows = training_rows(df)
    features = available_features(df)
    X = build_features(rows, features)
    y = (rows["result"] == "Fail").astype(int).to_numpy()
    n_fails = int(y.sum())
    folds = min(MAX_FOLDS, n_fails)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    records, oof = [], {}
    splits = list(cv.split(X, y))
    for name, pipeline in _candidates(random_state).items():
        # One fit per fold gives both the CV scores and honest out-of-fold probabilities.
        proba = np.zeros(len(y), dtype=float)
        fold_scores: dict[str, list[float]] = {m: [] for m in METRICS}
        for train_idx, test_idx in splits:
            fitted = clone(pipeline).fit(X.iloc[train_idx], y[train_idx])
            p = fitted.predict_proba(X.iloc[test_idx])[:, 1]
            proba[test_idx] = p
            for metric, value in _fold_metrics(y[test_idx], p).items():
                fold_scores[metric].append(value)
        records.append(
            {
                "model": name,
                **{m: float(np.mean(v)) for m, v in fold_scores.items()},
                "f1_std": float(np.std(fold_scores["f1"])),
            }  # fmt: skip
        )
        oof[name] = proba
    metrics = pd.DataFrame(records)
    chosen = metrics.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]["model"]

    final = clone(_candidates(random_state)[chosen]).fit(X, y)
    estimator = final.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        raw = np.asarray(estimator.feature_importances_, dtype=float)
    else:
        raw = np.abs(np.asarray(estimator.coef_[0], dtype=float))
    importance = pd.DataFrame(
        {
            "feature": features,
            "label": [FEATURE_LABELS[f] for f in features],
            "importance": raw / raw.sum() if raw.sum() else raw,
        }  # fmt: skip
    ).sort_values("importance", ascending=False, ignore_index=True)

    predicted = (oof[chosen] >= 0.5).astype(int)
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    confusion = pd.DataFrame(cm, index=["Pass", "Fail"], columns=["Pass", "Fail"])
    report = ModelReport(
        chosen=str(chosen),
        metrics=metrics,
        confusion=confusion,
        importance=importance,
        features=features,
        n_rows=int(len(rows)),
        n_fails=n_fails,
        folds=folds,
        trained_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )
    imputed = X.fillna(X.median())
    return RiskModel(
        pipeline=final,
        features=features,
        report=report,
        feature_means={f: float(imputed[f].mean()) for f in features},
        feature_stds={f: float(imputed[f].std(ddof=0)) for f in features},
        oof_proba=pd.Series(oof[chosen], index=row_keys(rows)),
    )


def save_model(model: RiskModel, path: Path | None = None) -> Path:
    path = path or model_dir() / MODEL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MODEL_VERSION,
        "pipeline": model.pipeline,
        "features": model.features,
        "report": model.report,
        "feature_means": model.feature_means,
        "feature_stds": model.feature_stds,
    }
    joblib.dump(payload, path)
    return path


def load_model(path: Path | None = None) -> RiskModel | None:
    """Load a model this app saved. Never load files from users: joblib can run code."""
    path = path or model_dir() / MODEL_FILENAME
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - a corrupt or incompatible file should not break the app
        log.warning("Ignoring unreadable model file %s: %s", path, exc)
        return None
    if not isinstance(payload, dict) or payload.get("version") != MODEL_VERSION:
        return None
    report = payload["report"]
    report.source = "saved"
    return RiskModel(payload["pipeline"], list(payload["features"]), report, payload["feature_means"],
                     payload["feature_stds"])  # fmt: skip


# --- Combined assessment --------------------------------------------------------------


@dataclass
class RiskResult:
    students: pd.DataFrame
    method: str  # "rules" | "rules+ml"
    model_report: ModelReport | None = None
    fallback_reason: str | None = None

    @property
    def counts(self) -> pd.Series:
        return self.students["risk_level"].value_counts().reindex(list(LEVELS), fill_value=0)

    @property
    def at_risk_count(self) -> int:
        return int(self.students["risk_level"].isin([HIGH, MEDIUM]).sum())


def assess(
    df: pd.DataFrame,
    config: AnalysisConfig,
    model: RiskModel | None = None,
    fallback_reason: str | None = None,
) -> RiskResult:
    """Score every student in ``df``; add ML fail probabilities when a model is given."""
    students = rule_scores(df, config)
    students["ml_prob"] = np.nan
    students["ml_subject"] = None
    ml_reasons: dict[str, list[str]] = {}
    usable = model is not None and len(set(model.features) & set(df.columns)) >= 2

    if usable and not df.empty:
        # Score only subjects the student has not already failed: the rules report those, and the
        # model's job is to flag the *next* likely fail (or, mid-semester, every subject).
        candidates = df[~final_fails(df)]
        proba = model.predict_proba(candidates) if not candidates.empty else pd.Series(dtype=float)
        if model.oof_proba is not None and not candidates.empty:
            # On the training data itself, use out-of-fold probabilities: honest, not memorised.
            oof = pd.Series(model.oof_proba.reindex(row_keys(candidates)).to_numpy(), index=candidates.index)
            proba = oof.fillna(proba)
        if not proba.empty:
            work = candidates.assign(_proba=proba)
            by_roll = work.loc[work.groupby("roll_no")["_proba"].idxmax()].set_index("roll_no")
            students["ml_prob"] = students["roll_no"].map(by_roll["_proba"]).astype(float)
            students["ml_subject"] = students["roll_no"].map(by_roll["subject"])
            for roll, row in by_roll.iterrows():
                if row["_proba"] >= ML_MEDIUM_AT:
                    factors = model.explain(build_features(row.to_frame().T, model.features).iloc[0])
                    text = f"Model: {row['_proba'] * 100:.0f}% chance of failing {row['subject']}"
                    ml_reasons[roll] = [text + (f" ({'; '.join(factors)})" if factors else "")]

    def combine(row) -> pd.Series:
        ml_level = level_for_probability(row["ml_prob"])
        rule_level = row["rule_level"]
        level = min((rule_level, ml_level), key=LEVELS.index)
        ml_points = 0.0 if pd.isna(row["ml_prob"]) else row["ml_prob"] * 100
        reasons = list(row["rule_reasons"]) + ml_reasons.get(row["roll_no"], [])
        return pd.Series({"risk_level": level, "risk_score": round(max(row["rule_score"], ml_points)),
                          "reasons": reasons})  # fmt: skip

    if students.empty:
        students = students.assign(risk_level=pd.Series(dtype=str), risk_score=pd.Series(dtype=int),
                                   reasons=pd.Series(dtype=object))  # fmt: skip
    else:
        students = pd.concat([students, students.apply(combine, axis=1)], axis=1)
    order = {level: i for i, level in enumerate(LEVELS)}
    students = students.sort_values(
        ["risk_level", "risk_score", "avg_score"], key=lambda s: s.map(order) if s.name == "risk_level" else s,
        ascending=[True, False, True],
    ).reset_index(drop=True)  # fmt: skip
    return RiskResult(
        students=students,
        method="rules+ml" if usable else "rules",
        model_report=model.report if usable else None,
        fallback_reason=None if usable else fallback_reason,
    )
