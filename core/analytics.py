"""Deterministic analytics on the derived marks table (no LLM involved).

All functions are pure: DataFrame in, DataFrame/dict out. Rates are in
percent (0-100). "Graded" rows are those with a Pass/Fail result; rows marked
Incomplete never count towards pass or fail rates.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from core.config import MARK_COMPONENTS

GRADE_BANDS = (("O", 90.0), ("A+", 80.0), ("A", 70.0), ("B+", 60.0), ("B", 50.0), ("C", 0.0))
GRADE_ORDER = ["O", "A+", "A", "B+", "B", "C", "F"]
ATTENDANCE_BANDS = ((0, 60, "<60%"), (60, 75, "60-75%"), (75, 85, "75-85%"), (85, 101, "85%+"))
TRENDS = ("Improved", "Stable", "Declined")


def _graded(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["result"].isin(["Pass", "Fail"])]


def _rate(mask: pd.Series) -> float:
    return float(mask.mean() * 100) if len(mask) else float("nan")


def apply_filters(
    df: pd.DataFrame,
    semesters: Iterable[str] | None = None,
    sections: Iterable[str] | None = None,
    subjects: Iterable[str] | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for column, values in (("semester", semesters), ("section", sections), ("subject", subjects)):
        if values:
            mask &= df[column].isin(list(values))
    return df[mask]


def kpis(df: pd.DataFrame, at_risk_count: int | None = None) -> dict:
    graded = _graded(df)
    absent_cols = [f"{c}_absent" for c in MARK_COMPONENTS if f"{c}_absent" in df]
    absentees = int(df.loc[df[absent_cols].any(axis=1), "roll_no"].nunique()) if absent_cols else 0
    return {
        "students": int(df["roll_no"].nunique()),
        "records": int(len(df)),
        "subjects": int(df["subject"].nunique()),
        "pass_rate": _rate(graded["result"] == "Pass"),
        "fail_rate": _rate(graded["result"] == "Fail"),
        "fails": int((graded["result"] == "Fail").sum()),
        "avg_score": float(df["score_pct"].mean()) if len(df) else float("nan"),
        "avg_attendance": float(df["attendance_pct"].mean()) if "attendance_pct" in df else float("nan"),
        "absentees": absentees,
        "at_risk": at_risk_count,
    }


def group_summary(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Students, average score, pass/fail rates and attendance per group."""
    columns = [*by, "students", "records", "avg_score", "pass_rate", "fail_rate", "fails", "avg_attendance"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    work = df.assign(
        _graded=df["result"].isin(["Pass", "Fail"]),
        _fail=df["result"].eq("Fail"),
        _pass=df["result"].eq("Pass"),
    )
    if "attendance_pct" not in work:
        work["attendance_pct"] = np.nan
    grouped = work.groupby(by, observed=True, sort=True)
    out = grouped.agg(
        students=("roll_no", "nunique"),
        records=("roll_no", "size"),
        avg_score=("score_pct", "mean"),
        graded=("_graded", "sum"),
        fails=("_fail", "sum"),
        passes=("_pass", "sum"),
        avg_attendance=("attendance_pct", "mean"),
    ).reset_index()
    graded = out["graded"].replace(0, np.nan)
    out["pass_rate"] = out["passes"] / graded * 100
    out["fail_rate"] = out["fails"] / graded * 100
    out["fails"] = out["fails"].astype(int)
    return out[columns]


def fail_rate_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Section x subject fail-rate (%) table; NaN where no graded rows exist."""
    summary = group_summary(df, ["section", "subject"])
    if summary.empty:
        return pd.DataFrame()
    return summary.pivot(index="section", columns="subject", values="fail_rate")


def grade_band(score: float, pass_pct: float) -> str | None:
    if pd.isna(score):
        return None
    if score < pass_pct:
        return "F"
    for label, floor in GRADE_BANDS:
        if score >= max(floor, pass_pct):
            return label
    return "C"


def grade_distribution(df: pd.DataFrame, pass_pct: float) -> pd.DataFrame:
    bands = df["score_pct"].map(lambda s: grade_band(s, pass_pct)).dropna()
    counts = bands.value_counts().reindex(GRADE_ORDER, fill_value=0)
    total = counts.sum()
    out = counts.rename_axis("grade").reset_index(name="count")
    out["share"] = out["count"] / total * 100 if total else 0.0
    return out


def mst_change(df: pd.DataFrame, tolerance: float = 5.0) -> pd.DataFrame:
    """Per student-subject MST-1 -> MST-2 change in percentage points."""
    columns = ["roll_no", "student_name", "section", "subject", "mst1_pct", "mst2_pct", "change", "trend"]
    if not {"mst1_pct", "mst2_pct"} <= set(df.columns):
        return pd.DataFrame(columns=columns)
    work = df.dropna(subset=["mst1_pct", "mst2_pct"]).copy()
    work["change"] = work["mst2_pct"] - work["mst1_pct"]
    work["trend"] = np.select(
        [work["change"] > tolerance, work["change"] < -tolerance], ["Improved", "Declined"], default="Stable"
    )
    return work[columns].reset_index(drop=True)


def mst_trend_summary(df: pd.DataFrame, tolerance: float = 5.0, by: str | None = "subject") -> pd.DataFrame:
    changes = mst_change(df, tolerance)
    if changes.empty:
        return pd.DataFrame(columns=[*([by] if by else []), *TRENDS])
    keys = [by, "trend"] if by else ["trend"]
    counts = changes.groupby(keys).size().unstack(fill_value=0) if by else changes["trend"].value_counts().to_frame().T
    counts = counts.reindex(columns=list(TRENDS), fill_value=0)
    return counts.reset_index() if by else counts.reset_index(drop=True)


def attendance_performance(df: pd.DataFrame, threshold: float = 75.0) -> dict:
    """Correlation between attendance and score, plus banded averages."""
    if "attendance_pct" not in df:
        return {"r": float("nan"), "n": 0, "bands": pd.DataFrame(), "below": {}, "above": {}}
    paired = df.dropna(subset=["attendance_pct", "score_pct"])
    r = float(paired["attendance_pct"].corr(paired["score_pct"])) if len(paired) >= 3 else float("nan")
    labels = [label for _, _, label in ATTENDANCE_BANDS]
    edges = [ATTENDANCE_BANDS[0][0]] + [hi for _, hi, _ in ATTENDANCE_BANDS]
    banded = paired.assign(band=pd.cut(paired["attendance_pct"], bins=edges, labels=labels, right=False))
    bands = group_summary(banded, ["band"]) if not banded.empty else pd.DataFrame()
    if not bands.empty:
        bands = bands.set_index("band").reindex(labels).reset_index()

    def side(mask: pd.Series) -> dict:
        part = paired[mask]
        graded = _graded(part)
        return {
            "records": int(len(part)),
            "avg_score": float(part["score_pct"].mean()) if len(part) else float("nan"),
            "fail_rate": _rate(graded["result"] == "Fail"),
        }

    return {
        "r": r,
        "n": int(len(paired)),
        "bands": bands,
        "below": side(paired["attendance_pct"] < threshold),
        "above": side(paired["attendance_pct"] >= threshold),
    }


def student_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per student across the subjects in ``df``."""
    columns = [
        "roll_no", "student_name", "section", "subjects", "avg_score", "min_score",
        "fails", "avg_attendance", "weakest_subject",
    ]  # fmt: skip
    if df.empty:
        return pd.DataFrame(columns=columns)
    work = df.assign(_fail=df["result"].eq("Fail"))
    if "attendance_pct" not in work:
        work["attendance_pct"] = np.nan
    grouped = work.groupby("roll_no", sort=False)
    out = grouped.agg(
        student_name=("student_name", "first"),
        section=("section", "first"),
        subjects=("subject", "nunique"),
        avg_score=("score_pct", "mean"),
        min_score=("score_pct", "min"),
        fails=("_fail", "sum"),
        avg_attendance=("attendance_pct", "mean"),
    )
    scored = work.dropna(subset=["score_pct"])
    weakest = scored.loc[scored.groupby("roll_no")["score_pct"].idxmin(), ["roll_no", "subject"]]
    out["weakest_subject"] = weakest.set_index("roll_no")["subject"]
    out["fails"] = out["fails"].astype(int)
    return out.reset_index()[columns]


def top_bottom(df: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = student_summary(df).dropna(subset=["avg_score"])
    ordered = summary.sort_values(["avg_score", "roll_no"], ascending=[False, True])
    bottom = summary.sort_values(["avg_score", "roll_no"], ascending=[True, True])
    return ordered.head(n).reset_index(drop=True), bottom.head(n).reset_index(drop=True)


def subject_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Difficulty index = average of (100 - average score) and fail rate.

    0 means everyone scores 100% and nobody fails; 100 means everyone scores
    0% and everyone fails. Ranked hardest first.
    """
    summary = group_summary(df, ["subject"])
    if summary.empty:
        return summary.assign(difficulty_index=pd.Series(dtype=float), rank=pd.Series(dtype=int))
    summary["difficulty_index"] = (0.5 * (100 - summary["avg_score"]) + 0.5 * summary["fail_rate"].fillna(0)).round(1)
    summary = summary.sort_values("difficulty_index", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def student_profile(df: pd.DataFrame, roll_no: str) -> dict:
    """Everything the Student Profile view needs for one student."""
    rows = df[df["roll_no"] == roll_no]
    if rows.empty:
        raise KeyError(f"Unknown roll number: {roll_no}")
    section = rows["section"].iloc[0]
    class_avg = df.groupby("subject")["score_pct"].mean()
    section_avg = df[df["section"] == section].groupby("subject")["score_pct"].mean()
    detail = rows.set_index("subject")
    detail = detail.assign(
        class_avg=class_avg.reindex(detail.index),
        section_avg=section_avg.reindex(detail.index),
    )
    detail["vs_class"] = detail["score_pct"] - detail["class_avg"]
    summary = student_summary(rows).iloc[0].to_dict()
    overall_class = student_summary(df)["avg_score"].mean()
    return {
        "roll_no": roll_no,
        "student_name": rows["student_name"].iloc[0],
        "section": section,
        "summary": summary,
        "class_avg_score": float(overall_class),
        "subjects": detail.reset_index(),
    }
