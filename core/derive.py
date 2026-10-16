"""Derived columns that depend on the marks scheme (and so on the sidebar).

Kept separate from cleaning so moving the pass-mark slider only recomputes
these cheap columns, not the whole ingestion pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.cleaning import _component_sum
from core.config import INTERNAL_COMPONENTS, MARK_COMPONENTS, AnalysisConfig

RESULT_PASS, RESULT_FAIL, RESULT_INCOMPLETE = "Pass", "Fail", "Incomplete"


def add_derived(clean: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    df = clean.copy()
    index = df.index

    if any(c in df for c in INTERNAL_COMPONENTS):
        computed_internal, _ = _component_sum(df, INTERNAL_COMPONENTS)
        if "internal_total" in df:
            df["internal_total"] = df["internal_total"].fillna(computed_internal)
        else:
            df["internal_total"] = computed_internal

    if "end_term" in df and "internal_total" in df:
        end = df["end_term"].where(~df.get("end_term_absent", pd.Series(False, index=index)), 0.0)
        computed_total = df["internal_total"] + end
        if "total" in df:
            df["total"] = df["total"].fillna(computed_total)
        else:
            df["total"] = computed_total

    for component in MARK_COMPONENTS:
        if component in df:
            df[f"{component}_pct"] = df[component] / config.max_marks(component) * 100

    if "internal_total" in df:
        df["internal_pct"] = df["internal_total"] / config.internal_max * 100

    score = pd.Series(np.nan, index=index, dtype=float)
    if "total" in df:
        score = df["total"] / config.total_max * 100
    if "internal_pct" in df:
        score = score.fillna(df["internal_pct"])
    if "end_term_pct" in df:
        score = score.fillna(df["end_term_pct"])
    df["score_pct"] = score.clip(lower=0).round(2)

    failed = df["score_pct"] < config.pass_pct
    if "end_term_absent" in df:
        failed = failed | df["end_term_absent"]
    df["result"] = np.where(df["score_pct"].isna(), RESULT_INCOMPLETE, np.where(failed, RESULT_FAIL, RESULT_PASS))
    df["result"] = df["result"].astype(str)
    return df


def assessment_stage(df: pd.DataFrame) -> str:
    """'final' when end-term marks exist for most rows, else 'mid-semester'."""
    if "end_term" in df and (df["end_term"].notna() | df.get("end_term_absent", False)).mean() >= 0.5:
        return "final"
    if "total" in df and df["total"].notna().mean() >= 0.5 and "internal_total" not in df:
        return "final"
    return "mid-semester"
