"""Everything a report needs, gathered once (no Streamlit), plus the executive summary.

The PDF and Excel builders only read a :class:`ReportData`, so both reports
always show the same numbers as each other and as the dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from core import analytics as an
from core.config import AnalysisConfig
from core.derive import assessment_stage

AT_RISK_LIMIT = 40  # rows in the PDF's at-risk table (the Excel sheet has everyone)


@dataclass
class ReportMeta:
    college: str = ""
    department: str = ""
    semester: str = ""
    faculty: str = ""
    title: str = "Student Performance Report"
    scope: str = "All students"
    include_names: bool = True
    generated: str = field(default_factory=lambda: datetime.now().strftime("%d %b %Y, %H:%M"))


@dataclass
class ReportData:
    meta: ReportMeta
    config: AnalysisConfig
    stage: str
    kpis: dict
    by_section: pd.DataFrame
    by_subject: pd.DataFrame
    matrix: pd.DataFrame
    grades: pd.DataFrame
    mst_trend: pd.DataFrame
    attendance: dict
    risk_students: pd.DataFrame | None
    risk_method: str | None
    risk_model_name: str | None
    co_table: pd.DataFrame | None
    co_settings: object | None
    quality_notes: list[str]
    clean: pd.DataFrame
    top: pd.DataFrame
    bottom: pd.DataFrame

    @property
    def at_risk(self) -> pd.DataFrame:
        if self.risk_students is None:
            return pd.DataFrame()
        flagged = self.risk_students[self.risk_students["risk_level"].isin(["High", "Medium"])]
        return flagged.reset_index(drop=True)


def _fmt(value: float | None, suffix: str = "%", digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:.{digits}f}{suffix}"


def build_report_data(
    df: pd.DataFrame,
    config: AnalysisConfig,
    meta: ReportMeta,
    risk=None,
    co_result: dict | None = None,
    quality=None,
) -> ReportData:
    """``df`` is the derived data in view; ``risk`` a RiskResult; ``co_result`` {'table', 'settings'}."""
    risk_students = risk.students if risk is not None else None
    at_risk_count = risk.at_risk_count if risk is not None else None
    quality_notes = []
    if quality is not None:
        quality_notes = [f"{i.message} ({i.count})" for i in quality.issues][:8]
    top, bottom = an.top_bottom(df, n=5)
    model_name = None
    if risk is not None and risk.model_report is not None:
        model_name = risk.model_report.chosen
    return ReportData(
        meta=meta,
        config=config,
        stage=assessment_stage(df),
        kpis=an.kpis(df, at_risk_count=at_risk_count),
        by_section=an.group_summary(df, ["section"]),
        by_subject=an.subject_difficulty(df),
        matrix=an.fail_rate_matrix(df),
        grades=an.grade_distribution(df, config.pass_pct),
        mst_trend=an.mst_trend_summary(df, config.mst_change_tolerance),
        attendance=an.attendance_performance(df, config.attendance_threshold),
        risk_students=risk_students,
        risk_method=risk.method if risk is not None else None,
        risk_model_name=model_name,
        co_table=(co_result or {}).get("table"),
        co_settings=(co_result or {}).get("settings"),
        quality_notes=quality_notes,
        clean=df,
        top=top,
        bottom=bottom,
    )


# --- Executive summary --------------------------------------------------------------


def key_facts(data: ReportData) -> list[str]:
    """Aggregate facts only — no student names — safe to send to an AI model."""
    k = data.kpis
    facts = [
        f"Scope: {data.meta.scope}. Stage: {'final results' if data.stage == 'final' else 'mid-semester'}.",
        f"{k['students']} students, {k['subjects']} subjects, {k['records']} student-subject records.",
        f"Pass rate {_fmt(k['pass_rate'])}, average score {_fmt(k['avg_score'])}, {k['fails']} failed records, "
        f"pass mark {data.config.pass_pct:g}%.",
    ]
    if not data.by_subject.empty:
        hardest = data.by_subject.iloc[0]
        easiest = data.by_subject.iloc[-1]
        facts.append(
            f"Hardest subject: {hardest['subject']} (fail rate {_fmt(hardest['fail_rate'])}, average "
            f"{_fmt(hardest['avg_score'])}). Easiest: {easiest['subject']} (fail rate {_fmt(easiest['fail_rate'])})."
        )
    if len(data.by_section) > 1:
        worst = data.by_section.sort_values("fail_rate", ascending=False).iloc[0]
        best = data.by_section.sort_values("fail_rate").iloc[0]
        facts.append(
            f"Section {worst['section']} has the highest fail rate ({_fmt(worst['fail_rate'])}); "
            f"Section {best['section']} the lowest ({_fmt(best['fail_rate'])})."
        )
    att = data.attendance
    if att.get("n") and att["below"].get("records") and att["above"].get("records"):
        facts.append(
            f"Below {data.config.attendance_threshold:g}% attendance the fail rate is "
            f"{_fmt(att['below']['fail_rate'])} vs {_fmt(att['above']['fail_rate'])} above it (r = {att['r']:.2f})."
        )
    if not data.mst_trend.empty:
        totals = data.mst_trend[["Improved", "Stable", "Declined"]].sum()
        facts.append(
            f"MST-1 to MST-2: {int(totals['Declined'])} student-subject pairs declined, "
            f"{int(totals['Improved'])} improved."
        )
    if data.risk_students is not None:
        counts = data.risk_students["risk_level"].value_counts()
        facts.append(
            f"At-risk students: {int(counts.get('High', 0))} high and {int(counts.get('Medium', 0))} medium risk "
            f"({'rules plus ML model' if data.risk_method == 'rules+ml' else 'rule-based'})."
        )
    if data.co_table is not None and not data.co_table.empty:
        below = data.co_table[~data.co_table["meets_target"]]["co"].tolist()
        facts.append(
            f"CO attainment: {len(data.co_table) - len(below)} of {len(data.co_table)} COs meet the target"
            + (f"; below target: {', '.join(below)}." if below else ".")
        )
    return facts


def template_summary(data: ReportData) -> str:
    """A readable paragraph built from the numbers (used when no AI model is configured)."""
    k = data.kpis
    parts = [
        f"This report covers {k['students']} students across {k['subjects']} subject"
        f"{'s' if k['subjects'] != 1 else ''} ({data.meta.scope.lower() if data.meta.scope else 'all students'})."
        f" The overall pass rate is {_fmt(k['pass_rate'])} with an average score of {_fmt(k['avg_score'])}."
    ]
    if not data.by_subject.empty and len(data.by_subject) > 1:
        hardest = data.by_subject.iloc[0]
        parts.append(
            f"{hardest['subject']} is the most difficult subject, with a fail rate of {_fmt(hardest['fail_rate'])}."
        )
    if len(data.by_section) > 1:
        worst = data.by_section.sort_values("fail_rate", ascending=False).iloc[0]
        parts.append(f"Section {worst['section']} needs the most attention ({_fmt(worst['fail_rate'])} fail rate).")
    att = data.attendance
    if att.get("n") and att["below"].get("records") and att["above"].get("records"):
        parts.append(
            f"Attendance matters: students below {data.config.attendance_threshold:g}% attendance fail "
            f"{_fmt(att['below']['fail_rate'])} of the time, against {_fmt(att['above']['fail_rate'])} for the rest."
        )
    if data.risk_students is not None:
        counts = data.risk_students["risk_level"].value_counts()
        high, medium = int(counts.get("High", 0)), int(counts.get("Medium", 0))
        if high or medium:
            parts.append(
                f"{high} students are at high risk and {medium} at medium risk; the attached list gives the "
                "reasons for each so mentors can follow up."
            )
    if data.co_table is not None and not data.co_table.empty:
        below = data.co_table[~data.co_table["meets_target"]]["co"].tolist()
        if below:
            verb = "falls" if len(below) == 1 else "fall"
            parts.append(f"{', '.join(below)} {verb} below the CO attainment target.")
        else:
            parts.append("All course outcomes meet the attainment target.")
    return " ".join(parts)


def llm_summary(model, data: ReportData) -> str:
    """Ask the configured AI model for a short summary. Only aggregate facts are sent."""
    prompt = (
        "Write an executive summary (90-140 words, one paragraph, plain text, no headings or bullet points) for a "
        "Head of Department about this class's results. Use only the facts below; do not invent numbers. "
        "End with one or two concrete, practical recommendations.\n\nFacts:\n- " + "\n- ".join(key_facts(data))
    )
    reply = model.complete(
        [
            {"role": "system", "content": "You are an academic analyst writing concise, factual reports."},
            {"role": "user", "content": prompt},
        ]
    )
    text = (reply.content or "").strip()
    if not text:
        raise ValueError("The AI model returned an empty summary.")
    return text
