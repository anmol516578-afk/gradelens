"""⚠️ At-Risk tab, plus the cached risk computation the whole app uses (KPI card, profiles, tables)."""

from __future__ import annotations

import io
import logging
from functools import partial

import joblib
import pandas as pd
import streamlit as st

from core.config import AnalysisConfig
from ml import risk_model as rm
from ui import charts, state
from ui import components as C
from ui.context import AppContext
from ui.theme import PLOTLY_CONFIG, risk_label

log = logging.getLogger("gradelens.risk")

REASON_GROUPS = (
    ("Failed", "Failed a subject"),
    ("Internal", "Internal marks below pass"),
    ("Attendance", "Low attendance"),
    ("MST-2 fell", "Sharp MST-2 drop"),
    ("Just above", "Just above pass in 2+ subjects"),
    ("Absent", "Absent in a test"),
    ("Model", "Model predicts a fail"),
)


# --- Cached computation ------------------------------------------------------------------


def _saved_model_mtime() -> float:
    path = rm.model_dir() / rm.MODEL_FILENAME
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner=False, max_entries=2)
def _cached_saved_model(mtime: float) -> rm.RiskModel | None:
    return rm.load_model() if mtime else None


@st.cache_resource(show_spinner=False, max_entries=4)
def _cached_train(df_all: pd.DataFrame, config_key: tuple) -> tuple[rm.RiskModel | None, str | None]:
    """Train once per sheet. A failure is cached too, so it is not retried on every click."""
    try:
        return rm.train(df_all), None
    except Exception as exc:  # noqa: BLE001 - the model is optional; rules still work
        log.exception("Risk model training failed")
        return None, f"the model could not be trained on this sheet ({type(exc).__name__}: {exc})"


def _training_key(config: AnalysisConfig) -> tuple:
    """Only what changes the training labels: the pass mark and the maxima.

    Keying on the whole config retrained the model (~3 s on 6,000 rows) every time the
    attendance slider moved, though attendance plays no part in training (DEBUG_LOG #44).
    """
    return (config.pass_pct, config.max_assignment, config.max_mst1, config.max_mst2, config.max_end_term,
            config.max_internal_total, config.max_total)  # fmt: skip


def get_model(df_all: pd.DataFrame, config: AnalysisConfig) -> tuple[rm.RiskModel | None, str | None]:
    """Train on this sheet when it has final results; otherwise fall back to a saved model, then rules."""
    ok, reason = rm.can_train(df_all)
    if ok:
        with st.spinner("Training the at-risk model on this sheet…"):
            return _cached_train(df_all, _training_key(config))
    saved = _cached_saved_model(_saved_model_mtime())
    if saved is not None and len(set(saved.features) & set(df_all.columns)) >= 2:
        return saved, None
    return None, reason


def _model_token(model: rm.RiskModel | None) -> tuple:
    if model is None:
        return ("none",)
    r = model.report
    return (r.source, r.trained_at, r.chosen, r.n_rows, tuple(model.features))


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_assess(df: pd.DataFrame, config_key: tuple, token: tuple, reason: str | None, _model) -> rm.RiskResult:
    return rm.assess(df, AnalysisConfig(**dict(config_key)), _model, fallback_reason=reason)


def compute_risk(df: pd.DataFrame, df_all: pd.DataFrame, config: AnalysisConfig) -> rm.RiskResult | None:
    """Risk for the students in view. Never raises: the rest of the app must work regardless."""
    try:
        model, reason = get_model(df_all, config)
        return _cached_assess(df, config.to_key(), _model_token(model), reason, model)
    except Exception:  # noqa: BLE001
        log.exception("Risk scoring with the model failed; falling back to rules")
    try:
        return rm.assess(df, config, None, fallback_reason="the model failed on this sheet")
    except Exception:  # noqa: BLE001
        log.exception("Rule-based risk scoring failed")
        return None


def reason_counts(students: pd.DataFrame) -> pd.DataFrame:
    flagged = students[students["risk_level"].isin([rm.HIGH, rm.MEDIUM])]
    rows = []
    for prefix, label in REASON_GROUPS:
        n = int(flagged["reasons"].map(lambda rs, p=prefix: any(r.startswith(p) for r in rs)).sum())
        if n:
            rows.append({"reason": label, "students": n})
    return pd.DataFrame(rows, columns=["reason", "students"])


# --- Rendering -------------------------------------------------------------------------


def _on_select(table_key: str, rolls_key: str) -> None:
    event = st.session_state.get(table_key)
    rows = event["selection"]["rows"] if event else []
    rolls = st.session_state.get(rolls_key, [])
    if rows and rows[0] < len(rolls):
        C.open_student(rolls[rows[0]])


def _save_model(model: rm.RiskModel) -> None:
    try:
        path = rm.save_model(model)
        st.session_state["risk_saved_msg"] = f"Saved to {path.name}. Mid-semester uploads will now use it."
    except OSError as exc:
        st.session_state["risk_saved_msg"] = f"Couldn't save the model: {exc}"


def _forget_model() -> None:
    path = rm.model_dir() / rm.MODEL_FILENAME
    path.unlink(missing_ok=True)
    st.session_state["risk_saved_msg"] = "Saved model removed."


def _model_bytes(model: rm.RiskModel) -> bytes:
    buffer = io.BytesIO()
    joblib.dump({"version": rm.MODEL_VERSION, "pipeline": model.pipeline, "features": model.features,
                 "report": model.report, "feature_means": model.feature_means,
                 "feature_stds": model.feature_stds}, buffer)  # fmt: skip
    return buffer.getvalue()


def _banner(risk: rm.RiskResult) -> None:
    if risk.method == "rules+ml" and risk.model_report is not None:
        r = risk.model_report
        m = r.chosen_metrics
        where = "trained on this sheet" if r.source == "trained" else f"saved model from {r.trained_at}"
        st.success(
            f"**Rules + ML model** — {r.chosen.lower()} ({where}), F1 {m['f1']:.2f} and ROC-AUC {m['roc_auc']:.2f} "
            f"in {r.folds}-fold cross-validation. Risk level is the higher of the two.",
            icon="🤖",
        )
    else:
        st.info(
            f"**Rule-based scores.** The ML model isn't used because {risk.fallback_reason or 'none is available'}. "
            "The rules below still flag students early.",
            icon="📏",
        )


def _how_it_works(config: AnalysisConfig) -> None:
    with st.expander("How the risk score works"):
        p = rm.POINTS
        pass_mark, required, internal_cap = config.pass_pct, config.attendance_threshold, p["internal_cap"]
        st.markdown(
            f"""
Each student gets points for early-warning signals across the subjects in view (capped at 100):

| Signal | Points |
|---|---|
| Failed a subject | {p["failed_subject"]} each (max {p["failed_cap"]}) |
| Internal marks below the {pass_mark:g}% pass mark | {p["internal_below_pass"]} per subject (max {internal_cap}) |
| Attendance below {required:g}% | {p["attendance_low"]} ({p["attendance_very_low"]} if 15+ points below) |
| Attendance low in some subjects only | {p["attendance_some_subjects"]} |
| MST-2 at least {rm.MST_DROP_AT:g} points below MST-1 | {p["mst_drop"]} per subject (max {p["mst_drop_cap"]}) |
| Passing by less than 10 points in 2+ subjects | {p["borderline_multiple"]} |
| Absent in an MST or assignment | {p["absent"]} |

**High** ≥ {rm.HIGH_AT} points · **Medium** ≥ {rm.MEDIUM_AT} · **Low** below that.

When the ML model is available it estimates each student's chance of failing a subject they have **not**
already failed, from attendance, assignment, MST-1 and MST-2 only. A probability of
{rm.ML_HIGH_AT:.0%}+ counts as High and {rm.ML_MEDIUM_AT:.0%}+ as Medium; the final level is the higher of the
rule level and the model level.
"""
        )


def _model_section(ctx: AppContext, risk: rm.RiskResult) -> None:
    st.divider()
    C.heading(
        "🤖 Prediction model", "Learns which early indicators come before a fail, so it can warn earlier next time."
    )
    report = risk.model_report
    if report is None:
        ok, reason = rm.can_train(ctx.derived_all)
        st.info(
            f"No model for this sheet: {risk.fallback_reason or reason}. Upload a sheet with **end-term results** "
            f"(at least {rm.MIN_ROWS} graded records and {rm.MIN_FAILS} fails) to train one, then save it to score "
            "future mid-semester sheets.",
            icon="ℹ️",
        )
        return

    metrics = report.metrics.copy()
    metrics.insert(0, "chosen", metrics["model"].eq(report.chosen).map({True: "✅", False: ""}))
    st.markdown(f"**Model comparison** · {report.n_rows:,} records, {report.n_fails} fails, "
                f"stratified {report.folds}-fold CV")  # fmt: skip
    pct = {"format": "%.2f"}
    st.dataframe(
        metrics[["chosen", "model", "f1", "roc_auc", "precision", "recall", "accuracy"]],
        hide_index=True,
        width="stretch",
        column_config={
            "chosen": st.column_config.TextColumn("", width="small"),
            "model": "Model",
            "f1": st.column_config.NumberColumn("F1", help="Balance of precision and recall (chosen by this)", **pct),
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", help="Ranking quality, 0.5 = random", **pct),
            "precision": st.column_config.NumberColumn(
                "Precision", help="Of predicted fails, share that failed", **pct
            ),
            "recall": st.column_config.NumberColumn("Recall", help="Of actual fails, share the model caught", **pct),
            "accuracy": st.column_config.NumberColumn("Accuracy", **pct),
        },
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Confusion matrix** (out-of-fold)")
        st.plotly_chart(charts.confusion_heatmap(report.confusion), config=PLOTLY_CONFIG, key="chart_confusion")
        cm = report.confusion
        caught, actual = int(cm.loc["Fail", "Fail"]), int(cm.loc["Fail"].sum())
        false_alarms = int(cm.loc["Pass", "Fail"])
        st.caption(f"Caught {caught} of {actual} actual fails; {false_alarms} false alarms among passes.")
    with right:
        st.markdown("**What the model relies on**")
        st.plotly_chart(charts.feature_importance_bar(report.importance), config=PLOTLY_CONFIG, key="chart_importance")

    model, _ = get_model(ctx.derived_all, ctx.config)
    b1, b2, b3 = st.columns(3)
    if report.source == "trained" and model is not None:
        b1.button("💾 Save for mid-semester use", key="risk_save", on_click=_save_model, args=(model,), width="stretch")
        b2.download_button("Download model (.joblib)", _model_bytes(model), file_name=rm.MODEL_FILENAME,
                           key="risk_download", width="stretch")  # fmt: skip
    if _saved_model_mtime():
        b3.button("Forget saved model", key="risk_forget", on_click=_forget_model, width="stretch")
    message = st.session_state.pop("risk_saved_msg", None)
    if message:
        st.toast(message, icon="💾")
    st.caption("Saved models are only ever loaded from the app's own models folder — never from uploads.")


def render(ctx: AppContext) -> None:
    risk: rm.RiskResult | None = ctx.risk
    C.heading(
        "Students who may need support",
        "⚖️ Decision support, not a final judgement — talk to the student and check the context before acting.",
    )
    if risk is None:
        st.warning("Risk scores couldn't be calculated for this sheet. Check the 🧹 Data Quality tab.", icon="⚠️")
        return
    if risk.students.empty:
        st.info("No students in view.", icon="🔍")
        return
    _banner(risk)

    counts = risk.counts
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Students by risk level**")
        st.plotly_chart(charts.risk_distribution(counts), config=PLOTLY_CONFIG, key="chart_risk_levels")
    with right:
        st.markdown("**Why students are flagged** (High + Medium)")
        reasons = reason_counts(risk.students)
        if reasons.empty:
            st.success("Nobody is flagged as High or Medium risk.", icon="🎉")
        else:
            fig = charts.reason_bar(reasons)
            st.plotly_chart(fig, config=PLOTLY_CONFIG, key="chart_risk_reasons")

    st.markdown("**Student list** · select a row to open the profile")
    f1, f2, f3 = st.columns([2, 1.4, 2], vertical_alignment="bottom")
    levels = f1.multiselect("Risk level", list(rm.LEVELS), default=[rm.HIGH, rm.MEDIUM], key="risk_levels",
                            format_func=risk_label)  # fmt: skip
    sections = [state.ALL, *sorted(risk.students["section"].unique())]
    section = f2.selectbox("Section", sections, key="risk_section")
    query = f3.text_input("Search name or roll no", key="risk_search", placeholder="e.g. Kulkarni")

    table = risk.students
    if levels:
        table = table[table["risk_level"].isin(levels)]
    if section != state.ALL:
        table = table[table["section"] == section]
    if query:
        q = query.strip().lower()
        table = table[table["student_name"].str.lower().str.contains(q, regex=False)
                      | table["roll_no"].str.lower().str.contains(q, regex=False)]  # fmt: skip
    table = table.reset_index(drop=True)
    view = table.assign(
        risk=table["risk_level"].map(risk_label),
        why=table["reasons"].map(lambda rs: " · ".join(rs) if rs else "No warning signs"),
        ml_pct=table["ml_prob"] * 100,
    )
    rolls_key = "_risk_rolls"
    st.session_state[rolls_key] = view["roll_no"].tolist()
    columns = ["risk", "risk_score", "roll_no", "student_name", "section", "why", "avg_score", "avg_attendance",
               "fails"]  # fmt: skip
    if risk.method == "rules+ml":
        columns.insert(2, "ml_pct")
    table_key = "risk_table"
    st.dataframe(
        view[columns],
        hide_index=True,
        width="stretch",
        height=min(460, 38 + 35 * max(len(view), 1)),
        key=table_key,
        on_select=partial(_on_select, table_key, rolls_key),
        selection_mode="single-row",
        column_config={
            "risk": st.column_config.TextColumn("Risk", width="small"),
            "risk_score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%d", width="small"
            ),  # fmt: skip
            "ml_pct": st.column_config.NumberColumn(
                "Model fail %", format="%.0f%%", help="Highest predicted chance of failing a subject not yet failed"
            ),
            "roll_no": "Roll No",
            "student_name": "Name",
            "section": st.column_config.TextColumn("Sec", width="small"),
            "why": st.column_config.TextColumn("Why", width="large"),
            "avg_score": st.column_config.NumberColumn("Avg score", format="%.1f%%"),
            "avg_attendance": st.column_config.NumberColumn("Attendance", format="%.0f%%"),
            "fails": st.column_config.NumberColumn("Fails", help="Subjects failed"),
        },
    )
    export = view[["roll_no", "student_name", "section", "risk_level", "risk_score", "why", "avg_score",
                   "avg_attendance", "fails"]].rename(columns={"why": "reasons"})  # fmt: skip
    st.caption(f"{len(view):,} student(s) shown.")
    C.download_csv(export, "at_risk_students.csv", "Download this list", key="dl_risk")
    _how_it_works(ctx.config)
    _model_section(ctx, risk)
