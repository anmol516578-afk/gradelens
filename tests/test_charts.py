import pandas as pd
import plotly.graph_objects as go

from core import analytics as an
from core.co_attainment import COSettings, compute_co_attainment, parse_question_marks
from ui import charts, theme


def test_dashboard_charts_build(derived, config):
    by_subject = an.group_summary(derived, ["subject"])
    fig = charts.fail_rate_bar(by_subject, "subject", horizontal=True, highlight="Python")
    assert isinstance(fig, go.Figure) and len(fig.data) == 1
    colors = dict(zip(fig.data[0].y, fig.data[0].marker.color, strict=True))
    assert colors["Python"] == theme.FAIL_COLOR and colors["DBMS"] == theme.FAIL_DIM
    assert fig.data[0].marker.opacity is None  # opacity arrays break click selection (DEBUG_LOG #6)

    heat = charts.fail_heatmap(an.fail_rate_matrix(derived))
    assert [t.type for t in heat.data] == ["heatmap", "scatter"]  # scatter overlay catches clicks
    assert len(heat.data[1].x) == 15 and heat.data[1].y[0].startswith("Section ")

    assert charts.score_histogram(derived, config.pass_pct).data
    grades = charts.grade_bar(an.grade_distribution(derived, config.pass_pct))
    assert list(grades.data[0].x) == an.GRADE_ORDER
    assert grades.data[0].marker.color[-1] == theme.STATUS["critical"]  # F is the fail status colour

    colors = theme.section_colors(["C", "A", "B"])
    scatter = charts.attendance_scatter(derived, config.attendance_threshold, config.pass_pct, colors)
    assert {t.name for t in scatter.data} == {"A", "B", "C", "Trend"}
    bands = an.attendance_performance(derived)["bands"]
    assert charts.attendance_band_bar(bands).data
    mst = charts.mst_trend_bar(an.mst_trend_summary(derived, config.mst_change_tolerance))
    assert [t.name for t in mst.data] == ["Declined", "Stable", "Improved"]


def test_section_colors_follow_entity_not_rank():
    full = theme.section_colors(["A", "B", "C"])
    assert full == {"A": theme.SERIES[0], "B": theme.SERIES[1], "C": theme.SERIES[2]}
    many = theme.section_colors([f"S{i:02d}" for i in range(10)])
    assert many["S09"] == theme.OTHER_COLOR  # a 9th+ series folds into 'other', never a new hue


def test_profile_charts(derived, config):
    profile = an.student_profile(derived, derived["roll_no"].iloc[0])
    subjects = profile["subjects"]
    radar = charts.profile_radar(subjects, profile["student_name"])
    assert len(radar.data) == 2 and radar.data[1].r[0] == radar.data[1].r[-1]  # closed outline
    assert len(charts.profile_components(subjects).data) == 3
    assert charts.profile_attendance(subjects, config.attendance_threshold).data


def test_co_and_risk_charts(bundle):
    marks = parse_question_marks(bundle.co_marks)
    table, _, _ = compute_co_attainment(marks, bundle.co_mapping)
    fig = charts.co_attainment_bar(table, COSettings())
    assert len(fig.layout.shapes) == 3  # level 1/2/3 lines
    risk = charts.risk_distribution(pd.Series({"High": 5, "Low": 100}))
    assert list(risk.data[0].y) == [5, 0, 100]
    imp = charts.feature_importance_bar(pd.DataFrame({"label": ["a", "b"], "importance": [0.2, 0.8]}))
    assert list(imp.data[0].y) == ["a", "b"]
    cm = charts.confusion_heatmap(pd.DataFrame([[10, 2], [1, 5]], index=["Pass", "Fail"], columns=["Pass", "Fail"]))
    assert cm.data[0].z.tolist() == [[10, 2], [1, 5]]


def test_auto_chart_rules():
    table = pd.DataFrame({"section": ["A", "B"], "fail_rate": [10.0, 20.0]})
    assert charts.auto_chart(table) is not None
    assert charts.auto_chart(pd.DataFrame({"a": [1, 2], "b": [3, 4]})) is None  # no category column
    assert charts.auto_chart(pd.DataFrame()) is None
    assert charts.auto_chart(None) is None
    long = pd.DataFrame({"name": [str(i) for i in range(50)], "v": range(50)})
    assert charts.auto_chart(long) is None
