"""End-to-end UI smoke tests with Streamlit's AppTest (no browser needed)."""

from __future__ import annotations

import io

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT
from ui import components as C

APP = str(ROOT / "app.py")
TIMEOUT = 90


def fresh() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    return at


def loaded(kind: str = "long") -> AppTest:
    at = fresh()
    at.selectbox(key="sample_choice").set_value(kind).run()
    at.button(key="load_sample").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def open_tab(at: AppTest, label: str) -> AppTest:
    at.session_state[C.TAB_KEY] = label
    at.run()
    assert not at.exception, (label, [e.value for e in at.exception])
    return at


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_empty_state_offers_sample_data():
    at = fresh()
    keys = [b.key for b in at.button]
    assert "load_sample_hero" in keys and "load_sample" in keys
    assert any("GradeLens" in m.value for m in at.markdown)


def test_hero_button_loads_sample():
    at = fresh()
    at.button(key="load_sample_hero").click().run()
    assert not at.exception
    assert metric(at, "Students") == "180"
    assert [t.label for t in at.tabs] == C.TABS


@pytest.mark.parametrize("label", C.TABS)
def test_every_tab_renders(label):
    at = open_tab(loaded(), label)
    assert not at.error, [e.value for e in at.error]


def test_wide_and_excel_samples_load():
    for kind in ("wide", "xlsx"):
        at = loaded(kind)
        assert metric(at, "Students") == "180", kind
    assert at.selectbox(key="sheet").value == "Marks"  # workbook sheet picker defaults to the marks sheet


def test_filters_and_pass_mark_update_kpis():
    at = loaded()
    assert metric(at, "Pass rate") == "89.4%"
    at.multiselect(key="f_sections").set_value(["B"]).run()
    assert metric(at, "Students") == "60"
    assert at.metric[1].delta is not None  # filtered view compares against the whole class
    at.button(key="reset_filters").click().run()
    assert metric(at, "Students") == "180"
    at.slider(key="cfg_pass_pct").set_value(50.0).run()
    assert metric(at, "Pass rate") != "89.4%"


def test_student_profile_navigation():
    at = open_tab(loaded(), C.TAB_PROFILE)
    first = at.selectbox(key="profile_student").value
    at.button(key="profile_next").click().run()
    assert at.selectbox(key="profile_student").value != first
    at.button(key="profile_prev").click().run()
    assert at.selectbox(key="profile_student").value == first
    assert any(m.label == "Average score" for m in at.metric)


def test_upload_csv_and_reshape():
    at = fresh()
    sheet = pd.DataFrame(
        {
            "Roll No": ["1", "2", "3", "4"],
            "Name": ["Asha Rao", "Ravi Das", "Meena Iyer", "Karan Gill"],
            "Sec": ["A", "a", "B", "B"],
            "Python - MST1": ["12", "AB", "5", "14"],
            "Python - MST2": ["13", "9", "4", "15"],
            "DBMS - MST1": ["10", "11", "3", "12"],
            "DBMS - MST2": ["11", "12", "2", "13"],
        }
    )
    at.sidebar.file_uploader[0].upload("class.csv", sheet.to_csv(index=False).encode(), "text/csv").run()
    assert not at.exception, [e.value for e in at.exception]
    assert metric(at, "Students") == "4"
    assert "Mid-semester" in " ".join(m.value for m in at.markdown)
    open_tab(at, C.TAB_QUALITY)
    assert metric(at, "Rows after cleaning") == "8"  # 4 students x 2 subjects


def test_upload_excel_with_title_rows():
    rows = [["Rayat Bahra University", "", ""], ["", "", ""], ["Roll No", "Subject", "Total"]]
    rows += [[str(i), "Python", str(30 + i * 5)] for i in range(8)]
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, header=False)
    at = fresh()
    at.sidebar.file_uploader[0].upload("marks.xlsx", buf.getvalue()).run()
    assert not at.exception
    assert metric(at, "Students") == "8"


def test_bad_file_shows_friendly_error():
    at = fresh()
    at.sidebar.file_uploader[0].upload("broken.xlsx", b"this is not excel").run()
    assert not at.exception
    assert any("Couldn't read that file" in e.value for e in at.error)


def test_unmappable_sheet_shows_mapping_editor():
    at = fresh()
    sheet = pd.DataFrame({"Colour": ["red", "blue"], "Hostel": ["H1", "H2"]})
    at.sidebar.file_uploader[0].upload("odd.csv", sheet.to_csv(index=False).encode()).run()
    assert not at.exception
    assert any("Roll No or Student Name" in e.value for e in at.error)
    assert len(at.button) >= 2  # Apply mapping / Reset buttons are offered
    assert any(b.label == "Apply mapping" for b in at.button)


def test_mapping_can_be_changed_and_reset():
    at = open_tab(loaded(), C.TAB_QUALITY)
    entries = list(at.session_state["mapping"])
    at.session_state["mapping"] = tuple((c, "ignore" if r == "attendance_pct" else r, s, h) for c, r, s, h in entries)
    at.run()
    assert not at.exception
    reset = next(b for b in at.button if b.label == "Reset to auto-detected")
    reset.click().run()
    assert not at.exception
    assert tuple(at.session_state["mapping"]) == tuple(at.session_state["mapping_auto"])


def test_marks_scheme_inferred_from_headers():
    at = fresh()
    sheet = pd.DataFrame(
        {"Roll No": [str(i) for i in range(30)], "Subject": ["Py"] * 30, "MST-1 (24)": [str(i % 25) for i in range(30)]}
    )
    at.sidebar.file_uploader[0].upload("m.csv", sheet.to_csv(index=False).encode()).run()
    assert not at.exception
    assert at.session_state["cfg_max_mst1"] == 24.0


def test_marks_scheme_table_follows_the_sheet():
    """A sheet with only MSTs is out of 30 internal marks, and says which components are missing."""
    at = fresh()
    sheet = pd.DataFrame(
        {
            "Roll No": [str(i) for i in range(6)],
            "Name": [f"S {i}" for i in range(6)],
            "Subject": ["Py"] * 6,
            "MST 1": ["15", "10", "8", "12", "5", "14"],
            "MST 2": ["15", "9", "7", "11", "6", "13"],
        }
    )
    at.sidebar.file_uploader[0].upload("mst.csv", sheet.to_csv(index=False).encode()).run()
    open_tab(at, C.TAB_QUALITY)
    table = next(df.value for df in at.dataframe if "Setting" in df.value.columns).set_index("Setting")["Value"]
    assert table["Internal total"] == "30" and table["Assignment"] == "not in this sheet"
    assert any("internal out of **30**" in c.value for c in at.sidebar.caption)
    assert metric(at, "Average score") == "69.4%"  # totals 30,19,15,23,11,27 -> 20.83 of 30
