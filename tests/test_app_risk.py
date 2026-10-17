"""At-Risk tab and risk-driven KPIs through AppTest."""

from __future__ import annotations

import pandas as pd

from ml import risk_model as rm
from tests.test_app_smoke import fresh, loaded, metric, open_tab
from ui import components as C


def test_kpi_and_tab_use_rules_plus_ml_on_sample():
    at = loaded()
    assert metric(at, "At-risk students") == "92"
    open_tab(at, C.TAB_RISK)
    assert not at.error
    assert any("Rules + ML model" in s.value for s in at.success)
    assert any("Decision support" in c.value for c in at.caption)
    assert any(b.key == "risk_save" for b in at.button)
    table = next(df for df in at.dataframe if "why" in df.value.columns)
    assert len(table.value) == 92  # default filter: High + Medium
    assert table.value["why"].str.len().gt(0).all()


def test_level_filter_and_search():
    at = open_tab(loaded(), C.TAB_RISK)
    at.multiselect(key="risk_levels").set_value(["High"]).run()
    table = next(df for df in at.dataframe if "why" in df.value.columns).value
    assert set(table["risk"]) == {"🔴 High"}
    at.text_input(key="risk_search").input("zzz-nobody").run()
    table = next(df for df in at.dataframe if "why" in df.value.columns).value
    assert table.empty


def test_save_model_then_mid_semester_upload_uses_it(isolated_model_dir):
    at = open_tab(loaded(), C.TAB_RISK)
    at.button(key="risk_save").click().run()
    assert (isolated_model_dir / rm.MODEL_FILENAME).exists()

    # Next: a mid-semester sheet (no end term) for new students.
    rows = []
    for i in range(30):
        weak = i % 5 == 0
        rows.append({"Roll No": f"2025BCA{i:03d}", "Name": f"New Student {i}", "Section": "A", "Subject": "Python",
                     "Attendance": 60 if weak else 90, "Assignment": 3 if weak else 8, "MST 1": 4 if weak else 12,
                     "MST 2": 3 if weak else 12})  # fmt: skip
    at2 = fresh()
    at2.sidebar.file_uploader[0].upload("midsem.csv", pd.DataFrame(rows).to_csv(index=False).encode()).run()
    assert not at2.exception
    open_tab(at2, C.TAB_RISK)
    assert any("saved model" in s.value for s in at2.success)
    assert metric(at2, "At-risk students") == "6"
    # Clean up so other tests see no saved model.
    (isolated_model_dir / rm.MODEL_FILENAME).unlink()


def test_small_upload_falls_back_to_rules():
    rows = [{"Roll No": str(i), "Name": f"S {i}", "Subject": "Python", "MST 1": 5 + i, "MST 2": 4 + i,
             "End Term": 20 + i} for i in range(8)]  # fmt: skip
    at = fresh()
    at.sidebar.file_uploader[0].upload("small.csv", pd.DataFrame(rows).to_csv(index=False).encode()).run()
    open_tab(at, C.TAB_RISK)
    assert any("Rule-based scores" in i.value and "too little data" in i.value for i in at.info)
