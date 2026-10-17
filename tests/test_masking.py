from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import pytest

from agent.masking import ID_COLUMN, Pseudonymizer


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "roll_no": ["2023BCA002", "2023BCA001", "2023BCA001", "12"],
            "student_name": ["Ravi Das", "Asha Rao", "Asha Rao", "Meena Iyer"],
            "section": ["A", "A", "A", "B"],
            "subject": ["Python", "Python", "DBMS", "Python"],
            "score_pct": [70.0, 55.0, 61.0, 30.0],
            "result": ["Pass", "Pass", "Pass", "Fail"],
        }
    )


def test_ids_are_stable_and_sorted_by_roll(frame):
    pseudo = Pseudonymizer.from_frame(frame)
    assert pseudo.roll_to_id == {"12": "S001", "2023BCA001": "S002", "2023BCA002": "S003"}
    # Built from the full data, so a filtered view keeps the same IDs.
    masked = pseudo.mask_frame(frame[frame["section"] == "A"])
    assert set(masked[ID_COLUMN]) == {"S002", "S003"}


def test_masked_frame_has_no_identity_columns(frame):
    masked = Pseudonymizer.from_frame(frame).mask_frame(frame)
    assert "roll_no" not in masked and "student_name" not in masked
    text = masked.to_csv()
    for name in frame["student_name"]:
        assert name not in text
    assert list(masked.columns)[0] == ID_COLUMN


def test_mask_text_handles_case_and_spacing(frame):
    pseudo = Pseudonymizer.from_frame(frame)
    masked = pseudo.mask_text("How is asha   RAO doing vs Ravi Das? Check 2023bca002 too.")
    assert masked == "How is S002 doing vs S003? Check S003 too."


def test_mask_text_leaves_ordinary_numbers_alone(frame):
    pseudo = Pseudonymizer.from_frame(frame)
    # "12" is a roll number, but too short to be identifying: it must not turn "top 12" into an ID.
    assert pseudo.mask_text("Show the top 12 students in Python") == "Show the top 12 students in Python"


def test_unmask_text_and_table(frame):
    pseudo = Pseudonymizer.from_frame(frame)
    assert pseudo.unmask_text("S002 scored lowest; S999 unknown") == "Asha Rao (2023BCA001) scored lowest; S999 unknown"
    table = pd.DataFrame({ID_COLUMN: ["S003", "S001"], "score_pct": [70.0, 30.0], "note": ["beat S001", None]})
    shown = pseudo.unmask_table(table)
    assert list(shown.columns) == ["roll_no", "student_name", "score_pct", "note"]
    assert shown["student_name"].tolist() == ["Ravi Das", "Meena Iyer"]
    assert shown["note"].iloc[0] == "beat Meena Iyer"
    assert pd.isna(shown["note"].iloc[1])


def test_unmask_figure_keeps_numeric_data_intact(frame):
    pseudo = Pseudonymizer.from_frame(frame)
    fig = px.bar(x=["S001", "S002"], y=[30.5, 55.25], title="Lowest: S001")
    original = json.loads(fig.to_json())
    restored = json.loads(pseudo.unmask_figure_json(fig.to_json()))
    assert restored["data"][0]["x"] == ["Meena Iyer", "Asha Rao"]
    assert restored["layout"]["title"]["text"] == "Lowest: Meena Iyer"
    # Plotly stores numbers as base64 "bdata"; it must survive byte-for-byte.
    assert restored["data"][0]["y"] == original["data"][0]["y"]
    import plotly.io as pio

    pio.from_json(json.dumps(restored))  # still a valid figure


def test_real_names_on_sample_data(derived):
    pseudo = Pseudonymizer.from_frame(derived)
    masked = pseudo.mask_frame(derived)
    assert masked[ID_COLUMN].nunique() == derived["roll_no"].nunique() == 180
    blob = masked.to_csv()
    assert not any(name in blob for name in pseudo.real_names())


def test_short_or_ambiguous_names_do_not_corrupt_questions():
    frame = pd.DataFrame(
        {
            "roll_no": ["R0001", "R0002", "R0003"],
            "student_name": ["A", "Python", "Kiran Bedi"],
            "section": ["A", "B", "B"],
            "subject": ["Python", "Python", "Python"],
            "score_pct": [50.0, 60.0, 70.0],
            "result": ["Pass", "Pass", "Pass"],
        }
    )
    pseudo = Pseudonymizer.from_frame(frame)
    masked = pseudo.mask_text("Fail rate in Section A for Python? And Kiran Bedi?")
    assert masked == "Fail rate in Section A for Python? And S003?"
