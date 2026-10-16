import pandas as pd

from core.sample_data import SAMPLE_FILES, generate_all, generate_clean, write_sample_files


def test_generation_is_reproducible():
    a, b = generate_clean(seed=1), generate_clean(seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not generate_clean(seed=2).equals(a)


def test_realistic_patterns(bundle):
    df = bundle.clean_long
    assert len(df) == 900 and df["roll_no"].nunique() == 180
    fail_rate = (df["result"] == "Fail").mean()
    assert 0.05 <= fail_rate <= 0.15
    by_section = df.groupby("section")["result"].apply(lambda s: (s == "Fail").mean())
    assert by_section.idxmax() == "B"
    by_subject = df.groupby("subject")["result"].apply(lambda s: (s == "Fail").mean())
    assert by_subject.idxmax() == "Discrete Maths"
    assert df[["attendance_pct", "total"]].corr().iloc[0, 1] > 0.3
    decline = (df["mst1"] - df["mst2"]) / 15 * 100 > 10
    assert decline.mean() > 0.05
    assert df["roll_no"].map(df.groupby("roll_no")["student_name"].nunique()).eq(1).all()


def test_messy_long_contains_the_promised_mess(bundle):
    messy = bundle.messy_long
    assert len(messy) == 901  # one duplicate row
    assert messy.duplicated().sum() == 1
    values = set(messy[["MST 1", "MST 2", "End Term"]].to_numpy().ravel())
    assert {"AB", "Absent", "-"} <= values
    assert "51" in set(messy["MST 2"])
    assert {"a", "Sec A", " A "} & set(messy["Section"])
    assert (messy["Attendance (%)"] == "").sum() > 0
    assert (messy["Assignment"] == "").sum() > 0


def test_wide_and_co_shapes(bundle):
    wide = bundle.messy_wide
    assert len(wide) == 180
    assert "Python - MST1" in wide.columns and "Discrete Maths - End Term" in wide.columns
    co = bundle.co_marks
    assert list(co.columns[-6:]) == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    assert set(bundle.co_mapping["co"]) == {"CO1", "CO2", "CO3", "CO4"}


def test_write_sample_files(tmp_path):
    paths = write_sample_files(tmp_path)
    assert set(paths) == set(SAMPLE_FILES)
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
    assert len(pd.read_csv(paths["long"])) == 901
    assert len(pd.read_csv(paths["co_mapping"])) == 6
    book = pd.ExcelFile(paths["xlsx"])
    assert book.sheet_names == ["Read Me", "Marks", "Wide"]


def test_generate_all_smaller_class():
    small = generate_all(seed=3, students_per_section=10)
    assert small.clean_long["roll_no"].nunique() == 30
