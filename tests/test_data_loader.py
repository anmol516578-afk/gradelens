import io

import pandas as pd
import pytest

from core.data_loader import LoadError, detect_header_row, list_sheets, read_path, read_table


def _xlsx_bytes(sheets: dict[str, pd.DataFrame], header=True) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False, header=header)
    return buf.getvalue()


def test_csv_keeps_markers_as_text():
    data = b"Roll No,Name,MST 1\n1,Asha Rao,AB\n2,Ravi Das,12\n"
    df = read_table(data, "marks.csv")
    assert list(df.columns) == ["Roll No", "Name", "MST 1"]
    assert df["MST 1"].tolist() == ["AB", "12"]


def test_title_rows_above_header_are_skipped():
    data = (
        b"Rayat Bahra University,,\n"
        b"Department of Computer Applications - Marks List,,\n"
        b",,\n"
        b"Roll No,Student Name,MST 1\n"
        b"1,Asha Rao,10\n"
    )
    df = read_table(data, "marks.csv")
    assert list(df.columns) == ["Roll No", "Student Name", "MST 1"]
    assert len(df) == 1


def test_semicolon_and_cp1252_csv():
    text = "Roll No;Student Name;Total\n1;José Kumar;55\n"
    df = read_table(text.encode("cp1252"), "marks.csv")
    assert df.loc[0, "Student Name"] == "José Kumar"
    assert df.loc[0, "Total"] == "55"


def test_xlsx_multi_sheet_selection():
    marks = pd.DataFrame({"Roll No": [1, 2], "Total": [55, 78]})
    notes = pd.DataFrame({"Info": ["hello"]})
    data = _xlsx_bytes({"Notes": notes, "Marks": marks})
    assert list_sheets(data, "book.xlsx") == ["Notes", "Marks"]
    df = read_table(data, "book.xlsx", sheet="Marks")
    assert df["Roll No"].tolist() == ["1", "2"]
    assert df["Total"].tolist() == ["55", "78"]


def test_list_sheets_for_csv():
    assert list_sheets(b"a,b\n1,2\n", "x.csv") == ["(CSV)"]


def test_empty_rows_and_columns_are_dropped():
    data = b"Roll No,Total,\n1,50,\n,,\n2,60,\n"
    df = read_table(data, "marks.csv")
    assert list(df.columns) == ["Roll No", "Total"]
    assert len(df) == 2


def test_duplicate_headers_are_deduplicated():
    df = read_table(b"Roll No,Total,Total\n1,50,60\n", "x.csv")
    assert list(df.columns) == ["Roll No", "Total", "Total.1"]


def test_errors_are_friendly():
    with pytest.raises(LoadError, match="Unsupported file type"):
        read_table(b"hello", "notes.txt")
    with pytest.raises(LoadError):
        read_table(b"Roll No,Total\n", "x.csv")
    with pytest.raises(LoadError, match="Excel"):
        list_sheets(b"not an excel file", "x.xlsx")
    with pytest.raises(LoadError, match="sheet"):
        read_table(_xlsx_bytes({"A": pd.DataFrame({"x": [1]})}), "x.xlsx", sheet="Missing")


def test_detect_header_row_defaults_to_zero_when_unclear():
    raw = pd.DataFrame([["1", "2"], ["3", "4"]])
    assert detect_header_row(raw) == 0


def test_xls_files_are_read(tmp_path):
    xlwt = pytest.importorskip("xlwt")
    book = xlwt.Workbook()
    sheet = book.add_sheet("Marks")
    for col, value in enumerate(["Roll No", "Total"]):
        sheet.write(0, col, value)
    sheet.write(1, 0, 1)
    sheet.write(1, 1, 55)
    path = tmp_path / "old.xls"
    book.save(str(path))
    df = read_path(path)
    assert list(df.columns) == ["Roll No", "Total"]
    assert df.loc[0, "Total"] in {"55", "55.0"}


def test_read_sample_files(tmp_path):
    from core.sample_data import write_sample_files

    paths = write_sample_files(tmp_path)
    long_df = read_path(paths["long"])
    assert len(long_df) == 901
    wide = read_path(paths["xlsx"], sheet="Wide")
    assert len(wide) == 180
