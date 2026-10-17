"""Awkward but realistic marks sheets, used to fuzz the whole app.

Each entry is ``name -> (filename, bytes)``. They cover layouts, encodings and
content that real college sheets contain, plus a few hostile values.
"""

from __future__ import annotations

import io

import pandas as pd

INJECTION_NAME = '=HYPERLINK("http://evil.example","Click me")'
HTML_SUBJECT = "<img src=x onerror=alert(1)>"


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _xlsx(df: pd.DataFrame, header: bool = True) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, header=header)
    return buffer.getvalue()


def _students(n: int, subjects=("Python", "DBMS"), sections=("A", "B")) -> list[dict]:
    rows = []
    for i in range(n):
        for j, subject in enumerate(subjects):
            weak = i % 7 == 0
            rows.append(
                {
                    "Roll No": f"2024BCA{i:03d}",
                    "Student Name": f"Student {i}",
                    "Section": sections[i % len(sections)],
                    "Subject": subject,
                    "Attendance": 55 + (i * 7 + j) % 45,
                    "MST 1": 3 if weak else 6 + (i + j) % 9,
                    "MST 2": 2 if weak else 5 + (i * 3 + j) % 10,
                    "End Term": 12 if weak else 25 + (i * 5 + j) % 35,
                }
            )
    return rows


def build() -> dict[str, tuple[str, bytes]]:
    sheets: dict[str, tuple[str, bytes]] = {}
    base = pd.DataFrame(_students(40))

    sheets["only_totals"] = (
        "totals.csv",
        _csv(
            pd.DataFrame(
                {
                    "Roll No": [f"R{i}" for i in range(12)],
                    "Name": [f"N {i}" for i in range(12)],
                    "Total": [25 + i * 6 for i in range(12)],
                }
            )
        ),
    )
    sheets["one_student"] = ("one.csv", _csv(base.head(1)))
    all_absent = base.copy()
    all_absent["MST 1"] = "AB"
    sheets["all_absent_mst1"] = ("absent.csv", _csv(all_absent))
    multi = pd.concat([base.assign(Semester="3"), base.assign(Semester="5", **{"Roll No": base["Roll No"] + "X"})])
    sheets["multi_semester"] = ("multisem.csv", _csv(multi))
    sheets["numeric_sections"] = ("numsec.csv", _csv(base.assign(Section=base.index % 3 + 1)))

    special = base.copy()
    special["Subject"] = special["Subject"].map({"Python": "DBMS & SQL <Lab>", "DBMS": "C++ / OOP's"})
    special.loc[0:3, "Student Name"] = ["Seán O'Brien", "आरव शर्मा", "Zoë Ångström", "Rahul $harma"]
    sheets["special_characters"] = ("special.csv", _csv(special))

    hostile = base.copy()
    hostile.loc[0, "Student Name"] = INJECTION_NAME
    hostile.loc[1, "Student Name"] = "+SUM(1,2)"
    hostile.loc[2, "Student Name"] = "@cmd"
    hostile["Subject"] = hostile["Subject"].replace({"Python": HTML_SUBJECT})
    hostile.loc[hostile["Section"] == "B", "Section"] = "<b>B</b>"
    sheets["hostile_values"] = ("hostile.csv", _csv(hostile))

    sheets["latin1_semicolon"] = (
        "latin1.csv",
        base.assign(**{"Student Name": "Jürgen Müller"}).to_csv(index=False, sep=";").encode("latin-1"),
    )
    sheets["utf8_bom"] = ("bom.csv", b"\xef\xbb\xbf" + _csv(base))
    sheets["percent_strings"] = ("pct.csv", _csv(base.assign(Attendance=base["Attendance"].astype(str) + "%")))
    sheets["decimal_marks"] = ("decimals.csv", _csv(base.assign(**{"MST 1": base["MST 1"] + 0.5})))
    negative = base.copy()
    negative.loc[0:2, "MST 2"] = -5
    sheets["negative_marks"] = ("negative.csv", _csv(negative))

    wide = {"Roll No": [f"W{i}" for i in range(20)], "Name": [f"Wide {i}" for i in range(20)], "Sec": ["A", "B"] * 10}
    for code in ("CS101", "CS102"):
        wide[f"{code} MST1"] = [(i * 3) % 16 for i in range(20)]
        wide[f"{code} MST2"] = [(i * 5) % 16 for i in range(20)]
    sheets["wide_subject_codes"] = ("wide.csv", _csv(pd.DataFrame(wide)))

    big_scheme = base.assign(
        **{"MST 1 (30)": base["MST 1"] * 2, "MST 2 (30)": base["MST 2"] * 2, "End Term (100)": base["End Term"] + 30}
    ).drop(columns=["MST 1", "MST 2", "End Term"])
    sheets["different_marks_scheme"] = ("scheme.csv", _csv(big_scheme))

    sheets["header_only"] = ("empty_rows.csv", b"Roll No,Name,MST 1,MST 2\n")
    sheets["zero_bytes"] = ("nothing.csv", b"")
    sheets["not_a_table"] = ("notes.csv", b"just some notes\nwith no columns\n")
    sheets["no_marks_columns"] = ("nomarks.csv", _csv(pd.DataFrame({"Roll No": ["1", "2"], "Hostel": ["H1", "H2"]})))
    sheets["names_only_no_roll"] = ("noroll.csv", _csv(base.drop(columns=["Roll No"])))
    sheets["duplicate_headers"] = ("dupes.csv", b"Roll No,Name,MST 1,MST 1,End Term\n1,A B,10,12,40\n2,C D,5,6,20\n")

    everyone_passes = base.assign(**{"MST 1": 14, "MST 2": 14, "End Term": 55})
    sheets["everyone_passes"] = ("allpass.csv", _csv(everyone_passes))
    everyone_fails = base.assign(**{"MST 1": 1, "MST 2": 1, "End Term": 5})
    sheets["everyone_fails"] = ("allfail.csv", _csv(everyone_fails))

    numeric_rolls = base.assign(**{"Roll No": base["Roll No"].str[-3:].astype(int) + 101})  # one int per student
    sheets["xlsx_numeric_rolls"] = ("numeric.xlsx", _xlsx(numeric_rolls))
    big = pd.DataFrame(_students(1200, subjects=("Python", "DBMS", "DS", "CN", "Maths"), sections=("A", "B", "C", "D")))
    sheets["large_6000_rows"] = ("large.csv", _csv(big))
    return sheets


SHEETS = build()
# Sheets that cannot be analysed and must produce a friendly message instead.
UNREADABLE = {"header_only", "zero_bytes", "not_a_table", "no_marks_columns"}
