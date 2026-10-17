"""Indian scripts in the PDF report (DEBUG_LOG #62): bundled fonts, script runs and shaping."""

from __future__ import annotations

import io

import pytest
from pypdf import PdfReader

from reports import pdf_report, scripts
from reports.data import ReportMeta, build_report_data
from reports.pdf_report import build_pdf

NAMES = {
    "Deva": "आरव शर्मा",
    "Beng": "অনন্যা দাস",
    "Guru": "ਗੁਰਪ੍ਰੀਤ ਸਿੰਘ",
    "Gujr": "હેત પટેલ",
    "Orya": "ସ୍ମିତା ମହାନ୍ତି",
    "Taml": "கார்த்திக் ராஜா",
    "Telu": "శ్రీనివాస్ రెడ్డి",
    "Knda": "ಅನನ್ಯ ಗೌಡ",
    "Mlym": "അഞ്ജലി നായർ",
}


def test_every_script_has_both_bundled_fonts():
    for _, _, stem in scripts.SCRIPTS:
        for weight in ("Regular", "Bold"):
            assert (scripts.FONT_DIR / f"{stem}-{weight}.ttf").stat().st_size > 10_000
    assert (scripts.FONT_DIR / "OFL.txt").exists()


def test_runs_split_by_script_and_keep_joiners_in_the_word():
    assert scripts.runs("Ram (राम), ਰਾਮ") == [
        (None, "Ram ("),
        ("Deva", "राम"),
        (None, "), "),
        ("Guru", "ਰਾਮ"),
    ]
    # Zero-width joiner/non-joiner belong to the Indian-script run they sit in.
    assert scripts.runs("क्‍ष") == [("Deva", "क्‍ष")]
    for script, name in NAMES.items():
        assert {s for s, _ in scripts.runs(name) if s} == {script}


def test_markup_escapes_and_wraps_only_indian_runs():
    assert scripts.markup("A & <B>") == "A &amp; &lt;B&gt;"
    out = scripts.markup("<b>राम</b> & co", bold=True)
    assert out == '&lt;b&gt;<font name="GL-Deva-Bold">राम</font>&lt;/b&gt; &amp; co'


@pytest.mark.parametrize("text", ["डॉ. मीरा, (क) Seán", 'शर्मा\'s "मीरा" [ब]-(क).', "ગુજરાતી, (હેત)"])
def test_punctuation_touching_indian_text_keeps_its_own_glyphs(text):
    """ReportLab shapes a word with its first font; "मीरा," used to turn the comma into "Ή"."""
    styles = pdf_report._styles()
    para = pdf_report._para(text, styles["base"])
    para.wrap(1000, 100)
    pieces = [(w.fontName, w.text) for w in para.blPara.lines[0].words]
    latin = "".join(t for font, t in pieces if not font.startswith("GL-"))
    expected = "".join(chunk for script, chunk in scripts.runs(text) if script is None)
    assert latin.replace(" ", "") == expected.replace(" ", "")
    # No unmapped (private-use) glyphs where a plain letter should be.
    assert not any("" <= ch <= "" for font, t in pieces if not font.startswith("GL-") for ch in t)


def test_single_font_words_are_shaped_by_reportlab_unchanged():
    from reportlab.platypus import paragraph

    assert scripts.shape_each_font_separately()
    assert getattr(paragraph.shapeFragWord, "gradelens_per_font", False)
    styles = pdf_report._styles()
    para = pdf_report._para("कि", styles["base"])
    para.wrap(1000, 100)
    (piece,) = para.blPara.lines[0].words
    # Shaped by HarfBuzz: the vowel sign "ि" is drawn before the consonant it follows in the text.
    assert piece.fontName == "GL-Deva" and type(piece.text).__name__ == "ShapedStr"
    assert len(piece.text) == 2 and piece.text[-1] == "क"


def test_pdf_embeds_a_font_for_each_indian_script(derived, config):
    frame = derived.copy()
    rolls = frame["roll_no"].drop_duplicates().tolist()
    for roll, name in zip(rolls[: len(NAMES)], NAMES.values(), strict=True):
        frame.loc[frame["roll_no"] == roll, "student_name"] = name
    meta = ReportMeta(college="राजकीय महाविद्यालय, जम्मू", department="ਕੰਪਿਊਟਰ ਵਿਭਾਗ", title="छात्र प्रदर्शन रिपोर्ट")
    data = build_report_data(frame, config, meta)
    pdf = build_pdf(data, "सारांश: " + ", ".join(NAMES.values()) + ".")
    reader = PdfReader(io.BytesIO(pdf))
    fonts = {
        str(font.get_object()["/BaseFont"]).split("+")[-1]
        for page in reader.pages
        for font in page["/Resources"].get("/Font", {}).values()
    }
    for _, _, stem in scripts.SCRIPTS:
        assert f"{stem}-Regular" in fonts, (stem, sorted(fonts))
    assert "NotoSansDevanagari-Bold" in fonts  # the title
