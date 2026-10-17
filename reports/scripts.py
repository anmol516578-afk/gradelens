"""Indian scripts in the PDF report and charts.

DejaVu Sans (bundled with matplotlib) covers Latin, but not Devanagari, Gurmukhi, Tamil, etc., which
rendered as empty boxes (DEBUG_LOG #62). Two things are needed for them:

1. **Fonts**: Noto fonts for nine Indian scripts are bundled in ``reports/fonts``.
2. **Shaping**: these scripts reorder and combine characters (e.g. the vowel sign in "कि" is drawn
   before the consonant). ReportLab does this through HarfBuzz (``uharfbuzz``), but only reliably
   when each font run contains characters that font has. The Noto fonts have no Latin letters or
   punctuation, so text is split into runs by script: Indian-script characters go to their Noto
   font, everything else (spaces, digits, commas, Latin) stays in DejaVu Sans.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

FONT_DIR = Path(__file__).with_name("fonts")

# (code, Unicode ranges, font file stem). Codes are ISO 15924.
SCRIPTS: tuple[tuple[str, tuple[tuple[int, int], ...], str], ...] = (
    ("Deva", ((0x0900, 0x097F), (0xA8E0, 0xA8FF), (0x1CD0, 0x1CFF)), "NotoSansDevanagari"),
    ("Beng", ((0x0980, 0x09FF),), "NotoSerifBengali"),
    ("Guru", ((0x0A00, 0x0A7F),), "NotoSansGurmukhi"),
    ("Gujr", ((0x0A80, 0x0AFF),), "NotoSansGujarati"),
    ("Orya", ((0x0B00, 0x0B7F),), "NotoSansOriya"),
    ("Taml", ((0x0B80, 0x0BFF),), "NotoSansTamil"),
    ("Telu", ((0x0C00, 0x0C7F),), "NotoSansTelugu"),
    ("Knda", ((0x0C80, 0x0CFF),), "NotoSansKannada"),
    ("Mlym", ((0x0D00, 0x0D7F),), "NotoSansMalayalam"),
)
JOINERS = {"‌", "‍"}  # zero-width (non-)joiner: part of the word they sit in
_INDIC = re.compile("[" + "".join(f"\\u{a:04x}-\\u{b:04x}" for _, ranges, _ in SCRIPTS for a, b in ranges) + "]")


def has_indic(text: str) -> bool:
    return bool(_INDIC.search(text or ""))


def script_of(char: str) -> str | None:
    code = ord(char)
    for script, ranges, _ in SCRIPTS:
        if any(a <= code <= b for a, b in ranges):
            return script
    return None


def runs(text: str) -> list[tuple[str | None, str]]:
    """Consecutive characters grouped by script (None = Latin and everything else)."""
    out: list[tuple[str | None, str]] = []
    for char in text:
        script = script_of(char)
        if script is None and char in JOINERS and out and out[-1][0] is not None:
            script = out[-1][0]
        if out and out[-1][0] == script:
            out[-1] = (script, out[-1][1] + char)
        else:
            out.append((script, char))
    return out


def reportlab_font(script: str, bold: bool) -> str:
    return f"GL-{script}{'-Bold' if bold else ''}"


@lru_cache(maxsize=1)
def register_reportlab_fonts() -> bool:
    """Register every bundled Indian-script font with ReportLab (once per process)."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for script, _, stem in SCRIPTS:
        for bold in (False, True):
            path = FONT_DIR / f"{stem}-{'Bold' if bold else 'Regular'}.ttf"
            pdfmetrics.registerFont(TTFont(reportlab_font(script, bold), str(path)))
    shape_each_font_separately()
    return True


def shape_each_font_separately() -> bool:
    """Make ReportLab shape every same-font stretch of a word on its own.

    ReportLab shapes a whole word (text between spaces) with the font of its *first* fragment
    (``reportlab.pdfbase.ttfonts.shapeFragWord``). A word that switches font with no space, such as
    "शर्मा," or "(ब)", was shaped with the wrong font and its glyphs overlapped. This wraps the
    function ``Paragraph`` calls so each font's stretch is shaped with that font; words in a single
    font go through ReportLab unchanged. Returns False if this ReportLab version has no such hook.
    """
    from reportlab.pdfbase import pdfmetrics, ttfonts
    from reportlab.platypus import paragraph

    original = getattr(paragraph, "shapeFragWord", None)
    if original is None or not hasattr(ttfonts, "makeShapedFragWord"):
        return False
    if getattr(original, "gradelens_per_font", False):
        return True

    def shape(word, *args, **kwargs):
        pieces = word[1:]
        if (
            isinstance(word, ttfonts.ShapedFragWord)
            or any(hasattr(frag, "cbDefn") for frag, _ in pieces)
            or len({frag.fontName for frag, _ in pieces}) < 2
        ):
            return original(word, *args, **kwargs)
        groups: list[list] = []
        for frag, text in pieces:
            if groups and groups[-1][-1][0].fontName == frag.fontName:
                groups[-1].append((frag, text))
            else:
                groups.append([(frag, text)])
        width, out, changed = 0.0, [], False
        for group in groups:
            sub = [sum(pdfmetrics.stringWidth(t, f.fontName, f.fontSize) for f, t in group), *group]
            result = original(sub, *args, **kwargs)
            changed = changed or result is not sub
            width += result[0]
            out.extend(result[1:])
        return ttfonts.makeShapedFragWord(word)([width, *out]) if changed else word

    shape.gradelens_per_font = True
    paragraph.shapeFragWord = shape
    return True


def markup(text: str, bold: bool = False) -> str:
    """Paragraph markup with each Indian-script run in its own font. Everything is XML-escaped."""
    if not has_indic(text):
        return escape(text)
    register_reportlab_fonts()
    parts = []
    for script, chunk in runs(text):
        if script is None:
            parts.append(escape(chunk))
        else:
            parts.append(f'<font name="{reportlab_font(script, bold)}">{escape(chunk)}</font>')
    return "".join(parts)


@lru_cache(maxsize=1)
def configure_matplotlib() -> list[str]:
    """Add the Indian-script fonts as fallbacks after DejaVu Sans for chart labels.

    Matplotlib draws the glyphs but does not shape them, so vowel signs in chart labels may sit
    next to (rather than around) their consonant; the PDF text itself is shaped correctly.
    """
    from matplotlib import font_manager, rcParams

    families = []
    for _, _, stem in SCRIPTS:
        path = FONT_DIR / f"{stem}-Regular.ttf"
        font_manager.fontManager.addfont(str(path))
        families.append(font_manager.FontProperties(fname=str(path)).get_name())
    rcParams["font.family"] = ["DejaVu Sans", *families]
    return families
