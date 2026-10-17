"""PDF report for the HOD (ReportLab + matplotlib charts).

Uses DejaVu Sans, which ships with matplotlib, so symbols such as "≥" and
"→" render correctly even in a slim Docker image with no system fonts.
"""

from __future__ import annotations

import io
import math
import os
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from core.palette import RISK_COLORS, SCORE_COLOR, STATUS
from reports import mpl_charts
from reports.data import AT_RISK_LIMIT, ReportData

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
BLUE = colors.HexColor(SCORE_COLOR)
INK = colors.HexColor("#1f1f1f")
MUTED = colors.HexColor("#6b6963")
ZEBRA = colors.HexColor("#f3f5f9")
RULE = colors.HexColor("#d9d7d0")

_FONTS_READY = False


def _register_fonts() -> tuple[str, str]:
    global _FONTS_READY
    if not _FONTS_READY:
        font_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("GL", os.path.join(font_dir, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("GL-Bold", os.path.join(font_dir, "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("GL", normal="GL", bold="GL-Bold", italic="GL", boldItalic="GL-Bold")
        _FONTS_READY = True
    return "GL", "GL-Bold"


def _styles() -> dict[str, ParagraphStyle]:
    regular, bold = _register_fonts()
    base = ParagraphStyle("base", fontName=regular, fontSize=9, leading=12.5, textColor=INK, alignment=TA_LEFT)
    return {
        "base": base,
        "title": ParagraphStyle("title", parent=base, fontName=bold, fontSize=20, leading=24, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", parent=base, fontSize=10.5, leading=14, textColor=MUTED),
        "h2": ParagraphStyle(
            "h2", parent=base, fontName=bold, fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=5, textColor=INK
        ),
        "small": ParagraphStyle("small", parent=base, fontSize=7.5, leading=10, textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=base, fontSize=7.8, leading=10),
        "header": ParagraphStyle(
            "header", parent=base, fontName=bold, fontSize=7.8, leading=10, textColor=colors.white
        ),
        "kpi_value": ParagraphStyle("kpi_value", parent=base, fontName=bold, fontSize=16, leading=19),
        "kpi_label": ParagraphStyle("kpi_label", parent=base, fontSize=7.5, leading=9.5, textColor=MUTED),
    }


def _pct(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:.{digits}f}%"


def _table(rows: list[list], widths: list[float], styles: dict, numeric_from: int = 1, extra: list | None = None):
    """Header row + zebra stripes; numeric columns right-aligned."""
    regular, bold = _register_fonts()
    data = [[Paragraph(escape(str(c)), styles["header"]) for c in rows[0]]]
    for row in rows[1:]:
        data.append([c if isinstance(c, Paragraph) else escape(str(c)) for c in row])
    table = Table(data, colWidths=widths, repeatRows=1)
    commands = [
        ("FONT", (0, 0), (-1, -1), regular, 7.8),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (numeric_from, 1), (-1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(2, len(data), 2):
        commands.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    table.setStyle(TableStyle(commands + (extra or [])))
    return table


def _image(png: bytes, width: float) -> Image:
    image = Image(io.BytesIO(png))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth, image.drawHeight = width, width * ratio
    return image


def _kpi_strip(data: ReportData, styles: dict) -> Table:
    k = data.kpis
    at_risk = "–" if k.get("at_risk") is None else f"{k['at_risk']:,}"
    cells = [
        ("Students", f"{k['students']:,}"),
        ("Pass rate", _pct(k["pass_rate"])),
        ("Average score", _pct(k["avg_score"])),
        ("At-risk students", at_risk),
        ("Absentees", f"{k['absentees']:,}"),
    ]
    row = [[Paragraph(value, styles["kpi_value"]), Paragraph(label, styles["kpi_label"])] for label, value in cells]
    table = Table([row], colWidths=[CONTENT_W / 5] * 5)
    table.setStyle(
        TableStyle(
            [("BOX", (i, 0), (i, 0), 0.6, RULE) for i in range(5)]
            + [
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _cover(data: ReportData, styles: dict) -> list:
    meta = data.meta
    story = [Paragraph(escape(meta.title or "Student Performance Report"), styles["title"])]
    org = " · ".join(x for x in (meta.college, meta.department) if x)
    if org:
        story.append(Paragraph(escape(org), styles["subtitle"]))
    details = [
        ("Semester", meta.semester or ", ".join(sorted(data.clean["semester"].astype(str).unique()))),
        ("Faculty", meta.faculty or "–"),
        ("Scope", meta.scope),
        ("Stage", "Final results" if data.stage == "final" else "Mid-semester (no end-term marks yet)"),
        ("Generated", meta.generated),
    ]
    rows = [[Paragraph(f"<b>{k}</b>", styles["small"]), Paragraph(escape(str(v)), styles["base"])] for k, v in details]
    table = Table(rows, colWidths=[28 * mm, CONTENT_W - 28 * mm])
    table.setStyle(
        TableStyle(
            [
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story += [Spacer(1, 6), table, Spacer(1, 8)]
    return story


def _section_table(data: ReportData, styles: dict) -> Table:
    rows = [["Section", "Students", "Average score", "Pass rate", "Fail rate", "Fails", "Attendance"]]
    for r in data.by_section.itertuples():
        rows.append(
            [
                r.section,
                f"{r.students:,}",
                _pct(r.avg_score),
                _pct(r.pass_rate),
                _pct(r.fail_rate),
                f"{r.fails:,}",
                _pct(r.avg_attendance, 0),
            ]
        )
    return _table(rows, [CONTENT_W * w for w in (0.16, 0.12, 0.16, 0.14, 0.14, 0.12, 0.16)], styles)


def _subject_table(data: ReportData, styles: dict) -> Table:
    rows = [["#", "Subject", "Difficulty", "Average score", "Fail rate", "Fails", "Attendance"]]
    for r in data.by_subject.itertuples():
        rows.append(
            [
                r.rank,
                Paragraph(escape(str(r.subject)), styles["cell"]),
                f"{r.difficulty_index:.1f}",
                _pct(r.avg_score),
                _pct(r.fail_rate),
                f"{r.fails:,}",
                _pct(r.avg_attendance, 0),
            ]
        )
    return _table(rows, [CONTENT_W * w for w in (0.06, 0.3, 0.12, 0.15, 0.12, 0.1, 0.15)], styles, numeric_from=2)


def _risk_section(data: ReportData, styles: dict) -> list:
    at_risk = data.at_risk
    story = [CondPageBreak(60 * mm), Paragraph("Students who may need support", styles["h2"])]
    method = "rules plus a machine-learning model" if data.risk_method == "rules+ml" else "transparent rules"
    if data.risk_model_name:
        method += f" ({data.risk_model_name.lower()})"
    story.append(
        Paragraph(
            f"Risk levels combine early-warning signals using {method}. They are decision support, not a judgement: "
            "please talk to each student before acting.",
            styles["small"],
        )
    )
    story.append(Spacer(1, 4))
    counts = data.risk_students["risk_level"].value_counts()
    chart = _image(mpl_charts.risk_levels(counts), 62 * mm)
    if at_risk.empty:
        story += [chart, Paragraph("No students are at high or medium risk.", styles["base"])]
        return story
    if not data.meta.include_names:
        by_section = at_risk.groupby(["section", "risk_level"]).size().unstack(fill_value=0)
        rows = [["Section", "High", "Medium"]]
        for section, r in by_section.iterrows():
            rows.append([section, int(r.get("High", 0)), int(r.get("Medium", 0))])
        story += [
            KeepTogether([chart, Spacer(1, 4), _table(rows, [40 * mm, 30 * mm, 30 * mm], styles)]),
            Paragraph("Student names are omitted from this copy of the report.", styles["small"]),
        ]
        return story
    level_styles = {
        level: ParagraphStyle(
            f"risk_{level}", parent=styles["cell"], fontName=_register_fonts()[1], textColor=colors.HexColor(color)
        )
        for level, color in RISK_COLORS.items()
    }
    rows = [["Roll No", "Name", "Sec", "Risk", "Score", "Main reasons"]]
    for r in at_risk.head(AT_RISK_LIMIT).itertuples():
        reasons = "; ".join(r.reasons[:2]) + (" …" if len(r.reasons) > 2 else "")
        rows.append(
            [
                r.roll_no,
                Paragraph(escape(str(r.student_name)), styles["cell"]),
                r.section,
                Paragraph(r.risk_level, level_styles[r.risk_level]),
                f"{int(r.risk_score)}",
                Paragraph(escape(reasons), styles["cell"]),
            ]
        )
    widths = [CONTENT_W * w for w in (0.13, 0.19, 0.06, 0.09, 0.07, 0.46)]
    table = _table(
        rows,
        widths,
        styles,
        numeric_from=4,
        extra=[("ALIGN", (5, 1), (5, -1), "LEFT"), ("VALIGN", (0, 1), (-1, -1), "TOP")],
    )
    story += [chart, Spacer(1, 4), table]
    if len(at_risk) > AT_RISK_LIMIT:
        story.append(
            Paragraph(
                f"Showing {AT_RISK_LIMIT} of {len(at_risk)} students at high or medium risk, highest "
                "first. The Excel report lists everyone.",
                styles["small"],
            )
        )
    return story


def _co_section(data: ReportData, styles: dict) -> list:
    co, settings = data.co_table, data.co_settings
    story = [CondPageBreak(70 * mm), Paragraph("Course Outcome attainment", styles["h2"])]
    story.append(
        Paragraph(
            f"A student attains a CO at ≥ {settings.student_threshold:g}% of its marks. "
            f"Level 3 when ≥ {settings.level3:g}% of students attain it, "
            f"Level 2 ≥ {settings.level2:g}%, Level 1 ≥ {settings.level1:g}%. "
            f"Target: Level {settings.target_level}.",
            styles["small"],
        )
    )
    rows = [["CO", "Questions", "Max marks", "Attained", "Assessed", "% attained", "Level", "Status"]]
    for r in co.itertuples():
        rows.append(
            [
                r.co,
                Paragraph(escape(str(r.questions)), styles["cell"]),
                f"{r.max_marks:g}",
                r.students_attained,
                r.students_assessed,
                _pct(r.pct_attained),
                int(r.level),
                "Meets target" if r.meets_target else "Below target",
            ]
        )
    status = [
        ("TEXTCOLOR", (7, i), (7, i), colors.HexColor(STATUS["good"] if r.meets_target else STATUS["critical"]))
        for i, r in enumerate(co.itertuples(), start=1)
    ]
    widths = [CONTENT_W * w for w in (0.08, 0.22, 0.1, 0.1, 0.1, 0.13, 0.08, 0.19)]
    story += [
        Spacer(1, 4),
        _image(mpl_charts.co_attainment(co, settings), CONTENT_W * 0.8),
        Spacer(1, 4),
        _table(rows, widths, styles, numeric_from=2, extra=status),
    ]
    return story


def _footer(canvas, doc, title: str) -> None:
    regular, _ = _register_fonts()
    canvas.saveState()
    canvas.setFont(regular, 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 9 * mm, f"GradeLens · {title}")
    canvas.drawRightString(PAGE_W - MARGIN, 9 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(MARGIN, 12 * mm, PAGE_W - MARGIN, 12 * mm)
    canvas.restoreState()


def build_pdf(data: ReportData, summary: str) -> bytes:
    styles = _styles()
    story = _cover(data, styles)
    story += [Paragraph("Executive summary", styles["h2"]), Paragraph(escape(summary), styles["base"]), Spacer(1, 8)]
    story += [_kpi_strip(data, styles), Spacer(1, 4)]

    story += [Paragraph("Results by section", styles["h2"]), _section_table(data, styles)]
    story += [
        Paragraph("Results by subject", styles["h2"]),
        _subject_table(data, styles),
        Paragraph(
            "Difficulty index = average of (100 − average score) and fail rate; higher is harder.", styles["small"]
        ),
    ]

    charts = [
        Paragraph("Fail rate by subject", styles["h2"]),
        _image(mpl_charts.fail_rate_by_subject(data.by_subject), CONTENT_W * 0.78),
    ]
    story.append(KeepTogether(charts))
    if not data.matrix.empty and data.matrix.size > 1:
        story.append(
            KeepTogether(
                [
                    Paragraph("Fail rate: section × subject", styles["h2"]),
                    _image(mpl_charts.fail_heatmap(data.matrix), CONTENT_W * 0.9),
                ]
            )
        )
    story.append(
        KeepTogether(
            [
                Paragraph("Grade distribution", styles["h2"]),
                _image(mpl_charts.grade_distribution(data.grades), CONTENT_W * 0.72),
            ]
        )
    )

    insights = []
    att = data.attendance
    if att.get("n") and att["below"].get("records") and att["above"].get("records"):
        insights.append(
            f"Attendance: below {data.config.attendance_threshold:g}% the fail rate is "
            f"{_pct(att['below']['fail_rate'])} vs {_pct(att['above']['fail_rate'])} above it "
            f"(correlation r = {att['r']:.2f})."
        )
    if not data.mst_trend.empty:
        t = data.mst_trend[["Improved", "Stable", "Declined"]].sum()
        insights.append(
            f"MST-1 → MST-2: {int(t['Improved'])} improved, {int(t['Stable'])} stable and "
            f"{int(t['Declined'])} declined (student-subject pairs)."
        )
    if insights:
        story.append(Paragraph("Other findings", styles["h2"]))
        story += [Paragraph("• " + escape(text), styles["base"]) for text in insights]

    if data.risk_students is not None:
        story += _risk_section(data, styles)

    if data.meta.include_names and not data.top.empty:
        rows = [["Top performers", "Avg", "Need the most support", "Avg"]]
        for t, b in zip(data.top.itertuples(), data.bottom.itertuples(), strict=False):
            rows.append(
                [
                    Paragraph(escape(f"{t.student_name} ({t.roll_no})"), styles["cell"]),
                    _pct(t.avg_score),
                    Paragraph(escape(f"{b.student_name} ({b.roll_no})"), styles["cell"]),
                    _pct(b.avg_score),
                ]
            )
        story += [
            CondPageBreak(45 * mm),
            Paragraph("Top and bottom five", styles["h2"]),
            _table(
                rows,
                [CONTENT_W * w for w in (0.37, 0.13, 0.37, 0.13)],
                styles,
                extra=[("ALIGN", (2, 1), (2, -1), "LEFT")],
            ),
        ]

    if data.co_table is not None and not data.co_table.empty and data.co_settings is not None:
        story += _co_section(data, styles)

    c = data.config
    method = [
        f"Marks scheme: assignment /{c.max_assignment:g}, MST-1 /{c.max_mst1:g}, MST-2 /{c.max_mst2:g}, "
        f"end term /{c.max_end_term:g}; total /{c.total_max:g}. Pass mark {c.pass_pct:g}%.",
        "Absent marks are treated as absences, not zeros; an end-term absence counts as a fail.",
    ]
    method += [f"Data note: {n}" for n in data.quality_notes]
    story += [CondPageBreak(30 * mm), Paragraph("Method and data notes", styles["h2"])]
    story += [Paragraph("• " + escape(text), styles["small"]) for text in method]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=14 * mm,
        bottomMargin=18 * mm,
        title=data.meta.title,
        author="GradeLens",
        subject=data.meta.scope,
    )
    on_page = lambda canvas, d: _footer(canvas, d, data.meta.title)  # noqa: E731
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buffer.getvalue()
