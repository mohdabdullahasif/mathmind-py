"""PDF exporter using ReportLab (no pdflatex required)."""
from __future__ import annotations

from pathlib import Path
from datetime import date
from typing import Any

from mathmind.models import Result


def export_pdf(result: Result, output_path: str, title: str = "MathMind Solution") -> str:
    """Export a Result to a PDF file. Returns the path."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import cm
    except ImportError:
        raise ImportError("reportlab not installed. Run: pip install reportlab")

    path = Path(output_path)
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    styles = getSampleStyleSheet()
    story: list[Any] = []

    # Title
    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(f"Generated: {date.today()}", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    # Problem
    story.append(Paragraph(f"<b>Expression:</b> {result.expression}", styles["Normal"]))
    if result.strategy:
        story.append(Paragraph(f"<b>Strategy:</b> {result.strategy} (confidence: {result.confidence:.0%})", styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    # Steps
    if result.steps:
        story.append(Paragraph("<b>Solution Steps:</b>", styles["Heading2"]))
        step_data = [["#", "Description", "Result"]]
        for s in result.steps:
            step_data.append([str(s.step_no), s.description, s.after or s.before])
        t = Table(step_data, colWidths=[1 * cm, 10 * cm, 6 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3 * cm))

    # Answer
    story.append(Paragraph(f"<b>Answer:</b> {result.value}", styles["Heading2"]))
    if result.latex:
        story.append(Paragraph(f"<b>LaTeX:</b> {result.latex}", styles["Code"]))

    doc.build(story)
    return str(path)


def export_worksheet(
    problems: list[str],
    answers: list[str],
    output_path: str,
    title: str = "MathMind Worksheet",
    student_name: str = "",
    include_answers: bool = True,
) -> str:
    """Generate a worksheet PDF with optional answer key."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import cm
    except ImportError:
        raise ImportError("reportlab not installed")

    path = Path(output_path)
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    styles = getSampleStyleSheet()
    story: list[Any] = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(f"Date: {date.today()}   Name: {student_name or '___________________'}", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    for i, (prob, ans) in enumerate(zip(problems, answers), 1):
        story.append(Paragraph(f"{i}. {prob}", styles["Normal"]))
        story.append(Spacer(1, 0.8 * cm))

    if include_answers:
        story.append(Paragraph("Answer Key", styles["Heading1"]))
        for i, ans in enumerate(answers, 1):
            story.append(Paragraph(f"{i}. {ans}", styles["Normal"]))

    doc.build(story)
    return str(path)
