"""
report_generator.py
--------------------
Automated PDF report generation using ReportLab.

Produces a shareable evaluation report summarizing the transcript,
semantic understanding score, fluency/clarity sub-scores, and
actionable feedback for the learner.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .audio_features import AudioFeatures
from .scoring import ScoreBreakdown


def _score_table(score: ScoreBreakdown, wpm: float) -> Table:
    data = [
        ["Metric", "Score"],
        ["Understanding", f"{score.understanding_score} / 100"],
        ["Fluency", f"{score.fluency_score} / 100"],
        ["Clarity", f"{score.clarity_score} / 100"],
        ["Overall", f"{score.overall_score} / 100"],
        ["Grade", score.grade],
        ["Speaking rate", f"{wpm} wpm"],
    ]
    table = Table(data, colWidths=[7 * cm, 7 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def generate_pdf_report(
    output_path: str,
    concept_title: str,
    transcript: str,
    reference_text: str,
    score: ScoreBreakdown,
    audio_features: AudioFeatures,
    wpm: float,
    matched_keywords: Optional[list] = None,
    missing_keywords: Optional[list] = None,
) -> str:
    """
    Build and save a PDF evaluation report.

    Returns
    -------
    str
        The output_path the report was written to.
    """
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "VBCUATitle", parent=styles["Title"], textColor=colors.HexColor("#1f2937")
    )
    heading_style = ParagraphStyle(
        "VBCUAHeading", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6
    )
    body_style = ParagraphStyle("VBCUABody", parent=styles["BodyText"], leading=15)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title="VBCUA Evaluation Report",
    )

    story = []
    story.append(Paragraph("Voice-Based Concept Understanding Analyser", title_style))
    story.append(Paragraph("Evaluation Report", styles["Heading3"]))
    story.append(
        Paragraph(
            f"Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            body_style,
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Concept: {concept_title}", heading_style))
    story.append(Paragraph("Score Summary", heading_style))
    story.append(_score_table(score, wpm))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Transcript", heading_style))
    story.append(Paragraph(transcript or "(no speech detected)", body_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Reference Explanation", heading_style))
    story.append(Paragraph(reference_text or "(none provided)", body_style))
    story.append(Spacer(1, 8))

    if matched_keywords:
        story.append(Paragraph("Key Concepts Covered", heading_style))
        story.append(Paragraph(", ".join(matched_keywords), body_style))

    if missing_keywords:
        story.append(Paragraph("Key Concepts Missed", heading_style))
        story.append(Paragraph(", ".join(missing_keywords), body_style))

    story.append(Paragraph("Audio Analysis", heading_style))
    audio_summary = (
        f"Duration: {audio_features.duration_sec}s | "
        f"Silence ratio: {audio_features.silence_ratio} | "
        f"Pitch mean: {audio_features.pitch_mean_hz} Hz | "
        f"Pitch variation: {audio_features.pitch_std_hz} Hz | "
        f"Estimated pauses: {audio_features.estimated_pause_count}"
    )
    story.append(Paragraph(audio_summary, body_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Feedback & Recommendations", heading_style))
    story.append(
        ListFlowable(
            [ListItem(Paragraph(f, body_style)) for f in score.feedback],
            bulletType="bullet",
        )
    )

    doc.build(story)
    return output_path
