"""
report_generator.py
--------------------
Automated PDF report generation using ReportLab for VBCUA.

Produces a shareable evaluation report containing:
  - Reference concept and student transcription
  - Audio waveform visualization
  - Score summary table (Understanding / Fluency / Clarity / Overall)
  - Keyword coverage (matched and missed)
  - Actionable feedback for the learner
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _score_table(score, wpm: float) -> Table:
    """Build a two-column metrics table from a ScoreBreakdown."""
    data = [
        ["Metric", "Value"],
        ["Semantic Similarity", f"{getattr(score, 'understanding_score', 0):.1f} / 100"],
        ["Fluency Score", f"{getattr(score, 'fluency_score', 0):.1f} / 100"],
        ["Clarity Score", f"{getattr(score, 'clarity_score', 0):.1f} / 100"],
        ["Overall Score", f"{getattr(score, 'overall_score', 0):.1f} / 100"],
        ["Understanding Level", getattr(score, 'understanding_level', getattr(score, 'grade', '—'))],
        ["Speaking Rate", f"{wpm} wpm"],
    ]
    table = Table(data, colWidths=[8 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1f2937")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#374151")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f9fafb"), colors.white]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    return table


def generate_pdf_report(
    output_path: str,
    concept_title: str,
    transcript: str,
    reference_text: str,
    score,
    audio_features,
    wpm: float,
    matched_keywords: Optional[List[str]] = None,
    missing_keywords: Optional[List[str]] = None,
    waveform_img_path: Optional[str] = None,
) -> str:
    """
    Build and save a PDF evaluation report.

    Parameters
    ----------
    output_path : str
        Destination PDF file path.
    concept_title : str
        Title of the evaluated concept.
    transcript : str
        Whisper-transcribed student explanation.
    reference_text : str
        Reference / expected explanation.
    score : ScoreBreakdown
        Scoring result from scoring_engine.compute_score.
    audio_features : AudioFeatures
        Extracted audio features.
    wpm : float
        Speaking rate (words per minute).
    matched_keywords : list[str], optional
        Keywords covered by the student.
    missing_keywords : list[str], optional
        Keywords missed by the student.
    waveform_img_path : str, optional
        Path to a pre-saved waveform PNG to embed in the report.

    Returns
    -------
    str
        The output_path the report was written to.
    """
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "VBCUATitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#1f2937"),
        fontSize=18,
        spaceAfter=4,
    )
    heading_style = ParagraphStyle(
        "VBCUAHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1f2937"),
        spaceBefore=14,
        spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "VBCUASub",
        parent=styles["Heading3"],
        textColor=colors.HexColor("#374151"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "VBCUABody",
        parent=styles["BodyText"],
        leading=16,
        textColor=colors.HexColor("#374151"),
    )
    caption_style = ParagraphStyle(
        "VBCUACaption",
        parent=styles["Normal"],
        textColor=colors.HexColor("#6b7280"),
        fontSize=9,
        spaceAfter=4,
    )

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

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Voice-Based Concept Understanding Analyser", title_style))
    story.append(Paragraph("Evaluation Report", styles["Heading3"]))
    story.append(Paragraph(
        f"Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        caption_style,
    ))
    story.append(Spacer(1, 10))

    # ── Concept & Scores ──────────────────────────────────────────────────────
    story.append(Paragraph(f"Concept: {concept_title}", heading_style))
    story.append(Paragraph("Evaluation Summary", sub_style))
    story.append(_score_table(score, wpm))
    story.append(Spacer(1, 14))

    # ── Reference Concept ─────────────────────────────────────────────────────
    story.append(Paragraph("Reference Concept", heading_style))
    story.append(Paragraph(reference_text or "(none provided)", body_style))
    story.append(Spacer(1, 10))

    # ── Student Transcription ─────────────────────────────────────────────────
    story.append(Paragraph("Student Transcription", heading_style))
    story.append(Paragraph(transcript or "(no speech detected)", body_style))
    story.append(Spacer(1, 10))

    # ── Waveform ──────────────────────────────────────────────────────────────
    if waveform_img_path and os.path.exists(waveform_img_path):
        story.append(Paragraph("Audio Visualization", heading_style))
        img = Image(waveform_img_path, width=16 * cm, height=4 * cm)
        story.append(img)
        story.append(Spacer(1, 10))

    # ── Audio features summary ────────────────────────────────────────────────
    story.append(Paragraph("Audio Analysis", heading_style))
    pause_ratio = getattr(audio_features, "pause_ratio", getattr(audio_features, "silence_ratio", 0.0))
    rms = getattr(audio_features, "rms_energy", getattr(audio_features, "rms_energy_mean", 0.0))
    audio_summary = (
        f"Duration: {audio_features.duration_sec} s  |  "
        f"Pause ratio: {pause_ratio}  |  "
        f"Confidence (RMS energy): {rms:.4f}  |  "
        f"Pitch mean: {audio_features.pitch_mean_hz} Hz  |  "
        f"Pitch variation: {audio_features.pitch_std_hz} Hz  |  "
        f"Estimated pauses: {audio_features.estimated_pause_count}"
    )
    story.append(Paragraph(audio_summary, body_style))
    story.append(Spacer(1, 10))

    # ── Keyword coverage ──────────────────────────────────────────────────────
    if matched_keywords:
        story.append(Paragraph("Key Concepts Covered", heading_style))
        story.append(Paragraph(", ".join(matched_keywords), body_style))
        story.append(Spacer(1, 6))

    if missing_keywords:
        story.append(Paragraph("Key Concepts Missed", heading_style))
        story.append(Paragraph(", ".join(missing_keywords), body_style))
        story.append(Spacer(1, 6))

    # ── Feedback ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Feedback & Recommendations", heading_style))
    feedback_items = getattr(score, "feedback", [])
    if feedback_items:
        story.append(ListFlowable(
            [ListItem(Paragraph(f, body_style)) for f in feedback_items],
            bulletType="bullet",
        ))

    doc.build(story)
    return output_path
