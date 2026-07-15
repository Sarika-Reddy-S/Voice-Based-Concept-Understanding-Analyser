"""
scoring.py
----------
Scoring engine that combines semantic similarity, fluency metrics,
and audio-derived delivery characteristics into a single interpretable
"Concept Understanding Score" plus sub-scores for feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .audio_features import AudioFeatures, compute_speaking_rate
from .semantic_analysis import SemanticResult

# Ideal speaking-rate band (words per minute) for clear explanations.
IDEAL_WPM_LOW = 110
IDEAL_WPM_HIGH = 160


@dataclass
class ScoreBreakdown:
    understanding_score: float     # 0-100, from semantic similarity
    fluency_score: float           # 0-100, from speaking rate + pauses
    clarity_score: float           # 0-100, from pitch stability + energy
    overall_score: float           # 0-100, weighted composite
    grade: str
    feedback: List[str] = field(default_factory=list)


def _score_speaking_rate(wpm: float) -> float:
    if wpm <= 0:
        return 0.0
    if IDEAL_WPM_LOW <= wpm <= IDEAL_WPM_HIGH:
        return 100.0
    if wpm < IDEAL_WPM_LOW:
        deficit = IDEAL_WPM_LOW - wpm
        return max(0.0, 100 - deficit * 1.5)
    excess = wpm - IDEAL_WPM_HIGH
    return max(0.0, 100 - excess * 1.2)


def _score_pauses(silence_ratio: float) -> float:
    # A moderate silence ratio (natural pauses) is fine; excessive silence
    # suggests hesitation or long dead air.
    ideal = 0.15
    penalty = abs(silence_ratio - ideal) * 200
    return max(0.0, 100 - penalty)


def _score_pitch_stability(pitch_std: float) -> float:
    # Some pitch variation indicates natural, expressive speech; very flat
    # (monotone) or wildly erratic pitch both reduce clarity.
    if pitch_std <= 0:
        return 40.0
    ideal_low, ideal_high = 15.0, 60.0
    if ideal_low <= pitch_std <= ideal_high:
        return 100.0
    if pitch_std < ideal_low:
        return max(0.0, 100 - (ideal_low - pitch_std) * 3)
    return max(0.0, 100 - (pitch_std - ideal_high) * 1.5)


def _grade_for(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Satisfactory"
    if score >= 40:
        return "Needs Improvement"
    return "Poor"


def compute_score(
    semantic_result: SemanticResult,
    audio_features: AudioFeatures,
    word_count: int,
) -> ScoreBreakdown:
    """
    Combine semantic and acoustic signals into an overall score.

    Weighting:
        Understanding (semantic similarity): 55%
        Fluency (speaking rate + pause pattern): 25%
        Clarity (pitch stability + energy consistency): 20%
    """
    understanding_score = round(semantic_result.overall_similarity * 100, 2)

    wpm = compute_speaking_rate(word_count, audio_features.duration_sec)
    rate_score = _score_speaking_rate(wpm)
    pause_score = _score_pauses(audio_features.silence_ratio)
    fluency_score = round((rate_score * 0.6 + pause_score * 0.4), 2)

    pitch_score = _score_pitch_stability(audio_features.pitch_std_hz)
    energy_consistency = 100 - min(
        100, (audio_features.rms_energy_std / (audio_features.rms_energy_mean + 1e-6)) * 50
    )
    clarity_score = round((pitch_score * 0.6 + max(0, energy_consistency) * 0.4), 2)

    overall = round(
        understanding_score * 0.55 + fluency_score * 0.25 + clarity_score * 0.20, 2
    )

    feedback = []
    if understanding_score < 60:
        feedback.append(
            "Conceptual coverage is limited — revisit the core definition and "
            "include the key terms you missed."
        )
    if semantic_result.missing_keywords:
        shown = ", ".join(semantic_result.missing_keywords[:5])
        feedback.append(f"Consider mentioning: {shown}.")
    if wpm and wpm < IDEAL_WPM_LOW:
        feedback.append("Try speaking a bit faster to sound more confident and fluent.")
    elif wpm and wpm > IDEAL_WPM_HIGH:
        feedback.append("Slow down slightly — you may be rushing through the explanation.")
    if audio_features.silence_ratio > 0.30:
        feedback.append("Reduce long pauses; they can make the explanation feel hesitant.")
    if pitch_score < 50:
        feedback.append("Vary your intonation a little more to sound more engaging.")
    if not feedback:
        feedback.append("Strong, clear, and well-structured explanation. Keep it up!")

    return ScoreBreakdown(
        understanding_score=understanding_score,
        fluency_score=fluency_score,
        clarity_score=clarity_score,
        overall_score=overall,
        grade=_grade_for(overall),
        feedback=feedback,
    )
