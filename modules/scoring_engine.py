"""
scoring_engine.py
-----------------
Understanding score calculation and classification for VBCUA.

Implements the `evaluate_understanding` function shown in the project
description, which combines semantic similarity, filler word ratio, and
audio features into a final score and qualitative classification:

    Strong Understanding  (score ≥ 80)
    Moderate Understanding (score ≥ 50)
    Poor Understanding    (score < 50)

Also provides a detailed `compute_score` for sub-score breakdowns
(Understanding / Fluency / Clarity) used in the Streamlit UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

# Colour constants matching the project description screenshots
COLOUR_STRONG   = "#2ecc71"
COLOUR_MODERATE = "#f39c12"
COLOUR_POOR     = "#e74c3c"

# Ideal speaking-rate band (words per minute)
IDEAL_WPM_LOW  = 110
IDEAL_WPM_HIGH = 160


@dataclass
class ScoreBreakdown:
    """Detailed score breakdown with sub-scores and feedback."""

    understanding_score: float    # 0–100
    fluency_score: float          # 0–100
    clarity_score: float          # 0–100
    overall_score: float          # 0–100
    grade: str
    understanding_level: str      # Strong / Moderate / Poor
    colour: str                   # hex colour for UI
    feedback: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Primary function — matches the exact logic shown in the project screenshot
# ---------------------------------------------------------------------------

def evaluate_understanding(
    similarity: float,
    filler_ratio: float,
    audio: dict,
) -> Tuple[int, str, str]:
    """
    Compute understanding score from similarity, filler ratio, and audio dict.

    This is the exact formula shown in the project description:

        score += 50 if similarity > 0.7 else 30 if similarity > 0.4 else 10
        score += 20 if filler_ratio < 0.05 else 10
        score += 15 if audio["pause_ratio"] < 0.25 else 5
        score += 15 if audio["rms_energy"]  > 0.01 else 5

    Parameters
    ----------
    similarity : float
        Cosine similarity score (0–1) from semantic_eval.
    filler_ratio : float
        Proportion of filler words in the transcript (0–1).
    audio : dict
        Must contain keys 'pause_ratio' and 'rms_energy'.

    Returns
    -------
    (score, level, colour)
        score  : int — total points (10–100)
        level  : str — 'Strong Understanding' | 'Moderate Understanding' | 'Poor Understanding'
        colour : str — hex colour code
    """
    score = 0

    # Semantic understanding component (max 50)
    if similarity > 0.7:
        score += 50
    elif similarity > 0.4:
        score += 30
    else:
        score += 10

    # Filler-word fluency component (max 20)
    score += 20 if filler_ratio < 0.05 else 10

    # Pause ratio component (max 15)
    score += 15 if audio.get("pause_ratio", 0.0) < 0.25 else 5

    # Energy confidence component (max 15)
    score += 15 if audio.get("rms_energy", 0.0) > 0.01 else 5

    if score >= 80:
        return score, "Strong Understanding", COLOUR_STRONG
    elif score >= 50:
        return score, "Moderate Understanding", COLOUR_MODERATE
    else:
        return score, "Poor Understanding", COLOUR_POOR


def understanding_level_for(overall_score: float) -> str:
    """Map a 0-100 score to the Strong / Moderate / Poor enum string."""
    if overall_score >= 75:
        return "Strong"
    if overall_score >= 50:
        return "Moderate"
    return "Poor"


def colour_for_level(level: str) -> str:
    """Return the hex colour associated with an understanding level."""
    mapping = {
        "Strong": COLOUR_STRONG,
        "Strong Understanding": COLOUR_STRONG,
        "Moderate": COLOUR_MODERATE,
        "Moderate Understanding": COLOUR_MODERATE,
        "Poor": COLOUR_POOR,
        "Poor Understanding": COLOUR_POOR,
    }
    return mapping.get(level, COLOUR_POOR)


# ---------------------------------------------------------------------------
# Helpers for sub-score computation
# ---------------------------------------------------------------------------

def _score_speaking_rate(wpm: float) -> float:
    if wpm <= 0:
        return 0.0
    if IDEAL_WPM_LOW <= wpm <= IDEAL_WPM_HIGH:
        return 100.0
    if wpm < IDEAL_WPM_LOW:
        return max(0.0, 100 - (IDEAL_WPM_LOW - wpm) * 1.5)
    return max(0.0, 100 - (wpm - IDEAL_WPM_HIGH) * 1.2)


def _score_pauses(pause_ratio: float) -> float:
    ideal = 0.15
    penalty = abs(pause_ratio - ideal) * 200
    return max(0.0, 100 - penalty)


def _score_filler_words(filler_ratio: float) -> float:
    if filler_ratio <= 0.03:
        return 100.0
    return max(0.0, 100 - (filler_ratio - 0.03) * 300)


def _score_pitch_stability(pitch_std: float) -> float:
    if pitch_std <= 0:
        return 40.0
    lo, hi = 15.0, 60.0
    if lo <= pitch_std <= hi:
        return 100.0
    if pitch_std < lo:
        return max(0.0, 100 - (lo - pitch_std) * 3)
    return max(0.0, 100 - (pitch_std - hi) * 1.5)


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
    semantic_result,
    audio_features,
    word_count: int,
    filler_ratio: float = 0.0,
) -> ScoreBreakdown:
    """
    Combine semantic and acoustic signals into a detailed ScoreBreakdown.

    Weights:
        Understanding (semantic similarity):   55 %
        Fluency (speaking rate + pauses + filler): 25 %
        Clarity (pitch stability + energy):    20 %
    """
    from .audio_utils import compute_speaking_rate

    understanding_score = round(semantic_result.overall_similarity * 100, 2)

    wpm = compute_speaking_rate(word_count, audio_features.duration_sec)
    rate_s = _score_speaking_rate(wpm)
    pause_s = _score_pauses(audio_features.pause_ratio)
    filler_s = _score_filler_words(filler_ratio)
    fluency_score = round(rate_s * 0.45 + pause_s * 0.30 + filler_s * 0.25, 2)

    pitch_s = _score_pitch_stability(audio_features.pitch_std_hz)
    energy_var = (audio_features.rms_energy_std / (audio_features.rms_energy + 1e-6)) * 50
    energy_consistency = 100 - min(100, energy_var)
    clarity_score = round(pitch_s * 0.6 + max(0, energy_consistency) * 0.4, 2)

    overall = round(
        understanding_score * 0.55 + fluency_score * 0.25 + clarity_score * 0.20, 2
    )

    level = understanding_level_for(overall)
    colour = colour_for_level(level)

    feedback: List[str] = []
    if understanding_score < 60:
        feedback.append(
            "Conceptual coverage is limited — revisit the core definition and "
            "include the key terms you missed."
        )
    if getattr(semantic_result, "missing_keywords", []):
        shown = ", ".join(semantic_result.missing_keywords[:5])
        feedback.append(f"Consider mentioning: {shown}.")
    if wpm and wpm < IDEAL_WPM_LOW:
        feedback.append("Try speaking a bit faster to sound more confident and fluent.")
    elif wpm and wpm > IDEAL_WPM_HIGH:
        feedback.append("Slow down slightly — you may be rushing through the explanation.")
    if audio_features.pause_ratio > 0.30:
        feedback.append("Reduce long pauses; they can make the explanation feel hesitant.")
    if filler_ratio > 0.08:
        feedback.append("Try to cut down on filler words like 'um', 'like', and 'you know'.")
    if pitch_s < 50:
        feedback.append("Vary your intonation a little more to sound more engaging.")
    if not feedback:
        feedback.append("Strong, clear, and well-structured explanation. Keep it up!")

    return ScoreBreakdown(
        understanding_score=understanding_score,
        fluency_score=fluency_score,
        clarity_score=clarity_score,
        overall_score=overall,
        grade=_grade_for(overall),
        understanding_level=level,
        colour=colour,
        feedback=feedback,
    )
