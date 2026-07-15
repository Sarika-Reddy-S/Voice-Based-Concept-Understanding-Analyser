"""
scoring.py
----------
BACKWARD-COMPATIBILITY shim — delegates to scoring_engine.py.

This file is preserved so existing imports of `modules.scoring` continue
to work. All new code should import from `modules.scoring_engine` directly.
"""

from .scoring_engine import (
    ScoreBreakdown,
    compute_score,
    evaluate_understanding,
    understanding_level_for,
    colour_for_level,
    _score_pauses,
    _score_speaking_rate,
    _score_filler_words,
    _score_pitch_stability,
    _grade_for,
    IDEAL_WPM_LOW,
    IDEAL_WPM_HIGH,
)

__all__ = [
    "ScoreBreakdown",
    "compute_score",
    "evaluate_understanding",
    "understanding_level_for",
    "colour_for_level",
    "_score_pauses",
    "_score_speaking_rate",
    "_score_filler_words",
    "_score_pitch_stability",
    "_grade_for",
    "IDEAL_WPM_LOW",
    "IDEAL_WPM_HIGH",
]
