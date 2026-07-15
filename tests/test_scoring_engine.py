"""
test_scoring_engine.py
-----------------------
Unit tests for modules/scoring_engine.py

Tests cover:
  - evaluate_understanding (the exact formula from the project description)
  - compute_score sub-scores
  - understanding_level_for classification
  - colour_for_level mapping
  - feedback generation
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.audio_utils import AudioFeatures
from modules.semantic_eval import SemanticResult
from modules.scoring_engine import (
    evaluate_understanding,
    compute_score,
    understanding_level_for,
    colour_for_level,
    COLOUR_STRONG,
    COLOUR_MODERATE,
    COLOUR_POOR,
    _score_pauses,
    _score_speaking_rate,
    _score_filler_words,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_audio_dict(**overrides) -> dict:
    base = {"pause_ratio": 0.15, "rms_energy": 0.05}
    base.update(overrides)
    return base


def make_audio_features(**overrides) -> AudioFeatures:
    defaults = dict(
        duration_sec=30.0,
        tempo_bpm=120.0,
        pause_ratio=0.15,
        rms_energy=0.05,
        rms_energy_std=0.01,
        zero_crossing_rate=0.08,
        pitch_mean_hz=150.0,
        pitch_std_hz=30.0,
        estimated_pause_count=3,
    )
    defaults.update(overrides)
    return AudioFeatures(**defaults)


# ── evaluate_understanding tests ──────────────────────────────────────────────

def test_strong_understanding_high_similarity():
    audio = make_audio_dict(pause_ratio=0.10, rms_energy=0.05)
    score, level, colour = evaluate_understanding(0.8, 0.02, audio)
    assert score >= 80
    assert level == "Strong Understanding"
    assert colour == COLOUR_STRONG


def test_moderate_understanding_mid_similarity():
    audio = make_audio_dict(pause_ratio=0.30, rms_energy=0.005)
    score, level, colour = evaluate_understanding(0.5, 0.10, audio)
    assert 50 <= score < 80
    assert level == "Moderate Understanding"
    assert colour == COLOUR_MODERATE


def test_poor_understanding_low_similarity():
    audio = make_audio_dict(pause_ratio=0.40, rms_energy=0.001)
    score, level, colour = evaluate_understanding(0.2, 0.20, audio)
    assert score < 50
    assert level == "Poor Understanding"
    assert colour == COLOUR_POOR


def test_evaluate_understanding_max_score():
    audio = make_audio_dict(pause_ratio=0.10, rms_energy=0.05)
    score, level, colour = evaluate_understanding(0.9, 0.01, audio)
    assert score == 100


def test_evaluate_understanding_min_score():
    audio = make_audio_dict(pause_ratio=0.50, rms_energy=0.001)
    score, _, _ = evaluate_understanding(0.1, 0.50, audio)
    assert score == 30  # 10 + 10 + 5 + 5


def test_similarity_boundary_0_7():
    audio = make_audio_dict()
    score_above, _, _ = evaluate_understanding(0.71, 0.0, audio)
    score_below, _, _ = evaluate_understanding(0.69, 0.0, audio)
    assert score_above > score_below


def test_filler_boundary_0_05():
    audio = make_audio_dict()
    s_low,  _, _ = evaluate_understanding(0.5, 0.04, audio)
    s_high, _, _ = evaluate_understanding(0.5, 0.06, audio)
    assert s_low > s_high


def test_pause_boundary_0_25():
    low_pause  = make_audio_dict(pause_ratio=0.20)
    high_pause = make_audio_dict(pause_ratio=0.30)
    s_low,  _, _ = evaluate_understanding(0.5, 0.0, low_pause)
    s_high, _, _ = evaluate_understanding(0.5, 0.0, high_pause)
    assert s_low > s_high


def test_rms_boundary_0_01():
    low_e  = make_audio_dict(rms_energy=0.005)
    high_e = make_audio_dict(rms_energy=0.02)
    s_low,  _, _ = evaluate_understanding(0.5, 0.0, low_e)
    s_high, _, _ = evaluate_understanding(0.5, 0.0, high_e)
    assert s_high > s_low


# ── understanding_level_for tests ────────────────────────────────────────────

def test_level_strong():
    assert understanding_level_for(80.0) == "Strong"


def test_level_moderate():
    assert understanding_level_for(60.0) == "Moderate"


def test_level_poor():
    assert understanding_level_for(30.0) == "Poor"


def test_level_boundary_75():
    assert understanding_level_for(75.0) == "Strong"
    assert understanding_level_for(74.9) == "Moderate"


def test_level_boundary_50():
    assert understanding_level_for(50.0) == "Moderate"
    assert understanding_level_for(49.9) == "Poor"


# ── colour_for_level tests ────────────────────────────────────────────────────

def test_colour_strong():
    assert colour_for_level("Strong") == COLOUR_STRONG


def test_colour_moderate():
    assert colour_for_level("Moderate") == COLOUR_MODERATE


def test_colour_poor():
    assert colour_for_level("Poor") == COLOUR_POOR


# ── compute_score tests ───────────────────────────────────────────────────────

def test_compute_score_high_similarity():
    sem = SemanticResult(overall_similarity=0.92)
    af = make_audio_features()
    result = compute_score(sem, af, word_count=70)
    assert result.understanding_score == 92.0
    assert result.overall_score > 70


def test_compute_score_zero_similarity():
    sem = SemanticResult(overall_similarity=0.0)
    af = make_audio_features()
    result = compute_score(sem, af, word_count=70)
    assert result.understanding_score == 0.0


def test_compute_score_feedback_not_empty():
    sem = SemanticResult(overall_similarity=0.3, missing_keywords=["entropy"])
    af = make_audio_features(pause_ratio=0.5)
    result = compute_score(sem, af, word_count=20)
    assert len(result.feedback) > 0


def test_compute_score_grade_assigned():
    sem = SemanticResult(overall_similarity=0.95)
    af = make_audio_features()
    result = compute_score(sem, af, word_count=70)
    assert result.grade in {"Excellent", "Good", "Satisfactory", "Needs Improvement", "Poor"}


def test_compute_score_high_filler_lowers_fluency():
    sem = SemanticResult(overall_similarity=0.9)
    af = make_audio_features()
    low_filler  = compute_score(sem, af, word_count=70, filler_ratio=0.0)
    high_filler = compute_score(sem, af, word_count=70, filler_ratio=0.25)
    assert high_filler.fluency_score < low_filler.fluency_score


# ── helper scorer unit tests ──────────────────────────────────────────────────

def test_score_speaking_rate_ideal():
    assert _score_speaking_rate(130) == 100.0


def test_score_speaking_rate_zero():
    assert _score_speaking_rate(0) == 0.0


def test_score_pauses_ideal():
    assert _score_pauses(0.15) == 100.0


def test_score_pauses_excessive():
    assert _score_pauses(0.9) < 50.0


def test_score_filler_no_fillers():
    assert _score_filler_words(0.0) == 100.0


def test_score_filler_heavy():
    assert _score_filler_words(0.40) < 50.0
