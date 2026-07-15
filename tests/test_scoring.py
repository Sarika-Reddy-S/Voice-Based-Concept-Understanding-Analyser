"""
test_scoring.py
---------------
Backward-compatible tests for the scoring module.
All logic now lives in scoring_engine; this file imports through the shim.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.audio_utils import AudioFeatures
from modules.scoring_engine import compute_score, _score_pauses, _score_speaking_rate
from modules.semantic_eval import SemanticResult


def make_features(**overrides) -> AudioFeatures:
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


def test_high_similarity_gives_high_understanding_score():
    semantic = SemanticResult(overall_similarity=0.92)
    features = make_features()
    score = compute_score(semantic, features, word_count=70)
    assert score.understanding_score == 92.0
    assert score.overall_score > 70


def test_zero_similarity_gives_zero_understanding_score():
    semantic = SemanticResult(overall_similarity=0.0)
    features = make_features()
    score = compute_score(semantic, features, word_count=70)
    assert score.understanding_score == 0.0


def test_ideal_speaking_rate_scores_100():
    assert _score_speaking_rate(130) == 100.0


def test_slow_speaking_rate_penalized():
    assert _score_speaking_rate(50) < 100.0


def test_fast_speaking_rate_penalized():
    assert _score_speaking_rate(220) < 100.0


def test_zero_wpm_scores_zero():
    assert _score_speaking_rate(0) == 0.0


def test_ideal_pause_ratio_scores_high():
    assert _score_pauses(0.15) == 100.0


def test_excessive_silence_penalized():
    assert _score_pauses(0.9) < 50.0


def test_grade_labels_assigned_correctly():
    semantic = SemanticResult(overall_similarity=0.95)
    features = make_features(pause_ratio=0.15, pitch_std_hz=30.0)
    score = compute_score(semantic, features, word_count=70)
    assert score.grade in {"Excellent", "Good"}


def test_feedback_not_empty():
    semantic = SemanticResult(overall_similarity=0.3, missing_keywords=["mitochondria"])
    features = make_features(pause_ratio=0.5)
    score = compute_score(semantic, features, word_count=20)
    assert len(score.feedback) > 0


def test_high_filler_ratio_lowers_fluency_score():
    semantic = SemanticResult(overall_similarity=0.9)
    features = make_features()
    low_filler  = compute_score(semantic, features, word_count=70, filler_ratio=0.0)
    high_filler = compute_score(semantic, features, word_count=70, filler_ratio=0.25)
    assert high_filler.fluency_score < low_filler.fluency_score


def test_high_filler_ratio_triggers_feedback():
    semantic = SemanticResult(overall_similarity=0.9)
    features = make_features()
    score = compute_score(semantic, features, word_count=70, filler_ratio=0.15)
    assert any("filler" in f.lower() for f in score.feedback)
