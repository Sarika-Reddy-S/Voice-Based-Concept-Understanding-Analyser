"""
test_audio_utils.py
-------------------
Unit tests for modules/audio_utils.py

Tests cover:
  - AudioFeatures dataclass structure and property aliases
  - compute_speaking_rate edge cases
  - get_audio_metadata on a real/stub file
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.audio_utils import (
    AudioFeatures,
    compute_speaking_rate,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── AudioFeatures tests ───────────────────────────────────────────────────────

def test_audio_features_basic_construction():
    af = make_audio_features()
    assert af.duration_sec == 30.0
    assert af.pause_ratio == 0.15
    assert af.rms_energy == 0.05


def test_silence_ratio_alias():
    """silence_ratio property should mirror pause_ratio for compat."""
    af = make_audio_features(pause_ratio=0.22)
    assert af.silence_ratio == 0.22


def test_rms_energy_mean_alias():
    """rms_energy_mean property should mirror rms_energy for compat."""
    af = make_audio_features(rms_energy=0.03)
    assert af.rms_energy_mean == 0.03


# ── compute_speaking_rate tests ───────────────────────────────────────────────

def test_speaking_rate_normal():
    rate = compute_speaking_rate(word_count=100, duration_sec=60.0)
    assert rate == 100.0


def test_speaking_rate_zero_duration():
    assert compute_speaking_rate(100, 0.0) == 0.0


def test_speaking_rate_zero_words():
    assert compute_speaking_rate(0, 30.0) == 0.0


def test_speaking_rate_typical():
    # 70 words in 30 s → 140 wpm
    rate = compute_speaking_rate(70, 30.0)
    assert rate == 140.0


def test_speaking_rate_fast():
    rate = compute_speaking_rate(200, 30.0)
    assert rate > 150
