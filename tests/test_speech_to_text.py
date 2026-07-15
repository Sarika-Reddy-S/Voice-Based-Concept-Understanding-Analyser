"""
test_speech_to_text.py
-----------------------
Unit tests for modules/speech_to_text.py

Tests cover the filler_word_ratio function (no Whisper model needed).
The Transcriber class itself is not unit-tested here due to requiring
the Whisper model download; it is covered by integration/functional tests.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.speech_to_text import filler_word_ratio, TranscriptionResult


# ── filler_word_ratio tests ───────────────────────────────────────────────────

def test_filler_ratio_empty():
    assert filler_word_ratio("") == 0.0


def test_filler_ratio_none_like():
    assert filler_word_ratio("   ") == 0.0


def test_filler_ratio_no_fillers():
    text = "Machine learning allows systems to learn from data."
    ratio = filler_word_ratio(text)
    assert ratio == 0.0


def test_filler_ratio_all_fillers():
    # um uh like um uh — all fillers
    text = "um uh like um uh"
    ratio = filler_word_ratio(text)
    assert ratio == 1.0


def test_filler_ratio_partial():
    # 2 fillers out of 6 words ≈ 0.33
    text = "um the machine um is learning"
    ratio = filler_word_ratio(text)
    assert 0.0 < ratio < 1.0


def test_filler_ratio_phrase_not_double_counted():
    # "you know" is a phrase; "well" is also a filler = 2 fillers, 6 words
    # Important: "know" and "you" are NOT double-counted individually
    text = "you know the concept very well"
    ratio = filler_word_ratio(text)
    # "you know" (1 phrase filler) + "well" (1 word filler) = 2/6 = 0.333
    # This confirms "know" is not separately counted after phrase removal
    assert ratio <= 0.34  # no double-counting — would be higher if "know" was recounted


def test_filler_ratio_range():
    text = "um basically like you know this is actually important"
    ratio = filler_word_ratio(text)
    assert 0.0 <= ratio <= 1.0


# ── TranscriptionResult tests ─────────────────────────────────────────────────

def test_transcription_result_word_count():
    tr = TranscriptionResult(text="hello world this is a test", language="en")
    assert tr.word_count == 6


def test_transcription_result_empty():
    tr = TranscriptionResult(text="", language="en")
    assert tr.word_count == 0
