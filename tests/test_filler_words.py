import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.filler_words import analyze_filler_words


def test_empty_transcript_returns_zeroes():
    stats = analyze_filler_words("")
    assert stats.total_words == 0
    assert stats.filler_word_count == 0
    assert stats.filler_ratio == 0.0


def test_detects_common_filler_words():
    text = "So, um, the mitochondria is, like, the powerhouse of the cell, you know."
    stats = analyze_filler_words(text)
    assert stats.filler_word_count > 0
    assert "um" in stats.breakdown
    assert "like" in stats.breakdown
    assert "you know" in stats.breakdown


def test_clean_transcript_has_low_filler_ratio():
    text = "Mitochondria generate energy for the cell through cellular respiration."
    stats = analyze_filler_words(text)
    assert stats.filler_word_count == 0
    assert stats.filler_ratio == 0.0


def test_filler_ratio_is_fraction_of_total_words():
    text = "um um um one two three four five six seven eight"
    stats = analyze_filler_words(text)
    assert stats.total_words == 11
    assert stats.filler_word_count == 3
    assert stats.filler_ratio == round(3 / 11, 4)


def test_multiword_phrase_not_double_counted():
    text = "you know what I mean"
    stats = analyze_filler_words(text)
    # "you know" and "i mean" should each count once as phrases,
    # not also count "know" or "mean" separately.
    assert stats.breakdown.get("you know") == 1
    assert stats.breakdown.get("i mean") == 1
    assert "know" not in stats.breakdown
    assert "mean" not in stats.breakdown
