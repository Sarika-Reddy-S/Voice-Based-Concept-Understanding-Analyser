"""
test_semantic_eval.py
---------------------
Unit tests for modules/semantic_eval.py

Tests cover helper utilities (sentence splitting, keyword extraction)
and the SemanticResult dataclass — without loading the SBERT model.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.semantic_eval import (
    SemanticResult,
    _split_sentences,
    _extract_keywords,
)


# ── _split_sentences tests ────────────────────────────────────────────────────

def test_split_sentences_basic():
    text = "Machine learning is powerful. It learns from data. Results improve over time."
    parts = _split_sentences(text)
    assert len(parts) == 3


def test_split_sentences_single():
    parts = _split_sentences("Only one sentence here")
    assert len(parts) == 1


def test_split_sentences_empty():
    parts = _split_sentences("")
    assert parts == []


# ── _extract_keywords tests ───────────────────────────────────────────────────

def test_extract_keywords_basic():
    kw = _extract_keywords("Machine learning allows systems to learn patterns from data.")
    assert "machine" in kw
    assert "learning" in kw
    assert "patterns" in kw


def test_extract_keywords_stopwords_removed():
    kw = _extract_keywords("the and with from that this")
    # All are stopwords → empty or very small set
    stopwords = {"the", "and", "with", "from", "that", "this"}
    assert kw.isdisjoint(stopwords)


def test_extract_keywords_min_length():
    kw = _extract_keywords("go do it so at")
    # All words < 4 chars → should be empty
    assert len(kw) == 0


# ── SemanticResult tests ──────────────────────────────────────────────────────

def test_semantic_result_percentage():
    result = SemanticResult(overall_similarity=0.75)
    assert result.percentage == 75.0


def test_semantic_result_clamped_zero():
    result = SemanticResult(overall_similarity=0.0)
    assert result.percentage == 0.0


def test_semantic_result_clamped_one():
    result = SemanticResult(overall_similarity=1.0)
    assert result.percentage == 100.0


def test_semantic_result_defaults():
    result = SemanticResult(overall_similarity=0.5)
    assert result.matched_keywords == []
    assert result.missing_keywords == []
    assert result.sentence_scores == []
