"""
test_semantic.py
-----------------
Tests for semantic analysis helpers (no SBERT model needed).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.semantic_eval import _split_sentences, _extract_keywords, SemanticResult


def test_split_sentences_multiple():
    text = "This is one. This is two. And three."
    parts = _split_sentences(text)
    assert len(parts) == 3


def test_split_sentences_empty():
    assert _split_sentences("") == []


def test_extract_keywords_removes_stopwords():
    kw = _extract_keywords("the cat sits with the dog from above")
    assert "from" not in kw
    assert "with" not in kw


def test_extract_keywords_min_length():
    kw = _extract_keywords("go at it on")
    assert len(kw) == 0


def test_semantic_result_percentage():
    r = SemanticResult(overall_similarity=0.65)
    assert r.percentage == 65.0
