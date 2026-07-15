import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.semantic_analysis import _extract_keywords, _split_sentences, SemanticResult


def test_split_sentences_basic():
    text = "Photosynthesis converts light energy. It occurs in chloroplasts. This is key."
    sentences = _split_sentences(text)
    assert len(sentences) == 3


def test_split_sentences_handles_empty_string():
    assert _split_sentences("") == []


def test_extract_keywords_filters_stopwords():
    text = "The mitochondria is the powerhouse of the cell and it produces energy"
    keywords = _extract_keywords(text)
    assert "mitochondria" in keywords
    assert "powerhouse" in keywords
    assert "the" not in keywords
    assert "and" not in keywords


def test_extract_keywords_respects_min_length():
    text = "AI is a big field of CS"
    keywords = _extract_keywords(text, min_len=4)
    assert "field" in keywords
    assert "big" not in keywords  # shorter than min_len


def test_semantic_result_percentage_property():
    result = SemanticResult(overall_similarity=0.8734)
    assert result.percentage == 87.34
