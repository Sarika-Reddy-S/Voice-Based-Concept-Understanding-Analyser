"""
semantic_eval.py
----------------
Semantic similarity evaluation using Sentence-BERT (SBERT) for VBCUA.

Compares a spoken explanation (transcribed to text) against a reference
concept definition and returns:
  - An overall cosine similarity score (0 – 1)
  - Sentence-level similarity breakdown
  - Keyword coverage: matched and missing reference keywords

The `semantic_similarity(transcript, reference)` function is the primary
entry point used by the Streamlit frontend and the scoring engine.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer, util


@dataclass
class SemanticResult:
    """Container for Sentence-BERT evaluation output."""

    overall_similarity: float          # 0–1 cosine similarity vs reference
    sentence_scores: List[float] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        return round(self.overall_similarity * 100, 2)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s.strip()]


def _extract_keywords(text: str, min_len: int = 4) -> set:
    _STOPWORDS = {
        "the", "and", "that", "this", "with", "from", "have", "which",
        "there", "their", "would", "could", "about", "into", "your",
        "some", "than", "then", "them", "these", "those", "were", "been",
    }
    words = re.findall(r"[A-Za-z']+", text.lower())
    return {w for w in words if len(w) >= min_len and w not in _STOPWORDS}


class SemanticEvaluator:
    """Wraps a Sentence-BERT model for concept-explanation similarity scoring."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def evaluate(self, spoken_text: str, reference_text: str) -> SemanticResult:
        """
        Compare a spoken explanation to a reference concept definition.

        Parameters
        ----------
        spoken_text : str
            Transcript from the speech-to-text module.
        reference_text : str
            The expected/correct explanation of the concept.

        Returns
        -------
        SemanticResult
        """
        if not (spoken_text or "").strip() or not (reference_text or "").strip():
            return SemanticResult(overall_similarity=0.0)

        embeddings = self.model.encode(
            [spoken_text, reference_text], convert_to_tensor=True
        )
        sim = float(util.cos_sim(embeddings[0], embeddings[1])[0][0])
        sim = max(0.0, min(1.0, sim))

        # Sentence-level breakdown
        spoken_sents = _split_sentences(spoken_text)
        sentence_scores: List[float] = []
        if spoken_sents:
            ref_emb = self.model.encode(reference_text, convert_to_tensor=True)
            sent_embs = self.model.encode(spoken_sents, convert_to_tensor=True)
            raw_scores = util.cos_sim(sent_embs, ref_emb).cpu().numpy().flatten()
            sentence_scores = [float(round(float(s), 4)) for s in raw_scores]

        ref_kw = _extract_keywords(reference_text)
        spk_kw = _extract_keywords(spoken_text)
        matched = sorted(ref_kw & spk_kw)
        missing = sorted(ref_kw - spk_kw)

        return SemanticResult(
            overall_similarity=sim,
            sentence_scores=sentence_scores,
            matched_keywords=matched,
            missing_keywords=missing,
        )


@functools.lru_cache(maxsize=1)
def get_semantic_evaluator(model_name: str = "all-MiniLM-L6-v2") -> SemanticEvaluator:
    """Cached factory — prevents SBERT reload on every Streamlit rerun."""
    return SemanticEvaluator(model_name=model_name)


def semantic_similarity(transcript: str, reference: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Convenience function: compute and return cosine similarity (0–1).

    Parameters
    ----------
    transcript : str
        Transcribed student explanation.
    reference : str
        Reference concept definition.
    model_name : str
        Sentence-BERT model to use.

    Returns
    -------
    float
        Cosine similarity score in [0, 1].
    """
    evaluator = get_semantic_evaluator(model_name)
    result = evaluator.evaluate(transcript, reference)
    return result.overall_similarity
