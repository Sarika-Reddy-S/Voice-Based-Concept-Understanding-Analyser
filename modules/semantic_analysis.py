"""
semantic_analysis.py
---------------------
Semantic similarity evaluation using Sentence-BERT (SBERT).

Compares a spoken explanation (already transcribed to text) against
a reference concept definition and returns a similarity score along
with sentence-level breakdowns useful for feedback.
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
    overall_similarity: float          # 0-1 cosine similarity vs reference
    sentence_scores: List[float] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        return round(self.overall_similarity * 100, 2)


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def _extract_keywords(text: str, min_len: int = 4) -> set:
    words = re.findall(r"[A-Za-z']+", text.lower())
    stopwords = {
        "the", "and", "that", "this", "with", "from", "have", "which",
        "there", "their", "would", "could", "about", "into", "your",
        "some", "than", "then", "them", "these", "those", "were", "been",
    }
    return {w for w in words if len(w) >= min_len and w not in stopwords}


class SemanticEvaluator:
    """Wraps a Sentence-BERT model for concept-explanation similarity scoring."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def evaluate(self, spoken_text: str, reference_text: str) -> SemanticResult:
        """
        Compare the spoken explanation to a reference concept definition.

        Parameters
        ----------
        spoken_text : str
            Transcript produced by the transcription module.
        reference_text : str
            The "gold standard" explanation of the concept being tested.

        Returns
        -------
        SemanticResult
        """
        if not spoken_text.strip() or not reference_text.strip():
            return SemanticResult(overall_similarity=0.0)

        overall_emb = self.model.encode(
            [spoken_text, reference_text], convert_to_tensor=True
        )
        overall_similarity = float(util.cos_sim(overall_emb[0], overall_emb[1])[0][0])
        overall_similarity = max(0.0, min(1.0, overall_similarity))

        spoken_sentences = _split_sentences(spoken_text)
        sentence_scores = []
        if spoken_sentences:
            ref_emb = self.model.encode(reference_text, convert_to_tensor=True)
            sent_embs = self.model.encode(spoken_sentences, convert_to_tensor=True)
            sims = util.cos_sim(sent_embs, ref_emb).cpu().numpy().flatten()
            sentence_scores = [float(round(s, 4)) for s in sims]

        ref_keywords = _extract_keywords(reference_text)
        spoken_keywords = _extract_keywords(spoken_text)
        matched = sorted(ref_keywords & spoken_keywords)
        missing = sorted(ref_keywords - spoken_keywords)

        return SemanticResult(
            overall_similarity=overall_similarity,
            sentence_scores=sentence_scores,
            matched_keywords=matched,
            missing_keywords=missing,
        )


@functools.lru_cache(maxsize=1)
def get_semantic_evaluator(model_name: str = "all-MiniLM-L6-v2") -> SemanticEvaluator:
    """Cached factory so Streamlit doesn't reload SBERT on every rerun."""
    return SemanticEvaluator(model_name=model_name)
