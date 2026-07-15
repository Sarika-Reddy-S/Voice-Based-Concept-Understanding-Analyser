"""
filler_words.py
----------------
Filler-word detection and speech-disfluency statistics.

Analyzes a transcript for common filler words/phrases ("um", "uh",
"like", "you know", etc.) and computes the ratio of filler words to
total words spoken — feeding the FILLER_WORD_STATS entity and the
fluency side of the scoring engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict

# Common English filler words / hedge phrases. Multi-word phrases are
# matched first so they aren't double-counted as their component words.
FILLER_PHRASES = [
    "you know", "i mean", "kind of", "sort of", "i guess",
]
FILLER_WORDS = [
    "um", "uh", "umm", "uhh", "hmm", "like", "actually",
    "basically", "literally", "so", "well", "right", "okay",
]


@dataclass
class FillerWordStats:
    total_words: int
    filler_word_count: int
    filler_ratio: float                    # filler_word_count / total_words
    breakdown: Dict[str, int] = field(default_factory=dict)


def analyze_filler_words(transcript_text: str) -> FillerWordStats:
    """
    Count filler words/phrases in a transcript and compute the filler ratio.

    Parameters
    ----------
    transcript_text : str
        The transcribed speech text.

    Returns
    -------
    FillerWordStats
    """
    if not transcript_text or not transcript_text.strip():
        return FillerWordStats(total_words=0, filler_word_count=0, filler_ratio=0.0)

    text = transcript_text.lower()
    total_words = len(re.findall(r"[A-Za-z']+", text))

    breakdown: Dict[str, int] = {}
    working_text = text

    # Multi-word phrases first, removing matches so component words
    # ("know", "mean", "guess") aren't recounted individually.
    for phrase in FILLER_PHRASES:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        matches = re.findall(pattern, working_text)
        if matches:
            breakdown[phrase] = len(matches)
            working_text = re.sub(pattern, " ", working_text)

    for word in FILLER_WORDS:
        pattern = r"\b" + re.escape(word) + r"\b"
        matches = re.findall(pattern, working_text)
        if matches:
            breakdown[word] = breakdown.get(word, 0) + len(matches)

    filler_word_count = sum(breakdown.values())
    filler_ratio = round(filler_word_count / total_words, 4) if total_words else 0.0

    return FillerWordStats(
        total_words=total_words,
        filler_word_count=filler_word_count,
        filler_ratio=filler_ratio,
        breakdown=dict(sorted(breakdown.items(), key=lambda kv: -kv[1])),
    )
