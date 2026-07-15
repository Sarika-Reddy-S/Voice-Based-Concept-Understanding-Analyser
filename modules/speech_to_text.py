"""
speech_to_text.py
-----------------
Whisper-based speech-to-text transcription module for VBCUA.

Provides:
  - `speech_to_text(audio_path)` — simple one-call transcription
  - `Transcriber` class — cached model wrapper for repeated calls
  - `filler_word_ratio(transcript)` — quick filler-word percentage

Whisper supports WAV, MP3, M4A and anything ffmpeg can decode.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import Optional

import whisper

# Common filler words / phrases for quick detection
_FILLER_PHRASES = ["you know", "i mean", "kind of", "sort of", "i guess"]
_FILLER_WORDS = [
    "um", "uh", "umm", "uhh", "hmm", "like", "actually",
    "basically", "literally", "so", "well", "right", "okay",
]


@dataclass
class TranscriptionResult:
    """Container for a Whisper transcription result."""

    text: str
    language: str
    segments: list = field(default_factory=list)
    duration: float = 0.0

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0


class Transcriber:
    """Thin wrapper around a Whisper model with lazy loading."""

    def __init__(self, model_size: str = "base"):
        """
        Parameters
        ----------
        model_size : str
            'tiny', 'base', 'small', 'medium', or 'large'.
            'base' balances speed and accuracy well for demos.
        """
        self.model_size = model_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Parameters
        ----------
        audio_path : str
            Path to a .wav/.mp3/.m4a file (anything ffmpeg can decode).
        language : Optional[str]
            Force a language code (e.g. 'en'). None = auto-detect.

        Returns
        -------
        TranscriptionResult
        """
        options: dict = {"task": "transcribe"}
        if language:
            options["language"] = language

        result = self.model.transcribe(audio_path, **options)

        duration = 0.0
        segs = result.get("segments") or []
        if segs:
            duration = segs[-1].get("end", 0.0)

        return TranscriptionResult(
            text=(result.get("text") or "").strip(),
            language=result.get("language", "unknown"),
            segments=segs,
            duration=duration,
        )


@functools.lru_cache(maxsize=4)
def get_transcriber(model_size: str = "base") -> Transcriber:
    """Cached factory — prevents model reloads on every Streamlit rerun."""
    return Transcriber(model_size=model_size)


def speech_to_text(audio_path: str, model_size: str = "base") -> str:
    """
    Convenience function: transcribe an audio file and return plain text.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    model_size : str
        Whisper model size (default 'base').

    Returns
    -------
    str
        Transcribed text (empty string if no speech detected).
    """
    transcriber = get_transcriber(model_size)
    result = transcriber.transcribe(audio_path)
    return result.text


def filler_word_ratio(transcript: str) -> float:
    """
    Compute the fraction of words that are filler words/phrases.

    Parameters
    ----------
    transcript : str
        Transcribed speech text.

    Returns
    -------
    float
        Ratio of filler tokens to total word tokens (0.0 – 1.0).
    """
    if not transcript or not transcript.strip():
        return 0.0

    text = transcript.lower()
    total_words = len(re.findall(r"[A-Za-z']+", text))
    if total_words == 0:
        return 0.0

    working = text
    filler_count = 0

    for phrase in _FILLER_PHRASES:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        matches = re.findall(pattern, working)
        filler_count += len(matches)
        working = re.sub(pattern, " ", working)

    for word in _FILLER_WORDS:
        pattern = r"\b" + re.escape(word) + r"\b"
        filler_count += len(re.findall(pattern, working))

    return round(filler_count / total_words, 4)
