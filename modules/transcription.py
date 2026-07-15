"""
transcription.py
-----------------
Speech-to-text transcription module built on OpenAI Whisper.

Provides a thin, cached wrapper around Whisper so the rest of the
application (Streamlit frontend, scoring engine) can request a
transcript without worrying about model lifecycle management.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Optional

import whisper


@dataclass
class TranscriptionResult:
    """Container for a Whisper transcription result."""

    text: str
    language: str
    segments: list = field(default_factory=list)
    duration: float = 0.0

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class Transcriber:
    """Wraps a Whisper model instance for repeated transcription calls."""

    def __init__(self, model_size: str = "base"):
        """
        Parameters
        ----------
        model_size : str
            One of Whisper's model sizes: tiny, base, small, medium, large.
            "base" is a reasonable default balancing speed and accuracy
            for classroom / demo hardware.
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
        Transcribe an audio file on disk.

        Parameters
        ----------
        audio_path : str
            Path to a .wav/.mp3/.m4a file (anything ffmpeg can decode).
        language : Optional[str]
            Force a language code (e.g. "en"). If None, Whisper auto-detects.

        Returns
        -------
        TranscriptionResult
        """
        options = {"task": "transcribe"}
        if language:
            options["language"] = language

        result = self.model.transcribe(audio_path, **options)

        duration = 0.0
        if result.get("segments"):
            duration = result["segments"][-1].get("end", 0.0)

        return TranscriptionResult(
            text=result.get("text", "").strip(),
            language=result.get("language", "unknown"),
            segments=result.get("segments", []),
            duration=duration,
        )


@functools.lru_cache(maxsize=1)
def get_transcriber(model_size: str = "base") -> Transcriber:
    """Cached factory so Streamlit doesn't reload the Whisper model on every rerun."""
    return Transcriber(model_size=model_size)
