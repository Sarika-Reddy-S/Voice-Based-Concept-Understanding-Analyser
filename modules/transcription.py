"""
transcription.py
-----------------
BACKWARD-COMPATIBILITY shim — delegates to speech_to_text.py.

This file is preserved so existing imports of `modules.transcription`
continue to work. All new code should import from `modules.speech_to_text`.
"""

from .speech_to_text import (
    TranscriptionResult,
    Transcriber,
    get_transcriber,
    speech_to_text,
    filler_word_ratio,
)

__all__ = [
    "TranscriptionResult",
    "Transcriber",
    "get_transcriber",
    "speech_to_text",
    "filler_word_ratio",
]
