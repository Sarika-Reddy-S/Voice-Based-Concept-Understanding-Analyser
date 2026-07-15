"""
audio_features.py
-----------------
BACKWARD-COMPATIBILITY shim — delegates to audio_utils.py.

This file is preserved so existing imports of `modules.audio_features`
continue to work. All new code should import from `modules.audio_utils`.
"""

from .audio_utils import (
    AudioFeatures,
    extract_audio_features,
    extract_features,
    generate_waveform_data,
    save_waveform,
    compute_speaking_rate,
    get_audio_metadata,
    load_audio,
)

__all__ = [
    "AudioFeatures",
    "extract_audio_features",
    "extract_features",
    "generate_waveform_data",
    "save_waveform",
    "compute_speaking_rate",
    "get_audio_metadata",
    "load_audio",
]
