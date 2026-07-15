"""
audio_utils.py
--------------
Audio loading, feature extraction, and waveform utilities for VBCUA.

This module provides the primary audio processing pipeline:
  - Loading and resampling audio files via Librosa
  - Extracting acoustic features (pause ratio, RMS energy, ZCR, duration)
  - Generating waveform data for Streamlit visualization
  - Saving waveform plots to disk for PDF embedding

Extracted features feed the scoring engine and the AUDIO_FEATURE
entity in the relational storage layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


@dataclass
class AudioFeatures:
    """Container for acoustic features extracted from an audio file."""
    duration_sec: float
    tempo_bpm: float
    pause_ratio: float          # proportion of audio frames below energy threshold
    rms_energy: float           # mean RMS energy
    rms_energy_std: float
    zero_crossing_rate: float   # mean ZCR
    pitch_mean_hz: float
    pitch_std_hz: float
    estimated_pause_count: int

    # Aliases for backward compatibility with existing code
    @property
    def silence_ratio(self) -> float:
        return self.pause_ratio

    @property
    def rms_energy_mean(self) -> float:
        return self.rms_energy


def load_audio(audio_path: str, target_sr: int = 16_000):
    """Load an audio file and resample to target_sr. Returns (y, sr)."""
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return y, sr


def extract_audio_features(audio_path: str) -> AudioFeatures:
    """
    Extract acoustic features relevant to speech fluency and delivery.

    Parameters
    ----------
    audio_path : str
        Path to an audio file (WAV, MP3, M4A, OGG, FLAC, …).

    Returns
    -------
    AudioFeatures
    """
    y, sr = load_audio(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    # Tempo — rough proxy for speaking rhythm
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # RMS energy — detect silence / pauses
    rms = librosa.feature.rms(y=y)[0]
    silence_threshold = float(np.percentile(rms, 20)) if len(rms) else 0.0
    silence_frames = rms < max(silence_threshold, 1e-4)
    pause_ratio = float(np.mean(silence_frames)) if len(silence_frames) else 0.0

    # Count contiguous silent regions as estimated pause count
    pause_count = 0
    prev = False
    for is_silent in silence_frames:
        if is_silent and not prev:
            pause_count += 1
        prev = bool(is_silent)

    # Pitch tracking
    f0, _, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    pitch_mean = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
    pitch_std = float(np.std(voiced_f0)) if voiced_f0.size else 0.0

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr)) if len(zcr) else 0.0

    rms_mean = float(np.mean(rms)) if len(rms) else 0.0
    rms_std = float(np.std(rms)) if len(rms) else 0.0

    return AudioFeatures(
        duration_sec=round(duration, 2),
        tempo_bpm=round(tempo, 2),
        pause_ratio=round(pause_ratio, 4),
        rms_energy=round(rms_mean, 6),
        rms_energy_std=round(rms_std, 6),
        zero_crossing_rate=round(zcr_mean, 6),
        pitch_mean_hz=round(pitch_mean, 2),
        pitch_std_hz=round(pitch_std, 2),
        estimated_pause_count=pause_count,
    )


# Keep backward-compatible alias
def extract_features(audio_path: str) -> AudioFeatures:
    """Alias for extract_audio_features (backward compatibility)."""
    return extract_audio_features(audio_path)


def compute_speaking_rate(word_count: int, duration_sec: float) -> float:
    """Return words-per-minute; guarded against divide-by-zero."""
    if duration_sec <= 0:
        return 0.0
    return round((word_count / duration_sec) * 60, 2)


def generate_waveform_data(audio_path: str, target_sr: int = 16_000):
    """Return (y, sr) suitable for waveform plotting in Streamlit."""
    return load_audio(audio_path, target_sr=target_sr)


def save_waveform(audio_path: str, output_dir: str | None = None) -> str:
    """
    Render and save a waveform PNG to output_dir (or alongside the audio).

    Returns the absolute path to the saved PNG.
    """
    y, sr = load_audio(audio_path)
    times = np.linspace(0, len(y) / sr, num=len(y))

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(times, y, linewidth=0.5, color="#2196F3")
    ax.set_title("Audio Waveform", fontsize=11)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.spines[:].set_color("#444")
    fig.tight_layout()

    stem = Path(audio_path).stem
    out_dir = Path(output_dir) if output_dir else Path(audio_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = str(out_dir / f"waveform_{stem}.png")
    fig.savefig(img_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return img_path


def get_audio_metadata(audio_path: str) -> dict:
    """Basic file metadata via SoundFile."""
    try:
        info = sf.info(audio_path)
        return {
            "samplerate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "frames": info.frames,
            "duration_sec": round(info.frames / info.samplerate, 2) if info.samplerate else 0.0,
        }
    except Exception:
        return {"samplerate": 0, "channels": 0, "format": "unknown", "duration_sec": 0.0}
