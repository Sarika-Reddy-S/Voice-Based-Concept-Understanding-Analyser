"""
audio_features.py
------------------
Audio signal processing and fluency-related feature extraction
using Librosa and SoundFile.

Extracts acoustic features that feed the fluency component of the
overall scoring engine: speaking rate, pause/silence ratio, pitch
variation, and energy stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf


@dataclass
class AudioFeatures:
    duration_sec: float
    tempo_bpm: float
    silence_ratio: float          # proportion of audio below energy threshold
    pitch_mean_hz: float
    pitch_std_hz: float
    rms_energy_mean: float
    rms_energy_std: float
    zero_crossing_rate: float
    estimated_pause_count: int


def load_audio(audio_path: str, target_sr: int = 16000):
    """Load audio file and resample to target_sr, returns (y, sr)."""
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return y, sr


def extract_features(audio_path: str) -> AudioFeatures:
    """
    Extract acoustic features relevant to fluency and delivery quality.

    Parameters
    ----------
    audio_path : str
        Path to the audio file to analyze.

    Returns
    -------
    AudioFeatures
    """
    y, sr = load_audio(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    # Tempo (used as a rough proxy for speaking rhythm/rate)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # Energy-based silence / pause detection
    rms = librosa.feature.rms(y=y)[0]
    silence_threshold = np.percentile(rms, 20) if len(rms) else 0.0
    silence_frames = rms < max(silence_threshold, 1e-4)
    silence_ratio = float(np.mean(silence_frames)) if len(silence_frames) else 0.0

    # Count contiguous silent regions as an estimate of pause count
    pause_count = 0
    prev = False
    for is_silent in silence_frames:
        if is_silent and not prev:
            pause_count += 1
        prev = is_silent

    # Pitch (fundamental frequency) tracking
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    pitch_mean = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
    pitch_std = float(np.std(voiced_f0)) if voiced_f0.size else 0.0

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    return AudioFeatures(
        duration_sec=round(float(duration), 2),
        tempo_bpm=round(tempo, 2),
        silence_ratio=round(silence_ratio, 4),
        pitch_mean_hz=round(pitch_mean, 2),
        pitch_std_hz=round(pitch_std, 2),
        rms_energy_mean=round(float(np.mean(rms)), 6) if len(rms) else 0.0,
        rms_energy_std=round(float(np.std(rms)), 6) if len(rms) else 0.0,
        zero_crossing_rate=round(float(np.mean(zcr)), 6) if len(zcr) else 0.0,
        estimated_pause_count=pause_count,
    )


def compute_speaking_rate(word_count: int, duration_sec: float) -> float:
    """Words per minute, guarded against divide-by-zero for very short clips."""
    if duration_sec <= 0:
        return 0.0
    return round((word_count / duration_sec) * 60, 2)


def generate_waveform_data(audio_path: str, target_sr: int = 16000):
    """Return (y, sr) for waveform plotting in the Streamlit frontend."""
    return load_audio(audio_path, target_sr=target_sr)


def get_audio_metadata(audio_path: str) -> dict:
    """Basic file metadata via SoundFile (sample rate, channels, format)."""
    info = sf.info(audio_path)
    return {
        "samplerate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
        "frames": info.frames,
        "duration_sec": round(info.frames / info.samplerate, 2) if info.samplerate else 0.0,
    }
