"""
modules/__init__.py
-------------------
Public API for the VBCUA modules package.

All primary entry-point functions are exported here so callers
can simply do:
    from modules import speech_to_text, semantic_similarity, evaluate_understanding
"""

from .audio_utils import (
    AudioFeatures,
    extract_audio_features,
    extract_features,           # backward-compat alias
    generate_waveform_data,
    save_waveform,
    compute_speaking_rate,
    get_audio_metadata,
)

from .speech_to_text import (
    TranscriptionResult,
    Transcriber,
    get_transcriber,
    speech_to_text,
    filler_word_ratio,
)

from .semantic_eval import (
    SemanticResult,
    SemanticEvaluator,
    get_semantic_evaluator,
    semantic_similarity,
)

from .scoring_engine import (
    ScoreBreakdown,
    evaluate_understanding,
    compute_score,
    understanding_level_for,
    colour_for_level,
)

from .filler_words import FillerWordStats, analyze_filler_words
from .data_storage import (
    init_db,
    get_or_create_user,
    start_session,
    end_session,
    save_audio_file,
    get_or_create_reference_concept,
    save_transcript,
    save_audio_feature,
    save_filler_word_stats,
    save_semantic_similarity,
    save_evaluation_result,
    save_report,
    get_session_history,
    get_result_detail,
)

from .report_generator import generate_pdf_report

__all__ = [
    # audio_utils
    "AudioFeatures",
    "extract_audio_features",
    "extract_features",
    "generate_waveform_data",
    "save_waveform",
    "compute_speaking_rate",
    "get_audio_metadata",
    # speech_to_text
    "TranscriptionResult",
    "Transcriber",
    "get_transcriber",
    "speech_to_text",
    "filler_word_ratio",
    # semantic_eval
    "SemanticResult",
    "SemanticEvaluator",
    "get_semantic_evaluator",
    "semantic_similarity",
    # scoring_engine
    "ScoreBreakdown",
    "evaluate_understanding",
    "compute_score",
    "understanding_level_for",
    "colour_for_level",
    # filler_words
    "FillerWordStats",
    "analyze_filler_words",
    # data_storage
    "init_db",
    "get_or_create_user",
    "start_session",
    "end_session",
    "save_audio_file",
    "get_or_create_reference_concept",
    "save_transcript",
    "save_audio_feature",
    "save_filler_word_stats",
    "save_semantic_similarity",
    "save_evaluation_result",
    "save_report",
    "get_session_history",
    "get_result_detail",
    # report_generator
    "generate_pdf_report",
]
