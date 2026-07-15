import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modules.data_storage as data_storage
from modules.filler_words import FillerWordStats


def setup_module(module):
    test_db = Path(__file__).resolve().parent / "test_vbcua.db"
    if test_db.exists():
        test_db.unlink()
    data_storage.DB_PATH = test_db
    data_storage.init_db()


def test_get_or_create_user_is_idempotent():
    uid1 = data_storage.get_or_create_user("Sarika")
    uid2 = data_storage.get_or_create_user("Sarika")
    assert uid1 == uid2


def test_start_and_end_session():
    uid = data_storage.get_or_create_user("Learner A")
    session_id = data_storage.start_session(uid)
    assert isinstance(session_id, int)
    data_storage.end_session(session_id, status="completed")


def test_reference_concept_reused_by_title():
    id1 = data_storage.get_or_create_reference_concept("Photosynthesis", "Plants convert light to energy.")
    id2 = data_storage.get_or_create_reference_concept("Photosynthesis", "Updated definition text.")
    assert id1 == id2


def test_full_evaluation_pipeline_persists_across_all_entities():
    uid = data_storage.get_or_create_user("Learner B")
    session_id = data_storage.start_session(uid)
    ref_id = data_storage.get_or_create_reference_concept("Newton's Laws", "Force equals mass times acceleration.")

    audio_id = data_storage.save_audio_file(uid, "clip.wav", "/tmp/clip.wav", duration_sec=12.5)
    transcript_id = data_storage.save_transcript(audio_id, "Force equals mass times acceleration basically.")
    data_storage.save_audio_feature(audio_id, pause_ratio=0.15, rms_energy=0.05, zero_crossing_rate=0.08, duration_sec=12.5)

    stats = FillerWordStats(total_words=8, filler_word_count=1, filler_ratio=0.125, breakdown={"basically": 1})
    data_storage.save_filler_word_stats(transcript_id, stats)
    data_storage.save_semantic_similarity(transcript_id, ref_id, 0.91)

    result_id = data_storage.save_evaluation_result(audio_id, ref_id, session_id, overall_score=88.0)
    data_storage.save_report(result_id, "/tmp/report.pdf", file_size_kb=42)
    data_storage.end_session(session_id, status="completed")

    detail = data_storage.get_result_detail(result_id)
    assert detail["result"]["understanding_level"] == "Strong"
    assert detail["audio_file"]["file_name"] == "clip.wav"
    assert detail["transcript"]["transcript_id"] == transcript_id
    assert detail["filler_word_stats"]["filler_word_count"] == 1
    assert detail["semantic_similarity"]["similarity_score"] == 0.91
    assert detail["report"]["file_size_kb"] == 42


def test_understanding_level_thresholds():
    assert data_storage.understanding_level_for(80) == "Strong"
    assert data_storage.understanding_level_for(60) == "Moderate"
    assert data_storage.understanding_level_for(30) == "Poor"


def test_session_history_reflects_new_result():
    history = data_storage.get_session_history(limit=5)
    assert any(h["concept_title"] == "Newton's Laws" for h in history)


def teardown_module(module):
    if data_storage.DB_PATH.exists():
        data_storage.DB_PATH.unlink()
