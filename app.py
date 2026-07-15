"""
app.py
------
Voice-Based Concept Understanding Analyser (VBCUA) — Streamlit frontend.

Ties together transcription, semantic evaluation, audio feature
extraction, filler-word analysis, scoring, PDF reporting, and the
full relational session/audio/evaluation persistence layer into a
single interactive web application.

Run with:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from modules import data_storage
from modules.audio_features import (
    extract_features,
    generate_waveform_data,
    get_audio_metadata,
)
from modules.filler_words import analyze_filler_words
from modules.report_generator import generate_pdf_report
from modules.scoring import compute_score
from modules.semantic_analysis import get_semantic_evaluator
from modules.transcription import get_transcriber

st.set_page_config(
    page_title="VBCUA — Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide",
)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def plot_waveform(y: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(10, 2.2))
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(times, y, linewidth=0.6, color="#2563eb")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    fig.tight_layout()
    return fig


def render_sidebar():
    st.sidebar.title("🎙️ VBCUA")
    st.sidebar.caption(
        "Voice-Based Concept Understanding Analyser — evaluates spoken "
        "conceptual explanations using speech-to-text, semantic similarity, "
        "and audio feature analysis."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Learner")
    user_name = st.sidebar.text_input("Your name", value=st.session_state.get("user_name", "Guest"))
    st.session_state["user_name"] = user_name

    st.sidebar.subheader("Model Settings")
    whisper_size = st.sidebar.selectbox(
        "Whisper model size", ["tiny", "base", "small", "medium"], index=1
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Session History")
    try:
        history = data_storage.get_session_history(limit=10)
    except Exception:
        history = []
    if history:
        for h in history:
            score_txt = f"{h['overall_score']:.1f}" if h.get("overall_score") is not None else "—"
            st.sidebar.write(
                f"#{h['result_id']} · {h['concept_title'] or 'Untitled'} · "
                f"{score_txt} ({h.get('understanding_level', '—')})"
            )
    else:
        st.sidebar.caption("No past sessions yet.")
    return whisper_size, user_name


def main():
    whisper_size, user_name = render_sidebar()

    st.title("Voice-Based Concept Understanding Analyser")
    st.write(
        "Upload a recording of yourself explaining a concept, provide the "
        "reference definition, and get an automated understanding, fluency, "
        "and clarity assessment with a downloadable PDF report."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        concept_title = st.text_input("Concept / Topic Title", placeholder="e.g. Photosynthesis")
        reference_text = st.text_area(
            "Reference Explanation",
            placeholder="Paste the correct/expected explanation of the concept here...",
            height=180,
        )
    with col2:
        audio_file = st.file_uploader(
            "Upload Audio Recording", type=["wav", "mp3", "m4a", "ogg", "flac"]
        )
        if audio_file is not None:
            st.audio(audio_file)

    analyze_clicked = st.button("Analyze Recording", type="primary", disabled=audio_file is None)

    if analyze_clicked and audio_file is not None:
        if not concept_title.strip() or not reference_text.strip():
            st.warning("Please provide both a concept title and a reference explanation before analyzing.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        # --- start relational session ---
        user_id = data_storage.get_or_create_user(user_name or "Guest")
        session_id = data_storage.start_session(user_id)

        try:
            with st.spinner("Transcribing speech..."):
                transcriber = get_transcriber(whisper_size)
                transcription = transcriber.transcribe(tmp_path)

            with st.spinner("Extracting audio features..."):
                features = extract_features(tmp_path)
                metadata = get_audio_metadata(tmp_path)

            with st.spinner("Detecting filler words..."):
                filler_stats = analyze_filler_words(transcription.text)

            with st.spinner("Running semantic similarity analysis..."):
                evaluator = get_semantic_evaluator()
                semantic_result = evaluator.evaluate(transcription.text, reference_text)

            score = compute_score(
                semantic_result, features, transcription.word_count, filler_stats.filler_ratio
            )

            # --- persist across the full ER schema ---
            ref_concept_id = data_storage.get_or_create_reference_concept(concept_title, reference_text)

            audio_id = data_storage.save_audio_file(
                user_id=user_id,
                file_name=audio_file.name,
                file_path=tmp_path,
                duration_sec=features.duration_sec,
                status="processed",
            )
            transcript_id = data_storage.save_transcript(audio_id, transcription.text)
            data_storage.save_audio_feature(
                audio_id=audio_id,
                pause_ratio=features.silence_ratio,
                rms_energy=features.rms_energy_mean,
                zero_crossing_rate=features.zero_crossing_rate,
                duration_sec=features.duration_sec,
            )
            data_storage.save_filler_word_stats(transcript_id, filler_stats)
            data_storage.save_semantic_similarity(
                transcript_id, ref_concept_id, semantic_result.overall_similarity
            )
            result_id = data_storage.save_evaluation_result(
                audio_id=audio_id,
                ref_concept_id=ref_concept_id,
                session_id=session_id,
                overall_score=score.overall_score,
                notes="; ".join(score.feedback),
            )

            st.success("Analysis complete!")

            # --- Results ---
            st.subheader("Transcript")
            st.write(transcription.text or "_(no speech detected)_")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Understanding", f"{score.understanding_score:.1f}")
            m2.metric("Fluency", f"{score.fluency_score:.1f}")
            m3.metric("Clarity", f"{score.clarity_score:.1f}")
            m4.metric(
                "Overall", f"{score.overall_score:.1f}",
                data_storage.understanding_level_for(score.overall_score),
            )

            st.subheader("Waveform")
            y, sr = generate_waveform_data(tmp_path)
            st.pyplot(plot_waveform(y, sr))

            with st.expander("Audio Feature Details"):
                st.json(
                    {
                        "duration_sec": features.duration_sec,
                        "tempo_bpm": features.tempo_bpm,
                        "pause_ratio": features.silence_ratio,
                        "pitch_mean_hz": features.pitch_mean_hz,
                        "pitch_std_hz": features.pitch_std_hz,
                        "estimated_pause_count": features.estimated_pause_count,
                        "format": metadata.get("format"),
                        "samplerate": metadata.get("samplerate"),
                    }
                )

            with st.expander("Filler Word Analysis"):
                st.write(f"Total words: {filler_stats.total_words}")
                st.write(f"Filler words: {filler_stats.filler_word_count}")
                st.write(f"Filler ratio: {filler_stats.filler_ratio:.2%}")
                if filler_stats.breakdown:
                    st.table(
                        {"word/phrase": list(filler_stats.breakdown.keys()),
                         "count": list(filler_stats.breakdown.values())}
                    )

            st.subheader("Concept Coverage")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Covered keywords**")
                st.write(", ".join(semantic_result.matched_keywords) or "—")
            with cc2:
                st.markdown("**Missed keywords**")
                st.write(", ".join(semantic_result.missing_keywords) or "—")

            st.subheader("Feedback")
            for f in score.feedback:
                st.markdown(f"- {f}")

            # --- PDF report ---
            wpm = round((transcription.word_count / features.duration_sec) * 60, 2) if features.duration_sec else 0
            report_path = str(REPORTS_DIR / f"vbcua_report_result_{result_id}.pdf")
            generate_pdf_report(
                output_path=report_path,
                concept_title=concept_title,
                transcript=transcription.text,
                reference_text=reference_text,
                score=score,
                audio_features=features,
                wpm=wpm,
                matched_keywords=semantic_result.matched_keywords,
                missing_keywords=semantic_result.missing_keywords,
            )
            file_size_kb = round(os.path.getsize(report_path) / 1024)
            data_storage.save_report(result_id, report_path, file_size_kb)

            with open(report_path, "rb") as f:
                st.download_button(
                    "Download PDF Report",
                    data=f.read(),
                    file_name=os.path.basename(report_path),
                    mime="application/pdf",
                )

            data_storage.end_session(session_id, status="completed")
        except Exception:
            data_storage.end_session(session_id, status="failed")
            raise
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
