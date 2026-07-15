"""
app.py
------
Voice-Based Concept Understanding Analyser (VBCUA) — Streamlit frontend.

Ties together transcription, semantic evaluation, audio feature
extraction, scoring, PDF reporting, and session persistence into a
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
            st.sidebar.write(f"#{h['id']} · {h['concept_title'] or 'Untitled'} · {score_txt}")
    else:
        st.sidebar.caption("No past sessions yet.")
    return whisper_size


def main():
    whisper_size = render_sidebar()

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
        if not reference_text.strip():
            st.warning("Please provide a reference explanation before analyzing.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Transcribing speech..."):
                transcriber = get_transcriber(whisper_size)
                transcription = transcriber.transcribe(tmp_path)

            with st.spinner("Extracting audio features..."):
                features = extract_features(tmp_path)
                metadata = get_audio_metadata(tmp_path)

            with st.spinner("Running semantic similarity analysis..."):
                evaluator = get_semantic_evaluator()
                semantic_result = evaluator.evaluate(transcription.text, reference_text)

            score = compute_score(semantic_result, features, transcription.word_count)

            # --- Persist session ---
            session_id = data_storage.create_session(concept_title, reference_text)
            data_storage.save_transcription(
                session_id, transcription.text, transcription.language, transcription.duration
            )
            data_storage.save_audio_features(session_id, features)
            data_storage.save_score(session_id, score)

            st.success("Analysis complete!")

            # --- Results ---
            st.subheader("Transcript")
            st.write(transcription.text or "_(no speech detected)_")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Understanding", f"{score.understanding_score:.1f}")
            m2.metric("Fluency", f"{score.fluency_score:.1f}")
            m3.metric("Clarity", f"{score.clarity_score:.1f}")
            m4.metric("Overall", f"{score.overall_score:.1f}", score.grade)

            st.subheader("Waveform")
            y, sr = generate_waveform_data(tmp_path)
            st.pyplot(plot_waveform(y, sr))

            with st.expander("Audio Feature Details"):
                st.json(
                    {
                        "duration_sec": features.duration_sec,
                        "tempo_bpm": features.tempo_bpm,
                        "silence_ratio": features.silence_ratio,
                        "pitch_mean_hz": features.pitch_mean_hz,
                        "pitch_std_hz": features.pitch_std_hz,
                        "estimated_pause_count": features.estimated_pause_count,
                        "format": metadata.get("format"),
                        "samplerate": metadata.get("samplerate"),
                    }
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
            report_path = str(REPORTS_DIR / f"vbcua_report_session_{session_id}.pdf")
            generate_pdf_report(
                output_path=report_path,
                concept_title=concept_title or "Untitled Concept",
                transcript=transcription.text,
                reference_text=reference_text,
                score=score,
                audio_features=features,
                wpm=wpm,
                matched_keywords=semantic_result.matched_keywords,
                missing_keywords=semantic_result.missing_keywords,
            )
            with open(report_path, "rb") as f:
                st.download_button(
                    "Download PDF Report",
                    data=f.read(),
                    file_name=os.path.basename(report_path),
                    mime="application/pdf",
                )
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
