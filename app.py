"""
app.py  (also runnable as main.py)
-----------------------------------
Voice-Based Concept Understanding Analyser (VBCUA) — Streamlit frontend.

Ties together:
  - speech_to_text     → Whisper transcription
  - semantic_eval      → Sentence-BERT similarity
  - audio_utils        → Librosa feature extraction + waveform
  - scoring_engine     → evaluate_understanding + compute_score
  - filler_words       → filler-word detection
  - data_storage       → full SQLite relational persistence (ER schema)
  - report_generator   → ReportLab PDF reports

Run with:
    streamlit run app.py
    # or:
    streamlit run main.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ── Module imports ────────────────────────────────────────────────────────────
from modules import data_storage
from modules.audio_utils import (
    extract_audio_features,
    generate_waveform_data,
    get_audio_metadata,
    save_waveform,
)
from modules.filler_words import analyze_filler_words
from modules.report_generator import generate_pdf_report
from modules.scoring_engine import (
    evaluate_understanding,
    compute_score,
    understanding_level_for,
    colour_for_level,
)
from modules.semantic_eval import get_semantic_evaluator
from modules.speech_to_text import get_transcriber, filler_word_ratio

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide",
)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0e1117;
}

/* Main title styling */
.vbcua-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    text-align: center;
    margin-bottom: 0.2rem;
}
.vbcua-subtitle {
    font-size: 0.95rem;
    color: #8b949e;
    text-align: center;
    margin-bottom: 2rem;
}

/* Card-style containers */
.metric-card {
    background: linear-gradient(135deg, #1c2333, #1a1f2e);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
}
.metric-label {
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Analysis completed banner */
.analysis-banner {
    background: linear-gradient(90deg, #1a4731, #1e5a3a);
    border: 1px solid #2ecc71;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #2ecc71;
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Score display card */
.score-display {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.score-number {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
}
.score-label {
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 0.3rem;
    text-transform: uppercase;
}
.understanding-level {
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

/* Transcript box */
.transcript-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem;
    color: #c9d1d9;
    font-size: 0.92rem;
    line-height: 1.65;
    min-height: 120px;
}

/* Section headers */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #c9d1d9;
    margin-bottom: 0.6rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #21262d;
}

/* Feedback items */
.feedback-item {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    padding: 0.6rem 0.9rem;
    border-radius: 0 6px 6px 0;
    color: #c9d1d9;
    font-size: 0.88rem;
    margin-bottom: 0.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #047857);
    color: white;
    border: none;
    border-radius: 8px;
    width: 100%;
    font-weight: 600;
}

/* Info banner */
.info-banner {
    background: #0d2137;
    border: 1px solid #1f6feb;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    color: #58a6ff;
    font-size: 0.88rem;
}

/* Metric row */
.sim-metric {
    background: #1c2333;
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
    border: 1px solid #30363d;
}
.sim-metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #58a6ff;
}
.sim-metric-label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def plot_waveform_fig(y: np.ndarray, sr: int) -> plt.Figure:
    """Return a dark-themed Matplotlib waveform figure."""
    fig, ax = plt.subplots(figsize=(10, 2.4))
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(times, y, linewidth=0.45, color="#2196F3")
    ax.set_xlabel("Time", color="#8b949e")
    ax.set_ylabel("Amplitude", color="#8b949e")
    ax.set_title("Audio Waveform", color="#c9d1d9")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    fig.tight_layout()
    return fig


def render_sidebar() -> tuple[str, str]:
    """Render sidebar and return (whisper_size, user_name)."""
    st.sidebar.markdown(
        "<div style='font-size:1.2rem;font-weight:700;color:#58a6ff'>🎙️ VBCUA</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "Voice-Based Concept Understanding Analyser — automated evaluation "
        "of spoken conceptual explanations using AI."
    )
    st.sidebar.markdown("---")

    st.sidebar.subheader("👤 Learner")
    user_name = st.sidebar.text_input(
        "Your name", value=st.session_state.get("user_name", "Student")
    )
    st.session_state["user_name"] = user_name

    st.sidebar.subheader("⚙️ Model Settings")
    whisper_size = st.sidebar.selectbox(
        "Whisper model size",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="Larger models are more accurate but slower.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Session History")
    try:
        history = data_storage.get_session_history(limit=10)
    except Exception:
        history = []
    if history:
        for h in history:
            score_txt = f"{h['overall_score']:.1f}" if h.get("overall_score") is not None else "—"
            level = h.get("understanding_level", "—")
            st.sidebar.markdown(
                f"<div style='font-size:0.8rem;color:#8b949e;padding:0.2rem 0'>"
                f"#{h['result_id']} · <b style='color:#c9d1d9'>{h.get('concept_title') or 'Untitled'}</b>"
                f"<br>Score: <b style='color:#58a6ff'>{score_txt}</b> · {level}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.caption("No past sessions yet.")

    return whisper_size, user_name


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    whisper_size, user_name = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='vbcua-title'>Voice-Based Concept Understanding Analyser</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='vbcua-subtitle'>Automated evaluation of spoken conceptual explanations using AI.</div>",
        unsafe_allow_html=True,
    )

    # ── Input Section ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='section-header'>📤 Upload Student Audio (WAV)</div>", unsafe_allow_html=True)
        audio_file = st.file_uploader(
            "Drag and drop or browse",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            label_visibility="collapsed",
            help="Supported formats: WAV, MP3, M4A, OGG, FLAC · Max 200 MB",
        )
        if audio_file is not None:
            st.audio(audio_file)

    with col_right:
        st.markdown("<div class='section-header'>📖 Concept Reference</div>", unsafe_allow_html=True)
        concept_title = st.text_input(
            "Concept / Topic Title",
            placeholder="e.g. Machine Learning",
            label_visibility="collapsed",
        )
        reference_text = st.text_area(
            "Reference Explanation",
            placeholder=(
                "Machine Learning is a subset of artificial intelligence that allows "
                "systems to learn patterns from data and improve performance without "
                "being explicitly programmed."
            ),
            height=150,
            label_visibility="collapsed",
        )

    # Info or prompt
    if audio_file is None:
        st.markdown(
            "<div class='info-banner'>📁 Upload an audio file to begin analysis.</div>",
            unsafe_allow_html=True,
        )

    # ── Analyze button ────────────────────────────────────────────────────────
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        analyze_clicked = st.button(
            "🔍 Analyze Concept Understanding",
            disabled=(audio_file is None),
        )

    # ── Analysis Pipeline ─────────────────────────────────────────────────────
    if analyze_clicked and audio_file is not None:
        if not concept_title.strip():
            st.warning("⚠️ Please provide a concept/topic title.")
            return
        if not reference_text.strip():
            st.warning("⚠️ Please paste the reference explanation.")
            return

        # Save uploaded audio to a temp file
        suffix = Path(audio_file.name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name

        # Start relational session
        user_id = data_storage.get_or_create_user(user_name or "Student")
        session_id = data_storage.start_session(user_id)

        try:
            with st.spinner("🎙️ Transcribing speech with Whisper…"):
                transcriber = get_transcriber(whisper_size)
                tr = transcriber.transcribe(audio_path)
                transcript_text = tr.text

            with st.spinner("📊 Extracting audio features…"):
                audio_features = extract_audio_features(audio_path)
                metadata = get_audio_metadata(audio_path)
                waveform_img = save_waveform(audio_path, output_dir=str(ASSETS_DIR))

            with st.spinner("🔍 Detecting filler words…"):
                filler_stats = analyze_filler_words(transcript_text)
                f_ratio = filler_stats.filler_ratio

            with st.spinner("🧠 Computing semantic similarity…"):
                evaluator = get_semantic_evaluator()
                sem_result = evaluator.evaluate(transcript_text, reference_text)

            # Primary scoring (matches screenshot formula exactly)
            audio_dict = {
                "pause_ratio": audio_features.pause_ratio,
                "rms_energy":  audio_features.rms_energy,
            }
            score_val, level, colour = evaluate_understanding(
                sem_result.overall_similarity, f_ratio, audio_dict
            )

            # Detailed sub-score breakdown
            detailed = compute_score(sem_result, audio_features, tr.word_count, f_ratio)

            # ── Persist to SQLite (full ER schema) ────────────────────────────
            ref_id = data_storage.get_or_create_reference_concept(concept_title, reference_text)
            audio_id = data_storage.save_audio_file(
                user_id=user_id,
                file_name=audio_file.name,
                file_path=audio_path,
                duration_sec=audio_features.duration_sec,
                status="processed",
            )
            transcript_id = data_storage.save_transcript(audio_id, transcript_text)
            data_storage.save_audio_feature(
                audio_id=audio_id,
                pause_ratio=audio_features.pause_ratio,
                rms_energy=audio_features.rms_energy,
                zero_crossing_rate=audio_features.zero_crossing_rate,
                duration_sec=audio_features.duration_sec,
            )
            data_storage.save_filler_word_stats(transcript_id, filler_stats)
            data_storage.save_semantic_similarity(transcript_id, ref_id, sem_result.overall_similarity)
            result_id = data_storage.save_evaluation_result(
                audio_id=audio_id,
                ref_concept_id=ref_id,
                session_id=session_id,
                overall_score=float(score_val),
                notes="; ".join(detailed.feedback),
            )

            # ── Results display ───────────────────────────────────────────────
            st.markdown(
                "<div class='analysis-banner'>✅ Analysis Completed</div>",
                unsafe_allow_html=True,
            )

            res_col1, res_col2 = st.columns([1.4, 1], gap="large")

            with res_col1:
                st.markdown("<div class='section-header'>📝 Transcribed Explanation</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='transcript-box'>{transcript_text or '<i>(no speech detected)</i>'}</div>",
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.markdown("<div class='section-header'>🎯 Final Evaluation</div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class='score-display'>
                        <div class='score-label'>Understanding Score</div>
                        <div class='score-number' style='color:{colour}'>{score_val}/100</div>
                        <div class='understanding-level' style='color:{colour}'>{level}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Metrics row ───────────────────────────────────────────────────
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f"<div class='sim-metric'>"
                    f"<div class='sim-metric-value'>{sem_result.overall_similarity:.2f}</div>"
                    f"<div class='sim-metric-label'>Semantic Similarity</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"<div class='sim-metric'>"
                    f"<div class='sim-metric-value'>{f_ratio:.2f}</div>"
                    f"<div class='sim-metric-label'>Filler Word Ratio</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"<div class='sim-metric'>"
                    f"<div class='sim-metric-value'>{audio_features.rms_energy:.4f}</div>"
                    f"<div class='sim-metric-label'>Confidence (Energy)</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── Waveform ──────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("<div class='section-header'>📈 Audio Waveform</div>", unsafe_allow_html=True)
            y_wave, sr_wave = generate_waveform_data(audio_path)
            st.pyplot(plot_waveform_fig(y_wave, sr_wave))

            # ── Detailed sub-scores ───────────────────────────────────────────
            with st.expander("📊 Detailed Score Breakdown"):
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Understanding", f"{detailed.understanding_score:.1f}/100")
                dc2.metric("Fluency", f"{detailed.fluency_score:.1f}/100")
                dc3.metric("Clarity", f"{detailed.clarity_score:.1f}/100")
                dc4.metric("Grade", detailed.grade)

            # ── Audio features ────────────────────────────────────────────────
            with st.expander("🔊 Audio Feature Details"):
                st.json({
                    "duration_sec": audio_features.duration_sec,
                    "tempo_bpm": audio_features.tempo_bpm,
                    "pause_ratio": audio_features.pause_ratio,
                    "rms_energy": audio_features.rms_energy,
                    "pitch_mean_hz": audio_features.pitch_mean_hz,
                    "pitch_std_hz": audio_features.pitch_std_hz,
                    "zero_crossing_rate": audio_features.zero_crossing_rate,
                    "estimated_pause_count": audio_features.estimated_pause_count,
                    "format": metadata.get("format"),
                    "samplerate": metadata.get("samplerate"),
                })

            # ── Filler words ──────────────────────────────────────────────────
            with st.expander("🗯️ Filler Word Analysis"):
                fa1, fa2, fa3 = st.columns(3)
                fa1.metric("Total Words", filler_stats.total_words)
                fa2.metric("Filler Words", filler_stats.filler_word_count)
                fa3.metric("Filler Ratio", f"{filler_stats.filler_ratio:.2%}")
                if filler_stats.breakdown:
                    st.table({
                        "Word / Phrase": list(filler_stats.breakdown.keys()),
                        "Count": list(filler_stats.breakdown.values()),
                    })

            # ── Concept coverage ──────────────────────────────────────────────
            with st.expander("🔑 Concept Keyword Coverage"):
                kc1, kc2 = st.columns(2)
                with kc1:
                    st.markdown("**✅ Covered keywords**")
                    st.write(", ".join(sem_result.matched_keywords) or "—")
                with kc2:
                    st.markdown("**❌ Missed keywords**")
                    st.write(", ".join(sem_result.missing_keywords) or "—")

            # ── Feedback ──────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("<div class='section-header'>💡 Feedback & Recommendations</div>", unsafe_allow_html=True)
            for fb in detailed.feedback:
                st.markdown(
                    f"<div class='feedback-item'>• {fb}</div>",
                    unsafe_allow_html=True,
                )

            # ── PDF report ────────────────────────────────────────────────────
            wpm = round((tr.word_count / audio_features.duration_sec) * 60, 2) if audio_features.duration_sec else 0
            report_path = str(REPORTS_DIR / f"vbcua_report_result_{result_id}.pdf")
            generate_pdf_report(
                output_path=report_path,
                concept_title=concept_title,
                transcript=transcript_text,
                reference_text=reference_text,
                score=detailed,
                audio_features=audio_features,
                wpm=wpm,
                matched_keywords=sem_result.matched_keywords,
                missing_keywords=sem_result.missing_keywords,
            )
            file_size_kb = round(os.path.getsize(report_path) / 1024)
            data_storage.save_report(result_id, report_path, file_size_kb)

            st.markdown("---")
            with open(report_path, "rb") as f:
                st.download_button(
                    "📥 Download PDF Report",
                    data=f.read(),
                    file_name=os.path.basename(report_path),
                    mime="application/pdf",
                )

            data_storage.end_session(session_id, status="completed")

        except Exception as exc:
            data_storage.end_session(session_id, status="failed")
            st.error(f"❌ Analysis failed: {exc}")
            raise
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
