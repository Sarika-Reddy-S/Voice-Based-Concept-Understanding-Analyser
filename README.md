# Voice-Based Concept Understanding Analyser (VBCUA)

Voice-Based Concept Understanding Analyser (VBCUA) is an AI-powered application that evaluates spoken conceptual explanations using speech-to-text transcription, semantic similarity analysis, and audio feature extraction. Built with **Streamlit**, **OpenAI Whisper**, **Sentence-BERT**, **Librosa**, and **ReportLab**, it assesses conceptual understanding, fluency, and communication clarity through intelligent scoring and automated reports.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Architecture](#architecture)
5. [Data Model (ER Design)](#data-model-entity-relationship-design)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Testing](#testing)
10. [Project Workflow](#project-workflow)
11. [Outcome](#outcome)
12. [Future Enhancements](#future-enhancements)
13. [License](#license)

---

## Overview

VBCUA lets a learner record or upload a spoken explanation of a concept, compares it against a reference explanation using semantic similarity, and analyzes delivery quality (speaking rate, pauses, pitch stability, energy confidence) from the raw audio. The result is a composite **Concept Understanding Score** (Strong / Moderate / Poor) plus actionable feedback and a downloadable PDF report — useful for self-study, viva/interview practice, or classroom assessment.

The system was developed using Python and modern AI/ML libraries for speech processing, semantic analysis, web application development, and report generation. Key technologies include **FastAPI**, **Streamlit**, **Librosa**, **Whisper**, **Sentence-BERT**, **ReportLab**, and **Visual Studio Code** for development support.

---

## References

- **Python**: https://www.python.org/downloads/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Streamlit**: https://docs.streamlit.io/
- **Librosa**: https://librosa.org/doc/latest/index.html
- **Whisper**: https://github.com/openai/whisper
- **Sentence-BERT**: https://www.sbert.net/docs/
- **ReportLab**: https://www.reportlab.com/docs/reportlab-userguide.pdf
- **Visual Studio Code**: https://code.visualstudio.com/

---

## Features

- 🎙️ Audio upload & in-browser playback (WAV, MP3, M4A, OGG, FLAC)
- 📝 Automatic speech-to-text transcription via **OpenAI Whisper**
- 🧠 Semantic similarity scoring against a reference explanation (**Sentence-BERT**)
- 📊 Audio feature extraction: speaking rate, pause ratio, RMS energy, pitch, ZCR (**Librosa**)
- 🗯️ Filler-word detection ("um", "uh", "like", "you know", …) and filler ratio
- 🎯 Scoring engine: `evaluate_understanding(similarity, filler_ratio, audio)` → Strong / Moderate / Poor
- 📈 Waveform visualization (dark-themed Matplotlib plot)
- 🗂️ Session history and full relational persistence (**SQLite**)
- 📄 One-click PDF report generation (**ReportLab**)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend UI | Streamlit |
| Speech-to-Text | OpenAI Whisper |
| Semantic Similarity | Sentence-Transformers (Sentence-BERT, all-MiniLM-L6-v2) |
| Audio Processing | Librosa, SoundFile, audioread |
| Numerical / Plotting | NumPy, Matplotlib |
| Report Generation | ReportLab |
| Testing | Pytest |
| Persistence | SQLite (stdlib `sqlite3`) |

---

## Architecture

```
Audio Upload
    │
    ▼
speech_to_text.py   ──►  Whisper transcription
    │
    ├──► filler_word_ratio()        (disfluency detection)
    │
    ▼
semantic_eval.py    ──►  Sentence-BERT cosine similarity vs reference concept
    │
    ▼
audio_utils.py      ──►  Librosa feature extraction (pause_ratio, rms_energy, ZCR, pitch)
    │
    ▼
scoring_engine.py   ──►  evaluate_understanding(similarity, filler_ratio, audio)
    │                     → score (0–100), level (Strong/Moderate/Poor), colour
    ▼
report_generator.py ──►  ReportLab PDF report
    │
    ▼
data_storage.py     ──►  SQLite (full ER schema)
```

---

## Data Model (Entity-Relationship Design)

The persistence layer (`modules/data_storage.py`) implements the following entities and relationships:

```
USER 1───N AUDIO_FILE (uploads)
AUDIO_FILE 1───1 TRANSCRIPT (generates)
AUDIO_FILE 1───1 AUDIO_FEATURE (analyzed for)
TRANSCRIPT 1───1 FILLER_WORD_STATS (analyzed for)
TRANSCRIPT ──── REFERENCE_CONCEPT  via SEMANTIC_SIMILARITY (compared with)
AUDIO_FILE, REFERENCE_CONCEPT ──► EVALUATION_RESULT (evaluated as)
EVALUATION_RESULT 1───1 REPORT (generates)
EVALUATION_RESULT N───1 SESSION (belongs to)
USER 1───N SESSION
```

| Entity | Key Fields |
|---|---|
| `USER` | user_id (PK), name, email, role, created_at |
| `SESSION` | session_id (PK), user_id (FK), started_at, ended_at, status |
| `AUDIO_FILE` | audio_id (PK), user_id (FK), file_name, file_path, duration_sec, uploaded_at, status |
| `REFERENCE_CONCEPT` | ref_concept_id (PK), concept_title, concept_text, created_at |
| `TRANSCRIPT` | transcript_id (PK), audio_id (FK), transcript_text, created_at |
| `AUDIO_FEATURE` | feature_id (PK), audio_id (FK), pause_ratio, rms_energy, zero_crossing_rate, duration_sec |
| `FILLER_WORD_STATS` | filler_id (PK), transcript_id (FK), filler_word_count, total_words, filler_ratio |
| `SEMANTIC_SIMILARITY` | similarity_id (PK), transcript_id (FK), ref_concept_id (FK), similarity_score |
| `EVALUATION_RESULT` | result_id (PK), audio_id (FK), ref_concept_id (FK), session_id (FK), overall_score, understanding_level (Strong/Moderate/Poor), notes |
| `REPORT` | report_id (PK), result_id (FK), pdf_path, generated_at, file_size_kb |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Voice-Based-Concept-Understanding-Analyser.git
cd Voice-Based-Concept-Understanding-Analyser

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note**: Whisper requires **ffmpeg** installed on your system.
> - macOS: `brew install ffmpeg`
> - Ubuntu: `sudo apt install ffmpeg`
> - Windows: https://ffmpeg.org/download.html

---

## Usage

```bash
streamlit run app.py
# or equivalently:
streamlit run main.py
```

In the browser tab that opens:

1. Enter your name in the sidebar.
2. Upload an audio recording (WAV / MP3 / M4A / OGG / FLAC).
3. Paste the reference concept explanation on the right panel.
4. Click **Analyze Concept Understanding**.
5. Review the transcription, understanding score (Strong/Moderate/Poor), semantic similarity, filler word ratio, confidence energy, and waveform.
6. Download the generated **PDF Report**.

---

## Project Structure

```
Voice-Based-Concept-Understanding-Analyser/
├── app.py                         # Streamlit front-end and main application logic
├── main.py                        # Entry-point alias (streamlit run main.py)
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── modules/
│   ├── __init__.py                # Public API exports
│   ├── audio_utils.py             # Audio loading and feature extraction utilities
│   ├── speech_to_text.py          # Whisper-based transcription logic
│   ├── semantic_eval.py           # Semantic similarity using Sentence-BERT
│   ├── scoring_engine.py          # Understanding score calculation and classification
│   ├── filler_words.py            # Filler-word / disfluency detection
│   ├── report_generator.py        # PDF report generation using ReportLab
│   ├── data_storage.py            # SQLite persistence layer (full ER schema)
│   │
│   │── (shims for backward compatibility)
│   ├── audio_features.py          → delegates to audio_utils.py
│   ├── transcription.py           → delegates to speech_to_text.py
│   ├── semantic_analysis.py       → delegates to semantic_eval.py
│   └── scoring.py                 → delegates to scoring_engine.py
│
├── tests/
│   ├── __init__.py
│   ├── test_audio_utils.py        # AudioFeatures, speaking rate
│   ├── test_speech_to_text.py     # filler_word_ratio, TranscriptionResult
│   ├── test_semantic_eval.py      # sentence splitting, keyword extraction
│   ├── test_scoring_engine.py     # evaluate_understanding, compute_score
│   ├── test_scoring.py            # legacy scoring tests
│   ├── test_semantic.py           # legacy semantic tests
│   ├── test_filler_words.py       # filler word analysis
│   └── test_data_storage.py       # SQLite CRUD operations
│
├── data/                          # SQLite DB (auto-created at runtime)
├── assets/                        # Waveform images (auto-generated)
└── reports/                       # Generated PDF reports
```

---

## Testing

```bash
pytest tests/ -v
```

Tests cover:
- **Scoring engine**: `evaluate_understanding` formula, sub-scores, grade labels, feedback generation
- **Audio utilities**: `AudioFeatures` dataclass, speaking-rate edge cases
- **Speech-to-text**: filler word ratio computation, `TranscriptionResult`
- **Semantic eval**: sentence splitting, keyword extraction, `SemanticResult`
- **Filler words**: detection accuracy, ratio computation
- **Data storage**: SQLite CRUD, session lifecycle

---

## Project Workflow

### Epic 1: Environment Setup
- **Story 1**: Python Environment and Dependency Installation
  - Created and activated `venv`; installed all `requirements.txt` dependencies
- **Story 2**: Project Structure Initialization
  - Organized modular structure: `app.py`, `modules/`, `tests/`, `data/`, `reports/`, `assets/`
- **Story 3**: Streamlit Application Initialization
  - Launched with `streamlit run main.py`; confirmed UI loads at `localhost:8501`

### Epic 2: Core Engine Development
- **Story 1**: Speech-to-Text Module Development (`modules/speech_to_text.py`)
  - Integrated OpenAI Whisper; handles WAV normalization and multi-format inputs
- **Story 2**: Semantic Understanding & Similarity Engine (`modules/semantic_eval.py`)
  - Sentence-BERT embeddings; cosine similarity; keyword coverage analysis
- **Story 3**: Audio Feature Extraction & Scoring Engine (`modules/audio_utils.py`, `modules/scoring_engine.py`)
  - Librosa features; exact `evaluate_understanding` formula; Strong/Moderate/Poor classification

### Epic 3: UI Development
- **Story 1**: User Interface Design and Visualization (`app.py`)
  - Dark-themed Streamlit layout with waveform, score card, metrics row
- **Story 2**: Input Handling and Session State Management
  - Audio uploader, session persistence, error handling
- **Story 3**: Output Rendering and Report Generation
  - Analysis banner, score display, feedback items, PDF download

### Epic 4: Testing & Deployment
- **Story 1**: Functional Testing and Validation — `pytest tests/ -v`
- **Story 2**: Performance Testing and Optimization — `@st.cache_resource` / `lru_cache`
- **Story 3**: Deployment Preparation — ready for Streamlit Community Cloud / Docker

---

## Outcome

By completing this project:

- Integrated **Whisper** (speech-to-text) and **Sentence-BERT** (semantic analysis) into a single pipeline
- Developed AI-driven pipelines for speech analysis, semantic scoring, and fluency evaluation
- Implemented the exact `evaluate_understanding` formula classifying answers as Strong / Moderate / Poor
- Generated automated PDF reports with educational insights and feedback
- Built a responsive, dark-themed Streamlit application matching the full project specification
- Persisted all data in a **SQLite** relational schema matching the ER diagram

---

## Future Enhancements

- Multi-language support for transcription and evaluation
- Live microphone recording directly in the browser
- Rubric-based / multi-reference scoring for open-ended answers
- User accounts with progress tracking over time
- Deployment to Streamlit Community Cloud / Docker container

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
