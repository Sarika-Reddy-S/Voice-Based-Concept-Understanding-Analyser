# Voice-Based Concept Understanding Analyser (VBCUA)

Voice-Based Concept Understanding Analyser (VBCUA) is an AI-powered application that evaluates spoken conceptual explanations using speech-to-text transcription, semantic similarity analysis, and audio feature extraction. Built with **Streamlit**, **OpenAI Whisper**, **Sentence-BERT**, **Librosa**, and **ReportLab**, it assesses conceptual understanding, fluency, and communication clarity through intelligent scoring and automated reports.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Architecture](#architecture)
5. [Instructions](#instructions)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Testing](#testing)
10. [Outcome](#outcome)
11. [Future Enhancements](#future-enhancements)
12. [License](#license)

---

## Overview

VBCUA lets a learner record or upload a spoken explanation of a concept, compares it against a reference explanation using semantic similarity, and analyzes delivery quality (speaking rate, pauses, pitch stability) from the raw audio. The result is a composite **Concept Understanding Score** plus actionable feedback and a downloadable PDF report — useful for self-study, viva/interview practice, or classroom assessment.

## Features

- 🎙️ Audio upload & in-browser playback
- 📝 Automatic speech-to-text transcription (OpenAI Whisper)
- 🧠 Semantic similarity scoring against a reference explanation (Sentence-BERT)
- 📊 Audio feature extraction: speaking rate, pauses, pitch, energy (Librosa)
- 🎯 Composite scoring engine (Understanding / Fluency / Clarity / Overall)
- 📈 Waveform visualization
- 🗂️ Session history and persistence (SQLite)
- 📄 One-click PDF report generation (ReportLab)

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Speech-to-text | OpenAI Whisper |
| Semantic similarity | Sentence-Transformers (Sentence-BERT) |
| Audio processing | Librosa, SoundFile |
| Numerical / plotting | NumPy, Matplotlib |
| Report generation | ReportLab |
| Testing | Pytest |
| Persistence | SQLite |

## Architecture

```
Audio Input → Whisper Transcription → Sentence-BERT Semantic Comparison
                    │                              │
                    ▼                              ▼
            Librosa Audio Features ──────► Scoring Engine ──► PDF Report
                                                    │
                                                    ▼
                                          SQLite Session Storage
```

---

## Instructions

The project was built following these development phases:

### 1. Environment Setup & Dependency Configuration
Configured the project using: Streamlit, OpenAI Whisper, Sentence-Transformers, Librosa & SoundFile, NumPy & Matplotlib, ReportLab, and Pytest (see [`requirements.txt`](requirements.txt)).

### 2. Model Selection & Architecture
Integrated AI and audio-processing frameworks for:
- Speech-to-text transcription
- Semantic similarity analysis
- Audio signal processing
- Fluency and scoring metrics

### 3. Core Backend Development
Developed modules for:
- Speech transcription (`modules/transcription.py`)
- Semantic evaluation (`modules/semantic_analysis.py`)
- Audio feature extraction (`modules/audio_features.py`)
- Scoring engine (`modules/scoring.py`)
- PDF report generation (`modules/report_generator.py`)

### 4. Data Persistence & Analysis Handling
Implemented storage for transcriptions, audio features, evaluation scores, and session data (`modules/data_storage.py`, SQLite-backed).

### 5. Streamlit Frontend UI Development
Built an interactive interface (`app.py`) with:
- Audio upload & playback
- Waveform visualization
- Real-time scoring
- Understanding analysis
- PDF report download

### 6. Testing & Deployment
Performed testing, optimization, validation, and deployment preparation (`tests/`).

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Voice-Based-Concept-Understanding-Analyser.git
cd Voice-Based-Concept-Understanding-Analyser

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# Note: Whisper requires ffmpeg to be installed on your system.
#   macOS:   brew install ffmpeg
#   Ubuntu:  sudo apt install ffmpeg
#   Windows: https://ffmpeg.org/download.html
```

## Usage

```bash
streamlit run app.py
```

Then, in the browser tab that opens:
1. Enter the concept/topic title and paste the reference explanation.
2. Upload an audio recording (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`).
3. Click **Analyze Recording**.
4. Review the transcript, scores, waveform, and feedback.
5. Download the generated PDF report.

## Project Structure

```
Voice-Based-Concept-Understanding-Analyser/
├── app.py                        # Streamlit frontend
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── modules/
│   ├── __init__.py
│   ├── transcription.py          # Whisper speech-to-text
│   ├── semantic_analysis.py      # Sentence-BERT similarity scoring
│   ├── audio_features.py         # Librosa feature extraction
│   ├── scoring.py                # Composite scoring engine
│   ├── report_generator.py       # ReportLab PDF generation
│   └── data_storage.py           # SQLite persistence layer
├── tests/
│   ├── __init__.py
│   ├── test_scoring.py
│   ├── test_semantic.py
│   └── test_data_storage.py
├── data/                         # SQLite DB (generated at runtime)
└── reports/                      # Generated PDF reports
```

## Testing

```bash
pytest tests/ -v
```

Tests cover the scoring engine (speaking-rate scoring, pause scoring, grading, feedback generation), semantic-analysis helper functions (sentence splitting, keyword extraction), and the SQLite persistence layer.

## Outcome

By completing this project, the following was achieved:
- Integrated Whisper and Sentence-BERT models into a single pipeline
- Developed AI pipelines for speech and semantic analysis
- Implemented fluency evaluation and automated scoring
- Generated PDF reports with educational insights
- Built a responsive Streamlit application for spoken concept assessment

## Future Enhancements

- Multi-language support for transcription and evaluation
- Live microphone recording directly in the browser
- Rubric-based / multi-reference scoring for open-ended answers
- User accounts and progress tracking over time
- Deployment to Streamlit Community Cloud / Docker container

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
