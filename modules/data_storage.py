"""
data_storage.py
----------------
SQLite-backed persistence layer for VBCUA, modeling the full
entity-relationship design:

    USER 1--N AUDIO_FILE (uploads)
    AUDIO_FILE 1--1 TRANSCRIPT (generates)
    AUDIO_FILE 1--1 AUDIO_FEATURE (analyzed for)
    TRANSCRIPT 1--1 FILLER_WORD_STATS (analyzed for)
    TRANSCRIPT N--N REFERENCE_CONCEPT via SEMANTIC_SIMILARITY (compared with)
    AUDIO_FILE N--1 REFERENCE_CONCEPT, evaluated as EVALUATION_RESULT
    EVALUATION_RESULT 1--1 REPORT (generates)
    EVALUATION_RESULT N--1 SESSION (belongs to)
    USER 1--N SESSION

Kept intentionally dependency-free (stdlib sqlite3 only) so the app
stays fully self-contained for local/offline use.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "vbcua.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS USER (
    user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name        VARCHAR(100) NOT NULL,
    email       VARCHAR(150),
    role        VARCHAR(20) DEFAULT 'learner',
    created_at  DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS SESSION (
    session_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    started_at  DATETIME NOT NULL,
    ended_at    DATETIME,
    status      VARCHAR(20) DEFAULT 'in_progress',
    FOREIGN KEY (user_id) REFERENCES USER (user_id)
);

CREATE TABLE IF NOT EXISTS AUDIO_FILE (
    audio_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    file_name    VARCHAR(255),
    file_path    VARCHAR(255),
    duration_sec FLOAT,
    uploaded_at  DATETIME NOT NULL,
    status       VARCHAR(20) DEFAULT 'uploaded',
    FOREIGN KEY (user_id) REFERENCES USER (user_id)
);

CREATE TABLE IF NOT EXISTS REFERENCE_CONCEPT (
    ref_concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_title  VARCHAR(255) NOT NULL UNIQUE,
    concept_text   TEXT NOT NULL,
    created_at     DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS TRANSCRIPT (
    transcript_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_id        INTEGER NOT NULL,
    transcript_text TEXT,
    created_at      DATETIME NOT NULL,
    FOREIGN KEY (audio_id) REFERENCES AUDIO_FILE (audio_id)
);

CREATE TABLE IF NOT EXISTS AUDIO_FEATURE (
    feature_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_id           INTEGER NOT NULL,
    pause_ratio        FLOAT,
    rms_energy         FLOAT,
    zero_crossing_rate FLOAT,
    duration_sec       FLOAT,
    created_at         DATETIME NOT NULL,
    FOREIGN KEY (audio_id) REFERENCES AUDIO_FILE (audio_id)
);

CREATE TABLE IF NOT EXISTS FILLER_WORD_STATS (
    filler_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id       INTEGER NOT NULL,
    filler_word_count   INTEGER,
    total_words         INTEGER,
    filler_ratio        FLOAT,
    created_at           DATETIME NOT NULL,
    FOREIGN KEY (transcript_id) REFERENCES TRANSCRIPT (transcript_id)
);

CREATE TABLE IF NOT EXISTS SEMANTIC_SIMILARITY (
    similarity_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id      INTEGER NOT NULL,
    ref_concept_id      INTEGER NOT NULL,
    similarity_score   FLOAT,
    created_at          DATETIME NOT NULL,
    FOREIGN KEY (transcript_id) REFERENCES TRANSCRIPT (transcript_id),
    FOREIGN KEY (ref_concept_id) REFERENCES REFERENCE_CONCEPT (ref_concept_id)
);

CREATE TABLE IF NOT EXISTS EVALUATION_RESULT (
    result_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_id              INTEGER NOT NULL,
    ref_concept_id         INTEGER NOT NULL,
    session_id            INTEGER,
    overall_score          FLOAT,
    understanding_level    VARCHAR(20),   -- Strong / Moderate / Poor
    created_at             DATETIME NOT NULL,
    notes                  TEXT,
    FOREIGN KEY (audio_id) REFERENCES AUDIO_FILE (audio_id),
    FOREIGN KEY (ref_concept_id) REFERENCES REFERENCE_CONCEPT (ref_concept_id),
    FOREIGN KEY (session_id) REFERENCES SESSION (session_id)
);

CREATE TABLE IF NOT EXISTS REPORT (
    report_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id     INTEGER NOT NULL,
    pdf_path      VARCHAR(255),
    generated_at  DATETIME NOT NULL,
    file_size_kb  INTEGER,
    FOREIGN KEY (result_id) REFERENCES EVALUATION_RESULT (result_id)
);
"""


@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def understanding_level_for(overall_score: float) -> str:
    """Map a 0-100 overall score to the Strong / Moderate / Poor enum."""
    if overall_score >= 75:
        return "Strong"
    if overall_score >= 50:
        return "Moderate"
    return "Poor"


# ---------------------------------------------------------------- USER ----
def get_or_create_user(name: str, email: Optional[str] = None, role: str = "learner") -> int:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT user_id FROM USER WHERE name = ?", (name,)).fetchone()
        if row:
            return row["user_id"]
        cur = conn.execute(
            "INSERT INTO USER (name, email, role, created_at) VALUES (?, ?, ?, ?)",
            (name, email, role, datetime.now().isoformat()),
        )
        return cur.lastrowid


# ------------------------------------------------------------- SESSION ----
def start_session(user_id: int) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO SESSION (user_id, started_at, status) VALUES (?, ?, 'in_progress')",
            (user_id, datetime.now().isoformat()),
        )
        return cur.lastrowid


def end_session(session_id: int, status: str = "completed") -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE SESSION SET ended_at = ?, status = ? WHERE session_id = ?",
            (datetime.now().isoformat(), status, session_id),
        )


# ----------------------------------------------------------- AUDIO_FILE ---
def save_audio_file(
    user_id: int, file_name: str, file_path: str, duration_sec: float, status: str = "processed"
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO AUDIO_FILE (user_id, file_name, file_path, duration_sec, uploaded_at, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, file_name, file_path, duration_sec, datetime.now().isoformat(), status),
        )
        return cur.lastrowid


# ------------------------------------------------------ REFERENCE_CONCEPT -
def get_or_create_reference_concept(concept_title: str, concept_text: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT ref_concept_id FROM REFERENCE_CONCEPT WHERE concept_title = ?",
            (concept_title,),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE REFERENCE_CONCEPT SET concept_text = ? WHERE ref_concept_id = ?",
                (concept_text, row["ref_concept_id"]),
            )
            return row["ref_concept_id"]
        cur = conn.execute(
            "INSERT INTO REFERENCE_CONCEPT (concept_title, concept_text, created_at) VALUES (?, ?, ?)",
            (concept_title, concept_text, datetime.now().isoformat()),
        )
        return cur.lastrowid


# ------------------------------------------------------------- TRANSCRIPT -
def save_transcript(audio_id: int, transcript_text: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO TRANSCRIPT (audio_id, transcript_text, created_at) VALUES (?, ?, ?)",
            (audio_id, transcript_text, datetime.now().isoformat()),
        )
        return cur.lastrowid


# --------------------------------------------------------- AUDIO_FEATURE --
def save_audio_feature(
    audio_id: int, pause_ratio: float, rms_energy: float, zero_crossing_rate: float, duration_sec: float
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO AUDIO_FEATURE
               (audio_id, pause_ratio, rms_energy, zero_crossing_rate, duration_sec, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (audio_id, pause_ratio, rms_energy, zero_crossing_rate, duration_sec, datetime.now().isoformat()),
        )
        return cur.lastrowid


# ------------------------------------------------------ FILLER_WORD_STATS -
def save_filler_word_stats(transcript_id: int, stats) -> int:
    payload = asdict(stats) if hasattr(stats, "__dataclass_fields__") else stats
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO FILLER_WORD_STATS
               (transcript_id, filler_word_count, total_words, filler_ratio, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                transcript_id,
                payload["filler_word_count"],
                payload["total_words"],
                payload["filler_ratio"],
                datetime.now().isoformat(),
            ),
        )
        return cur.lastrowid


# ------------------------------------------------------ SEMANTIC_SIMILARITY
def save_semantic_similarity(transcript_id: int, ref_concept_id: int, similarity_score: float) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO SEMANTIC_SIMILARITY
               (transcript_id, ref_concept_id, similarity_score, created_at)
               VALUES (?, ?, ?, ?)""",
            (transcript_id, ref_concept_id, similarity_score, datetime.now().isoformat()),
        )
        return cur.lastrowid


# ------------------------------------------------------- EVALUATION_RESULT
def save_evaluation_result(
    audio_id: int,
    ref_concept_id: int,
    session_id: Optional[int],
    overall_score: float,
    notes: str = "",
) -> int:
    level = understanding_level_for(overall_score)
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO EVALUATION_RESULT
               (audio_id, ref_concept_id, session_id, overall_score, understanding_level, created_at, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (audio_id, ref_concept_id, session_id, overall_score, level, datetime.now().isoformat(), notes),
        )
        return cur.lastrowid


# ------------------------------------------------------------------ REPORT
def save_report(result_id: int, pdf_path: str, file_size_kb: int) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO REPORT (result_id, pdf_path, generated_at, file_size_kb) VALUES (?, ?, ?, ?)",
            (result_id, pdf_path, datetime.now().isoformat(), file_size_kb),
        )
        return cur.lastrowid


# --------------------------------------------------------------- QUERIES --
def get_session_history(limit: int = 20):
    """Recent evaluation results, most recent first, for the sidebar history view."""
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT er.result_id, er.created_at, rc.concept_title,
                      er.overall_score, er.understanding_level
               FROM EVALUATION_RESULT er
               JOIN REFERENCE_CONCEPT rc ON rc.ref_concept_id = er.ref_concept_id
               ORDER BY er.result_id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_result_detail(result_id: int) -> Optional[dict]:
    with get_connection() as conn:
        result = conn.execute(
            "SELECT * FROM EVALUATION_RESULT WHERE result_id = ?", (result_id,)
        ).fetchone()
        if not result:
            return None
        audio = conn.execute(
            "SELECT * FROM AUDIO_FILE WHERE audio_id = ?", (result["audio_id"],)
        ).fetchone()
        transcript = conn.execute(
            "SELECT * FROM TRANSCRIPT WHERE audio_id = ?", (result["audio_id"],)
        ).fetchone()
        features = conn.execute(
            "SELECT * FROM AUDIO_FEATURE WHERE audio_id = ?", (result["audio_id"],)
        ).fetchone()
        filler = None
        similarity = None
        if transcript:
            filler = conn.execute(
                "SELECT * FROM FILLER_WORD_STATS WHERE transcript_id = ?", (transcript["transcript_id"],)
            ).fetchone()
            similarity = conn.execute(
                "SELECT * FROM SEMANTIC_SIMILARITY WHERE transcript_id = ?", (transcript["transcript_id"],)
            ).fetchone()
        report = conn.execute(
            "SELECT * FROM REPORT WHERE result_id = ?", (result_id,)
        ).fetchone()

        return {
            "result": dict(result),
            "audio_file": dict(audio) if audio else None,
            "transcript": dict(transcript) if transcript else None,
            "audio_feature": dict(features) if features else None,
            "filler_word_stats": dict(filler) if filler else None,
            "semantic_similarity": dict(similarity) if similarity else None,
            "report": dict(report) if report else None,
        }
