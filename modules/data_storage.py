"""
data_storage.py
----------------
Lightweight data persistence layer for VBCUA.

Uses a local SQLite database (no external services required) to store:
    - transcriptions
    - audio feature snapshots
    - evaluation scores
    - session metadata

This keeps the app fully self-contained for local/offline classroom use
while still giving users a history of past attempts.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "vbcua.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    concept_title TEXT,
    reference_text TEXT
);

CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    text TEXT,
    language TEXT,
    duration REAL,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);

CREATE TABLE IF NOT EXISTS audio_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    features_json TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    understanding_score REAL,
    fluency_score REAL,
    clarity_score REAL,
    overall_score REAL,
    grade TEXT,
    feedback_json TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);
"""


@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def create_session(concept_title: str, reference_text: str) -> int:
    init_db()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (created_at, concept_title, reference_text) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), concept_title, reference_text),
        )
        return cur.lastrowid


def save_transcription(session_id: int, text: str, language: str, duration: float) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO transcriptions (session_id, text, language, duration) VALUES (?, ?, ?, ?)",
            (session_id, text, language, duration),
        )


def save_audio_features(session_id: int, features) -> None:
    payload = asdict(features) if hasattr(features, "__dataclass_fields__") else features
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO audio_features (session_id, features_json) VALUES (?, ?)",
            (session_id, json.dumps(payload)),
        )


def save_score(session_id: int, score) -> None:
    payload = asdict(score) if hasattr(score, "__dataclass_fields__") else score
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO scores
               (session_id, understanding_score, fluency_score, clarity_score,
                overall_score, grade, feedback_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                payload["understanding_score"],
                payload["fluency_score"],
                payload["clarity_score"],
                payload["overall_score"],
                payload["grade"],
                json.dumps(payload.get("feedback", [])),
            ),
        )


def get_session_history(limit: int = 20):
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT s.id, s.created_at, s.concept_title, sc.overall_score, sc.grade
               FROM sessions s
               LEFT JOIN scores sc ON sc.session_id = s.id
               ORDER BY s.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_session_detail(session_id: int) -> Optional[dict]:
    with get_connection() as conn:
        session = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            return None
        transcription = conn.execute(
            "SELECT * FROM transcriptions WHERE session_id = ?", (session_id,)
        ).fetchone()
        features = conn.execute(
            "SELECT * FROM audio_features WHERE session_id = ?", (session_id,)
        ).fetchone()
        score = conn.execute(
            "SELECT * FROM scores WHERE session_id = ?", (session_id,)
        ).fetchone()

        return {
            "session": dict(session),
            "transcription": dict(transcription) if transcription else None,
            "audio_features": json.loads(features["features_json"]) if features else None,
            "score": dict(score) if score else None,
        }
