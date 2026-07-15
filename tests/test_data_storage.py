import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib

import modules.data_storage as data_storage


def setup_module(module):
    # Point the DB at a throwaway test file so tests never touch real data.
    test_db = Path(__file__).resolve().parent / "test_vbcua.db"
    if test_db.exists():
        test_db.unlink()
    data_storage.DB_PATH = test_db
    data_storage.init_db()


def test_create_session_returns_id():
    session_id = data_storage.create_session("Photosynthesis", "Plants convert light to energy.")
    assert isinstance(session_id, int)
    assert session_id > 0


def test_session_history_contains_created_session():
    session_id = data_storage.create_session("Newton's Laws", "Force equals mass times acceleration.")
    history = data_storage.get_session_history(limit=5)
    ids = [h["id"] for h in history]
    assert session_id in ids


def teardown_module(module):
    if data_storage.DB_PATH.exists():
        data_storage.DB_PATH.unlink()
