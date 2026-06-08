import os
import sqlite3
import tempfile
import pytest

# Set before any module import so DB_PATH in api.py picks it up at import time.
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
TEST_DB = _tmp.name
os.environ["SESSION_DB_PATH"] = TEST_DB

# Create schema immediately — clean_db fixture needs the table to exist before any test.
with sqlite3.connect(TEST_DB) as _conn:
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id         TEXT PRIMARY KEY,
            history    TEXT NOT NULL DEFAULT '[]',
            messages   TEXT NOT NULL DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


@pytest.fixture(autouse=True)
def clean_db():
    """Teardown: wipe sessions after every test so tests are independent."""
    yield
    with sqlite3.connect(TEST_DB) as conn:
        conn.execute("DELETE FROM sessions")
