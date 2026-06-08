import json
import os
import sqlite3
from datetime import datetime, timezone

VALID_TYPES = {"user", "feedback", "project", "reference"}


def _db_path() -> str:
    raw = os.environ.get("DB_PATH", "")
    if raw:
        return os.path.abspath(raw)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "data", "sessions.db"))


def _conn() -> sqlite3.Connection:
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                name        TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                type        TEXT NOT NULL,
                body        TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            )
        """)


def write_memory(name: str, description: str, mem_type: str, body: str) -> dict:
    if mem_type not in VALID_TYPES:
        raise ValueError(f"type must be one of {sorted(VALID_TYPES)}")
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        existing = conn.execute(
            "SELECT created_at FROM memories WHERE name = ?", (name,)
        ).fetchone()
        created_at = existing["created_at"] if existing else now
        conn.execute(
            """INSERT OR REPLACE INTO memories
               (name, description, type, body, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, description, mem_type, body, created_at, now),
        )
    return {"name": name, "action": "updated" if existing else "created"}


def read_memory(name: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM memories WHERE name = ?", (name,)
        ).fetchone()
    return dict(row) if row else None


def list_memories() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT name, description, type FROM memories ORDER BY type, name"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_memory(name: str) -> bool:
    with _conn() as conn:
        cursor = conn.execute("DELETE FROM memories WHERE name = ?", (name,))
    return cursor.rowcount > 0


def search_memories(query: str) -> list[dict]:
    pattern = f"%{query}%"
    with _conn() as conn:
        rows = conn.execute(
            """SELECT name, description, type, body FROM memories
               WHERE name LIKE ? OR description LIKE ? OR body LIKE ?
               ORDER BY type, name""",
            (pattern, pattern, pattern),
        ).fetchall()
    return [dict(r) for r in rows]


def export_memories() -> str:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY type, name"
        ).fetchall()
    payload = {
        "version": "1.1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "memories": [dict(r) for r in rows],
    }
    return json.dumps(payload, indent=2)


def import_memories(json_str: str) -> dict:
    data = json.loads(json_str)
    memories = data.get("memories", [])
    imported, skipped = 0, 0
    for m in memories:
        try:
            write_memory(m["name"], m["description"], m["type"], m["body"])
            imported += 1
        except Exception:
            skipped += 1
    return {"imported": imported, "skipped": skipped}
