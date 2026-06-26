"""SQLite database models for conversation persistence."""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from src.config import DATA_DIR

DB_PATH = DATA_DIR / "conversations.db"


def get_db() -> sqlite3.Connection:
    """Get a database connection (auto-creates tables)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT '',
            pet_name TEXT DEFAULT '',
            pet_breed TEXT DEFAULT '',
            pet_age TEXT DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
            content TEXT NOT NULL,
            metadata_json TEXT DEFAULT '{}',
            created_at REAL NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );
        CREATE INDEX IF NOT EXISTS idx_msg_conv
            ON messages(conversation_id, created_at);
    """)
    conn.commit()


def create_conversation(
    conv_id: str,
    pet_name: str = "",
    pet_breed: str = "",
    pet_age: str = "",
) -> dict:
    db = get_db()
    now = time.time()
    db.execute(
        "INSERT OR REPLACE INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?)",
        (conv_id, "", pet_name, pet_breed, pet_age, now, now),
    )
    db.commit()
    return get_conversation(conv_id)


def get_conversation(conv_id: str) -> dict | None:
    db = get_db()
    row = db.execute(
        "SELECT * FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()
    if not row:
        return None
    return dict(row)


def list_conversations(limit: int = 20) -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
):
    db = get_db()
    now = time.time()
    db.execute(
        "INSERT INTO messages VALUES (NULL, ?, ?, ?, ?, ?)",
        (conversation_id, role, content, json.dumps(metadata or {}), now),
    )
    db.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    db.commit()


def get_messages(conversation_id: str, limit: int = 50) -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ?",
        (conversation_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_conversation(conversation_id: str):
    db = get_db()
    db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    db.commit()
