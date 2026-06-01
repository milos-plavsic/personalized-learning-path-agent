from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "roadmaps.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS roadmaps (
            user_id TEXT PRIMARY KEY,
            goal TEXT,
            payload_json TEXT NOT NULL
        )"""
    )
    return c


def save_roadmap(user_id: str, goal: str, payload: dict) -> dict:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO roadmaps (user_id, goal, payload_json) VALUES (?,?,?)
               ON CONFLICT(user_id) DO UPDATE SET goal=excluded.goal, payload_json=excluded.payload_json""",
            (user_id, goal, json.dumps(payload)),
        )
    return {"user_id": user_id, "goal": goal, "payload": payload}


def get_roadmap(user_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT goal, payload_json FROM roadmaps WHERE user_id = ?", (user_id,)
        ).fetchone()
    if not row:
        return None
    return {"user_id": user_id, "goal": row[0], "payload": json.loads(row[1])}
