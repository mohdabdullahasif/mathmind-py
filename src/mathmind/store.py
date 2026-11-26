"""SQLite persistence for MathMind history, variables, and model registry."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mathmind.models import Result, Step

_DB_PATH = Path(os.environ.get("MATHMIND_DB", str(Path.home() / ".mathmind" / "mathmind.db")))


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or _DB_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        path = Path.cwd() / ".mathmind" / "mathmind.db"
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        channel TEXT NOT NULL DEFAULT 'typed',
        raw_input TEXT NOT NULL,
        expression TEXT NOT NULL,
        result TEXT,
        strategy TEXT,
        confidence REAL,
        latex TEXT
    );
    CREATE TABLE IF NOT EXISTS steps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        history_id INTEGER NOT NULL REFERENCES history(id) ON DELETE CASCADE,
        step_no INTEGER NOT NULL,
        description TEXT NOT NULL,
        before_expr TEXT,
        after_expr TEXT
    );
    CREATE TABLE IF NOT EXISTS variables (
        name TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        defined_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        history_id INTEGER REFERENCES history(id),
        kind TEXT NOT NULL,
        corrected_value TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS model_registry (
        name TEXT NOT NULL,
        version TEXT NOT NULL,
        metrics_json TEXT,
        trained_at TEXT NOT NULL,
        PRIMARY KEY (name, version)
    );
    """)
    conn.commit()


def save_result(result: Result, raw_input: str | None = None, db_path: Path | None = None) -> int:
    """Persist a Result and its steps. Returns the history row id."""
    conn = _connect(db_path)
    now = datetime.now(timezone.utc).isoformat()
    value_str = json.dumps(result.value) if not isinstance(result.value, str) else result.value
    cur = conn.execute(
        "INSERT INTO history (created_at, channel, raw_input, expression, result, strategy, confidence, latex) VALUES (?,?,?,?,?,?,?,?)",
        (now, result.channel, raw_input or result.expression, result.expression,
         value_str, result.strategy, result.confidence, result.latex),
    )
    history_id = cur.lastrowid
    for s in result.steps:
        conn.execute(
            "INSERT INTO steps (history_id, step_no, description, before_expr, after_expr) VALUES (?,?,?,?,?)",
            (history_id, s.step_no, s.description, s.before, s.after),
        )
    conn.commit()
    conn.close()
    return history_id  # type: ignore[return-value]


def get_history(limit: int = 50, db_path: Path | None = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT id, created_at, channel, raw_input, expression, result, strategy, confidence FROM history ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        dict(id=r[0], created_at=r[1], channel=r[2], raw_input=r[3],
             expression=r[4], result=r[5], strategy=r[6], confidence=r[7])
        for r in rows
    ]


def save_variable(name: str, value: float, db_path: Path | None = None) -> None:
    conn = _connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO variables (name, value, defined_at) VALUES (?,?,?)",
        (name, str(value), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def register_model(name: str, version: str, metrics: dict, db_path: Path | None = None) -> None:
    conn = _connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO model_registry (name, version, metrics_json, trained_at) VALUES (?,?,?,?)",
        (name, version, json.dumps(metrics), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
