"""SQLite Database Manager for Nirnaya Server.

Uses Write-Ahead Logging (WAL) mode for high concurrency non-blocking reads and writes.
"""
from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path
from typing import Generator

from .config import settings


def get_db_path() -> Path:
    p = settings.get_resolved_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def create_connection() -> sqlite3.Connection:
    """Creates a new SQLite connection configured with WAL mode."""
    db_file = get_db_path()
    conn = sqlite3.connect(str(db_file), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # Configure WAL mode and pragmas
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn


@contextlib.contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for acquiring and safely closing a database connection."""
    conn = create_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initializes tables and indexes if they do not already exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT UNIQUE NOT NULL,
                prefix TEXT NOT NULL,
                name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'client',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_used_at TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT UNIQUE NOT NULL,
                api_key_id INTEGER REFERENCES api_keys(id),
                model TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                duration_ms REAL NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_reqid ON request_logs(request_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_created ON request_logs(created_at);")

    # Seed initial admin key if provided
    from .models.api_keys import KeyManager
    if settings.INITIAL_ADMIN_KEY:
        KeyManager.ensure_admin_key(settings.INITIAL_ADMIN_KEY, name="Root Admin")
