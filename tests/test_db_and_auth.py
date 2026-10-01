"""Unit tests for SQLite WAL mode, API key hashing, auth verification, and audit logging."""
import os
import tempfile
from pathlib import Path
import pytest

from nirnaya_server.app.config import settings
from nirnaya_server.app.db import get_db, init_db
from nirnaya_server.app.models.api_keys import KeyManager, hash_key
from nirnaya_server.app.models.audit import AuditLogger


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Overrides the DB path with a temporary file for clean isolated testing."""
    test_db_file = tmp_path / "test_nirnaya.db"
    orig_path = settings.DB_PATH
    settings.DB_PATH = str(test_db_file)
    init_db()
    yield
    settings.DB_PATH = orig_path


def test_sqlite_wal_mode():
    """Verify SQLite WAL mode and pragmas are properly configured."""
    with get_db() as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.upper() == "WAL", f"Expected WAL mode, got {mode}"
        sync = conn.execute("PRAGMA synchronous;").fetchone()[0]
        # NORMAL synchronous is 1
        assert sync in (1, "1", "NORMAL")


def test_api_key_lifecycle():
    """Verify API key generation, SHA-256 storage, and constant-time verification."""
    # 1. Create a key
    raw_key, record = KeyManager.create_key(name="Test Service", role="client")
    assert raw_key.startswith("nir_live_")
    assert record["name"] == "Test Service"
    assert record["role"] == "client"
    assert record["is_active"] == 1

    # Ensure raw key is not stored directly in plain text
    with get_db() as conn:
        row = conn.execute("SELECT key_hash, prefix FROM api_keys WHERE id = ?", (record["id"],)).fetchone()
        assert row["key_hash"] == hash_key(raw_key)
        assert raw_key not in row["key_hash"]

    # 2. Verify valid key
    verified = KeyManager.verify_key(raw_key)
    assert verified is not None
    assert verified["id"] == record["id"]

    # 3. Verify invalid key returns None
    assert KeyManager.verify_key("nir_live_invalid_fake_key") is None
    assert KeyManager.verify_key("") is None

    # 4. Revocation
    revoked = KeyManager.revoke_key(record["id"])
    assert revoked is True
    # Once revoked, verify_key must reject
    assert KeyManager.verify_key(raw_key) is None


def test_initial_admin_key():
    """Verify ensure_admin_key idempotently seeds the root admin key."""
    admin_key = "nir_live_admin_secret_999"
    KeyManager.ensure_admin_key(admin_key, name="Root Superadmin")
    
    verified = KeyManager.verify_key(admin_key)
    assert verified is not None
    assert verified["role"] == "admin"
    assert verified["name"] == "Root Superadmin"

    # Calling it a second time should not create duplicate entries
    KeyManager.ensure_admin_key(admin_key, name="Root Superadmin")
    keys = [k for k in KeyManager.list_keys() if k["name"] == "Root Superadmin"]
    assert len(keys) == 1


def test_audit_logger():
    """Verify request telemetry is logged and queryable."""
    req_id = AuditLogger.generate_request_id()
    assert req_id.startswith("req_")

    AuditLogger.log_request(
        request_id=req_id,
        model="qwen3-0.6b-gguf",
        duration_ms=142.5,
        input_tokens=105,
        output_tokens=3,
        status_code=200,
    )

    logs = AuditLogger.get_recent_logs(limit=10)
    assert len(logs) >= 1
    latest = logs[0]
    assert latest["request_id"] == req_id
    assert latest["model"] == "qwen3-0.6b-gguf"
    assert latest["status_code"] == 200
    assert latest["duration_ms"] == 142.5
    assert latest["input_tokens"] == 105
    assert latest["output_tokens"] == 3
