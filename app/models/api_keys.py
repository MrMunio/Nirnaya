"""API Key data access and verification logic."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any, Dict, List, Optional, Tuple

from ..db import get_db


def hash_key(raw_key: str) -> str:
    """Computes a SHA-256 hex digest of the raw secret API key."""
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


class KeyManager:
    @staticmethod
    def generate_raw_key(prefix: str = "nir_live_") -> str:
        """Generates a high-entropy random API key."""
        random_suffix = secrets.token_urlsafe(32)
        return f"{prefix}{random_suffix}"

    @classmethod
    def create_key(
        cls,
        name: str,
        role: str = "client",
        custom_key: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Creates and stores a new API key. Returns the raw secret (only visible once) and record."""
        raw_key = custom_key if custom_key else cls.generate_raw_key()
        k_hash = hash_key(raw_key)
        prefix_display = raw_key[:12] + "..." if len(raw_key) > 12 else raw_key

        with get_db() as conn:
            cur = conn.execute(
                """
                INSERT INTO api_keys (key_hash, prefix, name, role, is_active)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(key_hash) DO UPDATE SET
                    prefix=excluded.prefix,
                    name=excluded.name,
                    role=excluded.role,
                    is_active=1
                RETURNING id, prefix, name, role, is_active, created_at
                """,
                (k_hash, prefix_display, name, role),
            )
            row = cur.fetchone()
            return raw_key, dict(row)

    @classmethod
    def ensure_admin_key(cls, admin_key: str, name: str = "Root Admin") -> None:
        """Ensures the initial admin key exists in the database."""
        k_hash = hash_key(admin_key)
        prefix_display = admin_key[:12] + "..." if len(admin_key) > 12 else admin_key
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (key_hash, prefix, name, role, is_active)
                VALUES (?, ?, ?, 'admin', 1)
                ON CONFLICT(key_hash) DO UPDATE SET
                    prefix=excluded.prefix,
                    name=excluded.name,
                    role='admin',
                    is_active=1
                """,
                (k_hash, prefix_display, name),
            )

    @classmethod
    def verify_key(cls, raw_key: str) -> Optional[Dict[str, Any]]:
        """Verifies raw key against active keys in database. Updates last_used_at timestamp."""
        if not raw_key:
            return None
        k_hash = hash_key(raw_key)
        with get_db() as conn:
            cur = conn.execute(
                """
                SELECT id, prefix, name, role, is_active, created_at
                FROM api_keys
                WHERE key_hash = ? AND is_active = 1
                """,
                (k_hash,),
            )
            row = cur.fetchone()
            if not row:
                return None
            key_data = dict(row)
            # Update last_used_at in background
            conn.execute(
                "UPDATE api_keys SET last_used_at = datetime('now') WHERE id = ?",
                (key_data["id"],),
            )
            return key_data

    @classmethod
    def list_keys(cls) -> List[Dict[str, Any]]:
        with get_db() as conn:
            cur = conn.execute(
                """
                SELECT id, prefix, name, role, is_active, created_at, last_used_at
                FROM api_keys
                ORDER BY id DESC
                """
            )
            return [dict(r) for r in cur.fetchall()]

    @classmethod
    def revoke_key(cls, key_id: int) -> bool:
        with get_db() as conn:
            cur = conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE id = ?",
                (key_id,),
            )
            return cur.rowcount > 0
