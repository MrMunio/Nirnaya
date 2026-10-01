"""Audit logging data access for Nirnaya Server."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from ..db import get_db


class AuditLogger:
    @staticmethod
    def generate_request_id() -> str:
        return f"req_{uuid.uuid4().hex[:16]}"

    @classmethod
    def log_request(
        cls,
        request_id: str,
        model: str,
        duration_ms: float,
        input_tokens: int,
        output_tokens: int,
        status_code: int = 200,
        api_key_id: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Records a request audit entry asynchronously or synchronously to SQLite."""
        try:
            with get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO request_logs (
                        request_id, api_key_id, model, status_code,
                        duration_ms, input_tokens, output_tokens, error_message
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        api_key_id,
                        model,
                        status_code,
                        round(duration_ms, 2),
                        input_tokens,
                        output_tokens,
                        error_message,
                    ),
                )
        except Exception as e:
            print(f"[AuditLogger] Failed to write log: {e}")

    @classmethod
    def get_recent_logs(cls, limit: int = 50) -> List[Dict[str, Any]]:
        with get_db() as conn:
            cur = conn.execute(
                """
                SELECT l.id, l.request_id, l.model, l.status_code, l.duration_ms,
                       l.input_tokens, l.output_tokens, l.error_message, l.created_at,
                       k.name as key_name, k.prefix as key_prefix
                FROM request_logs l
                LEFT JOIN api_keys k ON l.api_key_id = k.id
                ORDER BY l.id DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
