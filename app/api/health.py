"""Health and readiness probe endpoints."""
from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, Response, status

from ..config import settings
from ..db import get_db

router = APIRouter()


@router.get("/health", summary="Liveness Probe")
async def health() -> Dict[str, str]:
    """Returns 200 OK if service process is running."""
    return {"status": "ok", "service": "nirnaya-server"}


@router.get("/ready", summary="Readiness Probe")
async def ready(response: Response) -> Dict[str, Any]:
    """Checks database connectivity and readiness."""
    db_ok = False
    try:
        with get_db() as conn:
            cur = conn.execute("SELECT 1")
            if cur.fetchone():
                db_ok = True
    except Exception as e:
        db_ok = False

    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "database": "disconnected"}

    return {
        "status": "ready",
        "database": "connected",
        "default_model": settings.DEFAULT_MODEL,
    }
