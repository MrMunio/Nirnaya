"""Admin endpoints for API key generation, revocation, and audit logs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ...auth import require_admin
from ...models.api_keys import KeyManager
from ...models.audit import AuditLogger

router = APIRouter()


class CreateKeyRequest(BaseModel):
    name: str = Field(..., description="Descriptive identifier for the key (e.g. 'Production Worker')")
    role: str = Field("client", description="Role: 'client' or 'admin'")
    custom_key: Optional[str] = Field(None, description="Optional custom secret key string")


class KeyResponse(BaseModel):
    id: int
    prefix: str
    name: str
    role: str
    is_active: bool
    created_at: str
    raw_key: Optional[str] = None


@router.post("/keys", response_model=KeyResponse, summary="Create New API Key")
async def create_api_key(
    payload: CreateKeyRequest,
    admin_user: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    """Generates a new API key. Note: raw_key is only returned on initial creation."""
    raw_key, record = KeyManager.create_key(
        name=payload.name,
        role=payload.role,
        custom_key=payload.custom_key,
    )
    return {**record, "raw_key": raw_key}


@router.get("/keys", summary="List All API Keys")
async def list_api_keys(
    admin_user: Dict[str, Any] = Depends(require_admin),
) -> List[Dict[str, Any]]:
    return KeyManager.list_keys()


@router.delete("/keys/{key_id}", summary="Revoke API Key")
async def revoke_api_key(
    key_id: int,
    admin_user: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    success = KeyManager.revoke_key(key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with ID {key_id} not found.",
        )
    return {"status": "revoked", "key_id": key_id}


@router.get("/logs", summary="Get Request Audit Logs")
async def get_audit_logs(
    limit: int = Query(50, ge=1, le=500),
    admin_user: Dict[str, Any] = Depends(require_admin),
) -> List[Dict[str, Any]]:
    return AuditLogger.get_recent_logs(limit=limit)
