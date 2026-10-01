"""FastAPI authentication dependencies supporting Bearer token and X-API-Key."""
from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import Depends, Header, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings
from .models.api_keys import KeyManager

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_api_key(
    request: Request,
    auth_header: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Dict[str, Any]:
    """Validates API Key from Bearer Authorization header or X-API-Key header."""
    if not settings.REQUIRE_AUTH:
        return {"id": None, "name": "anonymous", "role": "admin", "prefix": "anonymous"}

    raw_token = None
    if auth_header and auth_header.credentials:
        raw_token = auth_header.credentials.strip()
    elif x_api_key:
        raw_token = x_api_key.strip()

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials. Provide 'Authorization: Bearer <token>' or 'X-API-Key' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    key_record = KeyManager.verify_key(raw_token)
    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API Key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Attach to request state for access in endpoints / middleware
    request.state.api_key = key_record
    return key_record


async def require_admin(
    current_key: Dict[str, Any] = Depends(get_current_api_key),
) -> Dict[str, Any]:
    """Ensures caller has admin permissions."""
    if current_key.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this operation.",
        )
    return current_key
