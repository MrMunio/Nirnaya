"""Models metadata and hardware capability endpoint."""
from __future__ import annotations

from typing import Any, Dict, List
from fastapi import APIRouter, Depends

from ...auth import get_current_api_key
from ...services.engine_service import engine_service

router = APIRouter()


@router.get("/models", summary="List Registered Models & Hardware Acceleration")
async def list_models(
    current_key: Dict[str, Any] = Depends(get_current_api_key),
) -> Dict[str, Any]:
    """Returns available models, the configured default model, and hardware status."""
    return {
        "models": engine_service.list_available_models(),
        "hardware": engine_service.get_hardware_status(),
    }
