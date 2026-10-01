"""API v1 router assembly."""
from fastapi import APIRouter

from .admin import router as admin_router
from .models import router as models_router
from .systemone import router as systemone_router

router = APIRouter()
router.include_router(systemone_router, tags=["Decision Engine"])
router.include_router(models_router, tags=["Models & Hardware"])
router.include_router(admin_router, prefix="/admin", tags=["Administration"])
