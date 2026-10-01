"""FastAPI application factory for Nirnaya Production Server."""
from __future__ import annotations

import contextlib
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .api.health import router as health_router
from .api.v1 import router as v1_router
from .config import settings
from .db import init_db
from .services.engine_service import engine_service


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle events: startup and shutdown."""
    print("=" * 60)
    print(f"🚀 Starting Nirnaya Decision Server (Environment: {settings.ENV})")
    print(f"📦 Default Model: {settings.DEFAULT_MODEL}")
    print(f"📂 Database: {settings.get_resolved_db_path()}")
    print("=" * 60)

    # 1. Initialize SQLite database & root admin key
    init_db()

    # 2. Warm up model if configured
    if settings.WARMUP_ON_STARTUP:
        try:
            engine_service.warmup()
        except Exception as exc:
            print(f"[Startup Warning] Warmup failed: {exc}")

    yield

    print("🛑 Shutting down Nirnaya Decision Server.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Nirnaya Decision Engine API",
        description=(
            "Production-grade, System-One typed probabilistic decision engine. "
            "Exposes high-speed, single-pass next-token readout for choice, score, "
            "and noul decisions strictly compatible with the TypeSafe AI Jev contract."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Allow CORS for frontends / web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router, tags=["Health"])
    app.include_router(v1_router, prefix="/v1")

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    return app


app = create_app()
