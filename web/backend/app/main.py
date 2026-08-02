"""Live-Document web backend — FastAPI application.

Run:
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import health, jobs, specs
from app.auth import router as auth_router, require_token
from app.config import (
    AUTH_ENABLED,
    AUTH_TOKEN,
    CORS_ORIGINS,
    DB_PATH,
    FRONTEND_DIST,
    SERVE_FRONTEND,
)
from app.services.job_manager import JobManager
from app.storage import JobStore

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = JobStore(DB_PATH)
    manager = JobManager(store)
    manager.start()
    app.state.manager = manager
    if AUTH_ENABLED:
        logger.info("Auth enabled. Access token: %s", AUTH_TOKEN)
    else:
        logger.warning(
            "Auth DISABLED (LIVE_DOC_AUTH_DISABLED=1) — public URLs will be unprotected!"
        )
    yield
    # allow in-flight work to finish (bounded), then cancel the worker
    await manager.shutdown()


app = FastAPI(
    title="Live-Document API",
    description="Document -> LearningSpec -> animation/video generation web API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public: token login only.
app.include_router(auth_router)
# Everything else under /api/v1 requires a valid token.
app.include_router(specs.router, dependencies=[Depends(require_token)])
app.include_router(jobs.router, dependencies=[Depends(require_token)])
app.include_router(health.router, dependencies=[Depends(require_token)])


if not (SERVE_FRONTEND and FRONTEND_DIST.is_dir()):
    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "name": "Live-Document API",
            "docs": "/docs",
            "health": "/api/v1/health",
            "specs": "/api/v1/specs",
            "jobs": "/api/v1/jobs",
        }


if SERVE_FRONTEND and FRONTEND_DIST.is_dir():
    # Register the static mount AFTER the API routers so /api/* keeps
    # priority, but BEFORE any catch-all root route (which is skipped when
    # the frontend is served, so "/" returns the SPA).
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIST), html=True),
        name="frontend",
    )
