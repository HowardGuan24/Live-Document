"""Health / GPU / engine availability endpoint."""

from __future__ import annotations

import platform
from functools import lru_cache

from fastapi import APIRouter

from app.schemas import EngineHealth, HealthResponse
from app.services import generative_service

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@lru_cache(maxsize=1)
def _module_available(module: str) -> bool:
    """Cheap cached import probe (imports are slow on cold start)."""
    try:
        __import__(module)
        return True
    except Exception:
        return False


@router.get("", response_model=HealthResponse)
def health() -> HealthResponse:
    gen = generative_service.probe()
    return HealthResponse(
        status="ok",
        python=platform.python_version(),
        gpu=gen,
        engines={
            "deterministic": EngineHealth(
                available=_module_available("manim"),
                detail="Manim DSL engine",
            ),
            "generative": EngineHealth(available=gen["available"], detail=gen["detail"]),
            "procedural": EngineHealth(
                available=_module_available("PIL"),
                detail="PIL fallback renderer",
            ),
        },
    )
