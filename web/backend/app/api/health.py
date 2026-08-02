"""Health / GPU / engine availability endpoint."""

from __future__ import annotations

import platform

from fastapi import APIRouter

from app.schemas import EngineHealth, HealthResponse
from app.services import generative_service

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health() -> HealthResponse:
    gen = generative_service.probe()
    return HealthResponse(
        status="ok",
        python=platform.python_version(),
        gpu=gen,
        engines={
            "deterministic": EngineHealth(available=True, detail="Manim DSL engine"),
            "generative": EngineHealth(available=gen["available"], detail=gen["detail"]),
            "procedural": EngineHealth(available=True, detail="PIL fallback renderer"),
        },
    )
