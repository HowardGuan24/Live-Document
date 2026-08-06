"""Compatibility wrapper for the removed deterministic animation engine."""

from __future__ import annotations

from typing import Any

from app.services.procedural_service import run_procedural


def run_deterministic(
    job_id: str,
    spec: dict[str, Any],
    style: dict[str, Any],
) -> dict[str, Any]:
    """Render through the maintained procedural backend."""
    result = run_procedural(job_id, spec, style)
    result["renderer"] = "procedural_deterministic_fallback"
    result.setdefault("metrics", {})["fallback_reason"] = (
        "the legacy deterministic renderer was removed during repository restaging"
    )
    return result
