"""Document planning endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas import DocumentRequest, SpecOut, SpecsResponse
from app.services.doc_planner_service import plan_document

router = APIRouter(prefix="/api/v1/specs", tags=["specs"])

logger = logging.getLogger("uvicorn.error")


@router.post("", response_model=SpecsResponse)
def create_specs(req: DocumentRequest) -> SpecsResponse:
    try:
        specs = plan_document(req.text)
    except Exception:
        logger.exception("document planning failed")
        raise HTTPException(status_code=500, detail="planning failed") from None
    suitable = sum(1 for s in specs if s.get("fallback_reason") is None)
    return SpecsResponse(
        specs=[SpecOut(**s) for s in specs],
        count=len(specs),
        suitable=suitable,
    )
