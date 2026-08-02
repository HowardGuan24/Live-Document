"""Pydantic schemas for the Live-Document web API."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

EngineName = Literal["deterministic", "generative", "procedural"]


class DocumentRequest(BaseModel):
    text: str = Field(min_length=1, description="Document / paragraph text to plan")
    filename: Optional[str] = None


class SpecOut(BaseModel):
    learning_goal: Optional[str] = None
    entities: list[str] = []
    state_variables: list[str] = []
    causal_steps: list[dict[str, Any]] = []
    invariants: list[str] = []
    comprehension_questions: list[str] = []
    fallback_reason: Optional[str] = None


class SpecsResponse(BaseModel):
    specs: list[SpecOut]
    count: int
    suitable: int


class JobCreate(BaseModel):
    text: Optional[str] = None
    spec: Optional[SpecOut] = None
    engine: EngineName = "deterministic"
    style: dict[str, Any] = Field(default_factory=dict)


class JobOut(BaseModel):
    id: str
    engine: str
    status: str
    progress: float
    message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    spec: Optional[SpecOut] = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: Optional[dict[str, Any]] = None


class JobListResponse(BaseModel):
    jobs: list[JobOut]
    total: int


class EngineHealth(BaseModel):
    available: bool
    detail: str


class HealthResponse(BaseModel):
    status: str
    python: str
    gpu: dict[str, Any]
    engines: dict[str, EngineHealth]
