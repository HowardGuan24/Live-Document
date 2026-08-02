"""Async job queue + worker for animation jobs (single-process, lightweight)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import JOBS_DIR
from app.services import animation_service, generative_service, procedural_service
from app.storage import JobStore


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, store: JobStore):
        self.store = store
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._mark_stale_failed()
            self._task = asyncio.create_task(self._worker(), name="job-worker")

    async def shutdown(self) -> None:
        """Drain the queue with a timeout, then cancel the worker task."""
        try:
            await asyncio.wait_for(self.queue.join(), timeout=15)
        except asyncio.TimeoutError:
            pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, RuntimeError):
                pass
            self._task = None

    def _mark_stale_failed(self) -> None:
        """Mark jobs left in pending/running by a previous process as failed."""
        for job in self.store.list(limit=500):
            if job.get("status") in ("pending", "running"):
                self.store.update(job["id"], {
                    "status": "failed",
                    "progress": 1.0,
                    "message": "server restarted before completion",
                    "error": {"type": "StaleJob", "message": "interrupted by server restart"},
                    "updated_at": utcnow(),
                })

    async def submit(self, job: dict[str, Any]) -> None:
        self.store.create(job)
        await self.queue.put(job["id"])

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await asyncio.to_thread(self._run, job_id)
            except Exception as exc:  # pragma: no cover - defensive
                self.store.update(job_id, {
                    "status": "failed",
                    "progress": 1.0,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "updated_at": utcnow(),
                })
            finally:
                self.queue.task_done()

    def _run(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        self.store.update(job_id, {
            "status": "running",
            "progress": 0.15,
            "message": "started",
            "updated_at": utcnow(),
        })
        spec = job.get("spec") or {}
        style = job.get("style") or {}
        try:
            engine = job.get("engine", "deterministic")
            fallback_reason: str | None = None
            if engine == "generative":
                try:
                    result = generative_service.run_generative(job_id, spec, style)
                except generative_service.GenerativeUnavailableError as exc:
                    result = procedural_service.run_procedural(job_id, spec, style)
                    fallback_reason = str(exc)
            elif engine == "procedural":
                result = procedural_service.run_procedural(job_id, spec, style)
            else:
                result = animation_service.run_deterministic(job_id, spec, style)

            artifacts = _collect_artifacts(job_id, result)
            metrics = dict(result.get("metrics") or {})
            if fallback_reason:
                metrics["fallback_reason"] = fallback_reason

            self.store.update(job_id, {
                "status": "completed",
                "progress": 1.0,
                "message": "done",
                "artifacts": artifacts,
                "metrics": metrics,
                "error": None,
                "updated_at": utcnow(),
            })
        except Exception as exc:
            self.store.update(job_id, {
                "status": "failed",
                "progress": 1.0,
                "message": "render failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "updated_at": utcnow(),
            })


def _collect_artifacts(job_id: str, result: dict[str, Any]) -> dict[str, str]:
    """Expose job artifacts as download URLs under /api/v1/jobs/<id>/files/<name>."""
    job_dir = JOBS_DIR / job_id
    base = f"/api/v1/jobs/{job_id}/files"
    artifacts: dict[str, str] = {}
    outputs = result.get("outputs") or {}
    for key, path in outputs.items():
        name = Path(str(path)).name
        artifacts[key] = f"{base}/{name}"
    for name in ("normalized_spec.json", "result.json"):
        if (job_dir / name).exists():
            artifacts.setdefault(name.removesuffix(".json"), f"{base}/{name}")
    return artifacts


def new_job(engine: str, spec: dict[str, Any], style: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:12],
        "engine": engine,
        "status": "pending",
        "progress": 0.0,
        "message": "queued",
        "created_at": utcnow(),
        "updated_at": utcnow(),
        "spec": spec,
        "artifacts": {},
        "metrics": {},
        "error": None,
        "style": style,
    }
