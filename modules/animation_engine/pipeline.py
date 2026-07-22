"""End-to-end JSON -> Manim -> MP4/GIF pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manim_renderer import render_manim
from .media import inspect_artifacts, mp4_to_gif
from .schema import load_animation_spec


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_animation(input_path: str | Path, output_root: str | Path = "outputs") -> dict[str, Any]:
    """Render one JSON file and return the same manifest written to disk."""

    started = time.perf_counter()
    input_path = Path(input_path).resolve()
    spec = load_animation_spec(input_path)
    job_dir = Path(output_root).resolve() / spec.id
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = job_dir / "result.json"

    manifest: dict[str, Any] = {
        "id": spec.id,
        "status": "rendering",
        "renderer": "manim_dsl",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "outputs": {},
        "metrics": {},
        "error": None,
    }
    normalized_path = job_dir / "normalized_spec.json"
    _write_json(normalized_path, asdict(spec))
    manifest["outputs"]["normalized_spec"] = str(normalized_path)
    _write_json(manifest_path, manifest)

    try:
        mp4_path = render_manim(spec, job_dir, input_path.parent)
        manifest["outputs"]["mp4"] = str(mp4_path)

        gif_path: Path | None = None
        if "gif" in spec.output.formats:
            gif_path = mp4_to_gif(
                mp4_path,
                job_dir / "animation.gif",
                fps=spec.output.gif_fps,
                width=min(spec.output.width, spec.output.max_gif_width),
            )
            manifest["outputs"]["gif"] = str(gif_path)

        manifest["metrics"] = inspect_artifacts(mp4_path, gif_path)
        manifest["metrics"].update(
            {
                "render_time_seconds": round(time.perf_counter() - started, 3),
                "width": spec.output.width,
                "height": spec.output.height,
                "fps": spec.output.fps,
                "gif_fps": spec.output.gif_fps if gif_path else None,
            }
        )
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(manifest_path, manifest)
        raise

    _write_json(manifest_path, manifest)
    return manifest
