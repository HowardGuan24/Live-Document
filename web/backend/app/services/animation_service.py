"""Deterministic engine service: LearningSpec -> animation DSL -> Manim MP4/GIF.

Reuses modules/animation_engine (DSL validation + Manim renderer + FFmpeg).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.config import JOBS_DIR, REPO_ROOT  # noqa: F401 (sys.path side-effect)
from modules.animation_engine.pipeline import generate_animation

ENTITY_COLORS = ["#48CAE4", "#7C9EFF", "#FF8FB1", "#FFD166", "#7AE582", "#B388EB"]


def _style_int(style: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    """Read an int from a client-supplied style dict, clamped to [lo, hi]."""
    try:
        val = int(style.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, val))


def _build_objects(spec: dict[str, Any], style: dict[str, Any]) -> list[dict[str, Any]]:
    goal = (spec.get("learning_goal") or "Concept animation")[:64]
    entities = [e for e in (spec.get("entities") or []) if e][:6]
    steps = spec.get("causal_steps") or []

    objects: list[dict[str, Any]] = [
        {
            "id": "title",
            "type": "text",
            "content": goal,
            "font_size": _style_int(style, "title_font_size", 32, 16, 72),
            "color": "#F7F9FC",
            "position": [0.0, 3.3, 0.0],
        }
    ]

    # Entities as boxes in a row
    n = len(entities)
    box_w, box_h, y = 2.6, 1.1, 1.1
    spacing = 3.4 if n <= 4 else 3.0
    start_x = -((n - 1) * spacing) / 2
    for i, ent in enumerate(entities):
        x = start_x + i * spacing
        objects.append(
            {
                "id": f"entity_{i}",
                "type": "rectangle",
                "width": box_w,
                "height": box_h,
                "color": ENTITY_COLORS[i % len(ENTITY_COLORS)],
                "stroke_width": 4,
                "position": [x, y, 0.0],
            }
        )
        objects.append(
            {
                "id": f"entity_{i}_label",
                "type": "text",
                "content": ent[:22],
                "font_size": 20,
                "color": "#0B1220",
                "position": [x, y, 0.0],
            }
        )

    # Causal steps: arrows between consecutive entities + caption below
    step_y = -1.8
    for idx, step in enumerate(steps[:5]):
        cause = (step.get("cause") or "")[:46]
        change = (step.get("change") or "")[:46]
        caption = f"{cause} → {change}"
        if idx < len(entities) - 1:
            x1 = start_x + idx * spacing
            x2 = start_x + (idx + 1) * spacing
            objects.append(
                {
                    "id": f"arrow_{idx}",
                    "type": "arrow",
                    "from": [x1 + box_w / 2, y, 0.0],
                    "to": [x2 - box_w / 2, y, 0.0],
                    "color": "#FFD166",
                    "stroke_width": 6,
                    "buff": 0.08,
                }
            )
        objects.append(
            {
                "id": f"step_{idx}",
                "type": "text",
                "content": caption,
                "font_size": 20,
                "color": "#FFD166",
                "position": [0.0, step_y - idx * 0.85, 0.0],
            }
        )
    return objects


def _build_timeline(spec: dict[str, Any]) -> list[dict[str, Any]]:
    entities = [e for e in (spec.get("entities") or []) if e][:6]
    steps = spec.get("causal_steps") or []

    timeline: list[dict[str, Any]] = [
        {"action": "write", "target": "title", "duration": 0.5}
    ]
    if entities:
        parallel = []
        for i in range(len(entities)):
            parallel.append({"action": "create", "target": f"entity_{i}", "duration": 0.4})
            parallel.append({"action": "write", "target": f"entity_{i}_label", "duration": 0.4})
        timeline.append({"parallel": parallel, "duration": 0.7})

    for idx in range(len(steps[:5])):
        actions = [{"action": "write", "target": f"step_{idx}", "duration": 0.45}]
        if idx < len(entities) - 1:
            actions.insert(0, {"action": "grow_arrow", "target": f"arrow_{idx}", "duration": 0.35})
        timeline.append({"parallel": actions, "duration": 0.55})

    timeline.append({"action": "wait", "duration": 0.4})
    return timeline


def build_dsl(spec: dict[str, Any], job_id: str, style: dict[str, Any]) -> dict[str, Any]:
    """Convert a LearningSpec into a valid animation DSL document."""
    output = {
        "width": _style_int(style, "width", 960, 320, 1920),
        "height": _style_int(style, "height", 540, 240, 1080),
        "fps": _style_int(style, "fps", 30, 12, 60),
        "gif_fps": _style_int(style, "gif_fps", 15, 5, 30),
        "formats": ["mp4", "gif"],
    }
    return {
        "id": job_id,
        "source_text": spec.get("learning_goal") or "",
        "explanation_goal": spec.get("learning_goal") or "",
        "background_color": style.get("background_color", "#10151C"),
        "output": output,
        "objects": _build_objects(spec, style),
        "timeline": _build_timeline(spec),
    }


def run_deterministic(
    job_id: str,
    spec: dict[str, Any],
    style: dict[str, Any],
) -> dict[str, Any]:
    """Render a deterministic Manim animation for a job; returns a result dict."""
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dsl = build_dsl(spec, job_id, style)
    dsl_path = job_dir / "input.json"
    dsl_path.write_text(json.dumps(dsl, ensure_ascii=False, indent=2), encoding="utf-8")

    started = time.perf_counter()
    manifest = generate_animation(dsl_path, output_root=JOBS_DIR)
    manifest.setdefault("metrics", {})["render_time_seconds"] = round(
        time.perf_counter() - started, 3
    )
    manifest["renderer"] = "manim_dsl"
    return manifest
