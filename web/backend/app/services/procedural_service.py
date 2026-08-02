"""Procedural fallback renderer: spec -> PIL GIF (no external model needed).

Keeps the web UI always functional even when Manim or the LTX/Wan stack is
missing; mirrors the repo's existing `success_fallback` semantics.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.config import JOBS_DIR

W, H = 960, 540
BG = (10, 18, 35)
TITLE = (235, 242, 255)
MUTED = (142, 164, 196)
ARROW = (255, 209, 102)
BOXES = [(72, 202, 228), (124, 158, 255), (255, 143, 177), (255, 209, 102), (122, 229, 130), (179, 136, 235)]


def _font(size: int, bold: bool = False) -> Any:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()


def _ease(v: float) -> float:
    v = min(1.0, max(0.0, v))
    return v * v * (3 - 2 * v)


def render_frame(
    frame: int, total: int, goal: str, entities: list[str], steps: list[dict[str, Any]]
) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    progress = frame / max(total - 1, 1)

    d.text((W // 2, 66), goal[:70], fill=TITLE, font=_font(36, bold=True), anchor="mm")
    d.text((W // 2, 112), "Live-Document · procedural fallback renderer", fill=MUTED, font=_font(18), anchor="mm")

    n = len(entities)
    if n == 0:
        d.text((W // 2, H // 2), "No dynamic content found", fill=MUTED, font=_font(26), anchor="mm")
        return img

    spacing = 240 if n <= 4 else 200
    start_x = W // 2 - (n - 1) * spacing // 2
    y = 250

    for i, ent in enumerate(entities):
        t = _ease((progress - i * 0.22) / 0.18)
        if t <= 0:
            continue
        cx = start_x + i * spacing
        half_w, half_h = 92 * t, 54 * t
        color = BOXES[i % len(BOXES)]
        d.rounded_rectangle(
            (cx - half_w, y - half_h, cx + half_w, y + half_h), radius=20, fill=color
        )
        d.text((cx, y), ent[:16], fill="#0B1220", font=_font(20, bold=True), anchor="mm")

    for i in range(len(steps[:5])):
        if i >= n - 1:
            break
        t = _ease((progress - (0.45 + i * 0.22)) / 0.18)
        if t <= 0:
            continue
        x1 = start_x + i * spacing + 92
        x2 = start_x + (i + 1) * spacing - 92
        ym = y
        end_x = x1 + (x2 - x1) * t
        d.line((x1, ym, end_x, ym), fill=ARROW, width=6)
        if t > 0.9:
            d.polygon([(end_x, ym), (end_x - 14, ym - 9), (end_x - 14, ym + 9)], fill=ARROW)

    # step captions
    for i, step in enumerate(steps[:5]):
        t = _ease((progress - (0.45 + i * 0.22)) / 0.18)
        if t <= 0:
            continue
        caption = f"{(step.get('cause') or '')[:40]} → {(step.get('change') or '')[:40]}"
        alpha = int(255 * t)
        d.text((W // 2, 430 + i * 52), caption, fill=(ARROW[0], ARROW[1], ARROW[2], alpha), font=_font(20), anchor="mm")

    return img


def run_procedural(job_id: str, spec: dict[str, Any], style: dict[str, Any]) -> dict[str, Any]:
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    goal = spec.get("learning_goal") or "Concept animation"
    entities = [e for e in (spec.get("entities") or []) if e][:6]
    steps = spec.get("causal_steps") or []

    started = time.perf_counter()
    total = 46
    frames = [
        render_frame(i, total, goal, entities, steps)
        for i in range(total)
    ]
    gif_path = job_dir / "animation.gif"
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True
    )
    render_time = round(time.perf_counter() - started, 3)

    return {
        "id": job_id,
        "status": "completed",
        "renderer": "procedural_fallback",
        "outputs": {"gif": str(gif_path)},
        "metrics": {
            "render_time_seconds": render_time,
            "width": W,
            "height": H,
            "frames": total,
            "fallback": True,
        },
        "error": None,
    }
