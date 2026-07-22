"""Generation backends: Wan2.1 for production and a local smoke-test renderer."""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .prompting import InputSpec


@dataclass(frozen=True)
class GenerationSettings:
    duration: float
    fps: int
    width: int
    height: int
    seed: int
    inference_steps: int
    guidance_scale: float
    model_id: str
    cpu_offload: bool = False


def wan_environment() -> dict[str, Any]:
    """Return availability details without importing model code at module import time."""

    details: dict[str, Any] = {
        "available": False,
        "torch_installed": False,
        "diffusers_installed": False,
        "accelerator_available": False,
        "rocm_version": None,
        "reason": None,
    }
    try:
        import torch

        details["torch_installed"] = True
        details["accelerator_available"] = bool(torch.cuda.is_available())
        details["rocm_version"] = getattr(torch.version, "hip", None)
    except ImportError:
        details["reason"] = "PyTorch is not installed"
        return details

    try:
        from diffusers import WanPipeline  # noqa: F401

        details["diffusers_installed"] = True
    except (ImportError, RuntimeError) as exc:
        details["reason"] = f"Diffusers WanPipeline is unavailable: {exc}"
        return details

    if not details["accelerator_available"]:
        details["reason"] = "PyTorch cannot see a CUDA/ROCm accelerator"
        return details

    details["available"] = True
    return details


def generate_wan(
    positive_prompt: str,
    negative_prompt: str,
    output_path: Path,
    settings: GenerationSettings,
) -> dict[str, Any]:
    """Generate a raw MP4 with the official Diffusers Wan pipeline."""

    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    # Wan temporal compression requires 4*k+1 frames.
    requested_frames = max(9, round(settings.duration * settings.fps))
    num_frames = max(9, math.ceil((requested_frames - 1) / 4) * 4 + 1)

    vae = AutoencoderKLWan.from_pretrained(
        settings.model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipeline = WanPipeline.from_pretrained(
        settings.model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(
        pipeline.scheduler.config,
        flow_shift=3.0,
    )
    if settings.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        # ROCm intentionally uses PyTorch's CUDA-compatible API surface.
        pipeline.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(settings.seed)
    frames = pipeline(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        height=settings.height,
        width=settings.width,
        num_frames=num_frames,
        num_inference_steps=settings.inference_steps,
        guidance_scale=settings.guidance_scale,
        generator=generator,
    ).frames[0]
    export_to_video(frames, str(output_path), fps=settings.fps)

    return {
        "backend": "wan",
        "workflow": "text-to-video with Wan2.1 via Diffusers",
        "model_id": settings.model_id,
        "num_frames": num_frames,
        "device": "cuda (PyTorch API; ROCm when torch.version.hip is set)",
        "rocm_version": getattr(torch.version, "hip", None),
    }


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _base_frame(width: int, height: int, content_type: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "#f7fafc")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (width * 0.045, height * 0.07, width * 0.955, height * 0.93),
        radius=max(8, width // 60),
        fill="#ffffff",
        outline="#d8e2ec",
        width=max(2, width // 320),
    )
    draw.text(
        (width * 0.075, height * 0.105),
        content_type.replace("_", " ").upper(),
        fill="#24364b",
        font=_font(max(14, height // 24)),
    )
    return image, draw


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: str, width: int) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = max(9, width * 3)
    points = [end]
    for offset in (2.55, -2.55):
        points.append((end[0] + head * math.cos(angle + offset), end[1] + head * math.sin(angle + offset)))
    draw.polygon(points, fill=color)


def _process_frame(draw: ImageDraw.ImageDraw, width: int, height: int, phase: float, gradient: bool) -> None:
    line_width = max(3, width // 180)
    if gradient:
        left, right = width * 0.13, width * 0.88
        top, bottom = height * 0.28, height * 0.79
        points = []
        for index in range(121):
            ratio = index / 120
            x = left + (right - left) * ratio
            y = top + (bottom - top) * (1 - math.exp(-4.2 * ratio))
            points.append((x, y))
        draw.line(points, fill="#4f6f8f", width=line_width)
        # Travel downhill only. Fade at both ends so the loop reset does not
        # visually teach the incorrect reverse action (climbing uphill).
        eased = phase * phase * (3 - 2 * phase)
        index = min(120, round(eased * 120))
        x, y = points[index]
        radius = max(8, width // 55)
        visibility = max(0.0, min(1.0, phase / 0.12, (1.0 - phase) / 0.18))
        fill = tuple(round(255 + (target - 255) * visibility) for target in (237, 106, 90))
        outline = tuple(round(255 + (target - 255) * visibility) for target in (158, 57, 48))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
        draw.text((left, bottom + height * 0.05), "START", fill="#6b7f93", font=_font(max(12, height // 30)))
        draw.text((right - width * 0.09, bottom + height * 0.05), "MINIMUM", fill="#2a8f68", font=_font(max(12, height // 30)))
        return

    y = height * 0.56
    xs = [width * 0.2, width * 0.5, width * 0.8]
    labels = ["INPUT", "CHANGE", "RESULT"]
    for index in range(2):
        _arrow(draw, (xs[index] + width * 0.075, y), (xs[index + 1] - width * 0.075, y), "#93a9bf", line_width)
    for x, label in zip(xs, labels):
        radius = width * 0.065
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#e8f2fb", outline="#3978a8", width=line_width)
        box = draw.textbbox((0, 0), label, font=_font(max(10, height // 35)))
        draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), label, fill="#244b68", font=_font(max(10, height // 35)))
    progress = (phase * 1.15) % 1.0
    x = xs[0] + (xs[-1] - xs[0]) * progress
    alpha = math.sin(math.pi * min(1.0, progress))
    radius = max(7, width // 70)
    color = (237, int(106 + 50 * alpha), 90)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _state_change_frame(draw: ImageDraw.ImageDraw, width: int, height: int, phase: float) -> None:
    y = height * 0.57
    x1, x2 = width * 0.3, width * 0.7
    progress = 0.5 - 0.5 * math.cos(2 * math.pi * phase)
    line_width = max(3, width // 180)
    draw.text((x1 - width * 0.065, height * 0.31), "BEFORE", fill="#6b7f93", font=_font(max(12, height // 30)))
    draw.text((x2 - width * 0.05, height * 0.31), "AFTER", fill="#6b7f93", font=_font(max(12, height // 30)))
    radius1 = width * 0.06
    draw.ellipse((x1 - radius1, y - radius1, x1 + radius1, y + radius1), fill="#8dc5e8", outline="#3978a8", width=line_width)
    _arrow(draw, (x1 + width * 0.1, y), (x2 - width * 0.1, y), "#758ba0", line_width)
    radius2 = width * (0.06 + 0.025 * progress)
    red = int(141 + (79 * progress))
    green = int(197 - (45 * progress))
    blue = int(232 - (119 * progress))
    draw.rounded_rectangle((x2 - radius2, y - radius2, x2 + radius2, y + radius2), radius=int(radius2 * (1 - 0.7 * progress)), fill=(red, green, blue), outline="#9e6a26", width=line_width)


def _data_flow_frame(draw: ImageDraw.ImageDraw, width: int, height: int, phase: float) -> None:
    y = height * 0.56
    centers = [width * 0.2, width * 0.5, width * 0.8]
    labels = ["SOURCE", "TRANSFORM", "OUTPUT"]
    line_width = max(3, width // 180)
    for index in range(2):
        _arrow(draw, (centers[index] + width * 0.1, y), (centers[index + 1] - width * 0.1, y), "#9eb2c5", line_width)
    for index, (x, label) in enumerate(zip(centers, labels)):
        fill = ["#e8f2fb", "#fff0d9", "#e5f5ec"][index]
        outline = ["#3978a8", "#bf7b22", "#2a8f68"][index]
        draw.rounded_rectangle((x - width * 0.09, y - height * 0.105, x + width * 0.09, y + height * 0.105), radius=12, fill=fill, outline=outline, width=line_width)
        font = _font(max(10, height // 38))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), label, fill="#33485d", font=font)
    for offset in (0.0, 0.28, 0.56):
        progress = (phase + offset) % 1.0
        x = centers[0] + (centers[-1] - centers[0]) * progress
        radius = max(5, width // 100)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#ed6a5a")


def _scene_frame(draw: ImageDraw.ImageDraw, width: int, height: int, phase: float) -> None:
    center = (width * 0.5, height * 0.57)
    orbit_x, orbit_y = width * 0.25, height * 0.2
    line_width = max(3, width // 180)
    draw.ellipse((center[0] - orbit_x, center[1] - orbit_y, center[0] + orbit_x, center[1] + orbit_y), outline="#b8c8d8", width=line_width)
    core = max(15, width // 32)
    draw.ellipse((center[0] - core, center[1] - core, center[0] + core, center[1] + core), fill="#ffd166", outline="#b87d14", width=line_width)
    angle = 2 * math.pi * phase
    x = center[0] + orbit_x * math.cos(angle)
    y = center[1] + orbit_y * math.sin(angle)
    radius = max(9, width // 55)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#4e9dcc", outline="#29678e", width=2)
    draw.text((width * 0.39, height * 0.84), "ONE RELATIONSHIP", fill="#6b7f93", font=_font(max(12, height // 30)))


def generate_procedural(
    spec: InputSpec,
    output_path: Path,
    settings: GenerationSettings,
) -> dict[str, Any]:
    """Render a deterministic explainer used for CI, smoke tests, and fallback."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    frame_count = max(2, round(settings.duration * settings.fps))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{settings.width}x{settings.height}",
        "-r",
        str(settings.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        lower_text = f"{spec.source_text} {spec.visual_goal} {spec.video_prompt}".lower()
        for index in range(frame_count):
            # Denominator excludes the duplicate endpoint for a clean periodic loop.
            phase = index / frame_count
            image, draw = _base_frame(settings.width, settings.height, spec.content_type)
            if spec.content_type == "process":
                gradient = any(term in lower_text for term in ("gradient", "loss curve", "梯度", "损失", "下坡"))
                _process_frame(draw, settings.width, settings.height, phase, gradient)
            elif spec.content_type == "state_change":
                _state_change_frame(draw, settings.width, settings.height, phase)
            elif spec.content_type == "data_flow":
                _data_flow_frame(draw, settings.width, settings.height, phase)
            else:
                _scene_frame(draw, settings.width, settings.height, phase)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.stderr:
            process.stderr.close()
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        if process.stderr:
            process.stderr.close()
        raise
    if return_code != 0:
        raise RuntimeError(f"procedural ffmpeg render failed: {stderr.strip()}")

    return {
        "backend": "procedural",
        "workflow": "deterministic explanation animation (smoke-test/fallback)",
        "model_id": None,
        "num_frames": frame_count,
        "device": "CPU",
        "rocm_version": None,
    }
