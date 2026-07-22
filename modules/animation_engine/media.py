"""FFmpeg conversion and lightweight artifact validation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .manim_renderer import RendererDependencyError


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RendererDependencyError(
            "FFmpeg is unavailable. Install FFmpeg or run: "
            "python -m pip install imageio-ffmpeg"
        ) from exc


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"FFmpeg failed with exit code {completed.returncode}:\n{details[-3000:]}")


def mp4_to_gif(mp4_path: Path, gif_path: Path, fps: int, width: int) -> Path:
    """Create a palette-optimized GIF from an MP4."""

    ffmpeg = find_ffmpeg()
    palette_path = gif_path.with_name("palette.png")
    scale_filter = f"fps={fps},scale={width}:-2:flags=lanczos"
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(mp4_path),
            "-vf",
            f"{scale_filter},palettegen=stats_mode=diff",
            "-frames:v",
            "1",
            str(palette_path),
        ]
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(mp4_path),
            "-i",
            str(palette_path),
            "-lavfi",
            f"{scale_filter}[x];[x][1:v]paletteuse=dither=sierra2_4a",
            "-loop",
            "0",
            str(gif_path),
        ]
    )
    palette_path.unlink(missing_ok=True)
    if not gif_path.is_file() or gif_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg completed without producing a valid GIF")
    return gif_path


def inspect_artifacts(mp4_path: Path, gif_path: Path | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "mp4_size_bytes": mp4_path.stat().st_size,
    }
    if gif_path:
        metrics["gif_size_bytes"] = gif_path.stat().st_size
        try:
            from PIL import Image

            with Image.open(gif_path) as image:
                metrics["gif_width"] = image.width
                metrics["gif_height"] = image.height
                metrics["gif_frames"] = getattr(image, "n_frames", 1)
            if metrics["gif_frames"] < 2:
                raise RuntimeError("Generated GIF contains fewer than two frames")
        except ImportError:
            metrics["gif_frames"] = None
    return metrics
