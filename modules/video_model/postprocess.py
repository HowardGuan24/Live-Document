"""ffmpeg normalization, embedding exports, and media inspection."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} is required but was not found on PATH")
    return path


def _run(command: list[str]) -> list[str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"ffmpeg post-processing failed: {message}")
    return command


def normalize_video(
    source: Path,
    output: Path,
    *,
    duration: float,
    fps: int,
    width: int,
    height: int,
    loop_mode: str,
) -> list[str]:
    ffmpeg = _binary("ffmpeg")
    base_filter = (
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1"
    )
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if loop_mode == "pingpong":
        half = duration / 2
        filter_graph = (
            f"[0:v]{base_filter},trim=duration={half},setpts=PTS-STARTPTS[forward];"
            f"[forward]split[a][b];[b]reverse[reverse];[a][reverse]concat=n=2:v=1:a=0[out]"
        )
        command.extend(["-filter_complex", filter_graph, "-map", "[out]", "-t", str(duration)])
    else:
        command.extend(
            [
                "-vf",
                f"{base_filter},tpad=stop_mode=clone:stop_duration={duration},"
                f"trim=duration={duration},setpts=PTS-STARTPTS",
            ]
        )
    command.extend(
        [
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return _run(command)


def export_webm(source: Path, output: Path) -> list[str]:
    return _run(
        [
            _binary("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "32",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )


def export_gif(source: Path, output: Path, max_width: int = 640, fps: int = 12) -> list[str]:
    filter_graph = (
        f"fps={fps},scale='min({max_width},iw)':-2:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
    )
    return _run(
        [
            _binary("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            filter_graph,
            "-loop",
            "0",
            str(output),
        ]
    )


def probe_media(path: Path) -> dict[str, Any]:
    command = [
            _binary("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size",
            "-of",
            "json",
            str(path),
        ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    data = json.loads(completed.stdout)
    stream = data.get("streams", [{}])[0]
    media_format = data.get("format", {})
    return {
        "path": str(path.resolve()),
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "average_frame_rate": stream.get("avg_frame_rate"),
        "frame_count": stream.get("nb_frames"),
        "duration_seconds": float(media_format["duration"]) if media_format.get("duration") else None,
        "size_bytes": int(media_format["size"]) if media_format.get("size") else path.stat().st_size,
        "sha256": sha256_file(path),
        "ffprobe_command": command,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
