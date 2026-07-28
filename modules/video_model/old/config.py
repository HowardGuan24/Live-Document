"""Versioned generation profiles and shared V2 configuration validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE_DIR = MODULE_ROOT / "benchmarks" / "configs"


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    description: str
    width: int
    height: int
    num_frames: int
    fps: int
    duration_seconds: float
    inference_steps: int
    guidance_scale: float
    prompt_template_version: str
    use_multiscale: bool = False
    vae_tiling: bool = False
    vae_slicing: bool = False
    cpu_offload: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_profile_data(data: Any, expected_name: str | None = None) -> GenerationProfile:
    if not isinstance(data, dict):
        raise ValueError("profile must be a JSON object")
    required = {
        "name": str,
        "description": str,
        "width": int,
        "height": int,
        "num_frames": int,
        "fps": int,
        "duration_seconds": (int, float),
        "inference_steps": int,
        "guidance_scale": (int, float),
        "prompt_template_version": str,
    }
    for field, expected_type in required.items():
        if field not in data or not isinstance(data[field], expected_type):
            raise ValueError(f"profile field '{field}' has an invalid or missing value")

    name = data["name"].strip()
    if expected_name and name != expected_name:
        raise ValueError(f"profile name '{name}' does not match filename '{expected_name}'")
    width, height = data["width"], data["height"]
    if width < 256 or height < 256 or width % 32 or height % 32:
        raise ValueError("profile width and height must be at least 256 and divisible by 32")
    num_frames = data["num_frames"]
    if num_frames < 9 or (num_frames - 1) % 8:
        raise ValueError("profile num_frames must follow LTX's 8*k+1 constraint")
    fps = data["fps"]
    if not 4 <= fps <= 30:
        raise ValueError("profile fps must be between 4 and 30")
    duration = float(data["duration_seconds"])
    if not 3 <= duration <= 8:
        raise ValueError("profile duration_seconds must be between 3 and 8")
    if not 1 <= data["inference_steps"] <= 50:
        raise ValueError("profile inference_steps must be between 1 and 50")
    if float(data["guidance_scale"]) <= 0:
        raise ValueError("profile guidance_scale must be positive")

    return GenerationProfile(
        name=name,
        description=data["description"].strip(),
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        duration_seconds=duration,
        inference_steps=data["inference_steps"],
        guidance_scale=float(data["guidance_scale"]),
        prompt_template_version=data["prompt_template_version"].strip(),
        use_multiscale=bool(data.get("use_multiscale", False)),
        vae_tiling=bool(data.get("vae_tiling", False)),
        vae_slicing=bool(data.get("vae_slicing", False)),
        cpu_offload=bool(data.get("cpu_offload", False)),
    )


def load_profile(name: str, profile_dir: str | Path = DEFAULT_PROFILE_DIR) -> GenerationProfile:
    if not name or any(part in name for part in ("/", "\\", "..")):
        raise ValueError(f"invalid profile name: {name!r}")
    path = Path(profile_dir) / f"{name}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        available = ", ".join(list_profiles(profile_dir)) or "none"
        raise ValueError(f"unknown profile '{name}'; available profiles: {available}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid profile JSON in {path}: {exc}") from exc
    return validate_profile_data(data, expected_name=name)


def list_profiles(profile_dir: str | Path = DEFAULT_PROFILE_DIR) -> list[str]:
    directory = Path(profile_dir)
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.json") if path.stem != "experiments")
