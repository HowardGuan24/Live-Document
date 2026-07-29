"""Load and validate sequence-pipeline specifications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STAGE_ROOT = Path(__file__).resolve().parents[2]


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "sequence_id",
        "title",
        "output_subdir",
        "canvas",
        "paths",
        "state_adapter",
        "semantic_builder",
        "composer",
        "projection",
        "render",
        "composite",
        "video_handoff",
        "common_visual",
        "common_negative",
        "anchor",
        "keyframes",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"sequence spec missing fields: {missing}")
    if not isinstance(spec["state_adapter"], str) or not spec["state_adapter"]:
        raise ValueError("state_adapter must be a non-empty module name")
    if not spec["keyframes"]:
        raise ValueError("sequence spec requires at least one keyframe")
    ids = [item["id"] for item in spec["keyframes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("keyframe ids must be unique")
    canvas = spec["canvas"]
    render = spec["render"]
    if (canvas["width"], canvas["height"]) != (
        render["width"],
        render["height"],
    ):
        raise ValueError("canvas and render dimensions must match")
    for item in spec["keyframes"]:
        for field in (
            "id",
            "display_frame",
            "state_frame",
            "output_filename",
            "meaning",
            "mechanism_delta",
            "stage_forbidden",
            "semantic_layers",
            "geometry_layers",
            "video_transition",
        ):
            if field not in item:
                raise ValueError(
                    f"keyframe {item.get('id', '?')} missing {field}"
                )
    spec["_spec_path"] = str(path.resolve())
    return spec


def resolve_stage_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else STAGE_ROOT / path


def default_output_root(spec: dict[str, Any]) -> Path:
    subdir = Path(spec["output_subdir"])
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError("output_subdir must stay below output/keyframe_render")
    return STAGE_ROOT / "output" / "keyframe_render" / subdir
