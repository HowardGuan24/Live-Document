"""Input validation and prompt construction for explanation videos."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONTENT_TYPE_ALIASES = {
    "process": "process",
    "formula": "process",
    "operation": "process",
    "state_change": "state_change",
    "state-change": "state_change",
    "data_flow": "data_flow",
    "dataflow": "data_flow",
    "data-flow": "data_flow",
    "scene": "scene",
    "analogy": "scene",
}

DEFAULT_PROMPT_TEMPLATE_VERSION = "ltx-explainer-v2.1"

TYPE_DIRECTIONS = {
    "process": (
        "Show one subject completing one ordered action from start to finish. "
        "Use a single obvious path, small incremental movement, and a clearly visible end state."
    ),
    "state_change": (
        "Show one object changing from a clearly distinct before state to an after state. "
        "Keep its identity, position, scale, and surrounding reference objects stable."
    ),
    "data_flow": (
        "Use a sparse left-to-right diagram with a source, one transformation, and a destination. "
        "Represent data as a few consistent particles moving along one unambiguous path."
    ),
    "scene": (
        "Use one concrete visual analogy with at most three major objects. "
        "Animate only the interaction that explains the concept and keep spatial relationships stable."
    ),
}

GLOBAL_DIRECTIONS = {
    "explainer-v1": (
        "Create a short educational explainer animation, not a cinematic scene. "
        "Use a clean flat diagram-like style, high contrast, a plain uncluttered background, "
        "one focal action, stable object shapes, and a fixed orthographic or front-facing camera. "
        "Use no cuts, no camera movement, no decorative motion, and no people. "
        "Leave text and labels out because they will be added deterministically in post-production."
    ),
    "ltx-explainer-v2.1": (
        "Create one continuous short educational shot. Show no more than three major objects, "
        "and preserve each object's color, shape, scale, and identity for the full shot. "
        "Use one literal, easy-to-follow action with a clear direction and a visible cause-and-effect relationship. "
        "Keep the camera completely fixed, front-facing, and wide enough to contain the whole action. "
        "Use a clean high-contrast diagram-like scene with a plain background and no decorative details. "
        "Do not render letters, labels, captions, numbers, logos, or watermarks."
    ),
}

LOOP_DIRECTION = (
    "The action should be loop-friendly: begin and end on visually compatible frames, "
    "with gentle motion and no abrupt appearance or disappearance."
)

NEGATIVE_PROMPT = (
    "cinematic camera, camera pan, camera zoom, camera shake, scene cut, photorealistic, "
    "busy background, clutter, decorative objects, multiple simultaneous actions, object duplication, "
    "object morphing, warped geometry, inconsistent shape, flicker, jitter, motion blur, noise, "
    "unreadable text, letters, words, subtitles, watermark, logo, low contrast"
)


@dataclass(frozen=True)
class InputSpec:
    """Validated hand-off from the document understanding module."""

    source_text: str
    visual_goal: str
    video_prompt: str
    content_type: str = "process"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return value.strip()


def validate_input(data: Any) -> InputSpec:
    if not isinstance(data, dict):
        raise ValueError("input JSON must contain an object at the top level")

    raw_type = data.get("content_type", "process")
    if not isinstance(raw_type, str):
        raise ValueError("'content_type' must be a string")
    content_type = CONTENT_TYPE_ALIASES.get(raw_type.strip().lower())
    if content_type is None:
        allowed = ", ".join(sorted({"process", "state_change", "data_flow", "scene"}))
        raise ValueError(f"unsupported content_type '{raw_type}'; expected one of: {allowed}")

    return InputSpec(
        source_text=_required_text(data, "source_text"),
        visual_goal=_required_text(data, "visual_goal"),
        video_prompt=_required_text(data, "video_prompt"),
        content_type=content_type,
    )


def load_input(path: str | Path) -> InputSpec:
    input_path = Path(path)
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"input file does not exist: {input_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {input_path}: {exc}") from exc
    return validate_input(data)


def build_prompts(
    spec: InputSpec,
    template_version: str = DEFAULT_PROMPT_TEMPLATE_VERSION,
) -> tuple[str, str]:
    """Use all semantic inputs while adding strict explanation-video constraints."""

    if template_version not in GLOBAL_DIRECTIONS:
        versions = ", ".join(sorted(GLOBAL_DIRECTIONS))
        raise ValueError(f"unknown prompt template version '{template_version}'; available: {versions}")

    positive = "\n".join(
        [
            GLOBAL_DIRECTIONS[template_version],
            TYPE_DIRECTIONS[spec.content_type],
            f"Source concept: {spec.source_text}",
            f"Teaching goal: {spec.visual_goal}",
            f"Requested visual: {spec.video_prompt}",
            LOOP_DIRECTION,
        ]
    )
    return positive, NEGATIVE_PROMPT
