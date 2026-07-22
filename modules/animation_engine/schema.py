"""Validation and normalization for the animation DSL.

The public input is JSON, but downstream code receives immutable-ish dataclasses
with all defaults applied. This module intentionally has no Manim dependency so
bad requests can be rejected before a renderer process starts.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_OBJECTS = {
    "arrow",
    "axes",
    "circle",
    "formula",
    "graph",
    "group",
    "image",
    "line",
    "point",
    "rectangle",
    "text",
}

SUPPORTED_ACTIONS = {
    "add",
    "change_color",
    "create",
    "fade_in",
    "fade_out",
    "follow_path",
    "formula_transform",
    "grow_arrow",
    "highlight",
    "highlight_parts",
    "move",
    "move_by",
    "rotate",
    "scale",
    "transform",
    "wait",
    "write",
}

DEPENDENT_OBJECTS = {"arrow", "graph", "group", "line"}
FUNCTIONS = {"cos", "gaussian", "linear", "quadratic", "sigmoid", "sin"}
ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


class DSLValidationError(ValueError):
    """Raised when an input cannot be safely interpreted as animation DSL."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Invalid animation DSL:\n- " + "\n- ".join(issues))


@dataclass(slots=True)
class OutputConfig:
    width: int = 768
    height: int = 432
    fps: int = 30
    gif_fps: int = 15
    max_gif_width: int = 768
    formats: list[str] = field(default_factory=lambda: ["mp4", "gif"])


@dataclass(slots=True)
class ObjectSpec:
    id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionSpec:
    action: str | None = None
    target: str | None = None
    duration: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    parallel: list["ActionSpec"] = field(default_factory=list)


@dataclass(slots=True)
class AnimationSpec:
    id: str
    objects: list[ObjectSpec]
    timeline: list[ActionSpec]
    output: OutputConfig = field(default_factory=OutputConfig)
    background_color: str = "#111318"
    source_text: str = ""
    explanation_goal: str = ""


def _identifier(value: Any, fallback: str = "animation") -> str:
    candidate = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or fallback)).strip("_")
    if not candidate or not re.match(r"^[A-Za-z_]", candidate):
        candidate = f"animation_{candidate}"
    return candidate[:64]


def _legacy_to_dsl(data: dict[str, Any]) -> dict[str, Any]:
    """Turn the repository's first ExplanationSpec version into safe DSL.

    It is deliberately a generic storyboard. Rich animations should use the
    explicit objects/timeline contract rather than guessing geometry here.
    """

    goal = str(data.get("explanation_goal") or data.get("goal") or "Explanation")
    source = str(data.get("source_span") or data.get("source_text") or "")
    raw_steps = data.get("steps") or ["show_source", "show_explanation"]
    steps = [str(step).replace("_", " ").strip() for step in raw_steps][:5]

    objects: list[dict[str, Any]] = [
        {
            "id": "title",
            "type": "text",
            "content": goal,
            "font_size": 34,
            "max_width": 11.5,
            "position": [0, 3.0, 0],
            "color": "#F4F7FB",
        }
    ]
    timeline: list[dict[str, Any]] = [
        {"action": "write", "target": "title", "duration": 0.8}
    ]

    if source:
        objects.append(
            {
                "id": "source",
                "type": "text",
                "content": source,
                "font_size": 24,
                "max_width": 11.5,
                "position": [0, 1.75, 0],
                "color": "#A9B4C4",
            }
        )
        timeline.append({"action": "fade_in", "target": "source", "duration": 0.6})

    start_y = 0.7
    for index, step in enumerate(steps):
        object_id = f"step_{index + 1}"
        objects.append(
            {
                "id": object_id,
                "type": "text",
                "content": f"{index + 1}. {step}",
                "font_size": 26,
                "max_width": 10.5,
                "position": [-0.2, start_y - index * 0.75, 0],
                "color": "#DCE6F2",
            }
        )
        timeline.append({"action": "fade_in", "target": object_id, "duration": 0.5})
        timeline.append({"action": "highlight", "target": object_id, "duration": 0.35})

    return {
        "id": data.get("id") or "legacy_explanation",
        "source_text": source,
        "explanation_goal": goal,
        "objects": objects,
        "timeline": timeline,
        "output": data.get("output", {}),
    }


def normalize_input(data: dict[str, Any]) -> dict[str, Any]:
    """Accept direct DSL, an ``animation`` envelope, or legacy ExplanationSpec."""

    if not isinstance(data, dict):
        raise DSLValidationError(["top-level JSON value must be an object"])

    if isinstance(data.get("animation"), dict):
        merged = dict(data["animation"])
        for key in ("id", "source_text", "source_span", "explanation_goal", "goal", "output"):
            if key in data and key not in merged:
                merged[key] = data[key]
        data = merged

    objects = data.get("objects")
    if objects and all(isinstance(item, dict) for item in objects) and "timeline" in data:
        return data

    if "steps" in data or "goal" in data or "explanation_goal" in data:
        return _legacy_to_dsl(data)

    return data


def _is_position(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) in {2, 3}
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    )


def _validate_output(raw: Any, issues: list[str]) -> OutputConfig:
    raw = raw if isinstance(raw, dict) else {}
    raw_formats = raw.get("formats", ["mp4", "gif"])
    if not isinstance(raw_formats, list):
        issues.append("output.formats must be an array")
        raw_formats = ["mp4", "gif"]
    config = OutputConfig(
        width=raw.get("width", 768),
        height=raw.get("height", 432),
        fps=raw.get("fps", 30),
        gif_fps=raw.get("gif_fps", 15),
        max_gif_width=raw.get("max_gif_width", raw.get("width", 768)),
        formats=list(raw_formats),
    )
    for name, value, lower, upper in (
        ("width", config.width, 160, 3840),
        ("height", config.height, 90, 2160),
        ("fps", config.fps, 1, 60),
        ("gif_fps", config.gif_fps, 1, 30),
        ("max_gif_width", config.max_gif_width, 160, 1920),
    ):
        if not isinstance(value, int) or not lower <= value <= upper:
            issues.append(f"output.{name} must be an integer between {lower} and {upper}")

    allowed_formats = {"gif", "mp4"}
    if not config.formats or not set(config.formats).issubset(allowed_formats):
        issues.append("output.formats may only contain 'mp4' and 'gif'")
    return config


def _parse_objects(raw_objects: Any, issues: list[str]) -> list[ObjectSpec]:
    if not isinstance(raw_objects, list) or not raw_objects:
        issues.append("objects must be a non-empty array")
        return []

    objects: list[ObjectSpec] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_objects):
        path = f"objects[{index}]"
        if not isinstance(raw, dict):
            issues.append(f"{path} must be an object")
            continue
        object_id = raw.get("id")
        object_type = raw.get("type")
        if not isinstance(object_id, str) or not ID_PATTERN.fullmatch(object_id):
            issues.append(f"{path}.id must be a valid identifier")
            continue
        if object_id in ids:
            issues.append(f"duplicate object id: {object_id}")
            continue
        ids.add(object_id)
        if object_type not in SUPPORTED_OBJECTS:
            issues.append(f"{path}.type is unsupported: {object_type!r}")
            continue

        properties = {key: value for key, value in raw.items() if key not in {"id", "type"}}
        if "position" in properties and not _is_position(properties["position"]):
            issues.append(f"{path}.position must contain two or three finite numbers")
        if object_type in {"text", "formula"} and not isinstance(properties.get("content"), str):
            issues.append(f"{path}.content must be a string")
        if object_type == "formula":
            render_mode = properties.get("render_mode", "latex")
            if render_mode not in {"latex", "text"}:
                issues.append(f"{path}.render_mode must be 'latex' or 'text'")
            isolate = properties.get("isolate", [])
            if not isinstance(isolate, list) or not all(isinstance(part, str) for part in isolate):
                issues.append(f"{path}.isolate must be an array of strings")
        if object_type == "graph" and properties.get("function") not in FUNCTIONS:
            issues.append(f"{path}.function must be one of {sorted(FUNCTIONS)}")
        if object_type in {"arrow", "line"}:
            for key in ("from", "to"):
                endpoint = properties.get(key)
                if not isinstance(endpoint, str) and not _is_position(endpoint):
                    issues.append(f"{path}.{key} must be an object id or coordinate vector")
        if object_type == "graph" and not isinstance(properties.get("axes"), str):
            issues.append(f"{path}.axes must reference an axes object")
        if object_type == "image" and not isinstance(properties.get("path"), str):
            issues.append(f"{path}.path must be a string")
        objects.append(ObjectSpec(object_id, object_type, properties))
    return objects


def _parse_action(raw: Any, path: str, issues: list[str]) -> ActionSpec | None:
    if not isinstance(raw, dict):
        issues.append(f"{path} must be an object")
        return None

    if "parallel" in raw:
        children = raw["parallel"]
        if not isinstance(children, list) or not children:
            issues.append(f"{path}.parallel must be a non-empty array")
            return None
        parsed = [
            child
            for index, item in enumerate(children)
            if (child := _parse_action(item, f"{path}.parallel[{index}]", issues)) is not None
        ]
        duration = raw.get("duration", 1.0)
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0 or duration > 30:
            issues.append(f"{path}.duration must be a finite number in (0, 30]")
            duration = 1.0
        if any(child.action == "wait" for child in parsed):
            issues.append(f"{path}.parallel cannot contain a wait action")
        return ActionSpec(duration=float(duration), parallel=parsed)

    action = raw.get("action")
    if action not in SUPPORTED_ACTIONS:
        issues.append(f"{path}.action is unsupported: {action!r}")
        return None
    duration = raw.get("duration", 1.0)
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0 or duration > 30:
        issues.append(f"{path}.duration must be a finite number in (0, 30]")
        duration = 1.0
    target = raw.get("target")
    if action != "wait" and not isinstance(target, str):
        issues.append(f"{path}.target is required")
    properties = {key: value for key, value in raw.items() if key not in {"action", "target", "duration"}}
    required_properties = {
        "change_color": ("color",),
        "follow_path": ("path",),
        "formula_transform": ("replacement",),
        "highlight_parts": ("parts",),
        "move": ("to",),
        "move_by": ("vector",),
        "scale": ("factor",),
        "transform": ("replacement",),
    }
    for key in required_properties.get(action, ()):
        if key not in properties:
            issues.append(f"{path}.{key} is required for action {action!r}")
    return ActionSpec(action, target, float(duration), properties)


def _validate_references(objects: list[ObjectSpec], timeline: list[ActionSpec], issues: list[str]) -> None:
    ids = {item.id for item in objects}
    by_id = {item.id: item for item in objects}

    for item in objects:
        for key in ("axes", "from", "to"):
            reference = item.properties.get(key)
            if item.type in DEPENDENT_OBJECTS and isinstance(reference, str) and reference not in ids:
                issues.append(f"object {item.id!r} references unknown object {reference!r}")
        if item.type == "group":
            members = item.properties.get("members")
            if not isinstance(members, list) or not members:
                issues.append(f"group {item.id!r} requires a non-empty members array")
            elif any(member not in ids for member in members):
                issues.append(f"group {item.id!r} contains an unknown member")
        if item.type == "graph" and by_id.get(item.properties.get("axes"), ObjectSpec("", "", {})).type != "axes":
            issues.append(f"graph {item.id!r} requires an axes object")
        relative_to = item.properties.get("relative_to")
        placement = item.properties.get("placement")
        if relative_to:
            if relative_to not in ids:
                issues.append(f"object {item.id!r} has unknown relative_to object {relative_to!r}")
            if placement not in {"above", "below", "left_of", "right_of"}:
                issues.append(f"object {item.id!r} has invalid relative placement {placement!r}")
            if item.type in DEPENDENT_OBJECTS or by_id.get(relative_to, ObjectSpec("", "", {})).type in DEPENDENT_OBJECTS:
                issues.append("relative layout may only connect independent objects")
        if item.type == "group" and item.properties.get("arrange"):
            member_types = {by_id.get(member, ObjectSpec("", "", {})).type for member in item.properties.get("members", [])}
            if member_types & {"arrow", "line"}:
                issues.append(f"arranged group {item.id!r} cannot contain arrows or lines")

    def check_action(action: ActionSpec, path: str) -> None:
        if action.parallel:
            for index, child in enumerate(action.parallel):
                check_action(child, f"{path}.parallel[{index}]")
            return
        if action.target and action.target not in ids:
            issues.append(f"{path} references unknown target {action.target!r}")
        for key in ("to", "path", "replacement"):
            value = action.properties.get(key)
            if isinstance(value, str) and value not in ids:
                issues.append(f"{path}.{key} references unknown object {value!r}")
        if action.action in {"move", "move_by"}:
            value = action.properties.get("to" if action.action == "move" else "vector")
            if not isinstance(value, str) and not _is_position(value):
                issues.append(f"{path} requires a destination object or coordinate vector")
        if action.action == "scale" and not isinstance(action.properties.get("factor"), (int, float)):
            issues.append(f"{path}.factor must be numeric")
        if action.action == "change_color" and not isinstance(action.properties.get("color"), str):
            issues.append(f"{path}.color must be a string")
        if action.action == "formula_transform":
            replacement = by_id.get(action.properties.get("replacement"))
            target = by_id.get(action.target)
            if target and target.type != "formula":
                issues.append(f"{path}.target must be a formula object")
            if replacement and replacement.type != "formula":
                issues.append(f"{path}.replacement must be a formula object")
            key_map = action.properties.get("key_map", {})
            if not isinstance(key_map, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in key_map.items()
            ):
                issues.append(f"{path}.key_map must map strings to strings")
        if action.action == "highlight_parts":
            target = by_id.get(action.target)
            parts = action.properties.get("parts")
            if target and target.type != "formula":
                issues.append(f"{path}.target must be a formula object")
            if not isinstance(parts, list) or not parts or not all(isinstance(part, str) for part in parts):
                issues.append(f"{path}.parts must be a non-empty array of strings")

    for index, action in enumerate(timeline):
        check_action(action, f"timeline[{index}]")


def parse_animation_spec(data: dict[str, Any]) -> AnimationSpec:
    data = normalize_input(data)
    issues: list[str] = []

    animation_id = _identifier(data.get("id"))
    objects = _parse_objects(data.get("objects"), issues)
    raw_timeline = data.get("timeline")
    if not isinstance(raw_timeline, list) or not raw_timeline:
        issues.append("timeline must be a non-empty array")
        timeline: list[ActionSpec] = []
    else:
        timeline = [
            action
            for index, raw in enumerate(raw_timeline)
            if (action := _parse_action(raw, f"timeline[{index}]", issues)) is not None
        ]

    output = _validate_output(data.get("output"), issues)
    _validate_references(objects, timeline, issues)

    if issues:
        raise DSLValidationError(issues)

    return AnimationSpec(
        id=animation_id,
        objects=objects,
        timeline=timeline,
        output=output,
        background_color=str(data.get("background_color", "#111318")),
        source_text=str(data.get("source_text") or data.get("source_span") or ""),
        explanation_goal=str(data.get("explanation_goal") or data.get("goal") or ""),
    )


def load_animation_spec(path: str | Path) -> AnimationSpec:
    input_path = Path(path)
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DSLValidationError([f"input file does not exist: {input_path}"]) from exc
    except json.JSONDecodeError as exc:
        raise DSLValidationError([f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"]) from exc
    return parse_animation_spec(data)


def wrap_text(value: str, width: int = 48) -> str:
    """Wrap human text while preserving explicit line breaks."""

    return "\n".join(
        textwrap.fill(line, width=width, break_long_words=True) if line else ""
        for line in value.splitlines() or [value]
    )
