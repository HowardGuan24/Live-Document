#!/usr/bin/env python3
"""Stage 5 Phase 4 deterministic split teaching-layout renderer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCHEMA_PATH = Path(__file__).with_name("schema.json")
PHASE3_SCHEMA_PATH = Path(__file__).parents[1] / "phase3" / "schema.json"
PHASE1_SCHEMA_PATH = Path(__file__).parents[1] / "phase1" / "schema.json"
RUNTIME_PATH = Path(__file__)
RUNTIME_VERSION = "stage5-phase4-runtime-4"
OUTPUT_SIZE = (880, 600)
MECHANISM_SIZE = OUTPUT_SIZE
ANNOTATION_XY = (12, 12)
ANNOTATION_WIDTH = 390
ANNOTATION_MAX_HEIGHT = 150
ANNOTATION_OVERLAP_LIMIT = 0.03
CAPTION_OVERLAP_LIMIT = 0.02
TITLE_FONT_SIZE = 17
LEGEND_FONT_SIZE = 13
CAPTION_FONT_SIZE = 15
TITLE_FONT_MIN = 17
LEGEND_FONT_MIN = 13
CAPTION_FONT_MIN = 14
TITLE_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
BODY_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
PALETTE = (
    (46, 134, 193), (230, 126, 34), (72, 72, 70),
    (191, 179, 125), (88, 166, 91), (153, 102, 204),
)
STYLE_SPECS: dict[str, dict[str, Any]] = {
    "candidate-light-glass": {
        "annotation_fill": (238, 231, 215, 102), "caption_fill": (238, 231, 215, 102),
        "title": (25, 27, 29), "body": (35, 37, 39), "border": (83, 76, 64),
    },
    "candidate-dark-glass": {
        "annotation_fill": (27, 31, 36, 112), "caption_fill": (27, 31, 36, 112),
        "title": (247, 246, 242), "body": (233, 232, 228), "border": (196, 197, 198),
    },
}


class Phase4Error(RuntimeError):
    """A deterministic Phase 4 contract or rendering failure."""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def repository_root() -> Path:
    for parent in (RUNTIME_PATH.resolve(), *RUNTIME_PATH.resolve().parents):
        if (parent / "modules" / "video_model" / "stage5").is_dir():
            return parent
    raise Phase4Error("repository root could not be resolved")


def artifact_ref(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if relative_to is not None:
        name = resolved.relative_to(relative_to.resolve()).as_posix()
    else:
        try:
            name = resolved.relative_to(repository_root()).as_posix()
        except ValueError:
            name = str(resolved)
    return {"path": name, "sha256": sha256_file(resolved), "size_bytes": resolved.stat().st_size}


def resolve_ref(ref: Mapping[str, Any], *, base: Path | None = None) -> Path:
    value = Path(ref["path"])
    if value.is_absolute():
        return value
    if value.parts and value.parts[0] == "modules":
        return repository_root() / value
    if base is not None:
        return base / value
    return repository_root() / value


def verify_ref(ref: Mapping[str, Any], *, base: Path | None = None, label: str) -> Path:
    path = resolve_ref(ref, base=base)
    if not path.is_file():
        raise Phase4Error(f"{label} is missing: {path}")
    if path.stat().st_size != ref["size_bytes"] or sha256_file(path) != ref["sha256"]:
        raise Phase4Error(f"{label} hash or size mismatch")
    return path


def validate_definition(document: Mapping[str, Any], definition: str) -> None:
    from jsonschema import Draft202012Validator

    schema = load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    contract = {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{definition}"}
    Draft202012Validator(contract).validate(document)


def validate_against_schema(document: Mapping[str, Any], schema_path: Path) -> None:
    from jsonschema import Draft202012Validator

    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)


def check_artifact_binding(path: Path, ref: Mapping[str, Any], label: str) -> None:
    if sha256_file(path) != ref["sha256"] or path.stat().st_size != ref["size_bytes"]:
        raise Phase4Error(f"explicit {label} is not the bound artifact")


def _rect_valid(rect: Sequence[int]) -> bool:
    x0, y0, x1, y1 = rect
    return 0 <= x0 < x1 <= OUTPUT_SIZE[0] and 0 <= y0 < y1 <= OUTPUT_SIZE[1]


def _rect_overlap(a: Sequence[int], b: Sequence[int]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _support_frames(array: np.ndarray) -> list[int]:
    if array.ndim == 1:
        return np.flatnonzero(np.asarray(array) != 0).astype(int).tolist()
    axes = tuple(range(1, array.ndim))
    return np.flatnonzero(np.any(np.asarray(array) != 0, axis=axes)).astype(int).tolist()


def validate_presentation_semantics(
    presentation: Mapping[str, Any], contract: Mapping[str, Any], manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> tuple[list[Mapping[str, Any]], dict[int, Mapping[str, Any]], dict[str, list[int]]]:
    validate_definition(presentation, "presentation")
    if presentation["layout_preset"] != "split_annotation_caption_v2":
        raise Phase4Error("only the split annotation/caption product layout is accepted")
    field_descriptors = {item["name"]: item for item in manifest["semantic_fields"]}
    fields = [item["semantic_field"] for item in presentation["legend"]]
    roles = [item["visual_role_id"] for item in presentation["legend"]]
    labels = [item["label"] for item in presentation["legend"]]
    if len(fields) != len(set(fields)) or len(roles) != len(set(roles)) or len(labels) != len(set(labels)):
        raise Phase4Error("semantic fields, visual roles, and learner labels must each be unique")
    unknown = sorted(set(fields) - set(field_descriptors))
    if unknown:
        raise Phase4Error(f"unknown semantic field in legend: {unknown}")
    stages = presentation["stages"]
    if len(stages) != len(contract["stages"]):
        raise Phase4Error("presentation stage count differs from Phase 1 semantic contract")
    if [item["semantic_stage_index"] for item in stages] != list(range(len(stages))):
        raise Phase4Error("semantic stage indices must be consecutive and ordered")
    frame_count = manifest["timeline"]["frame_count"]
    expected_start = 0
    stage_by_frame: dict[int, Mapping[str, Any]] = {}
    for stage in stages:
        if stage["start_frame"] != expected_start or stage["end_frame"] < stage["start_frame"]:
            raise Phase4Error("stage ranges must be ordered, contiguous, and nonempty")
        for index in range(stage["start_frame"], stage["end_frame"] + 1):
            stage_by_frame[index] = stage
        expected_start = stage["end_frame"] + 1
    if expected_start != frame_count:
        raise Phase4Error("stage ranges do not cover the complete Phase 3 timeline")
    support = {field: _support_frames(np.asarray(arrays[field])) for field in fields}
    for item in presentation["legend"]:
        field = item["semantic_field"]
        if not support[field]:
            raise Phase4Error(f"legend role is never visible in Phase 3 state: {field}")
        visible = set(item["visible_stage_indices"])
        causal = set(item["causally_active_stage_indices"])
        if not causal.issubset(visible):
            raise Phase4Error(f"causally active stages must be visible stages: {item['visual_role_id']}")
        actual = {
            stage["semantic_stage_index"]
            for stage in stages
            if any(stage["start_frame"] <= frame <= stage["end_frame"] for frame in support[field])
        }
        if visible != actual:
            raise Phase4Error(f"visual role visibility is incompatible with saved state stages: {item['visual_role_id']}")
    for stage in stages:
        stage_index = stage["semantic_stage_index"]
        causal_spatial = [
            item for item in presentation["legend"]
            if stage_index in item["causally_active_stage_indices"] and len(field_descriptors[item["semantic_field"]]["state_shape"]) == 2
        ]
        if not causal_spatial:
            raise Phase4Error(f"stage {stage_index} has no spatial causal role for overlap protection")
    return stages, stage_by_frame, support


def _load_prototype(path: Path):
    spec = importlib.util.spec_from_file_location(f"phase4_source_{sha256_file(path)[:12]}", path)
    if spec is None or spec.loader is None:
        raise Phase4Error("frozen prototype import could not be constructed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame_state(index: int, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for descriptor in manifest["semantic_fields"]:
        value = np.asarray(arrays[descriptor["name"]])[index]
        state[descriptor["name"]] = value.item() if value.ndim == 0 else value.copy()
    for descriptor in manifest.get("runtime_fields", []):
        source_key = descriptor.get("source_state_key")
        if source_key:
            value = np.asarray(arrays[descriptor["name"]])[index]
            state[source_key] = value.item() if value.ndim == 0 else value.copy()
    return state


def _generic_semantic_frame(
    index: int, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray], role_colors: Mapping[str, tuple[int, int, int]]
) -> Image.Image:
    spatial = [item for item in manifest["semantic_fields"] if len(item["state_shape"]) == 2]
    height, width = spatial[0]["state_shape"] if spatial else (120, 176)
    canvas = np.full((height, width, 3), (239, 241, 243), dtype=np.uint8)
    for order, descriptor in enumerate(spatial):
        field = descriptor["name"]
        mask = np.asarray(arrays[field][index]) != 0
        color = role_colors.get(field, (190 - (order * 17) % 70,) * 3)
        canvas[mask] = color
    image = Image.fromarray(canvas).resize(MECHANISM_SIZE, Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    scalar_fields = [item for item in manifest["semantic_fields"] if not item["state_shape"] and item["name"] in role_colors]
    for row, descriptor in enumerate(scalar_fields[:4]):
        values = np.asarray(arrays[descriptor["name"]], dtype=float)
        low, high, value = float(np.min(values)), float(np.max(values)), float(values[index])
        fraction = 0.0 if high == low else (value - low) / (high - low)
        x0, y0, x1 = 24, 548 + row * 12, 320
        draw.rectangle((x0, y0, x1, y0 + 6), fill=(210, 213, 216))
        draw.rectangle((x0, y0, x0 + round((x1 - x0) * fraction), y0 + 6), fill=role_colors[descriptor["name"]])
    return image


def _probe_semantic_frame(module: Any, state: Mapping[str, Any], temp_dir: Path, index: int) -> Image.Image:
    path = temp_dir / f"probe-{index:06d}.png"
    module.render_semantic_probe([dict(state)], str(path))
    with Image.open(path) as source:
        source.load()
        image = source.convert("RGB").resize(MECHANISM_SIZE, Image.Resampling.NEAREST)
    path.unlink()
    return image


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            current = candidate
        else:
            if not current:
                raise Phase4Error(f"text token does not fit teaching region: {word}")
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_lines(
    draw: ImageDraw.ImageDraw, lines: Sequence[str], box: Sequence[int], font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int], line_height: int,
) -> None:
    if len(lines) * line_height > box[3] - box[1]:
        raise Phase4Error("teaching copy does not fit its measured region")
    for row, line in enumerate(lines):
        draw.text((box[0], box[1] + row * line_height), line, font=font, fill=fill)


def _resize_mask(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, "L").resize(OUTPUT_SIZE, Image.Resampling.NEAREST)
    return np.asarray(image) != 0


def critical_semantic_mask(
    presentation: Mapping[str, Any], stage: Mapping[str, Any], arrays: Mapping[str, np.ndarray], frame_index: int
) -> tuple[np.ndarray, list[str]]:
    stage_index = stage["semantic_stage_index"]
    fields, masks = [], []
    for item in presentation["legend"]:
        field = item["semantic_field"]
        if stage_index not in item["causally_active_stage_indices"]:
            continue
        value = np.asarray(arrays[field][frame_index])
        if value.ndim != 2:
            continue
        fields.append(field)
        masks.append(_resize_mask(value != 0))
    if not masks:
        raise Phase4Error(f"frame {frame_index} has no active spatial critical evidence")
    union = np.logical_or.reduce(masks)
    if not np.any(union):
        raise Phase4Error(f"frame {frame_index} critical semantic evidence is empty")
    return union, fields


def _layout_geometry(presentation: Mapping[str, Any], stage: Mapping[str, Any], *, compact: bool) -> tuple[dict[str, Any], list[Mapping[str, Any]], dict[str, Any]]:
    scratch = Image.new("RGB", OUTPUT_SIZE)
    draw = ImageDraw.Draw(scratch)
    title_font = ImageFont.truetype(str(TITLE_FONT), TITLE_FONT_SIZE)
    legend_font = ImageFont.truetype(str(BODY_FONT), LEGEND_FONT_SIZE)
    caption_font = ImageFont.truetype(str(BODY_FONT), CAPTION_FONT_SIZE)
    if min(TITLE_FONT_SIZE, LEGEND_FONT_SIZE, CAPTION_FONT_SIZE) <= 0 or TITLE_FONT_SIZE < TITLE_FONT_MIN or LEGEND_FONT_SIZE < LEGEND_FONT_MIN or CAPTION_FONT_SIZE < CAPTION_FONT_MIN:
        raise Phase4Error("readable minimum teaching font size was violated")
    current_stage = stage["semantic_stage_index"]
    visible = [item for item in presentation["legend"] if current_stage in item["visible_stage_indices"]]
    if not visible or len(visible) > 4:
        raise Phase4Error("top-left annotation requires one through four active legend rows")
    padding = 8 if compact else 10
    legend_row_height = 18 if compact else 20
    title_line_height, caption_line_height = 21, 18
    x0, y0 = ANNOTATION_XY
    x1 = x0 + ANNOTATION_WIDTH
    inner_left, inner_right = x0 + padding, x1 - padding
    title_lines = _wrap(draw, stage["title"], title_font, inner_right - inner_left)
    if len(title_lines) > 2:
        raise Phase4Error("stage title exceeds the two-line maximum")
    title_region = [inner_left, y0 + padding, inner_right, y0 + padding + len(title_lines) * title_line_height]
    legend_top = title_region[3] + (3 if compact else 4)
    legend_region = [inner_left, legend_top, inner_right, legend_top + len(visible) * legend_row_height]
    annotation_region = [x0, y0, x1, legend_region[3] + padding]
    caption_lines = _wrap(draw, stage["caption"], caption_font, 700)
    if len(caption_lines) > 2:
        raise Phase4Error("bottom caption exceeds the two-line maximum")
    caption_padding_x = 8 if compact else 12
    caption_padding_y = 6 if compact else 8
    text_width = max(draw.textbbox((0, 0), line, font=caption_font)[2] for line in caption_lines)
    caption_width = text_width + caption_padding_x * 2
    caption_height = len(caption_lines) * caption_line_height + caption_padding_y * 2
    caption_x0 = (OUTPUT_SIZE[0] - caption_width) // 2
    caption_y1 = 588
    caption_region = [caption_x0, caption_y1 - caption_height, caption_x0 + caption_width, caption_y1]
    caption_text_region = [
        caption_region[0] + caption_padding_x, caption_region[1] + caption_padding_y,
        caption_region[2] - caption_padding_x, caption_region[3] - caption_padding_y,
    ]
    metrics = {
        "semantic_stage_index": current_stage,
        "annotation_region": annotation_region,
        "title_region": title_region,
        "legend_region": legend_region,
        "caption_region": caption_region,
        "visible_visual_role_ids": [item["visual_role_id"] for item in visible],
        "annotation_width": annotation_region[2] - annotation_region[0],
        "annotation_height": annotation_region[3] - annotation_region[1],
        "caption_width": caption_region[2] - caption_region[0],
        "caption_height": caption_region[3] - caption_region[1],
        "title_line_count": len(title_lines), "caption_line_count": len(caption_lines),
        "adaptive_compaction": compact,
    }
    if not 350 <= metrics["annotation_width"] <= 410 or metrics["annotation_height"] > ANNOTATION_MAX_HEIGHT:
        raise Phase4Error(f"annotation does not fit 350-410 by <=150: {metrics['annotation_width']}x{metrics['annotation_height']}")
    if metrics["caption_width"] >= OUTPUT_SIZE[0] or metrics["caption_height"] > 60:
        raise Phase4Error("caption backing is not content-sized within its formal bounds")
    for name, region in (
        ("annotation", annotation_region), ("title", title_region), ("legend", legend_region),
        ("caption", caption_region), ("caption text", caption_text_region),
    ):
        if not _rect_valid(region):
            raise Phase4Error(f"{name} region is out of bounds")
    if _rect_overlap(annotation_region, caption_region):
        raise Phase4Error("annotation and caption regions must be separate")
    text = {
        "title_lines": title_lines, "caption_lines": caption_lines,
        "fonts": (title_font, legend_font, caption_font), "caption_text_region": caption_text_region,
        "legend_row_height": legend_row_height,
    }
    return metrics, visible, text


def _overlap(mask: np.ndarray, annotation: Sequence[int], caption: Sequence[int]) -> dict[str, Any]:
    total = int(np.count_nonzero(mask))
    annotation_count = int(np.count_nonzero(mask[annotation[1]:annotation[3], annotation[0]:annotation[2]]))
    caption_count = int(np.count_nonzero(mask[caption[1]:caption[3], caption[0]:caption[2]]))
    return {
        "critical_pixel_count": total,
        "annotation_critical_pixels": annotation_count,
        "caption_critical_pixels": caption_count,
        "annotation_coverage_ratio": annotation_count / total,
        "caption_coverage_ratio": caption_count / total,
    }


def split_layout(
    presentation: Mapping[str, Any], stage: Mapping[str, Any], critical_mask: np.ndarray, frame_index: int
) -> tuple[dict[str, Any], list[Mapping[str, Any]], dict[str, Any], dict[str, Any], int]:
    metrics, visible, text = _layout_geometry(presentation, stage, compact=False)
    overlap = _overlap(critical_mask, metrics["annotation_region"], metrics["caption_region"])
    alpha_reduction = 0
    if overlap["annotation_coverage_ratio"] > ANNOTATION_OVERLAP_LIMIT or overlap["caption_coverage_ratio"] > CAPTION_OVERLAP_LIMIT:
        metrics, visible, text = _layout_geometry(presentation, stage, compact=True)
        overlap = _overlap(critical_mask, metrics["annotation_region"], metrics["caption_region"])
        alpha_reduction = 12
    if overlap["annotation_coverage_ratio"] > ANNOTATION_OVERLAP_LIMIT or overlap["caption_coverage_ratio"] > CAPTION_OVERLAP_LIMIT:
        raise Phase4Error(
            f"frame {frame_index} critical overlap exceeds thresholds after compaction: "
            f"annotation={overlap['annotation_coverage_ratio']:.6f}, caption={overlap['caption_coverage_ratio']:.6f}"
        )
    return metrics, visible, text, overlap, alpha_reduction


def compose_frame(
    mechanism: Image.Image, presentation: Mapping[str, Any], stage: Mapping[str, Any], colors: Mapping[str, tuple[int, int, int]],
    critical_mask: np.ndarray, style_id: str, frame_index: int,
) -> tuple[Image.Image, list[str], dict[str, Any], dict[str, Any]]:
    if style_id not in STYLE_SPECS:
        raise Phase4Error(f"unknown review style: {style_id}")
    metrics, visible, text, overlap, alpha_reduction = split_layout(presentation, stage, critical_mask, frame_index)
    style = STYLE_SPECS[style_id]
    annotation_fill = (*style["annotation_fill"][:3], style["annotation_fill"][3] - alpha_reduction)
    caption_fill = (*style["caption_fill"][:3], style["caption_fill"][3] - alpha_reduction)
    canvas = mechanism.convert("RGBA")
    overlay = Image.new("RGBA", OUTPUT_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(metrics["annotation_region"], radius=3, fill=annotation_fill, outline=(*style["border"], 92), width=1)
    draw.rounded_rectangle(metrics["caption_region"], radius=3, fill=caption_fill, outline=(*style["border"], 72), width=1)
    title_font, legend_font, caption_font = text["fonts"]
    _draw_lines(draw, text["title_lines"], metrics["title_region"], title_font, (*style["title"], 255), 21)
    for row, item in enumerate(visible):
        y = metrics["legend_region"][1] + row * text["legend_row_height"]
        draw.rectangle((metrics["legend_region"][0], y + 3, metrics["legend_region"][0] + 12, y + 15), fill=(*colors[item["semantic_field"]], 255))
        draw.text((metrics["legend_region"][0] + 18, y), item["label"], font=legend_font, fill=(*style["body"], 255))
    _draw_lines(draw, text["caption_lines"], text["caption_text_region"], caption_font, (*style["body"], 255), 18)
    composed = Image.alpha_composite(canvas, overlay).convert("RGB")
    return composed, metrics["visible_visual_role_ids"], metrics, overlap


def _derive_role_colors(
    presentation: Mapping[str, Any], manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
    mechanisms: Sequence[Image.Image], support: Mapping[str, list[int]],
) -> tuple[dict[str, tuple[int, int, int]], list[dict[str, Any]]]:
    colors, evidence = {}, []
    descriptors = {item["name"]: item for item in manifest["semantic_fields"]}
    for order, item in enumerate(presentation["legend"]):
        field, frame = item["semantic_field"], support[item["semantic_field"]][0]
        descriptor = descriptors[field]
        if len(descriptor["state_shape"]) == 2:
            mask = Image.fromarray((np.asarray(arrays[field][frame]) != 0).astype(np.uint8) * 255).resize(MECHANISM_SIZE, Image.Resampling.NEAREST)
            pixels = np.asarray(mechanisms[frame])[np.asarray(mask) != 0]
            selected = Counter(map(tuple, pixels.tolist())).most_common(1)[0][0] if len(pixels) else PALETTE[order]
        else:
            selected = PALETTE[order]
        color = tuple(int(value) for value in selected)
        colors[field] = color
        evidence.append({**item, "state_shape": descriptor["state_shape"], "support_frames": support[field], "selected_rgb": list(color), "source_frame_index": frame})
    return colors, evidence


def _render_pass(
    presentation: Mapping[str, Any], manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray], module: Any,
    adapter: str, temp_dir: Path, style_id: str, colors: Mapping[str, tuple[int, int, int]] | None = None,
) -> tuple[list[Image.Image], list[Image.Image], list[list[str]], list[dict[str, Any]], list[dict[str, Any]], str]:
    role_colors = colors or {item["semantic_field"]: PALETTE[index] for index, item in enumerate(presentation["legend"])}
    stage_by_frame = {
        index: stage for stage in presentation["stages"]
        for index in range(stage["start_frame"], stage["end_frame"] + 1)
    }
    mechanisms, teaching, visible_roles, layouts, overlaps = [], [], [], [], []
    digest = hashlib.sha256()
    for index in range(manifest["timeline"]["frame_count"]):
        state = _frame_state(index, manifest, arrays)
        mechanism = _probe_semantic_frame(module, state, temp_dir, index) if adapter == "frozen_public_semantic_probe_v1" else _generic_semantic_frame(index, manifest, arrays, role_colors)
        critical_mask, _ = critical_semantic_mask(presentation, stage_by_frame[index], arrays, index)
        composed, visible, layout, overlap = compose_frame(mechanism, presentation, stage_by_frame[index], role_colors, critical_mask, style_id, index)
        digest.update(index.to_bytes(8, "big")); digest.update(composed.tobytes())
        mechanisms.append(mechanism); teaching.append(composed); visible_roles.append(visible); layouts.append(layout); overlaps.append(overlap)
    return mechanisms, teaching, visible_roles, layouts, overlaps, digest.hexdigest()


def style_parameters(style_id: str) -> dict[str, Any]:
    style = STYLE_SPECS[style_id]
    return {
        "style_id": style_id,
        "backing_alpha": style["annotation_fill"][3] / 255,
        "caption_backing_alpha": style["caption_fill"][3] / 255,
        "annotation_fill_rgba": list(style["annotation_fill"]), "caption_fill_rgba": list(style["caption_fill"]),
        "title_rgb": list(style["title"]), "body_rgb": list(style["body"]), "border_rgb": list(style["border"]),
        "corner_radius": 3, "blur_used": False,
    }


def write_failure(output_dir: Path, error: Exception) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "schema_version": "stage5-phase4-failure-1", "phase": "phase4",
        "failure_class": error.__class__.__name__, "message": str(error),
        "return_target": "phase4_runtime" if isinstance(error, Phase4Error) else "implementation",
    }
    path = output_dir / "failure.json"
    if not path.exists():
        path.write_bytes(canonical_bytes(failure))


def render_teaching_sequence(
    *, semantic_contract: Path, sequence_archive: Path, sequence_manifest: Path, presentation_path: Path,
    style_id: str, output_directory: Path,
) -> None:
    if output_directory.exists():
        raise Phase4Error("output already exists and will not be reused")
    output_directory.mkdir(parents=True)
    try:
        contract, manifest, presentation = load_json(semantic_contract), load_json(sequence_manifest), load_json(presentation_path)
        validate_against_schema(contract, PHASE1_SCHEMA_PATH); validate_against_schema(manifest, PHASE3_SCHEMA_PATH)
        verify_ref(manifest["source"]["semantic_contract"], label="Phase 3 semantic contract")
        check_artifact_binding(semantic_contract, manifest["source"]["semantic_contract"], "semantic contract")
        check_artifact_binding(sequence_archive, manifest["sequence_archive"], "sequence archive")
        check_artifact_binding(sequence_manifest, presentation["sequence_manifest"], "sequence manifest")
        check_artifact_binding(semantic_contract, presentation["semantic_contract"], "presentation semantic contract")
        if style_id not in presentation["review_style_ids"]:
            raise Phase4Error("requested style is absent from the presentation review styles")
        with np.load(sequence_archive, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        expected_names = {item["name"] for item in manifest["semantic_fields"]} | {item["name"] for item in manifest.get("runtime_fields", [])}
        if set(arrays) != expected_names:
            raise Phase4Error("Phase 3 archive keys differ from its manifest")
        for descriptor in [*manifest["semantic_fields"], *manifest.get("runtime_fields", [])]:
            value = arrays[descriptor["name"]]
            if list(value.shape) != descriptor["archive_shape"] or value.dtype.str != descriptor["dtype"] or array_digest(value) != descriptor["content_sha256"]:
                raise Phase4Error(f"Phase 3 archive descriptor mismatch: {descriptor['name']}")
        stages, _, support = validate_presentation_semantics(presentation, contract, manifest, arrays)
        prototype_path = verify_ref(manifest["source"]["prototype"], label="frozen prototype")
        module = _load_prototype(prototype_path)
        adapter = "frozen_public_semantic_probe_v1" if callable(getattr(module, "render_semantic_probe", None)) else "generic_semantic_field_v1"
        with tempfile.TemporaryDirectory(prefix="phase4-layout-v2-") as temp_name:
            temp_dir = Path(temp_name)
            mechanisms0, _, _, _, _, _ = _render_pass(presentation, manifest, arrays, module, adapter, temp_dir, style_id)
            colors, role_evidence = _derive_role_colors(presentation, manifest, arrays, mechanisms0, support)
            mechanisms, frames, visible_roles, frame_layouts, overlaps, first_digest = _render_pass(presentation, manifest, arrays, module, adapter, temp_dir, style_id, colors)
            _, replay_frames, _, replay_layouts, replay_overlaps, second_digest = _render_pass(presentation, manifest, arrays, module, adapter, temp_dir, style_id, colors)
        if first_digest != second_digest or any(a.tobytes() != b.tobytes() for a, b in zip(frames, replay_frames)):
            raise Phase4Error("deterministic teaching replay mismatch")
        if frame_layouts != replay_layouts or overlaps != replay_overlaps:
            raise Phase4Error("deterministic split-layout metrics replay mismatch")
        frames_dir = output_directory / "frames"; frames_dir.mkdir()
        frame_records = []
        for index, (mechanism, frame, roles, layout, overlap) in enumerate(zip(mechanisms, frames, visible_roles, frame_layouts, overlaps)):
            frame_path = frames_dir / f"frame-{index:06d}.png"
            buffer = io.BytesIO(); frame.save(buffer, format="PNG", optimize=False, compress_level=9); frame_path.write_bytes(buffer.getvalue())
            annotation, caption = layout["annotation_region"], layout["caption_region"]
            annotation_pixels = np.ascontiguousarray(np.asarray(frame)[annotation[1]:annotation[3], annotation[0]:annotation[2]])
            caption_pixels = np.ascontiguousarray(np.asarray(frame)[caption[1]:caption[3], caption[0]:caption[2]])
            frame_records.append({
                "frame_index": index, "artifact": artifact_ref(frame_path, relative_to=output_directory),
                "stage_index": next(stage["semantic_stage_index"] for stage in stages if stage["start_frame"] <= index <= stage["end_frame"]),
                "visible_visual_role_ids": roles, "mechanism_rgb_sha256": sha256_bytes(mechanism.tobytes()),
                "annotation_region": annotation, "caption_region": caption,
                "annotation_rgb_sha256": sha256_bytes(annotation_pixels.tobytes()),
                "caption_rgb_sha256": sha256_bytes(caption_pixels.tobytes()),
                "critical_overlap": overlap, "width": 880, "height": 600, "mode": "RGB",
            })
        stage_layouts = []
        for stage in stages:
            frame_index = stage["start_frame"]
            stage_layouts.append(frame_layouts[frame_index])
        max_annotation = max(item["annotation_coverage_ratio"] for item in overlaps)
        max_caption = max(item["caption_coverage_ratio"] for item in overlaps)
        checks = [
            {"check_id": "EXPLICIT_INPUT_BINDINGS", "passed": True, "evidence": "semantic contract, sequence manifest, archive, and presentation bindings matched"},
            {"check_id": "SAVED_SEQUENCE_ONLY", "passed": True, "evidence": "no state evaluator or validator was called"},
            {"check_id": "ROLE_BINDINGS_UNCHANGED", "passed": True, "evidence": f"roles={len(role_evidence)}; exact semantic field and visual role identities retained"},
            {"check_id": "SPLIT_LAYOUT", "passed": True, "evidence": "title/legend annotation and caption occupy disjoint top-left and bottom regions"},
            {"check_id": "CRITICAL_OVERLAP", "passed": True, "evidence": f"max annotation={max_annotation:.6f}; max caption={max_caption:.6f}"},
            {"check_id": "NO_OPAQUE_WHITE_PANEL", "passed": True, "evidence": f"style={style_id}; backing alpha={STYLE_SPECS[style_id]['annotation_fill'][3] / 255:.6f}"},
            {"check_id": "DETERMINISTIC_REPLAY", "passed": True, "evidence": first_digest},
            {"check_id": "PHASE_BOUNDARY", "passed": True, "evidence": "candidate output contains only frames and teaching-manifest.json"},
        ]
        teaching_manifest = {
            "schema_version": "stage5-phase4-teaching-manifest-4", "artifact_type": "teaching_sequence", "phase": "phase4",
            "semantic_contract": artifact_ref(semantic_contract), "sequence_manifest": artifact_ref(sequence_manifest), "sequence_archive": artifact_ref(sequence_archive),
            "source_presentation": artifact_ref(presentation_path), "presentation": artifact_ref(presentation_path),
            "renderer": {"adapter": adapter, "state_source": "saved_phase3_sequence_only", "prototype": artifact_ref(prototype_path), "runtime_sha256": sha256_file(RUNTIME_PATH)},
            "timeline": {key: manifest["timeline"][key] for key in ("frame_count", "fps", "duration_seconds")} | {"stage_count": len(stages)},
            "layout": {
                "preset_id": "split_annotation_caption_v2", "style_id": style_id, "style_parameters": style_parameters(style_id),
                "output_width": 880, "output_height": 600, "mechanism_viewport": [0, 0, 880, 600],
                "annotation_anchor": "top_left", "annotation_width_range": [350, 410], "annotation_max_height": 150,
                "caption_placement": "bottom_center_content_sized",
                "critical_overlap_thresholds": {"annotation": 0.03, "caption": 0.02},
                "max_observed_annotation_coverage_ratio": max_annotation,
                "max_observed_caption_coverage_ratio": max_caption,
                "stage_layouts": stage_layouts,
            },
            "frames": frame_records, "legend_bindings": role_evidence,
            "layout_checks": {
                "canvas_is_880x600": True, "no_sidecar": True, "no_opaque_white_panel": True,
                "annotation_caption_separate": True, "annotation_bounds": True, "caption_bounds": True,
                "text_fits": True, "readable_minimum_fonts": True, "stage_specific_legend": True,
                "no_empty_legend_rows": True, "copy_matches_presentation": True, "critical_overlap_within_thresholds": True,
            },
            "deterministic_replay": {"algorithm": "sha256-frame-index-rgb-bytes-v1", "first_digest": first_digest, "second_digest": second_digest, "matched": True},
            "checks": checks, "human_review_status": "pending_human_review", "status": "complete",
        }
        validate_definition(teaching_manifest, "teaching_manifest")
        manifest_path = output_directory / "teaching-manifest.json"; manifest_path.write_bytes(canonical_bytes(teaching_manifest))
        validate_teaching_manifest(manifest_path)
    except Exception as error:
        for child in list(output_directory.iterdir()):
            if child.is_dir(): shutil.rmtree(child)
            else: child.unlink()
        write_failure(output_directory, error)
        raise


def validate_teaching_manifest(path: Path) -> None:
    document = load_json(path); validate_definition(document, "teaching_manifest")
    base = path.parent
    for ref in [item["artifact"] for item in document["frames"]]:
        verify_ref(ref, base=base, label=f"teaching artifact {ref['path']}")
    for name in ("source_presentation", "presentation", "semantic_contract", "sequence_manifest", "sequence_archive"):
        verify_ref(document[name], label=name)
    verify_ref(document["renderer"]["prototype"], label="renderer prototype")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-presentation"); validate.add_argument("document", type=Path)
    validate = commands.add_parser("validate-teaching-manifest"); validate.add_argument("document", type=Path)
    render = commands.add_parser("render-teaching")
    render.add_argument("--semantic-contract", type=Path, required=True)
    render.add_argument("--sequence-archive", type=Path, required=True)
    render.add_argument("--sequence-manifest", type=Path, required=True)
    render.add_argument("--presentation", type=Path, required=True)
    render.add_argument("--style-id", choices=tuple(STYLE_SPECS), required=True)
    render.add_argument("--output-dir", "--output-directory", dest="output_directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-presentation":
            validate_definition(load_json(args.document), "presentation"); return 0
        if args.command == "validate-teaching-manifest":
            validate_teaching_manifest(args.document); return 0
        if args.command == "render-teaching":
            render_teaching_sequence(
                semantic_contract=args.semantic_contract, sequence_archive=args.sequence_archive,
                sequence_manifest=args.sequence_manifest, presentation_path=args.presentation,
                style_id=args.style_id, output_directory=args.output_directory,
            ); return 0
    except Exception as error:
        print(f"phase4: {error}", file=sys.stderr); return 1
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
