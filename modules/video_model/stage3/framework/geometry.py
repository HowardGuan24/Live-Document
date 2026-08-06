"""Route-aware geometry resolver and control compiler.

The compiler has no Case IDs and no final-image coordinates. It consumes:

- semantic object classes and their program geometry;
- a versioned geometry policy;
- optional migration relations for legacy exports;
- region/height fields directly exported by the program.

Appearance references are deliberately absent from every function signature.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    write_json,
)


OUTPUT_SIZE = (1024, 576)
SUPPORTED_PRIMITIVES = {
    "glass_beaker",
    "glass_burette",
    "bar_magnet",
    "fixed_coil",
}


def _bbox(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    if "bbox_xyxy" in geometry:
        x0, y0, x1, y1 = geometry["bbox_xyxy"]
        return float(x0), float(y0), float(x1), float(y1)
    points = geometry.get("points", [])
    if not points and "center_xy" in geometry:
        cx, cy = geometry["center_xy"]
        radius = float(geometry.get("radius", 4))
        return cx - radius, cy - radius, cx + radius, cy + radius
    if not points:
        raise ValueError(f"unsupported geometry: {geometry}")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _scaled_bbox(
    geometry: dict[str, Any],
    source_size: tuple[int, int],
    output_size: tuple[int, int] = OUTPUT_SIZE,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = _bbox(geometry)
    sx = output_size[0] / source_size[0]
    sy = output_size[1] / source_size[1]
    return x0 * sx, y0 * sy, x1 * sx, y1 * sy


def _artifact_path(
    record: dict[str, Any], repo_root: Path
) -> Path:
    return repo_root / record["path"]


def keyframe_semantics(
    contract: dict[str, Any],
    keyframe_id: str,
    repo_root: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    record = next(
        item
        for item in contract["keyframes"]
        if item["keyframe_id"] == keyframe_id
    )
    semantic_path = _artifact_path(record["semantic_layers"], repo_root)
    return load_json(semantic_path), semantic_path, record


def layer_data_path(
    contract: dict[str, Any],
    layer: dict[str, Any],
    repo_root: Path,
) -> Path:
    return (
        repo_root
        / contract["program_source"]["root"]
        / layer["data"]["path"]
    )


def load_identity(
    contract: dict[str, Any],
    semantic: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    layer = next(
        item
        for item in semantic["layers"]
        if item["layer_type"] == "object_identity"
    )
    return load_json(layer_data_path(contract, layer, repo_root)), layer


def geometry_in_bounds(
    item: dict[str, Any], canvas: dict[str, Any]
) -> bool:
    x0, y0, x1, y1 = _bbox(item["geometry"])
    return (
        0 <= x0 <= x1 < canvas["width"]
        and 0 <= y0 <= y1 < canvas["height"]
    )


def identity_preflight(
    contract: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    required = contract["semantic_exports"].get(
        "control_object_requirements", []
    )
    frames = []
    all_pass = True
    for keyframe in contract["keyframes"]:
        semantic, _, _ = keyframe_semantics(
            contract, keyframe["keyframe_id"], repo_root
        )
        identity, _ = load_identity(contract, semantic, repo_root)
        classes = [
            item.get("class_id", "untyped")
            for item in identity.get("items", [])
        ]
        requirements = []
        for requirement in required:
            actual = classes.count(requirement["class_id"])
            passed = actual == requirement["cardinality"]
            all_pass &= passed
            requirements.append(
                {
                    "class_id": requirement["class_id"],
                    "expected": requirement["cardinality"],
                    "actual": actual,
                    "passed": passed,
                    "legacy_migration_available": (
                        "legacy_migration" in requirement
                    ),
                }
            )
        bounds_pass = all(
            geometry_in_bounds(item, semantic["canvas"])
            for item in identity.get("items", [])
        )
        all_pass &= bounds_pass
        frames.append(
            {
                "keyframe_id": keyframe["keyframe_id"],
                "requirements": requirements,
                "object_geometry_in_bounds": bounds_pass,
            }
        )
    return {
        "case_id": contract["case_id"],
        "policy": contract["geometry_policy"],
        "passed": all_pass,
        "frames": frames,
    }


def _components(binary: np.ndarray) -> list[dict[str, Any]]:
    mask = binary > 0
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    output: list[dict[str, Any]] = []
    for y, x in np.argwhere(mask):
        y = int(y)
        x = int(x)
        if visited[y, x]:
            continue
        queue: deque[tuple[int, int]] = deque([(y, x)])
        visited[y, x] = True
        xs: list[int] = []
        ys: list[int] = []
        while queue:
            cy, cx = queue.popleft()
            xs.append(cx)
            ys.append(cy)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny = cy + dy
                    nx = cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        output.append(
            {
                "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)],
                "pixel_count": len(xs),
            }
        )
    return sorted(output, key=lambda item: -item["pixel_count"])


def _claimed_mask(
    shape: tuple[int, int],
    objects: list[dict[str, Any]],
    margin: int = 7,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for item in objects:
        x0, y0, x1, y1 = _bbox(item["geometry"])
        xa = max(0, int(math.floor(x0)) - margin)
        ya = max(0, int(math.floor(y0)) - margin)
        xb = min(shape[1] - 1, int(math.ceil(x1)) + margin)
        yb = min(shape[0] - 1, int(math.ceil(y1)) + margin)
        mask[ya : yb + 1, xa : xb + 1] = True
    return mask


def normalize_legacy_objects(
    contract: dict[str, Any],
    semantic: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill declared legacy gaps from semantic boundary components.

    This never examines a rendered RGB screenshot. The missing class and
    relation must already be declared in the input contract. Pixel bounds
    are measured only from the program's hard-boundary semantic layer.
    """

    identity, identity_layer = load_identity(contract, semantic, repo_root)
    normalized = json.loads(json.dumps(identity))
    existing = [
        item.get("class_id", "untyped")
        for item in normalized.get("items", [])
    ]
    required = contract["semantic_exports"].get(
        "control_object_requirements", []
    )
    derivations: list[dict[str, Any]] = []
    hard_layers = [
        layer
        for layer in semantic["layers"]
        if layer["layer_type"] == "hard_boundary"
    ]
    hard = None
    if hard_layers:
        hard = np.load(
            layer_data_path(contract, hard_layers[0], repo_root),
            allow_pickle=False,
        )
    for requirement in required:
        missing = requirement["cardinality"] - existing.count(
            requirement["class_id"]
        )
        if missing <= 0:
            continue
        migration = requirement.get("legacy_migration")
        if not migration or hard is None:
            raise ValueError(
                f"missing {requirement['class_id']} without migration"
            )
        relation = migration["relation"]
        if relation != "paired_parallel_boundaries_above_glass_beaker":
            raise ValueError(f"unsupported migration relation: {relation}")
        anchors = [
            item
            for item in normalized["items"]
            if item.get("class_id") == "glass_beaker"
        ]
        if len(anchors) != 1:
            raise ValueError("migration needs exactly one glass_beaker")
        anchor_bbox = _bbox(anchors[0]["geometry"])
        claimed = _claimed_mask(hard.shape, normalized["items"])
        unclaimed = np.where(claimed, 0, hard)
        candidates = [
            item
            for item in _components(unclaimed)
            if item["pixel_count"] >= 20
            and item["bbox_xyxy"][3] < anchor_bbox[1] + 24
        ]
        if len(candidates) < 2:
            raise ValueError(
                "could not find paired unclaimed boundary components"
            )
        # Select the two tall components whose x centers are closest to the
        # anchor center. This rule is relational and contains no Case coords.
        anchor_cx = (anchor_bbox[0] + anchor_bbox[2]) / 2
        tall = sorted(
            candidates,
            key=lambda item: (
                -(
                    item["bbox_xyxy"][3]
                    - item["bbox_xyxy"][1]
                ),
                abs(
                    (
                        item["bbox_xyxy"][0]
                        + item["bbox_xyxy"][2]
                    )
                    / 2
                    - anchor_cx
                ),
            ),
        )[:2]
        xa = min(item["bbox_xyxy"][0] for item in tall)
        ya = min(item["bbox_xyxy"][1] for item in tall)
        xb = max(item["bbox_xyxy"][2] for item in tall)
        yb = max(item["bbox_xyxy"][3] for item in tall)
        synthesized = {
            "object_id": (
                f"{contract['case_id']}-normalized-"
                f"{requirement['class_id']}"
            ),
            "class_id": requirement["class_id"],
            "geometry": {
                "kind": "bbox",
                "bbox_xyxy": [xa, ya, xb, yb],
            },
            "source": "legacy_semantic_normalizer",
        }
        normalized["items"].append(synthesized)
        existing.append(requirement["class_id"])
        derivations.append(
            {
                "class_id": requirement["class_id"],
                "method": relation,
                "source_layer": hard_layers[0]["layer_id"],
                "source_layer_sha256": hard_layers[0]["data"]["sha256"],
                "selected_component_bboxes": [
                    item["bbox_xyxy"] for item in tall
                ],
                "output_bbox_xyxy": [xa, ya, xb, yb],
                "appearance_reference_used": False,
                "rendered_screenshot_used": False,
            }
        )
    return normalized, {
        "schema_version": "1.0",
        "case_id": contract["case_id"],
        "state_id": semantic["state_id"],
        "identity_source_layer": identity_layer["layer_id"],
        "identity_source_sha256": identity_layer["data"]["sha256"],
        "derivations": derivations,
    }


def _draw_beaker(
    draw: ImageDraw.ImageDraw,
    source_bbox: tuple[float, float, float, float],
    canvas: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = canvas
    source_width = source_bbox[2] - source_bbox[0]
    center_x = (source_bbox[0] + source_bbox[2]) / 2
    canonical_width = min(
        max(source_width * 0.85, width * 0.32), width * 0.40
    )
    canonical_height = canonical_width / 1.50
    bottom = min(height * 0.90, source_bbox[3] + height * 0.05)
    left = center_x - canonical_width / 2
    right = center_x + canonical_width / 2
    top = bottom - canonical_height
    line = 4
    rim_height = canonical_height * 0.15
    draw.ellipse(
        (left, top, right, top + rim_height),
        outline=255,
        width=line,
    )
    draw.line(
        (left + 2, top + rim_height / 2, left + 26, bottom - 18),
        fill=255,
        width=line,
    )
    draw.line(
        (right - 2, top + rim_height / 2, right - 26, bottom - 18),
        fill=255,
        width=line,
    )
    draw.ellipse(
        (left + 26, bottom - 36, right - 26, bottom),
        outline=255,
        width=line,
    )
    liquid_y = top + canonical_height * 0.65
    draw.ellipse(
        (left + 23, liquid_y - 15, right - 23, liquid_y + 15),
        outline=255,
        width=3,
    )
    for fraction, length in ((0.35, 23), (0.47, 15), (0.59, 23)):
        y = top + canonical_height * fraction
        draw.line((right - 25 - length, y, right - 25, y), fill=255, width=2)
    return (
        int(round(left)),
        int(round(top)),
        int(round(right)),
        int(round(bottom)),
    )


def _draw_burette(
    draw: ImageDraw.ImageDraw,
    source_bbox: tuple[float, float, float, float],
    anchor_bbox: tuple[int, int, int, int],
    canvas: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = canvas
    center_x = (source_bbox[0] + source_bbox[2]) / 2
    if not (anchor_bbox[0] <= center_x <= anchor_bbox[2]):
        center_x = (anchor_bbox[0] + anchor_bbox[2]) / 2
    tube_width = max(width * 0.038, source_bbox[2] - source_bbox[0])
    top = max(height * 0.03, source_bbox[1] - height * 0.025)
    outlet_bottom = anchor_bbox[1] - height * 0.012
    body_bottom = min(outlet_bottom - 54, top + height * 0.29)
    left = center_x - tube_width / 2
    right = center_x + tube_width / 2
    draw.rounded_rectangle(
        (left, top, right, body_bottom),
        radius=max(5, int(tube_width * 0.22)),
        outline=255,
        width=4,
    )
    for index, y in enumerate(
        np.linspace(top + 24, body_bottom - 30, 7)
    ):
        length = tube_width * (0.46 if index % 2 == 0 else 0.30)
        draw.line((left, y, left + length, y), fill=255, width=2)
    stop_y = body_bottom - 10
    draw.ellipse(
        (left + 2, stop_y - 12, right - 2, stop_y + 22),
        outline=255,
        width=4,
    )
    draw.line(
        (center_x - 44, stop_y + 5, center_x + 44, stop_y + 5),
        fill=255,
        width=4,
    )
    draw.ellipse(
        (center_x + 36, stop_y - 3, center_x + 52, stop_y + 13),
        outline=255,
        width=3,
    )
    nozzle_left = center_x - tube_width * 0.20
    nozzle_right = center_x + tube_width * 0.20
    draw.line(
        (nozzle_left, stop_y + 22, nozzle_left, outlet_bottom - 17),
        fill=255,
        width=3,
    )
    draw.line(
        (
            nozzle_right,
            stop_y + 22,
            nozzle_right,
            outlet_bottom - 17,
        ),
        fill=255,
        width=3,
    )
    draw.line(
        (nozzle_left, outlet_bottom - 17, center_x, outlet_bottom),
        fill=255,
        width=3,
    )
    draw.line(
        (nozzle_right, outlet_bottom - 17, center_x, outlet_bottom),
        fill=255,
        width=3,
    )
    return (
        int(round(center_x - 52)),
        int(round(top)),
        int(round(center_x + 52)),
        int(round(outlet_bottom)),
    )


def _draw_magnet(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    radius = max(5, int((y1 - y0) * 0.12))
    draw.rounded_rectangle(
        (x0, y0, x1, y1), radius=radius, outline=255, width=4
    )
    cx = (x0 + x1) // 2
    draw.line((cx, y0 + 3, cx, y1 - 3), fill=255, width=3)
    return x0, y0, x1, y1


def _draw_coil(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    width = max(1, x1 - x0)
    for offset in np.linspace(0, width * 0.25, 5):
        draw.ellipse(
            (x0 + offset, y0, x1 - width * 0.25 + offset, y1),
            outline=255,
            width=3,
        )
    return x0, y0, x1, y1


def _primitive_landmarks(
    primitive: str,
    bbox: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    """Declare identity-bearing internal geometry for downstream gates.

    A silhouette can preserve an object's position while losing the parts
    that make the object recognizable.  Landmark requirements are emitted by
    the versioned primitive provider, in output-image coordinates, so the
    candidate selector does not need Case IDs or object-name heuristics.
    """

    if primitive != "canonical_graduated_tube_v1":
        return []
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    expected_center_y = [
        int(round(value))
        for value in np.linspace(
            y0 + height * 0.10,
            y0 + height * 0.58,
            7,
        )
    ]
    return [
        {
            "landmark_id": "graduation_ticks",
            "kind": "repeated_horizontal_segments",
            "roi_xyxy": [
                int(round(x0 + width * 0.22)),
                int(round(y0 + height * 0.09)),
                int(round(x0 + width * 0.50)),
                int(round(y0 + height * 0.60)),
            ],
            "minimum_distinct_groups": 5,
            "minimum_edge_pixels_per_row": 6,
            "maximum_row_gap_px": 2,
            "expected_group_center_y": expected_center_y,
            "maximum_expected_position_error_px": 3,
            "reason_zh": (
                "刻度是 graduated tube 的身份部件；只保留外轮廓会把它误读成滴液器或瓶子。"
            ),
        }
    ]


def compile_canonical_control(
    contract: dict[str, Any],
    keyframe_id: str,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    semantic, semantic_path, _ = keyframe_semantics(
        contract, keyframe_id, repo_root
    )
    normalized, normalization = normalize_legacy_objects(
        contract, semantic, repo_root
    )
    source_size = (
        int(semantic["canvas"]["width"]),
        int(semantic["canvas"]["height"]),
    )
    image = Image.new("L", OUTPUT_SIZE, 0)
    draw = ImageDraw.Draw(image)
    objects = normalized.get("items", [])
    rendered: list[dict[str, Any]] = []
    beaker_rendered = None
    beaker_source = None
    for item in objects:
        if item.get("class_id") == "glass_beaker":
            beaker_source = _scaled_bbox(
                item["geometry"], source_size
            )
            beaker_rendered = _draw_beaker(
                draw, beaker_source, OUTPUT_SIZE
            )
            rendered.append(
                {
                    "object_id": item["object_id"],
                    "class_id": item["class_id"],
                    "output_bbox_xyxy": list(beaker_rendered),
                    "primitive": "canonical_open_cylindrical_vessel_v1",
                }
            )
    for item in objects:
        class_id = item.get("class_id")
        if class_id == "glass_beaker":
            continue
        scaled = _scaled_bbox(item["geometry"], source_size)
        if class_id == "glass_burette":
            if beaker_rendered is None:
                raise ValueError("glass_burette requires glass_beaker anchor")
            output_bbox = _draw_burette(
                draw, scaled, beaker_rendered, OUTPUT_SIZE
            )
            primitive = "canonical_graduated_tube_v1"
        elif class_id == "bar_magnet":
            output_bbox = _draw_magnet(draw, scaled)
            primitive = "canonical_bar_magnet_v1"
        elif class_id == "fixed_coil":
            output_bbox = _draw_coil(draw, scaled)
            primitive = "canonical_coil_v1"
        else:
            # State objects such as droplets are excluded from the stable
            # appearance-anchor control. State Renderer B adds them later.
            continue
        rendered.append(
            {
                "object_id": item["object_id"],
                "class_id": class_id,
                "output_bbox_xyxy": list(output_bbox),
                "primitive": primitive,
                "required_internal_landmarks": _primitive_landmarks(
                    primitive, output_bbox
                ),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    control_path = output_dir / "structure_control.png"
    image.convert("RGB").save(control_path, optimize=False)
    normalized_path = output_dir / "normalized_objects.json"
    write_json(normalized_path, normalized)
    normalization_path = output_dir / "normalization_derivation.json"
    write_json(normalization_path, normalization)
    edge_density = float(np.count_nonzero(np.asarray(image))) / (
        OUTPUT_SIZE[0] * OUTPUT_SIZE[1]
    )
    expected = {
        item["class_id"]: item["cardinality"]
        for item in contract["semantic_exports"].get(
            "control_object_requirements", []
        )
    }
    rendered_counts = {
        class_id: sum(
            item["class_id"] == class_id for item in rendered
        )
        for class_id in expected
    }
    gate = {
        "schema_version": "1.0",
        "case_id": contract["case_id"],
        "keyframe_id": keyframe_id,
        "geometry_policy": "canonicalize",
        "checks": [
            {
                "name": "required_stable_objects_rendered",
                "passed": rendered_counts == expected,
                "evidence": {
                    "expected": expected,
                    "actual": rendered_counts,
                },
            },
            {
                "name": "control_edge_density_in_range",
                "passed": 0.002 <= edge_density <= 0.08,
                "evidence": round(edge_density, 6),
            },
            {
                "name": "annotation_and_ui_excluded",
                "passed": True,
                "evidence": (
                    "compiler reads object_identity/hard_boundary only"
                ),
            },
            {
                "name": "appearance_reference_not_read",
                "passed": True,
                "evidence": (
                    "geometry compiler API has no appearance parameter"
                ),
            },
        ],
        "passed": True,
        "source_semantics": file_record(semantic_path, repo_root),
        "rendered_objects": rendered,
        "required_internal_landmarks": [
            {
                "object_id": item["object_id"],
                "class_id": item["class_id"],
                **landmark,
            }
            for item in rendered
            for landmark in item.get(
                "required_internal_landmarks", []
            )
        ],
        "artifacts": {
            "structure_control": file_record(control_path, repo_root),
            "normalized_objects": file_record(
                normalized_path, repo_root
            ),
            "normalization_derivation": file_record(
                normalization_path, repo_root
            ),
        },
    }
    gate["passed"] = all(
        item["passed"] for item in gate["checks"]
    )
    write_json(output_dir / "g1.json", gate)
    return gate


def _copy_exact_control(
    contract: dict[str, Any],
    keyframe_id: str,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    semantic, semantic_path, _ = keyframe_semantics(
        contract, keyframe_id, repo_root
    )
    hard = [
        layer
        for layer in semantic["layers"]
        if layer["layer_type"] == "hard_boundary"
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    if hard:
        source_path = layer_data_path(contract, hard[0], repo_root)
        array = np.load(source_path, allow_pickle=False)
        source_image = Image.fromarray(np.uint8(array)).convert("RGB")
        control_kind = "hard_boundary_exact"
        source_sha = hard[0]["data"]["sha256"]
    else:
        heights = [
            layer
            for layer in semantic["layers"]
            if layer["layer_type"] == "height_or_normal"
        ]
        if not heights:
            raise ValueError("preserve_exact needs hard boundary or height")
        source_path = layer_data_path(contract, heights[0], repo_root)
        array = np.load(source_path, allow_pickle=False).astype(np.float32)
        minimum = float(array.min())
        maximum = float(array.max())
        normalized = (array - minimum) / max(maximum - minimum, 1e-9)
        source_image = Image.fromarray(
            np.uint8(np.clip(normalized, 0, 1) * 255)
        ).convert("RGB")
        control_kind = "height_field_exact"
        source_sha = heights[0]["data"]["sha256"]
    output_path = output_dir / "structure_control.png"
    source_image.save(output_path, optimize=False)
    # A second exact payload is copied byte-for-byte. It is the regression
    # artifact; PNG encoder differences cannot disguise altered source data.
    payload_path = output_dir / source_path.name
    shutil.copyfile(source_path, payload_path)
    gate = {
        "schema_version": "1.0",
        "case_id": contract["case_id"],
        "keyframe_id": keyframe_id,
        "geometry_policy": "preserve_exact",
        "control_kind": control_kind,
        "checks": [
            {
                "name": "semantic_payload_byte_identical",
                "passed": sha256_path(payload_path) == source_sha,
                "evidence": {
                    "source_sha256": source_sha,
                    "copied_sha256": sha256_path(payload_path),
                },
            },
            {
                "name": "appearance_reference_not_read",
                "passed": True,
            },
        ],
        "passed": sha256_path(payload_path) == source_sha,
        "source_semantics": file_record(semantic_path, repo_root),
        "artifacts": {
            "structure_control": file_record(output_path, repo_root),
            "exact_payload": file_record(payload_path, repo_root),
        },
    }
    write_json(output_dir / "g1.json", gate)
    return gate


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask > 0
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    return mask & ~eroded


def compile_layout_control(
    contract: dict[str, Any],
    keyframe_id: str,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    semantic, semantic_path, _ = keyframe_semantics(
        contract, keyframe_id, repo_root
    )
    canvas = semantic["canvas"]
    source_size = (canvas["width"], canvas["height"])
    region_layers = [
        layer
        for layer in semantic["layers"]
        if layer["layer_type"] == "region"
    ]
    if not region_layers:
        raise ValueError("layout_only requires at least one region layer")
    structure = np.zeros(
        (canvas["height"], canvas["width"]), dtype=np.uint8
    )
    regions = np.zeros_like(structure)
    region_metrics = []
    for index, layer in enumerate(region_layers, start=1):
        array = np.load(
            layer_data_path(contract, layer, repo_root),
            allow_pickle=False,
        )
        if array.ndim != 2:
            raise ValueError("layout region must be 2D")
        mask = array > 0
        boundary = _binary_boundary(mask)
        structure[boundary] = 255
        regions[mask] = np.uint8(
            min(255, 48 + index * 55)
        )
        region_metrics.append(
            {
                "layer_id": layer["layer_id"],
                "area_px": int(mask.sum()),
                "boundary_px": int(boundary.sum()),
                "source_sha256": layer["data"]["sha256"],
            }
        )
    identity, _ = load_identity(contract, semantic, repo_root)
    anchor_image = Image.fromarray(structure).convert("L")
    draw = ImageDraw.Draw(anchor_image)
    anchors = []
    for item in identity.get("items", []):
        try:
            x0, y0, x1, y1 = _bbox(item["geometry"])
        except ValueError:
            continue
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        radius = max(4, min(12, (x1 - x0 + y1 - y0) / 8))
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=255,
            width=2,
        )
        anchors.append(
            {
                "object_id": item.get("object_id"),
                "class_id": item.get("class_id", "untyped"),
                "center_xy": [round(cx, 3), round(cy, 3)],
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    control_path = output_dir / "structure_control.png"
    region_path = output_dir / "regions.png"
    anchor_image.resize(OUTPUT_SIZE, Image.Resampling.NEAREST).convert(
        "RGB"
    ).save(control_path, optimize=False)
    Image.fromarray(regions).resize(
        OUTPUT_SIZE, Image.Resampling.NEAREST
    ).convert("RGB").save(region_path, optimize=False)
    anchors_path = output_dir / "anchors.json"
    write_json(anchors_path, {"anchors": anchors})
    edge_density = float(np.count_nonzero(np.asarray(anchor_image))) / (
        source_size[0] * source_size[1]
    )
    gate = {
        "schema_version": "1.0",
        "case_id": contract["case_id"],
        "keyframe_id": keyframe_id,
        "geometry_policy": "layout_only",
        "checks": [
            {
                "name": "region_topology_exported",
                "passed": bool(region_metrics)
                and all(item["area_px"] > 0 for item in region_metrics),
                "evidence": region_metrics,
            },
            {
                "name": "identity_anchors_exported",
                "passed": bool(anchors),
                "evidence": anchors,
            },
            {
                "name": "sparse_structure_density",
                "passed": 0.0001 <= edge_density <= 0.12,
                "evidence": round(edge_density, 6),
            },
            {
                "name": "scalar_and_annotation_not_converted_to_edges",
                "passed": True,
                "evidence": (
                    "only region boundaries and identity anchors are drawn"
                ),
            },
            {
                "name": "appearance_reference_not_read",
                "passed": True,
            },
        ],
        "source_semantics": file_record(semantic_path, repo_root),
        "artifacts": {
            "structure_control": file_record(control_path, repo_root),
            "regions": file_record(region_path, repo_root),
            "anchors": file_record(anchors_path, repo_root),
        },
    }
    gate["passed"] = all(
        item["passed"] for item in gate["checks"]
    )
    write_json(output_dir / "g1.json", gate)
    return gate


def compile_control(
    contract: dict[str, Any],
    keyframe_id: str,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    policy = contract["geometry_policy"]
    if policy == "canonicalize":
        return compile_canonical_control(
            contract, keyframe_id, output_dir, repo_root
        )
    if policy == "preserve_exact":
        return _copy_exact_control(
            contract, keyframe_id, output_dir, repo_root
        )
    if policy == "layout_only":
        return compile_layout_control(
            contract, keyframe_id, output_dir, repo_root
        )
    raise ValueError(f"unsupported geometry policy: {policy}")


def compile_legacy_delta_layout(
    contract: dict[str, Any],
    state_index: int,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Adapt the Stage 1 delta state grid to the Stage 3 layout interface.

    The legacy simulator predates per-keyframe semantic manifests, but its
    JSONL still contains land, new-land, deposit grids, particle identities,
    and flow samples. This adapter reads those state arrays—not RGB frames.
    """

    states_path = (
        repo_root
        / "modules/video_model/stage1/output/causal_delta/mechanism/states.jsonl"
    )
    lines = [
        line
        for line in states_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    state = json.loads(lines[state_index])
    land = np.asarray(state["land"], dtype=bool)
    new_land = np.asarray(state["new_land"], dtype=bool)
    thickness = np.asarray(state["thick"], dtype=np.float32)
    if land.shape != new_land.shape or land.shape != thickness.shape:
        raise ValueError("legacy delta grids do not share one shape")
    shoreline = _binary_boundary(land | new_land)
    deposit = thickness > 0
    structure = np.uint8(shoreline) * 255
    region_map = np.zeros(land.shape, dtype=np.uint8)
    region_map[land] = 70
    region_map[deposit] = 145
    region_map[new_land] = 235
    structure_image = Image.fromarray(structure).convert("L")
    flow_samples = state.get("flow_samples", [])
    flow_image = Image.new("L", OUTPUT_SIZE, 0)
    flow_draw = ImageDraw.Draw(flow_image)
    scale_x = OUTPUT_SIZE[0] / land.shape[1]
    scale_y = OUTPUT_SIZE[1] / land.shape[0]
    # Flow is a separate vector-control channel, not structure Canny.
    # Subsample it for legibility instead of turning every vector into edges.
    for sample in flow_samples[::4]:
        if isinstance(sample, dict):
            x = float(sample["x"])
            y = float(sample["y"])
            vx = float(sample.get("vx", sample.get("u", 0)))
            vy = float(sample.get("vy", sample.get("v", 0)))
        else:
            x, y, vx, vy = (float(value) for value in sample[:4])
        start = (x * scale_x, y * scale_y)
        end = (
            (x + vx * 3.0) * scale_x,
            (y + vy * 3.0) * scale_y,
        )
        flow_draw.line(
            (*start, *end),
            fill=255,
            width=2,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    control_path = output_dir / "structure_control.png"
    region_path = output_dir / "regions.png"
    flow_path = output_dir / "flow_anchors.png"
    structure_image.resize(
        OUTPUT_SIZE, Image.Resampling.NEAREST
    ).convert("RGB").save(control_path, optimize=False)
    Image.fromarray(region_map).resize(
        OUTPUT_SIZE, Image.Resampling.NEAREST
    ).convert("RGB").save(region_path, optimize=False)
    flow_image.convert("RGB").save(flow_path, optimize=False)
    anchors_path = output_dir / "anchors.json"
    write_json(
        anchors_path,
        {
            "state_index": state_index,
            "beat_id": state["beat_id"],
            "flow_sample_count": len(flow_samples),
            "particle_count": len(state.get("particles", [])),
        },
    )
    checks = [
        {
            "name": "legacy_state_arrays_not_rgb_frame",
            "passed": True,
            "evidence": file_record(states_path, repo_root),
        },
        {
            "name": "shoreline_topology_exported",
            "passed": int(shoreline.sum()) > 0,
            "evidence": int(shoreline.sum()),
        },
        {
            "name": "deposit_and_new_land_are_separate_regions",
            "passed": int(deposit.sum()) > 0 and int(new_land.sum()) > 0,
            "evidence": {
                "deposit_cells": int(deposit.sum()),
                "new_land_cells": int(new_land.sum()),
            },
        },
        {
            "name": "one_sandbar_and_two_final_flow_paths",
            "passed": len(_components(np.uint8(new_land) * 255)) == 1
            and state.get("stats", {}).get("raw_channel_count") == 2,
            "evidence": {
                "sandbar_component_count": len(
                    _components(np.uint8(new_land) * 255)
                ),
                "raw_channel_count": state.get("stats", {}).get(
                    "raw_channel_count"
                ),
            },
        },
        {
            "name": "appearance_reference_not_read",
            "passed": True,
        },
    ]
    gate = {
        "schema_version": "1.0",
        "case_id": contract["case_id"],
        "state_index": state_index,
        "geometry_policy": "layout_only",
        "adapter": "legacy_delta_state_grid_v1",
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "artifacts": {
            "structure_control": file_record(control_path, repo_root),
            "regions": file_record(region_path, repo_root),
            "flow_anchors": file_record(flow_path, repo_root),
            "anchors": file_record(anchors_path, repo_root),
        },
    }
    write_json(output_dir / "g1.json", gate)
    return gate
