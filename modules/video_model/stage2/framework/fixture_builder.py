"""Create deterministic, model-free fixtures for every Stage 2 case."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .contracts import (
    artifact_record,
    load_json,
    sha256_path,
    validate_concept_spec,
    validate_fixture_manifest,
    validate_layer_manifest,
    validate_sequence_spec,
    validate_states,
    write_json,
)


CANVAS_WIDTH = 320
CANVAS_HEIGHT = 180
REPRESENTATIVE_STATE_ID = "state_02"

DISCIPLINE_PALETTES = {
    "mathematics": ((237, 232, 218), (47, 91, 96)),
    "physics": ((219, 234, 236), (35, 89, 119)),
    "chemistry": ((233, 227, 239), (110, 62, 119)),
    "biology": ((226, 237, 218), (54, 108, 72)),
    "geography": ((232, 224, 204), (72, 108, 92)),
}

LAYER_COPY = {
    "hard_boundary": {
        "title_zh": "需要严格保留的硬边界",
        "meaning_zh": "表示可以形成清楚轮廓的几何；不包含软浓度或教学文字。",
        "model_input_policy": "allowed_by_route",
        "final_role_zh": "约束大尺度轮廓和拓扑。",
    },
    "region": {
        "title_zh": "对象或允许修改区域",
        "meaning_zh": "白色区域表示本状态中的对象范围或允许改变的像素范围。",
        "model_input_policy": "allowed_by_route",
        "final_role_zh": "限制局部材质增强，保护区域外像素。",
    },
    "scalar_field": {
        "title_zh": "连续标量场",
        "meaning_zh": "每个像素保存一个连续数值，例如浓度、温度、振幅或压力。",
        "model_input_policy": "allowed_by_route",
        "final_role_zh": "决定软变化的空间强弱，默认不转成密集 Canny。",
    },
    "vector_field": {
        "title_zh": "方向与速度场",
        "meaning_zh": "每个采样位置保存二维方向和大小；箭头只用于人类审计。",
        "model_input_policy": "allowed_by_route",
        "final_role_zh": "约束运动方向并为视频 evaluator 提供依据。",
    },
    "height_or_normal": {
        "title_zh": "高度或表面法线",
        "meaning_zh": "保存大尺度表面起伏，预览图把法线方向编码成颜色。",
        "model_input_policy": "allowed_by_route",
        "final_role_zh": "让程序保留大形状，图像模型只补材料细节。",
    },
    "object_identity": {
        "title_zh": "带稳定身份的对象",
        "meaning_zh": "每个对象有跨帧不变的 ID、类别、中心和边界框。",
        "model_input_policy": "allowed_by_route",
        "final_role_zh": "防止模型复制、吞掉或交换需要追踪的对象。",
    },
    "annotation": {
        "title_zh": "教学叠加层",
        "meaning_zh": "保存箭头和重点提示的程序指令，写实底图完成后再叠加。",
        "model_input_policy": "never",
        "final_role_zh": "只用于最终教学解释，不让生成模型写文字。",
    },
}

def _stable_seed(case_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(case_id.encode("utf-8")).digest()[:4], "big"
    )


def _normalized(value: np.ndarray) -> np.ndarray:
    minimum = float(value.min())
    maximum = float(value.max())
    if math.isclose(minimum, maximum):
        return np.zeros_like(value, dtype=np.float32)
    return np.asarray(
        (value - minimum) / (maximum - minimum), dtype=np.float32
    )


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int],
    width: int = 2,
) -> None:
    draw.line((start, end), fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(math.hypot(dx, dy), 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 5 + width
    left = (
        end[0] - ux * size + px * size * 0.55,
        end[1] - uy * size + py * size * 0.55,
    )
    right = (
        end[0] - ux * size - px * size * 0.55,
        end[1] - uy * size - py * size * 0.55,
    )
    draw.polygon((end, left, right), fill=fill)


def _fixture_coordinates(seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    yy, xx = np.mgrid[0:CANVAS_HEIGHT, 0:CANVAS_WIDTH]
    x = xx.astype(np.float32) / (CANVAS_WIDTH - 1)
    y = yy.astype(np.float32) / (CANVAS_HEIGHT - 1)
    phase = (seed % 360) * math.pi / 180.0
    return x, y, phase


def _numeric_layer(layer_type: str, seed: int) -> np.ndarray:
    x, y, phase = _fixture_coordinates(seed)
    center_x = 0.42 + 0.08 * math.sin(phase)
    center_y = 0.50 + 0.09 * math.cos(phase)
    radius_x = 0.23
    radius_y = 0.27
    ellipse = ((x - center_x) / radius_x) ** 2 + (
        (y - center_y) / radius_y
    ) ** 2
    if layer_type == "hard_boundary":
        return np.asarray(np.abs(ellipse - 1.0) < 0.045, dtype=np.uint8) * 255
    if layer_type == "region":
        return np.asarray(ellipse <= 1.0, dtype=np.uint8) * 255
    if layer_type == "scalar_field":
        gaussian = np.exp(
            -(
                ((x - center_x) / 0.24) ** 2
                + ((y - center_y) / 0.30) ** 2
            )
            * 2.1
        )
        wave = 0.18 * (np.sin(9 * x + phase) + 1.0)
        return _normalized(gaussian + wave + 0.18 * x)
    if layer_type == "vector_field":
        u = 0.75 + 0.20 * np.cos(2 * math.pi * y + phase)
        v = 0.28 * np.sin(2 * math.pi * x + phase)
        return np.stack((u, v), axis=-1).astype(np.float32)
    if layer_type == "height_or_normal":
        hill = np.exp(
            -(
                ((x - 0.53) / 0.25) ** 2
                + ((y - 0.52) / 0.32) ** 2
            )
            * 2.0
        )
        ripples = 0.22 * np.sin(14 * x + phase) * np.cos(8 * y)
        return np.asarray(hill + ripples, dtype=np.float32)
    raise ValueError(f"not a numeric fixture layer: {layer_type}")


def _preview_numeric(layer_type: str, array: np.ndarray) -> Image.Image:
    if layer_type == "hard_boundary":
        rgb = np.repeat(array[:, :, None], 3, axis=2)
        return Image.fromarray(rgb, mode="RGB")
    if layer_type == "region":
        mask = array.astype(np.float32) / 255.0
        rgb = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.uint8(22 * mask)
        rgb[:, :, 1] = np.uint8(170 * mask)
        rgb[:, :, 2] = np.uint8(114 * mask)
        return Image.fromarray(rgb, mode="RGB")
    if layer_type == "scalar_field":
        value = _normalized(array)
        rgb = np.stack(
            (
                np.clip(2.1 * value - 0.55, 0, 1),
                np.clip(1.65 - 2.0 * np.abs(value - 0.5), 0, 1),
                np.clip(1.35 - 1.8 * value, 0, 1),
            ),
            axis=-1,
        )
        return Image.fromarray(np.uint8(rgb * 255), mode="RGB")
    if layer_type == "vector_field":
        image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (11, 32, 44))
        draw = ImageDraw.Draw(image)
        for py in range(14, CANVAS_HEIGHT, 22):
            for px in range(14, CANVAS_WIDTH, 24):
                u, v = array[py, px]
                norm = max(float(math.hypot(float(u), float(v))), 1e-5)
                scale = 12.0
                end = (
                    px + float(u) / norm * scale,
                    py + float(v) / norm * scale,
                )
                _draw_arrow(
                    draw,
                    (px, py),
                    end,
                    fill=(99, 210, 226),
                    width=1,
                )
        return image
    if layer_type == "height_or_normal":
        grad_y, grad_x = np.gradient(array.astype(np.float32))
        normal = np.stack((-grad_x, -grad_y, np.ones_like(array)), axis=-1)
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal /= np.maximum(norm, 1e-6)
        rgb = np.uint8(np.clip((normal * 0.5 + 0.5) * 255, 0, 255))
        return Image.fromarray(rgb, mode="RGB")
    raise ValueError(f"cannot preview numeric layer: {layer_type}")


def _objects_payload(
    case: dict[str, Any],
    seed: int,
    object_count: int,
) -> dict[str, Any]:
    count = object_count
    items = []
    for index in range(count):
        angle = 2 * math.pi * index / max(count, 1) + (seed % 17) * 0.01
        center_x = int(160 + math.cos(angle) * (42 + 4 * (index % 2)))
        center_y = int(90 + math.sin(angle) * 38)
        radius = 8 + index % 3
        items.append(
            {
                "object_id": f"{case['case_id']}-object-{index + 1:02d}",
                "class_id": f"{case['slug']}_fixture_object",
                "center_xy": [center_x, center_y],
                "bbox_xyxy": [
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ],
                "visible": True,
            }
        )
    return {
        "schema_version": "1.0",
        "coordinate_system": "pixel_xy_top_left",
        "items": items,
    }


def _preview_objects(payload: dict[str, Any]) -> Image.Image:
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (9, 21, 27))
    draw = ImageDraw.Draw(image)
    colors = (
        (247, 177, 75),
        (90, 200, 184),
        (222, 104, 106),
        (125, 156, 232),
    )
    for index, item in enumerate(payload["items"]):
        draw.ellipse(
            item["bbox_xyxy"],
            fill=colors[index % len(colors)],
            outline=(255, 255, 255),
            width=2,
        )
        cx, cy = item["center_xy"]
        draw.line((cx - 3, cy, cx + 3, cy), fill=(20, 35, 42), width=1)
        draw.line((cx, cy - 3, cx, cy + 3), fill=(20, 35, 42), width=1)
    return image


def _annotations_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "composite_after_generation": True,
        "items": [
            {
                "annotation_id": f"{case['case_id']}-annotation-01",
                "kind": "arrow",
                "start_xy": [54, 132],
                "end_xy": [126, 94],
                "meaning_zh": "指出代表状态中的主要变化区域",
            },
            {
                "annotation_id": f"{case['case_id']}-annotation-02",
                "kind": "arrow",
                "start_xy": [268, 46],
                "end_xy": [201, 79],
                "meaning_zh": "指出需要在报告中解释的数据来源",
            },
        ],
    }


def _preview_annotations(payload: dict[str, Any]) -> Image.Image:
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for index, item in enumerate(payload["items"]):
        color = (255, 208, 77) if index == 0 else (255, 112, 89)
        _draw_arrow(
            draw,
            tuple(item["start_xy"]),
            tuple(item["end_xy"]),
            fill=color,
            width=3,
        )
    return image


def _semantic_record(
    *,
    layer_id: str,
    layer_type: str,
    case: dict[str, Any],
    data_path: Path,
    preview_path: Path,
    fixture_root: Path,
    data_shape: list[int],
    data_dtype: str,
    data_range: list[float],
    data_encoding: str,
) -> dict[str, Any]:
    copy = LAYER_COPY[layer_type]
    data_record = artifact_record(data_path, fixture_root)
    data_record.update(
        {
            "encoding": data_encoding,
            "shape": data_shape,
            "dtype": data_dtype,
            "value_range": data_range,
        }
    )
    preview_record = artifact_record(preview_path, fixture_root)
    preview_record.update(
        {
            "encoding": "png_rgb_preview",
            "shape": [CANVAS_HEIGHT, CANVAS_WIDTH, 3],
            "dtype": "uint8",
            "value_range": [0, 255],
        }
    )
    return {
        "layer_id": layer_id,
        "layer_type": layer_type,
        "title_zh": copy["title_zh"],
        "meaning_zh": copy["meaning_zh"],
        "source_zh": (
            f"Phase 1 根据 {case['case_id']} 的固定 fixture 参数直接计算；"
            "它验证数据契约，不代表完成的案例机制模拟。"
        ),
        "data": data_record,
        "preview": preview_record,
        "model_input_policy": copy["model_input_policy"],
        "used_as_model_input": False,
        "final_role_zh": copy["final_role_zh"],
    }


def _build_layers(
    case: dict[str, Any],
    template: dict[str, Any],
    fixture_root: Path,
) -> tuple[dict[str, Any], dict[str, Image.Image], dict[str, np.ndarray]]:
    seed = _stable_seed(case["case_id"])
    layers_root = fixture_root / "layers"
    layers_root.mkdir(parents=True, exist_ok=True)
    records = []
    previews: dict[str, Image.Image] = {}
    numeric: dict[str, np.ndarray] = {}
    for layer_type in case["primary_layer_types"]:
        layer_id = f"{case['case_id'].lower()}_{layer_type}"
        data_path: Path
        preview_path = layers_root / f"{layer_type}_preview.png"
        if layer_type in {
            "hard_boundary",
            "region",
            "scalar_field",
            "vector_field",
            "height_or_normal",
        }:
            array = _numeric_layer(layer_type, seed)
            data_path = layers_root / f"{layer_type}.npy"
            np.save(data_path, array, allow_pickle=False)
            preview = _preview_numeric(layer_type, array)
            preview.save(preview_path)
            data_shape = list(array.shape)
            data_dtype = str(array.dtype)
            data_range = [float(array.min()), float(array.max())]
            data_encoding = "numpy_npy"
            numeric[layer_type] = array
        elif layer_type == "object_identity":
            payload = _objects_payload(
                case,
                seed,
                int(template["fixture_object_count"]),
            )
            data_path = layers_root / "object_identity.json"
            write_json(data_path, payload)
            preview = _preview_objects(payload)
            preview.save(preview_path)
            data_shape = [len(payload["items"])]
            data_dtype = "json_object_records"
            data_range = [0, len(payload["items"])]
            data_encoding = "json"
        elif layer_type == "annotation":
            payload = _annotations_payload(case)
            data_path = layers_root / "annotation.json"
            write_json(data_path, payload)
            preview = _preview_annotations(payload)
            preview.save(preview_path)
            data_shape = [len(payload["items"])]
            data_dtype = "json_annotation_records"
            data_range = [0, len(payload["items"])]
            data_encoding = "json"
        else:
            raise ValueError(f"unsupported fixture layer: {layer_type}")
        previews[layer_type] = preview
        records.append(
            _semantic_record(
                layer_id=layer_id,
                layer_type=layer_type,
                case=case,
                data_path=data_path,
                preview_path=preview_path,
                fixture_root=fixture_root,
                data_shape=data_shape,
                data_dtype=data_dtype,
                data_range=data_range,
                data_encoding=data_encoding,
            )
        )
    manifest = {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "state_id": REPRESENTATIVE_STATE_ID,
        "canvas": {
            "width": CANVAS_WIDTH,
            "height": CANVAS_HEIGHT,
            "coordinate_system": "pixel_xy_top_left",
        },
        "layers": records,
    }
    return manifest, previews, numeric


def _composite_fixture_frames(
    case: dict[str, Any],
    previews: dict[str, Image.Image],
) -> tuple[Image.Image, Image.Image]:
    base_color, accent = DISCIPLINE_PALETTES[case["discipline"]]
    yy = np.linspace(0, 1, CANVAS_HEIGHT, dtype=np.float32)[:, None]
    base = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.float32)
    for channel, value in enumerate(base_color):
        base[:, :, channel] = value * (0.93 + 0.07 * (1 - yy))
    clean = Image.fromarray(np.uint8(np.clip(base, 0, 255)), mode="RGB")
    order = [
        "height_or_normal",
        "scalar_field",
        "region",
        "vector_field",
        "hard_boundary",
        "object_identity",
    ]
    strengths = {
        "height_or_normal": 0.52,
        "scalar_field": 0.36,
        "region": 0.34,
        "vector_field": 0.58,
        "hard_boundary": 0.82,
        "object_identity": 0.86,
    }
    clean_array = np.asarray(clean, dtype=np.float32)
    for layer_type in order:
        if layer_type not in previews:
            continue
        overlay = np.asarray(previews[layer_type], dtype=np.float32)
        intensity = overlay.max(axis=2) / 255.0
        if layer_type in {"height_or_normal", "scalar_field", "vector_field"}:
            intensity = np.maximum(intensity, 0.22)
        alpha = (
            intensity[:, :, None] * strengths[layer_type]
        ).astype(np.float32)
        clean_array = clean_array * (1 - alpha) + overlay * alpha
    clean = Image.fromarray(
        np.uint8(np.clip(clean_array, 0, 255)), mode="RGB"
    )
    draw = ImageDraw.Draw(clean)
    draw.rectangle(
        (8, 8, CANVAS_WIDTH - 9, CANVAS_HEIGHT - 9),
        outline=accent,
        width=2,
    )
    program = clean.copy()
    if "annotation" in previews:
        annotation = np.asarray(previews["annotation"], dtype=np.float32)
        program_array = np.asarray(program, dtype=np.float32)
        mask = annotation.max(axis=2, keepdims=True) > 0
        program_array = np.where(mask, annotation, program_array)
        program = Image.fromarray(
            np.uint8(np.clip(program_array, 0, 255)), mode="RGB"
        )
    return clean, program


def _build_specs(
    case: dict[str, Any],
    template: dict[str, Any],
    fixture_root: Path,
    layer_ids: list[str],
) -> tuple[Path, Path, Path]:
    progress_values = [0.0, 1 / 3, 2 / 3, 1.0]
    segments = []
    keyframes = []
    states = []
    for index, (meaning, progress) in enumerate(
        zip(template["beats_zh"], progress_values)
    ):
        state_id = f"state_{index:02d}"
        segments.append(
            {
                "segment_id": f"segment_{index:02d}",
                "order": index,
                "meaning_zh": meaning,
                "mechanism_condition_zh": (
                    f"Phase 1 fixture_progress = {progress:.6f}；"
                    "Phase 2 必须替换为该案例的真实机制条件。"
                ),
                "only_major_change_zh": meaning,
            }
        )
        keyframes.append(
            {
                "keyframe_id": f"keyframe_{index:02d}",
                "order": index,
                "state_id": state_id,
                "meaning_zh": meaning,
                "selection_condition_zh": (
                    f"仅用于契约冒烟：fixture_progress = {progress:.6f}"
                ),
                "only_major_change_zh": meaning,
                "forbidden_zh": [
                    "把 Phase 1 fixture 当成完成的科学模拟",
                    "让模型改变程序定义的数量、位置或拓扑",
                ],
            }
        )
        states.append(
            {
                "state_id": state_id,
                "order": index,
                "progress": progress,
                "meaning_zh": meaning,
                "fixture_only": True,
            }
        )
    concept = {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "title_zh": case["title_zh"],
        "discipline": case["discipline"],
        "learning_goal_zh": template["learning_goal_zh"],
        "assumptions_zh": template["assumptions_zh"],
        "segments": segments,
        "forbidden_shortcuts_zh": [
            "用漂亮画面掩盖机制条件缺失",
            "让图像模型重新决定程序事实",
            "让一个视频过渡跨越多个主要变化",
        ],
    }
    sequence = {
        "schema_version": "1.0",
        "sequence_id": f"{case['slug']}_phase1_fixture",
        "case_id": case["case_id"],
        "canvas": {
            "width": CANVAS_WIDTH,
            "height": CANVAS_HEIGHT,
            "coordinate_system": "pixel_xy_top_left",
        },
        "camera": {
            "view_zh": "固定教学视图（Phase 1 契约 fixture）",
            "locked": True,
        },
        "state_source": "states.jsonl",
        "keyframes": keyframes,
        "semantic_layer_ids": layer_ids,
        "fixed_across_sequence_zh": template[
            "fixed_across_sequence_zh"
        ],
        "model_policy": {
            "image_model_role_zh": template["image_model_role_zh"],
            "video_model_role_zh": template["video_model_role_zh"],
            "annotations_after_generation": True,
        },
    }
    concept_path = fixture_root / "concept_spec.json"
    sequence_path = fixture_root / "sequence_spec.json"
    states_path = fixture_root / "states.jsonl"
    write_json(concept_path, concept)
    write_json(sequence_path, sequence)
    states_path.write_text(
        "".join(
            json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n"
            for state in states
        ),
        encoding="utf-8",
    )
    return concept_path, sequence_path, states_path


def build_fixture(
    case: dict[str, Any],
    template: dict[str, Any],
    fixture_root: Path,
) -> dict[str, Any]:
    fixture_root.mkdir(parents=True, exist_ok=True)
    layers, previews, numeric = _build_layers(
        case, template, fixture_root
    )
    layer_ids = [item["layer_id"] for item in layers["layers"]]
    concept_path, sequence_path, states_path = _build_specs(
        case, template, fixture_root, layer_ids
    )
    layers_path = fixture_root / "semantic_layers.json"
    write_json(layers_path, layers)
    clean, program = _composite_fixture_frames(case, previews)
    clean_path = fixture_root / "clean_frame.png"
    program_path = fixture_root / "program_frame.png"
    clean.save(clean_path)
    program.save(program_path)

    hard_layers = [
        layer
        for layer in layers["layers"]
        if layer["layer_type"] == "hard_boundary"
    ]
    if hard_layers:
        control_root = fixture_root / "control"
        control_root.mkdir(parents=True, exist_ok=True)
        control_path = control_root / "hard_boundary_candidate.png"
        boundary = numeric["hard_boundary"]
        Image.fromarray(boundary, mode="L").convert("RGB").save(control_path)
        control = {
            "route": "sparse_hard_boundary_candidate",
            "reason_zh": (
                "本案例声明了硬边界，因此保存一张稀疏控制候选；"
                "Phase 1 没有把它输入任何模型。"
            ),
            "input_layer_ids": [hard_layers[0]["layer_id"]],
            "used_as_model_input": False,
            "control_preview": artifact_record(control_path, fixture_root),
            "edge_fraction": round(float((boundary > 0).mean()), 6),
        }
        # fixture_manifest.schema.json keeps only portable contract fields.
        control_for_manifest = {
            key: value
            for key, value in control.items()
            if key != "edge_fraction"
        }
    else:
        control = {
            "route": "off",
            "reason_zh": (
                "本 fixture 没有声明硬边界；连续场或对象身份不能为了使用 "
                "ControlNet 被强行转成 Canny。"
            ),
            "input_layer_ids": [],
            "used_as_model_input": False,
            "control_preview": None,
        }
        control_for_manifest = control

    manifest = {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "classification": (
            "model-free contract fixture, not a finished scientific animation"
        ),
        "representative_state_id": REPRESENTATIVE_STATE_ID,
        "concept_spec": artifact_record(concept_path, fixture_root),
        "sequence_spec": artifact_record(sequence_path, fixture_root),
        "states": artifact_record(states_path, fixture_root),
        "clean_frame": artifact_record(clean_path, fixture_root),
        "program_frame": artifact_record(program_path, fixture_root),
        "semantic_layers": artifact_record(layers_path, fixture_root),
        "control": control_for_manifest,
        "model_runs": {"image": 0, "video": 0},
    }
    manifest_path = fixture_root / "fixture_manifest.json"
    write_json(manifest_path, manifest)

    concept = load_json(concept_path)
    sequence = load_json(sequence_path)
    validate_concept_spec(concept, expected_case_id=case["case_id"])
    validate_sequence_spec(
        sequence,
        expected_case_id=case["case_id"],
        expected_layer_ids=set(layer_ids),
    )
    validate_states(states_path)
    validate_layer_manifest(layers, fixture_root)
    validate_fixture_manifest(manifest, fixture_root)
    return {
        "case_id": case["case_id"],
        "title_zh": case["title_zh"],
        "discipline": case["discipline"],
        "sentinel": case["sentinel"],
        "fixture_root": str(fixture_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_path(manifest_path),
        "layer_types": case["primary_layer_types"],
        "layer_count": len(layers["layers"]),
        "control": control,
        "program_frame": manifest["program_frame"],
        "clean_frame": manifest["clean_frame"],
        "layers": layers["layers"],
        "model_runs": {"image": 0, "video": 0},
    }
