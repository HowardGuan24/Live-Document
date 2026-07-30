"""Five deterministic sentinel programs used by Stage 2 Phase 2.

The generic runner knows nothing about triangles, waves, titration, cells, or
mountains. Those mechanism decisions stay in this case-plugin module.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw


WIDTH = 640
HEIGHT = 360
KEYFRAME_PROGRESS = (0.0, 1 / 3, 2 / 3, 1.0)


@dataclass(frozen=True)
class LayerSample:
    layer_id: str
    layer_type: str
    title_zh: str
    meaning_zh: str
    data: np.ndarray | dict[str, Any]
    preview: Image.Image
    model_input_policy: str
    final_role_zh: str


@dataclass(frozen=True)
class ProgramSample:
    state: dict[str, Any]
    clean_frame: Image.Image
    program_frame: Image.Image
    layers: tuple[LayerSample, ...]


@dataclass(frozen=True)
class SentinelProgram:
    case_id: str
    title_zh: str
    primary_mechanism_zh: str
    sample: Callable[[float], ProgramSample]
    validate: Callable[[list[ProgramSample]], list[dict[str, Any]]]


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _segment(progress: float) -> tuple[int, float]:
    scaled = min(max(progress, 0.0), 1.0) * 3.0
    index = min(int(scaled), 2)
    local = scaled - index
    if progress >= 1.0:
        return 2, 1.0
    return index, local


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: tuple[int, int, int],
    width: int = 3,
) -> None:
    draw.line((start, end), fill=fill, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(math.hypot(dx, dy), 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 7 + width
    points = (
        end,
        (
            end[0] - ux * size + px * size * 0.52,
            end[1] - uy * size + py * size * 0.52,
        ),
        (
            end[0] - ux * size - px * size * 0.52,
            end[1] - uy * size - py * size * 0.52,
        ),
    )
    draw.polygon(points, fill=fill)


def _edge(binary: np.ndarray) -> np.ndarray:
    mask = binary > 0
    inner = (
        np.roll(mask, 1, 0)
        & np.roll(mask, -1, 0)
        & np.roll(mask, 1, 1)
        & np.roll(mask, -1, 1)
    )
    edge = mask & ~inner
    edge[[0, -1], :] = False
    edge[:, [0, -1]] = False
    return np.uint8(edge) * 255


def _binary_preview(
    array: np.ndarray, color: tuple[int, int, int]
) -> Image.Image:
    mask = np.asarray(array > 0, dtype=np.float32)
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for channel, value in enumerate(color):
        rgb[:, :, channel] = np.uint8(mask * value)
    return Image.fromarray(rgb, mode="RGB")


def _scalar_preview(array: np.ndarray) -> Image.Image:
    value = array.astype(np.float32)
    minimum, maximum = float(value.min()), float(value.max())
    value = (
        np.zeros_like(value)
        if math.isclose(minimum, maximum)
        else (value - minimum) / (maximum - minimum)
    )
    rgb = np.stack(
        (
            np.clip(2.1 * value - 0.55, 0, 1),
            np.clip(1.65 - 2.0 * np.abs(value - 0.5), 0, 1),
            np.clip(1.35 - 1.8 * value, 0, 1),
        ),
        axis=-1,
    )
    return Image.fromarray(np.uint8(rgb * 255), mode="RGB")


def _normal_preview(height: np.ndarray) -> Image.Image:
    grad_y, grad_x = np.gradient(height.astype(np.float32))
    normal = np.stack((-grad_x, -grad_y, np.ones_like(height)), axis=-1)
    normal /= np.maximum(
        np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6
    )
    return Image.fromarray(
        np.uint8(np.clip((normal * 0.5 + 0.5) * 255, 0, 255)),
        mode="RGB",
    )


def _vector_preview(array: np.ndarray) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (13, 34, 43))
    draw = ImageDraw.Draw(image)
    for y in range(18, HEIGHT, 30):
        for x in range(18, WIDTH, 34):
            u, v = (float(item) for item in array[y, x])
            norm = max(math.hypot(u, v), 1e-5)
            _draw_arrow(
                draw,
                (x, y),
                (x + 14 * u / norm, y + 14 * v / norm),
                (101, 211, 226),
                1,
            )
    return image


def _object_preview(payload: dict[str, Any]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (10, 25, 31))
    draw = ImageDraw.Draw(image)
    colors = (
        (242, 171, 67),
        (82, 191, 177),
        (221, 95, 104),
        (116, 146, 224),
        (211, 116, 193),
        (127, 195, 93),
    )
    for index, item in enumerate(payload["items"]):
        geometry = item["geometry"]
        color = colors[index % len(colors)]
        if geometry["kind"] == "polygon":
            draw.polygon(
                [tuple(point) for point in geometry["points"]],
                fill=color,
                outline=(255, 255, 255),
            )
        elif geometry["kind"] == "ellipse":
            draw.ellipse(
                geometry["bbox_xyxy"],
                fill=color,
                outline=(255, 255, 255),
                width=2,
            )
        elif geometry["kind"] == "polyline":
            draw.line(
                [tuple(point) for point in geometry["points"]],
                fill=color,
                width=4,
            )
    return image


def _annotation_layer(
    case_id: str,
    arrows: list[tuple[tuple[int, int], tuple[int, int], str]],
) -> LayerSample:
    payload = {
        "schema_version": "1.0",
        "composite_after_generation": True,
        "items": [
            {
                "annotation_id": f"{case_id}-annotation-{index + 1:02d}",
                "kind": "arrow",
                "start_xy": list(start),
                "end_xy": list(end),
                "meaning_zh": meaning,
            }
            for index, (start, end, meaning) in enumerate(arrows)
        ],
    }
    image = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    colors = ((255, 203, 67), (255, 99, 78), (101, 222, 205))
    for index, item in enumerate(payload["items"]):
        _draw_arrow(
            draw,
            tuple(item["start_xy"]),
            tuple(item["end_xy"]),
            colors[index % len(colors)],
            3,
        )
    return LayerSample(
        layer_id=f"{case_id.lower()}_annotation",
        layer_type="annotation",
        title_zh="教学箭头",
        meaning_zh="程序在写实底图之后叠加的重点箭头。",
        data=payload,
        preview=image,
        model_input_policy="never",
        final_role_zh="帮助解释机制，不输入图像模型。",
    )


def _overlay_annotations(
    clean: Image.Image, annotation: LayerSample, labels: list[tuple[int, int, str]]
) -> Image.Image:
    result = clean.copy()
    array = np.asarray(result, dtype=np.uint8).copy()
    overlay = np.asarray(annotation.preview, dtype=np.uint8)
    mask = overlay.max(axis=2) > 0
    array[mask] = overlay[mask]
    result = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(result)
    for x, y, text in labels:
        box = draw.textbbox((x, y), text)
        draw.rectangle(
            (box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2),
            fill=(13, 38, 43),
        )
        draw.text((x, y), text, fill=(244, 247, 238))
    return result


# ---------------------------------------------------------------------------
# MATH-02: Pythagorean dissection

_A = 90.0
_B = 120.0
_S = _A + _B
_C = 150.0
_SQUARE_ORIGIN = np.array([390.0, 65.0])
_LOCAL_TRIANGLE = np.array([[0.0, 0.0], [_A, 0.0], [0.0, _B]])
_TRIANGLE_COLORS = (
    (221, 102, 83),
    (226, 153, 62),
    (77, 156, 150),
    (91, 121, 183),
)


def _pose_from_target(points: np.ndarray) -> tuple[float, np.ndarray]:
    axis = (points[1] - points[0]) / _A
    angle = math.atan2(float(axis[1]), float(axis[0]))
    return angle, points[0]


def _apply_pose(angle: float, translation: np.ndarray) -> np.ndarray:
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    return _LOCAL_TRIANGLE @ rotation.T + translation


def _interpolate_pose(
    first: tuple[float, np.ndarray],
    last: tuple[float, np.ndarray],
    amount: float,
) -> tuple[float, np.ndarray]:
    first_angle, first_translation = first
    last_angle, last_translation = last
    delta = (last_angle - first_angle + math.pi) % (2 * math.pi) - math.pi
    return (
        first_angle + delta * amount,
        first_translation * (1 - amount) + last_translation * amount,
    )


def _sequential_poses(
    first: list[tuple[float, np.ndarray]],
    last: list[tuple[float, np.ndarray]],
    local_progress: float,
    order: tuple[int, ...],
) -> list[tuple[float, np.ndarray]]:
    """Move one rigid piece at a time so the proof stays visually traceable."""

    rank = {piece_index: position for position, piece_index in enumerate(order)}
    poses = []
    for piece_index, (first_pose, last_pose) in enumerate(zip(first, last)):
        amount = _smoothstep(
            local_progress * len(order) - rank[piece_index]
        )
        poses.append(_interpolate_pose(first_pose, last_pose, amount))
    return poses


def _math_layouts() -> tuple[list[tuple[float, np.ndarray]], ...]:
    start = [
        (0.0, np.array([25.0, 35.0])),
        (0.0, np.array([165.0, 35.0])),
        (0.0, np.array([25.0, 195.0])),
        (0.0, np.array([165.0, 195.0])),
    ]
    a_targets = [
        np.array([[0, 0], [_A, 0], [0, _B]], dtype=float),
        np.array([[_S, 0], [_S, _A], [_A, 0]], dtype=float),
        np.array([[_S, _S], [_B, _S], [_S, _A]], dtype=float),
        np.array([[0, _S], [0, _B], [_B, _S]], dtype=float),
    ]
    b_targets = [
        np.array([[_S, 0], [_S, _A], [_A, 0]], dtype=float),
        np.array([[_A, _A], [_A, 0], [_S, _A]], dtype=float),
        np.array([[0, _A], [_A, _A], [0, _S]], dtype=float),
        np.array([[_A, _S], [0, _S], [_A, _A]], dtype=float),
    ]
    layout_a = [
        _pose_from_target(points + _SQUARE_ORIGIN) for points in a_targets
    ]
    layout_b = [
        _pose_from_target(points + _SQUARE_ORIGIN) for points in b_targets
    ]
    return start, layout_a, layout_b


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return abs(
        float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    ) / 2


def _polygon_mask(polygons: list[np.ndarray]) -> np.ndarray:
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    for points in polygons:
        draw.polygon([tuple(point) for point in points], fill=255)
    return np.asarray(image, dtype=np.uint8)


def _polygon_outlines(
    polygons: list[np.ndarray], *, width: int = 2
) -> np.ndarray:
    """Preserve every piece boundary, including edges shared by two pieces."""
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    for points in polygons:
        closed = [tuple(point) for point in points]
        closed.append(closed[0])
        draw.line(closed, fill=255, width=width, joint="curve")
    return np.asarray(image, dtype=np.uint8)


def _sample_math(progress: float) -> ProgramSample:
    start, layout_a, layout_b = _math_layouts()
    segment, local = _segment(progress)
    if segment == 0:
        poses = _sequential_poses(
            start, layout_a, local, (1, 0, 3, 2)
        )
        stage = "assemble_c2"
    elif segment == 1:
        poses = _sequential_poses(
            layout_a, start, local, (2, 3, 0, 1)
        )
        stage = "return_to_staging"
    else:
        poses = _sequential_poses(
            start, layout_b, local, (1, 0, 3, 2)
        )
        stage = "assemble_a2_b2"
    if math.isclose(progress, 1 / 3):
        stage = "assembled_c2"
    elif math.isclose(progress, 2 / 3):
        stage = "same_pieces_staged"
    elif math.isclose(progress, 1.0):
        stage = "assembled_a2_b2"
    polygons = [_apply_pose(angle, translation) for angle, translation in poses]
    triangle_mask = _polygon_mask(polygons)
    outer = Image.new("L", (WIDTH, HEIGHT), 0)
    draw_outer = ImageDraw.Draw(outer)
    square_box = (
        int(_SQUARE_ORIGIN[0]),
        int(_SQUARE_ORIGIN[1]),
        int(_SQUARE_ORIGIN[0] + _S),
        int(_SQUARE_ORIGIN[1] + _S),
    )
    draw_outer.rectangle(square_box, fill=255)
    outer_array = np.asarray(outer, dtype=np.uint8)
    leftover = np.uint8((outer_array > 0) & (triangle_mask == 0)) * 255
    # Edging the union of all triangles erases shared internal edges.  Those
    # edges encode the piece count and are mechanism-critical, so retain the
    # outline of every semantic object separately.
    hard = np.maximum(_edge(outer_array), _polygon_outlines(polygons))

    objects = {
        "schema_version": "1.0",
        "coordinate_system": "pixel_xy_top_left",
        "items": [
            {
                "object_id": f"MATH-02-triangle-{index + 1:02d}",
                "class_id": "congruent_right_triangle",
                "geometry": {
                    "kind": "polygon",
                    "points": [
                        [round(float(x), 4), round(float(y), 4)]
                        for x, y in polygon
                    ],
                },
                "area_px2": round(_polygon_area(polygon), 4),
            }
            for index, polygon in enumerate(polygons)
        ],
    }
    clean = Image.new("RGB", (WIDTH, HEIGHT), (236, 231, 217))
    draw = ImageDraw.Draw(clean)
    draw.rectangle(square_box, fill=(249, 247, 235), outline=(42, 78, 82), width=3)
    region_color = (218, 221, 184)
    region_image = Image.new("RGB", (WIDTH, HEIGHT), region_color)
    clean.paste(region_image, mask=Image.fromarray(leftover, mode="L"))
    draw = ImageDraw.Draw(clean)
    for color, polygon in zip(_TRIANGLE_COLORS, polygons):
        draw.polygon(
            [tuple(point) for point in polygon],
            fill=color,
            outline=(255, 250, 235),
            width=2,
        )
    annotation = _annotation_layer(
        "MATH-02",
        [
            ((330, 52), (430, 105), "指出当前剩余面积"),
            ((322, 320), (480, 270), "追踪同一组拼图片"),
        ],
    )
    labels = []
    for index, polygon in enumerate(polygons):
        center = polygon.mean(axis=0)
        labels.append((int(center[0]) - 7, int(center[1]) - 6, f"T{index + 1}"))
    if progress < 0.30:
        labels.append((460, 166, "(a+b)^2 board"))
    elif progress < 0.38:
        labels.append((485, 166, "c^2"))
    elif 0.62 < progress < 0.71:
        labels.append((452, 166, "same 4 pieces"))
    elif progress > 0.96:
        labels.extend(((425, 102, "a^2"), (515, 205, "b^2")))
    program = _overlay_annotations(clean, annotation, labels)
    state = {
        "case_id": "MATH-02",
        "progress": round(progress, 6),
        "stage": stage,
        "triangle_count": 4,
        "triangle_area_px2": _A * _B / 2,
        "outer_area_px2": _S * _S,
        "remaining_area_px2": _S * _S - 4 * (_A * _B / 2),
        "expected_c2_px2": _C * _C,
        "expected_a2_plus_b2_px2": _A * _A + _B * _B,
        "objects": objects["items"],
    }
    layers = (
        LayerSample(
            "math02_hard_boundary",
            "hard_boundary",
            "拼图与外框硬边界",
            "精确保存大正方形和四个全等三角形的轮廓。",
            hard,
            _binary_preview(hard, (255, 255, 255)),
            "allowed_by_route",
            "防止图像模型改变三角形数量、边长或外框。",
        ),
        LayerSample(
            "math02_remaining_region",
            "region",
            "当前剩余面积",
            "外框内未被四个三角形占用的区域。",
            leftover,
            _binary_preview(leftover, (97, 194, 151)),
            "allowed_by_route",
            "显示 c² 或 a²+b² 的面积证据。",
        ),
        LayerSample(
            "math02_piece_region",
            "region",
            "四块拼图占用区",
            "四个三角形当前占用的像素并集；共享边仍由 hard_boundary 单独保存。",
            triangle_mask,
            _binary_preview(triangle_mask, (225, 150, 73)),
            "allowed_by_route",
            "限定材质增强只能发生在拼图内部，空区和背景保持逐像素不变。",
        ),
        LayerSample(
            "math02_piece_identity",
            "object_identity",
            "四块拼图身份",
            "四个三角形的稳定 ID、顶点和面积。",
            objects,
            _object_preview(objects),
            "allowed_by_route",
            "验证只发生刚体移动，没有复制或拉伸。",
        ),
        annotation,
    )
    return ProgramSample(state, clean, program, layers)


def _validate_math(samples: list[ProgramSample]) -> list[dict[str, Any]]:
    expected_area = _A * _B / 2
    side_lengths = sorted((_A, _B, _C))
    rigid = True
    identities = []
    for sample in samples:
        items = sample.state["objects"]
        identities.append([item["object_id"] for item in items])
        for item in items:
            points = np.asarray(item["geometry"]["points"], dtype=float)
            lengths = sorted(
                math.dist(points[index], points[(index + 1) % 3])
                for index in range(3)
            )
            rigid &= math.isclose(
                item["area_px2"], expected_area, abs_tol=0.1
            ) and all(
                math.isclose(actual, expected, abs_tol=0.1)
                for actual, expected in zip(lengths, side_lengths)
            )
    return [
        {
            "name": "four_stable_triangle_identities",
            "passed": all(ids == identities[0] for ids in identities)
            and len(identities[0]) == 4,
            "evidence": identities[0],
        },
        {
            "name": "triangles_move_rigidly",
            "passed": rigid,
            "evidence": {
                "area_px2": expected_area,
                "side_lengths_px": side_lengths,
            },
        },
        {
            "name": "pythagorean_remaining_area",
            "passed": math.isclose(_C * _C, _A * _A + _B * _B),
            "evidence": {
                "c2": _C * _C,
                "a2_plus_b2": _A * _A + _B * _B,
            },
        },
    ]


# ---------------------------------------------------------------------------
# PHYS-01: coherent two-source water-wave interference


_WAVE_SOURCES = np.array([[235.0, 180.0], [405.0, 180.0]])
_WAVELENGTH = 30.0
_WAVE_NUMBER = 2 * math.pi / _WAVELENGTH


def _wave_fields(progress: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    radius = 60.0 + 105.0 * progress
    temporal_phase = 2 * math.pi * progress * 2.2
    waves = []
    amplitudes = []
    active_masks = []
    for source_x, source_y in _WAVE_SOURCES:
        distance = np.hypot(x - source_x, y - source_y)
        active = 1.0 / (1.0 + np.exp((distance - radius) / 1.8))
        amplitude = active / np.sqrt(1.0 + distance / 70.0)
        wave = amplitude * np.cos(
            _WAVE_NUMBER * distance - temporal_phase
        )
        waves.append(wave)
        amplitudes.append(amplitude)
        active_masks.append(active > 0.5)
    height = np.asarray(waves[0] + waves[1], dtype=np.float32)
    phase_difference = _WAVE_NUMBER * (
        np.hypot(x - _WAVE_SOURCES[0, 0], y - _WAVE_SOURCES[0, 1])
        - np.hypot(x - _WAVE_SOURCES[1, 0], y - _WAVE_SOURCES[1, 1])
    )
    envelope = np.sqrt(
        np.maximum(
            0,
            amplitudes[0] ** 2
            + amplitudes[1] ** 2
            + 2
            * amplitudes[0]
            * amplitudes[1]
            * np.cos(phase_difference),
        )
    ).astype(np.float32)
    overlap = np.asarray(active_masks[0] & active_masks[1], dtype=np.uint8)
    return height, envelope, overlap, radius


def _sample_wave(progress: float) -> ProgramSample:
    height, envelope, overlap, radius = _wave_fields(progress)
    grad_y, grad_x = np.gradient(height)
    light = np.clip(
        0.54 + height * 0.22 - grad_x * 0.85 - grad_y * 0.35,
        0,
        1,
    )
    rgb = np.stack(
        (
            42 + 70 * light,
            120 + 92 * light,
            145 + 95 * light,
        ),
        axis=-1,
    )
    clean = Image.fromarray(np.uint8(np.clip(rgb, 0, 255)), mode="RGB")
    draw = ImageDraw.Draw(clean)
    draw.rectangle((7, 7, WIDTH - 8, HEIGHT - 8), outline=(222, 244, 235), width=3)
    for x, y in _WAVE_SOURCES:
        draw.ellipse(
            (x - 7, y - 7, x + 7, y + 7),
            fill=(234, 159, 60),
            outline=(255, 250, 225),
            width=2,
        )
    node_mask = np.uint8(
        (overlap > 0) & (envelope < 0.16)
    ) * 255
    antinode_mask = np.uint8(
        (overlap > 0) & (envelope > 1.05)
    ) * 255
    annotation = _annotation_layer(
        "PHYS-01",
        [
            ((80, 55), (235, 180), "固定振源一"),
            ((555, 55), (405, 180), "固定振源二"),
            ((320, 330), (320, 220), "重叠区中的节点与腹线"),
        ],
    )
    program = _overlay_annotations(
        clean,
        annotation,
        [
            (220, 192, "S1"),
            (410, 192, "S2"),
            (16, 16, f"R={radius:.0f}px"),
        ],
    )
    program_array = np.asarray(program, dtype=np.uint8).copy()
    node_edge = _edge(node_mask)
    antinode_edge = _edge(antinode_mask)
    program_array[node_edge > 0] = (255, 225, 89)
    program_array[antinode_edge > 0] = (255, 102, 80)
    program = Image.fromarray(program_array, mode="RGB")
    objects = {
        "schema_version": "1.0",
        "coordinate_system": "pixel_xy_top_left",
        "items": [
            {
                "object_id": f"PHYS-01-source-{index + 1:02d}",
                "class_id": "coherent_point_source",
                "geometry": {
                    "kind": "ellipse",
                    "bbox_xyxy": [
                        source[0] - 7,
                        source[1] - 7,
                        source[0] + 7,
                        source[1] + 7,
                    ],
                },
                "frequency_relative": 1.0,
                "phase_radians": 0.0,
            }
            for index, source in enumerate(_WAVE_SOURCES)
        ],
    }
    overlap_values = envelope[overlap > 0]
    state = {
        "case_id": "PHYS-01",
        "progress": round(progress, 6),
        "source_count": 2,
        "sources_xy": _WAVE_SOURCES.tolist(),
        "wavefront_radius_px": round(radius, 5),
        "wavelength_px": _WAVELENGTH,
        "overlap_pixel_count": int(overlap.sum()),
        "node_pixel_count": int(node_mask.astype(bool).sum()),
        "antinode_pixel_count": int(antinode_mask.astype(bool).sum()),
        "overlap_envelope_min": (
            round(float(overlap_values.min()), 6)
            if overlap_values.size
            else None
        ),
        "overlap_envelope_max": (
            round(float(overlap_values.max()), 6)
            if overlap_values.size
            else None
        ),
    }
    layers = (
        LayerSample(
            "phys01_amplitude_envelope",
            "scalar_field",
            "干涉振幅包络",
            "显示哪些重叠位置长期振幅较大或接近零。",
            envelope,
            _scalar_preview(envelope),
            "allowed_by_route",
            "验证相长区、相消区和节点拓扑。",
        ),
        LayerSample(
            "phys01_surface_height",
            "height_or_normal",
            "瞬时水面高度",
            "两列波在当前时刻线性叠加后的高度。",
            height,
            _normal_preview(height),
            "allowed_by_route",
            "程序保留波峰波谷，图像模型以后只补水面材质。",
        ),
        LayerSample(
            "phys01_source_identity",
            "object_identity",
            "两个相干振源",
            "保存两个固定点源的位置、频率和相位。",
            objects,
            _object_preview(objects),
            "allowed_by_route",
            "防止增加第三个振源或移动源位置。",
        ),
        annotation,
    )
    return ProgramSample(state, clean, program, layers)


def _validate_wave(samples: list[ProgramSample]) -> list[dict[str, Any]]:
    source_positions = [sample.state["sources_xy"] for sample in samples]
    overlap = [sample.state["overlap_pixel_count"] for sample in samples]
    node_counts = [sample.state["node_pixel_count"] for sample in samples]
    antinode_counts = [
        sample.state["antinode_pixel_count"] for sample in samples
    ]
    return [
        {
            "name": "two_sources_are_fixed",
            "passed": all(
                positions == source_positions[0]
                for positions in source_positions
            )
            and len(source_positions[0]) == 2,
            "evidence": source_positions[0],
        },
        {
            "name": "wavefronts_progress_into_overlap",
            "passed": overlap[0] == 0
            and all(b >= a for a, b in zip(overlap, overlap[1:]))
            and overlap[-1] > 0,
            "evidence": overlap,
        },
        {
            "name": "nodes_and_antinodes_exist_after_overlap",
            "passed": node_counts[-1] > 0 and antinode_counts[-1] > 0,
            "evidence": {
                "nodes": node_counts,
                "antinodes": antinode_counts,
            },
        },
        {
            "name": "boundary_reflection_excluded",
            "passed": samples[-1].state["wavefront_radius_px"] < 175,
            "evidence": samples[-1].state["wavefront_radius_px"],
        },
    ]


# ---------------------------------------------------------------------------
# CHEM-01: strong-acid/strong-base titration with phenolphthalein


_ACID_MOLES = 0.0005
_INITIAL_VOLUME_L = 0.100
_BASE_MOLARITY = 0.1
_TITRATION_KEY_VOLUMES_ML = (0.0, 0.05, 4.8, 5.01)
_TITRATION_KEY_PLUME = (0.0, 1.0, 0.0, 0.0)


def _bulk_ph(base_volume_ml: float) -> float:
    base_moles = _BASE_MOLARITY * base_volume_ml / 1000.0
    total_volume = _INITIAL_VOLUME_L + base_volume_ml / 1000.0
    difference = base_moles - _ACID_MOLES
    if abs(difference) < 1e-12:
        return 7.0
    if difference < 0:
        return -math.log10((-difference) / total_volume)
    poh = -math.log10(difference / total_volume)
    return 14.0 - poh


def _titration_parameters(progress: float) -> tuple[float, float]:
    index, local = _segment(progress)
    amount = _smoothstep(local)
    volume = (
        _TITRATION_KEY_VOLUMES_ML[index] * (1 - amount)
        + _TITRATION_KEY_VOLUMES_ML[index + 1] * amount
    )
    plume = (
        _TITRATION_KEY_PLUME[index] * (1 - amount)
        + _TITRATION_KEY_PLUME[index + 1] * amount
    )
    return volume, plume


def _sample_titration(progress: float) -> ProgramSample:
    base_volume_ml, plume_strength = _titration_parameters(progress)
    bulk_ph = _bulk_ph(base_volume_ml)
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    beaker_left, beaker_right = 180, 460
    beaker_top, beaker_bottom = 72, 305
    level_y = int(238 - base_volume_ml * 0.8)
    liquid = (
        (x >= beaker_left + 8)
        & (x <= beaker_right - 8)
        & (y >= level_y)
        & (y <= beaker_bottom - 8)
    )
    plume_center_x = 320 + 18 * math.sin(progress * math.pi * 5)
    plume_center_y = level_y + 34
    # During the local-color-to-clear segment, represent mixing as a
    # spreading scalar field: spatial variance grows while peak amplitude
    # and its integrated proxy both fall. The scale is exactly 1 at K1, so
    # the frozen Phase 2 endpoints remain unchanged.
    mixing_local = float(
        np.clip((progress - 1 / 3) * 3, 0.0, 1.0)
    )
    mixing_amount = _smoothstep(mixing_local)
    plume_spread_factor = 1.0 + 0.9 * mixing_amount
    effective_plume_amplitude = (
        plume_strength / plume_spread_factor
    )
    gaussian = np.exp(
        -(
            (
                (x - plume_center_x)
                / (48 * plume_spread_factor)
            )
            ** 2
            + (
                (y - plume_center_y)
                / (72 * plume_spread_factor)
            )
            ** 2
        )
        * 2.0
    )
    ph_field = np.full((HEIGHT, WIDTH), bulk_ph, dtype=np.float32)
    ph_field = np.where(
        liquid,
        np.maximum(
            ph_field,
            bulk_ph + effective_plume_amplitude * gaussian * 9.0,
        ),
        0.0,
    ).astype(np.float32)
    indicator = 1.0 / (1.0 + np.exp(-(ph_field - 8.2) / 0.22))
    indicator *= liquid
    # Phenolphthalein itself has a sharp transition. For the clean teaching
    # frame, add a low-opacity concentration halo only between K1 and K2 so
    # viewers can see the diluted plume spread before becoming colorless.
    # The halo is zero at both endpoints and cannot alter frozen keyframes.
    diffusion_halo_strength = (
        1.2 * mixing_amount * (1.0 - mixing_amount)
    )
    visible_indicator = np.maximum(
        indicator,
        diffusion_halo_strength * gaussian * liquid,
    )

    boundary_image = Image.new("L", (WIDTH, HEIGHT), 0)
    boundary_draw = ImageDraw.Draw(boundary_image)
    boundary_draw.line(
        (
            beaker_left,
            beaker_top,
            beaker_left,
            beaker_bottom,
            beaker_right,
            beaker_bottom,
            beaker_right,
            beaker_top,
        ),
        fill=255,
        width=4,
    )
    boundary_draw.line((300, 20, 300, 94), fill=255, width=5)
    boundary_draw.line((340, 20, 340, 94), fill=255, width=5)
    hard = np.asarray(boundary_image, dtype=np.uint8)
    region = np.uint8(liquid) * 255

    clean = Image.new("RGB", (WIDTH, HEIGHT), (227, 222, 231))
    draw = ImageDraw.Draw(clean, "RGBA")
    draw.rectangle((300, 12, 340, 95), fill=(225, 239, 240, 120), outline=(75, 91, 100, 255), width=3)
    draw.rectangle(
        (beaker_left, beaker_top, beaker_right, beaker_bottom),
        fill=(220, 240, 244, 68),
        outline=(67, 93, 102, 255),
        width=4,
    )
    fluid = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    fluid[:, :, :3] = np.array([179, 218, 224], dtype=np.uint8)
    fluid[:, :, 3] = np.uint8(liquid * 150)
    pink = np.zeros_like(fluid)
    pink[:, :, :3] = np.array([232, 88, 153], dtype=np.uint8)
    pink[:, :, 3] = np.uint8(
        np.clip(visible_indicator * 175, 0, 175)
    )
    clean = Image.alpha_composite(clean.convert("RGBA"), Image.fromarray(fluid, mode="RGBA"))
    clean = Image.alpha_composite(clean, Image.fromarray(pink, mode="RGBA")).convert("RGB")
    draw = ImageDraw.Draw(clean, "RGBA")
    if plume_strength > 0.08:
        drop_y = int(58 + 70 * min(progress * 3, 1))
        draw.ellipse((313, drop_y - 7, 327, drop_y + 7), fill=(200, 229, 238, 220))
    draw.line((beaker_left + 8, level_y, beaker_right - 8, level_y), fill=(239, 252, 250, 190), width=2)

    objects_items = [
        {
            "object_id": "CHEM-01-vessel-01",
            "class_id": "glass_beaker",
            "geometry": {
                "kind": "polygon",
                "points": [
                    [beaker_left, beaker_top],
                    [beaker_right, beaker_top],
                    [beaker_right, beaker_bottom],
                    [beaker_left, beaker_bottom],
                ],
            },
        }
    ]
    if plume_strength > 0.08:
        objects_items.append(
            {
                "object_id": "CHEM-01-drop-current",
                "class_id": "base_solution_drop",
                "geometry": {
                    "kind": "ellipse",
                    "bbox_xyxy": [313, 51, 327, 65],
                },
            }
        )
    objects = {
        "schema_version": "1.0",
        "coordinate_system": "pixel_xy_top_left",
        "items": objects_items,
    }
    annotation = _annotation_layer(
        "CHEM-01",
        [
            ((95, 95), (320, level_y + 42), "局部 pH 与颜色区域"),
            ((545, 285), (440, level_y), "液面与总体积"),
        ],
    )
    program = _overlay_annotations(
        clean,
        annotation,
        [
            (20, 18, f"Vb={base_volume_ml:.2f}mL"),
            (20, 36, f"bulk pH={bulk_ph:.2f}"),
        ],
    )
    base_moles = _BASE_MOLARITY * base_volume_ml / 1000.0
    state = {
        "case_id": "CHEM-01",
        "progress": round(progress, 6),
        "base_volume_ml": round(base_volume_ml, 6),
        "base_moles": round(base_moles, 10),
        "acid_moles_initial": _ACID_MOLES,
        "bulk_ph": round(bulk_ph, 6),
        "plume_strength": round(plume_strength, 6),
        "plume_spread_factor": round(plume_spread_factor, 6),
        "plume_peak_amplitude": round(
            effective_plume_amplitude, 6
        ),
        "plume_integrated_proxy": round(
            plume_strength * plume_spread_factor, 6
        ),
        "diffusion_halo_strength": round(
            diffusion_halo_strength, 6
        ),
        "indicator_mean_inside_liquid": round(
            float(indicator[liquid].mean()), 6
        ),
        "liquid_level_y": level_y,
    }
    layers = (
        LayerSample(
            "chem01_apparatus_boundary",
            "hard_boundary",
            "滴定器材硬边界",
            "固定滴定管与烧杯轮廓，不包含颜色羽流。",
            hard,
            _binary_preview(hard, (255, 255, 255)),
            "allowed_by_route",
            "保持玻璃器材和镜头稳定。",
        ),
        LayerSample(
            "chem01_liquid_region",
            "region",
            "当前液体区域",
            "由加入总体积计算的液面和液体范围。",
            region,
            _binary_preview(region, (77, 188, 173)),
            "allowed_by_route",
            "把透明液体材质限制在烧杯内部。",
        ),
        LayerSample(
            "chem01_ph_field",
            "scalar_field",
            "局部 pH 场",
            "每个液体像素的 pH；非液体区域为零。",
            ph_field,
            _scalar_preview(ph_field),
            "allowed_by_route",
            "决定局部变色与终点持续颜色。",
        ),
        LayerSample(
            "chem01_object_identity",
            "object_identity",
            "滴定器材与当前液滴",
            "固定容器身份，并在出现时追踪当前液滴。",
            objects,
            _object_preview(objects),
            "allowed_by_route",
            "防止模型改变器材或凭空增加液滴。",
        ),
        annotation,
    )
    return ProgramSample(state, clean, program, layers)


def _validate_titration(
    samples: list[ProgramSample],
) -> list[dict[str, Any]]:
    volumes = [sample.state["base_volume_ml"] for sample in samples]
    indicator = [
        sample.state["indicator_mean_inside_liquid"] for sample in samples
    ]
    levels = [sample.state["liquid_level_y"] for sample in samples]
    return [
        {
            "name": "added_base_volume_is_monotonic",
            "passed": all(b >= a for a, b in zip(volumes, volumes[1:])),
            "evidence": volumes,
        },
        {
            "name": "liquid_level_tracks_added_volume",
            "passed": all(b <= a for a, b in zip(levels, levels[1:])),
            "evidence": levels,
        },
        {
            "name": "local_color_then_clear_then_endpoint",
            "passed": indicator[1] > indicator[0]
            and indicator[2] < indicator[1]
            and indicator[3] > indicator[2],
            "evidence": indicator,
        },
        {
            "name": "endpoint_is_slight_base_excess",
            "passed": samples[-1].state["base_moles"] > _ACID_MOLES
            and samples[-1].state["bulk_ph"] > 8.2,
            "evidence": {
                "base_moles": samples[-1].state["base_moles"],
                "acid_moles": _ACID_MOLES,
                "bulk_ph": samples[-1].state["bulk_ph"],
            },
        },
    ]


# ---------------------------------------------------------------------------
# BIO-01: simplified animal-cell mitosis with stable chromosome identities


_CHROMOSOME_COUNT = 6
_INITIAL_CHROMOSOMES = np.array(
    [
        [270, 132],
        [350, 128],
        [294, 180],
        [356, 186],
        [278, 228],
        [342, 232],
    ],
    dtype=float,
)
_META_Y = np.linspace(115, 245, _CHROMOSOME_COUNT)


def _chromosome_positions(progress: float) -> tuple[list[dict[str, Any]], str]:
    if progress <= 1 / 3:
        amount = _smoothstep(progress * 3)
        positions = (
            _INITIAL_CHROMOSOMES * (1 - amount)
            + np.stack(
                (np.full(_CHROMOSOME_COUNT, 320.0), _META_Y),
                axis=1,
            )
            * amount
        )
        items = [
            {
                "object_id": f"BIO-01-chromosome-{index + 1:02d}",
                "parent_id": None,
                "shape": "replicated_x",
                "center_xy": position.tolist(),
                "destination": "unassigned",
            }
            for index, position in enumerate(positions)
        ]
        return items, "prophase_to_metaphase"
    split_amount = _smoothstep((progress - 1 / 3) * 1.5)
    final_amount = _smoothstep(max(0.0, (progress - 2 / 3) * 3))
    items = []
    for index, y in enumerate(_META_Y):
        for side, suffix in ((-1, "A"), (1, "B")):
            anaphase_x = 320 + side * 92 * split_amount
            daughter_x = 210 if side < 0 else 430
            x = anaphase_x * (1 - final_amount) + daughter_x * final_amount
            # Keep all six inherited chromatids visibly distinct in each
            # daughter cell instead of letting nearby line segments overlap.
            target_y = 100.0 + index * 32.0
            current_y = y * (1 - final_amount) + target_y * final_amount
            items.append(
                {
                    "object_id": (
                        f"BIO-01-chromosome-{index + 1:02d}-{suffix}"
                    ),
                    "parent_id": f"BIO-01-chromosome-{index + 1:02d}",
                    "shape": "sister_chromatid",
                    "center_xy": [float(x), float(current_y)],
                    "destination": "left" if side < 0 else "right",
                }
            )
    stage = (
        "sister_separation"
        if progress <= 2 / 3
        else "anaphase_to_cytokinesis"
    )
    return items, stage


def _cell_region(progress: float) -> np.ndarray:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    if progress < 0.76:
        return np.uint8(
            ((x - 320) / 172) ** 2 + ((y - 180) / 137) ** 2 <= 1
        ) * 255
    split = _smoothstep((progress - 0.76) / 0.24)
    left_center = 320 * (1 - split) + 210 * split
    right_center = 320 * (1 - split) + 430 * split
    radius_x = 172 * (1 - split) + 105 * split
    left = ((x - left_center) / radius_x) ** 2 + (
        (y - 180) / 125
    ) ** 2 <= 1
    right = ((x - right_center) / radius_x) ** 2 + (
        (y - 180) / 125
    ) ** 2 <= 1
    return np.uint8(left | right) * 255


def _sample_mitosis(progress: float) -> ProgramSample:
    items, stage = _chromosome_positions(progress)
    region = _cell_region(progress)
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    vector = np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32)
    inside = region > 0
    vector[:, :, 0] = np.where(inside, np.sign(x - 320), 0)
    vector[:, :, 1] = np.where(
        inside, 0.12 * np.sin((y - 180) / 28), 0
    )
    objects = {
        "schema_version": "1.0",
        "coordinate_system": "pixel_xy_top_left",
        "items": [],
    }
    for item in items:
        cx, cy = item["center_xy"]
        if item["shape"] == "replicated_x":
            points = [
                [cx - 8, cy - 12],
                [cx, cy],
                [cx + 8, cy - 12],
                [cx, cy],
                [cx - 8, cy + 12],
                [cx, cy],
                [cx + 8, cy + 12],
            ]
        else:
            points = [[cx - 3, cy - 12], [cx + 3, cy + 12]]
        objects["items"].append(
            {
                **item,
                "geometry": {"kind": "polyline", "points": points},
            }
        )
    clean = Image.new("RGB", (WIDTH, HEIGHT), (218, 229, 207))
    draw = ImageDraw.Draw(clean, "RGBA")
    mask_image = Image.fromarray(region, mode="L")
    cytoplasm = Image.new("RGBA", (WIDTH, HEIGHT), (143, 187, 151, 185))
    clean_rgba = clean.convert("RGBA")
    clean_rgba.paste(cytoplasm, mask=mask_image)
    clean = clean_rgba.convert("RGB")
    draw = ImageDraw.Draw(clean, "RGBA")
    edge = _edge(region)
    edge_y, edge_x = np.where(edge > 0)
    for px, py in zip(edge_x[::2], edge_y[::2]):
        draw.point((int(px), int(py)), fill=(44, 101, 66, 255))
    draw.ellipse((195, 166, 215, 186), fill=(67, 105, 132, 255))
    draw.ellipse((425, 166, 445, 186), fill=(67, 105, 132, 255))
    colors = {
        "replicated_x": (110, 55, 132, 255),
        "sister_chromatid": (137, 60, 145, 255),
    }
    for item in objects["items"]:
        points = [tuple(point) for point in item["geometry"]["points"]]
        draw.line(points, fill=colors[item["shape"]], width=5)
    for pole_x in (205, 435):
        for item in objects["items"]:
            cx, cy = item["center_xy"]
            if (
                item["destination"] == "left"
                and pole_x == 205
                or item["destination"] == "right"
                and pole_x == 435
            ):
                draw.line((pole_x, 176, cx, cy), fill=(84, 133, 135, 95), width=1)
    annotation = _annotation_layer(
        "BIO-01",
        [
            ((72, 72), (205, 176), "左侧纺锤极"),
            ((565, 72), (435, 176), "右侧纺锤极"),
            ((320, 332), (320, 225), "染色体身份与分离"),
        ],
    )
    labels = [(16, 16, stage)]
    if progress <= 1 / 3:
        labels.append((300, 85, "equator"))
    program = _overlay_annotations(clean, annotation, labels)
    sister_items = [
        item for item in items if item["shape"] == "sister_chromatid"
    ]
    state = {
        "case_id": "BIO-01",
        "progress": round(progress, 6),
        "stage": stage,
        "replicated_chromosome_count": (
            len(items) if not sister_items else 0
        ),
        "sister_chromatid_count": len(sister_items),
        "left_destination_count": sum(
            item["destination"] == "left" for item in items
        ),
        "right_destination_count": sum(
            item["destination"] == "right" for item in items
        ),
        "objects": items,
    }
    layers = (
        LayerSample(
            "bio01_cell_region",
            "region",
            "细胞与子细胞区域",
            "细胞膜以内的区域，后期从一个连通体变为两个。",
            region,
            _binary_preview(region, (85, 185, 119)),
            "allowed_by_route",
            "控制细胞形变和分裂拓扑。",
        ),
        LayerSample(
            "bio01_spindle_direction",
            "vector_field",
            "纺锤体方向场",
            "表示染色单体分别向左右两极移动的方向。",
            vector,
            _vector_preview(vector),
            "allowed_by_route",
            "为动画和视频检查提供运动方向。",
        ),
        LayerSample(
            "bio01_chromosome_identity",
            "object_identity",
            "染色体与姐妹染色单体身份",
            "保存父染色体、姐妹 ID、当前位置和目的地。",
            objects,
            _object_preview(objects),
            "allowed_by_route",
            "防止遗传物质被模型复制或吞掉。",
        ),
        annotation,
    )
    return ProgramSample(state, clean, program, layers)


def _validate_mitosis(samples: list[ProgramSample]) -> list[dict[str, Any]]:
    stage_order = [sample.state["stage"] for sample in samples]
    parent_counts = [
        sample.state["replicated_chromosome_count"] for sample in samples
    ]
    sister_counts = [
        sample.state["sister_chromatid_count"] for sample in samples
    ]
    final = samples[-1].state
    parent_mappings = Counter(
        item["parent_id"]
        for item in final["objects"]
        if item["parent_id"]
    )
    return [
        {
            "name": "mitosis_stage_order",
            "passed": stage_order == [
                "prophase_to_metaphase",
                "prophase_to_metaphase",
                "sister_separation",
                "anaphase_to_cytokinesis",
            ],
            "evidence": stage_order,
        },
        {
            "name": "chromosome_to_sister_mapping",
            "passed": parent_counts[:2] == [6, 6]
            and sister_counts[2:] == [12, 12]
            and len(parent_mappings) == 6
            and set(parent_mappings.values()) == {2},
            "evidence": {
                "parent_counts": parent_counts,
                "sister_counts": sister_counts,
                "sisters_per_parent": dict(parent_mappings),
            },
        },
        {
            "name": "equal_daughter_allocation",
            "passed": final["left_destination_count"] == 6
            and final["right_destination_count"] == 6,
            "evidence": {
                "left": final["left_destination_count"],
                "right": final["right_destination_count"],
            },
        },
    ]


# ---------------------------------------------------------------------------
# GEO-02: orographic rainfall and rain shadow


def _terrain_profile() -> np.ndarray:
    x = np.linspace(0, 1, WIDTH, dtype=np.float32)
    mountain = np.exp(-((x - 0.52) / 0.19) ** 2 * 2.2)
    foothills = 0.12 * np.exp(-((x - 0.25) / 0.13) ** 2 * 2.0)
    return np.asarray(0.06 + 0.78 * mountain + foothills, dtype=np.float32)


_OROGRAPHIC_X = (0.14, 0.38, 0.51, 0.80)
_OROGRAPHIC_TEMP = (22.0, 16.0, 10.5, 20.0)
_OROGRAPHIC_HUMIDITY = (0.72, 0.96, 1.0, 0.43)
_OROGRAPHIC_RAIN = (0.0, 0.05, 1.0, 0.12)


def _orographic_parameters(progress: float) -> tuple[float, float, float, float]:
    index, local = _segment(progress)
    amount = _smoothstep(local)
    values = []
    for key_values in (
        _OROGRAPHIC_X,
        _OROGRAPHIC_TEMP,
        _OROGRAPHIC_HUMIDITY,
        _OROGRAPHIC_RAIN,
    ):
        values.append(
            key_values[index] * (1 - amount)
            + key_values[index + 1] * amount
        )
    return tuple(values)  # type: ignore[return-value]


def _sample_orographic(progress: float) -> ProgramSample:
    parcel_x, temperature, humidity, rain_strength = _orographic_parameters(
        progress
    )
    terrain = _terrain_profile()
    ground_y = 318 - terrain * 215
    x_pixels = np.arange(WIDTH)
    terrain_field = np.repeat(terrain[None, :], HEIGHT, axis=0)
    slope = np.gradient(ground_y)
    vector = np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32)
    vector[:, :, 0] = 1.0
    vector[:, :, 1] = np.repeat(slope[None, :] * 0.9, HEIGHT, axis=0)
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    terrain_region = np.uint8(
        y >= ground_y[None, :]
    ) * 255
    parcel_px = parcel_x * (WIDTH - 1)
    parcel_ground = float(np.interp(parcel_px, x_pixels, ground_y))
    parcel_y = max(62.0, parcel_ground - 72.0)
    parcel_objects = {
        "schema_version": "1.0",
        "coordinate_system": "pixel_xy_top_left",
        "items": [
            {
                "object_id": "GEO-02-air-parcel",
                "class_id": "moving_air_parcel",
                "geometry": {
                    "kind": "ellipse",
                    "bbox_xyxy": [
                        parcel_px - 7,
                        parcel_y - 7,
                        parcel_px + 7,
                        parcel_y + 7,
                    ],
                },
            }
        ],
    }
    cloud = np.exp(
        -(
            ((x - parcel_px) / 95) ** 2
            + ((y - parcel_y) / 48) ** 2
        )
        * 2.0
    ) * humidity
    rain_center_x = WIDTH * 0.43
    rain = (
        np.exp(
            -(
                ((x - rain_center_x) / 75) ** 2
                + ((y - 190) / 105) ** 2
            )
            * 1.6
        )
        * rain_strength
        * (x < WIDTH * 0.54)
    ).astype(np.float32)
    humidity_field = np.asarray(np.clip(cloud + rain * 0.7, 0, 1), dtype=np.float32)

    clean = Image.new("RGB", (WIDTH, HEIGHT), (163, 208, 224))
    draw = ImageDraw.Draw(clean, "RGBA")
    terrain_points = [(0, HEIGHT), *zip(x_pixels.tolist(), ground_y.tolist()), (WIDTH - 1, HEIGHT)]
    draw.polygon(terrain_points, fill=(111, 137, 82, 255))
    draw.line(
        list(zip(x_pixels.tolist(), ground_y.tolist())),
        fill=(72, 91, 61, 255),
        width=3,
    )
    cloud_mask = np.uint8(np.clip(cloud * 205, 0, 205))
    cloud_rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    cloud_rgba[:, :, :3] = np.array([236, 241, 239], dtype=np.uint8)
    cloud_rgba[:, :, 3] = cloud_mask
    clean = Image.alpha_composite(
        clean.convert("RGBA"), Image.fromarray(cloud_rgba, mode="RGBA")
    ).convert("RGB")
    draw = ImageDraw.Draw(clean, "RGBA")
    rain_points_y, rain_points_x = np.where(rain > 0.28)
    for px, py in zip(rain_points_x[::180], rain_points_y[::180]):
        draw.line((int(px), int(py), int(px - 3), int(py + 10)), fill=(54, 120, 181, 170), width=2)
    draw.ellipse(
        (parcel_px - 7, parcel_y - 7, parcel_px + 7, parcel_y + 7),
        fill=(243, 161, 67, 255),
        outline=(255, 248, 226, 255),
        width=2,
    )
    annotation = _annotation_layer(
        "GEO-02",
        [
            ((55, 85), (220, 150), "迎风侧湿空气抬升"),
            ((565, 82), (465, 150), "背风侧下沉增温"),
            ((325, 28), (335, 115), "凝结与迎风坡降水"),
        ],
    )
    program = _overlay_annotations(
        clean,
        annotation,
        [
            (12, 16, f"T={temperature:.1f}C"),
            (12, 34, f"RH={humidity:.2f}"),
            (12, 52, f"rain={rain_strength:.2f}"),
        ],
    )
    state = {
        "case_id": "GEO-02",
        "progress": round(progress, 6),
        "parcel_x_ratio": round(parcel_x, 6),
        "parcel_xy": [round(parcel_px, 4), round(parcel_y, 4)],
        "temperature_c": round(temperature, 4),
        "relative_humidity": round(humidity, 6),
        "rain_strength": round(rain_strength, 6),
        "terrain_sha256_surrogate": round(float(terrain.sum()), 8),
        "terrain_peak_x": int(np.argmax(terrain)),
        "rain_centroid_x": (
            round(float((rain * x).sum() / max(float(rain.sum()), 1e-9)), 4)
            if rain_strength > 0
            else None
        ),
    }
    layers = (
        LayerSample(
            "geo02_terrain_height",
            "height_or_normal",
            "固定山地高度",
            "所有帧共享的地形剖面，不能随云雨变化。",
            terrain_field,
            _normal_preview(terrain_field),
            "allowed_by_route",
            "锁定迎风坡、山顶和背风坡位置。",
        ),
        LayerSample(
            "geo02_terrain_region",
            "region",
            "固定山地区域",
            "由同一高度剖面直接填充的地表以下区域。",
            terrain_region,
            _binary_preview(terrain_region, (111, 137, 82)),
            "allowed_by_route",
            "把模型材质残差限制在山地内部，不覆盖天空、云雨或空气团。",
        ),
        LayerSample(
            "geo02_airflow",
            "vector_field",
            "地形约束气流",
            "主风从左向右，并随地形先抬升后下沉。",
            vector,
            _vector_preview(vector),
            "allowed_by_route",
            "验证云和空气不能逆主风运动。",
        ),
        LayerSample(
            "geo02_humidity_cloud_rain",
            "scalar_field",
            "湿度、云和降水强度",
            "连续场表示空气团附近湿度以及迎风坡降水。",
            humidity_field,
            _scalar_preview(humidity_field),
            "allowed_by_route",
            "限制云雨区域并验证背风坡变干。",
        ),
        LayerSample(
            "geo02_parcel_identity",
            "object_identity",
            "移动空气团身份",
            "保存同一个空气团的稳定 ID 和当前椭圆位置。",
            parcel_objects,
            _object_preview(parcel_objects),
            "allowed_by_route",
            "保护空气团不被静态山地材质覆盖。",
        ),
        annotation,
    )
    return ProgramSample(state, clean, program, layers)


def _validate_orographic(
    samples: list[ProgramSample],
) -> list[dict[str, Any]]:
    terrain = [
        sample.state["terrain_sha256_surrogate"] for sample in samples
    ]
    parcel_x = [sample.state["parcel_x_ratio"] for sample in samples]
    peak_x = samples[2].state["terrain_peak_x"]
    rain_centroid = samples[2].state["rain_centroid_x"]
    return [
        {
            "name": "terrain_is_fixed",
            "passed": all(value == terrain[0] for value in terrain),
            "evidence": terrain,
        },
        {
            "name": "air_parcel_moves_with_main_wind",
            "passed": all(b > a for a, b in zip(parcel_x, parcel_x[1:])),
            "evidence": parcel_x,
        },
        {
            "name": "main_rain_is_windward",
            "passed": rain_centroid is not None and rain_centroid < peak_x,
            "evidence": {
                "rain_centroid_x": rain_centroid,
                "terrain_peak_x": peak_x,
            },
        },
        {
            "name": "leeward_air_is_warmer_and_drier",
            "passed": samples[-1].state["relative_humidity"]
            < samples[2].state["relative_humidity"]
            and samples[-1].state["temperature_c"]
            > samples[2].state["temperature_c"],
            "evidence": {
                "windward": {
                    "temperature_c": samples[2].state["temperature_c"],
                    "relative_humidity": samples[2].state[
                        "relative_humidity"
                    ],
                },
                "leeward": {
                    "temperature_c": samples[-1].state["temperature_c"],
                    "relative_humidity": samples[-1].state[
                        "relative_humidity"
                    ],
                },
            },
        },
    ]


PROGRAMS = {
    program.case_id: program
    for program in (
        SentinelProgram(
            "MATH-02",
            "勾股定理的拼图重排证明",
            "四个全等三角形只做刚体运动，剩余面积从 c² 对应到 a²+b²。",
            _sample_math,
            _validate_math,
        ),
        SentinelProgram(
            "PHYS-01",
            "两个同步振源产生的水面波干涉",
            "两列相干波按线性叠加形成传播波前和稳定节点拓扑。",
            _sample_wave,
            _validate_wave,
        ),
        SentinelProgram(
            "CHEM-01",
            "酸碱滴定从局部变色到到达终点",
            "加入体积、化学计量、局部 pH 与指示剂颜色保持一致。",
            _sample_titration,
            _validate_titration,
        ),
        SentinelProgram(
            "BIO-01",
            "一个动物细胞的有丝分裂",
            "六个复制染色体映射到十二条姐妹染色单体并平均分配。",
            _sample_mitosis,
            _validate_mitosis,
        ),
        SentinelProgram(
            "GEO-02",
            "地形雨与背风坡雨影",
            "固定山地上的湿空气先抬升降水，再在背风侧下沉增温变干。",
            _sample_orographic,
            _validate_orographic,
        ),
    )
}
