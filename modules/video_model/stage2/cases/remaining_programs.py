"""Deterministic program plugins for the five Phase 4 expansion cases."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .sentinel_programs import (
    HEIGHT,
    WIDTH,
    LayerSample,
    ProgramSample,
    SentinelProgram,
    _annotation_layer,
    _binary_preview,
    _draw_arrow,
    _edge,
    _normal_preview,
    _object_preview,
    _overlay_annotations,
    _scalar_preview,
    _smoothstep,
    _vector_preview,
)


def _object_layer(
    layer_id: str,
    title: str,
    meaning: str,
    items: list[dict[str, Any]],
    final_role: str,
) -> LayerSample:
    payload = {
        "schema_version": "1.0",
        "coordinate_system": "pixel_xy_top_left",
        "items": items,
    }
    return LayerSample(
        layer_id,
        "object_identity",
        title,
        meaning,
        payload,
        _object_preview(payload),
        "allowed_by_route",
        final_role,
    )


def _line_mask(lines: list[list[tuple[float, float]]], width: int = 3) -> np.ndarray:
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    for points in lines:
        draw.line(points, fill=255, width=width, joint="curve")
    return np.asarray(image, dtype=np.uint8)


# ---------------------------------------------------------------------------
# MATH-01: a point on a unit circle generates a sine trace


def _sample_unit_circle(progress: float) -> ProgramSample:
    theta = 2 * math.pi * progress
    center = np.array([170.0, 180.0])
    radius = 105.0
    point = center + np.array(
        [radius * math.cos(theta), -radius * math.sin(theta)]
    )
    plot_left, plot_right = 340, 610
    plot_mid_y = 180
    trace_x = plot_left + (plot_right - plot_left) * progress
    trace_y = plot_mid_y - radius * math.sin(theta)

    clean = Image.new("RGB", (WIDTH, HEIGHT), (244, 240, 228))
    draw = ImageDraw.Draw(clean)
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        fill=(226, 236, 230),
        outline=(44, 86, 91),
        width=3,
    )
    draw.line((35, 180, 305, 180), fill=(96, 113, 111), width=2)
    draw.line((170, 45, 170, 315), fill=(96, 113, 111), width=2)
    draw.line((plot_left, 55, plot_left, 305), fill=(96, 113, 111), width=2)
    draw.line((plot_left, plot_mid_y, plot_right, plot_mid_y), fill=(96, 113, 111), width=2)
    draw.line(
        (center[0], center[1], point[0], point[1]),
        fill=(214, 112, 58),
        width=5,
    )
    draw.ellipse(
        (point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7),
        fill=(202, 70, 78),
    )
    sample_progress = np.linspace(0, progress, max(2, int(180 * progress) + 2))
    trace_points = [
        (
            plot_left + (plot_right - plot_left) * value,
            plot_mid_y - radius * math.sin(2 * math.pi * value),
        )
        for value in sample_progress
    ]
    draw.line(trace_points, fill=(42, 139, 133), width=4)
    draw.line(
        (point[0], point[1], trace_x, trace_y),
        fill=(111, 132, 143),
        width=1,
    )
    draw.ellipse(
        (trace_x - 5, trace_y - 5, trace_x + 5, trace_y + 5),
        fill=(202, 70, 78),
    )

    boundary = _line_mask(
        [
            [
                (
                    center[0] + radius * math.cos(angle),
                    center[1] + radius * math.sin(angle),
                )
                for angle in np.linspace(0, 2 * math.pi, 181)
            ],
            [(plot_left, 55), (plot_left, 305)],
            [(plot_left, plot_mid_y), (plot_right, plot_mid_y)],
        ],
        2,
    )
    objects = [
        {
            "object_id": "MATH-01-rotating-point",
            "class_id": "rotating_point",
            "geometry": {
                "kind": "ellipse",
                "bbox_xyxy": [
                    float(point[0] - 7),
                    float(point[1] - 7),
                    float(point[0] + 7),
                    float(point[1] + 7),
                ],
            },
        },
        {
            "object_id": "MATH-01-trace-head",
            "class_id": "sine_trace_head",
            "geometry": {
                "kind": "ellipse",
                "bbox_xyxy": [
                    float(trace_x - 5),
                    float(trace_y - 5),
                    float(trace_x + 5),
                    float(trace_y + 5),
                ],
            },
        },
    ]
    annotation = _annotation_layer(
        "MATH-01",
        [
            ((245, 42), (point[0], point[1]), "圆周点"),
            ((315, 330), (trace_x, trace_y), "相同纵坐标"),
            ((540, 330), (trace_x, trace_y), "只绘制已发生轨迹"),
        ],
    )
    program = _overlay_annotations(
        clean,
        annotation,
        [
            (16, 16, f"theta={theta:.2f} rad"),
            (465, 24, "y=sin(theta)"),
        ],
    )
    paper_rgb = np.asarray(clean, dtype=np.uint8)
    paper_region = np.uint8(
        np.all(
            paper_rgb == np.array([244, 240, 228], dtype=np.uint8),
            axis=2,
        )
    ) * 255
    state = {
        "case_id": "MATH-01",
        "progress": round(progress, 6),
        "theta_rad": round(theta, 8),
        "circle_point_xy": [round(float(v), 6) for v in point],
        "circle_normalized_y": round(math.sin(theta), 8),
        "curve_head_xy": [round(trace_x, 6), round(trace_y, 6)],
        "curve_normalized_y": round(
            (plot_mid_y - trace_y) / radius, 8
        ),
        "trace_fraction": round(progress, 6),
    }
    layers = (
        LayerSample(
            "math01_hard_boundary",
            "hard_boundary",
            "圆与坐标轴硬边界",
            "固定单位圆、绘图区坐标轴和镜头。",
            boundary,
            _binary_preview(boundary, (255, 255, 255)),
            "allowed_by_route",
            "防止圆变椭圆或绘图区漂移。",
        ),
        LayerSample(
            "math01_paper_region",
            "region",
            "未被数学图形占用的底纸",
            "只包含程序底色像素，不包含圆、坐标轴、半径、曲线或运动点。",
            paper_region,
            _binary_preview(paper_region, (224, 213, 187)),
            "allowed_by_route",
            "模型如需增加纸张质感，只能在空白底纸中贡献限幅细节。",
        ),
        _object_layer(
            "math01_point_identity",
            "圆周点与曲线头身份",
            "保存同一个运动点在圆和曲线中的对应位置。",
            objects,
            "验证两个红点纵坐标始终一致。",
        ),
        annotation,
    )
    return ProgramSample(state, clean, program, layers)


def _validate_unit_circle(samples: list[ProgramSample]) -> list[dict[str, Any]]:
    states = [sample.state for sample in samples]
    differences = [
        abs(state["circle_normalized_y"] - state["curve_normalized_y"])
        for state in states
    ]
    trace = [state["trace_fraction"] for state in states]
    return [
        {
            "name": "circle_and_curve_vertical_coordinates_match",
            "passed": max(differences) < 1e-6,
            "evidence": differences,
        },
        {
            "name": "one_turn_draws_one_period",
            "passed": math.isclose(states[-1]["theta_rad"], 2 * math.pi, abs_tol=1e-6)
            and states[-1]["trace_fraction"] == 1.0,
            "evidence": {
                "theta_end": states[-1]["theta_rad"],
                "trace_end": states[-1]["trace_fraction"],
            },
        },
        {
            "name": "trace_never_retracts",
            "passed": all(b > a for a, b in zip(trace, trace[1:])),
            "evidence": trace,
        },
    ]


# ---------------------------------------------------------------------------
# PHYS-02: induction depends on flux change, including stop and reversal


def _induction_state(progress: float) -> tuple[float, float, str]:
    if progress <= 1 / 3:
        amount = _smoothstep(progress * 3)
        return 92 + 178 * amount, amount, "approach"
    if progress <= 2 / 3:
        return 270.0, 0.0, "stopped"
    amount = _smoothstep((progress - 2 / 3) * 3)
    return 270 - 178 * amount, -amount, "withdraw"


def _sample_induction(progress: float) -> ProgramSample:
    magnet_x, current, stage = _induction_state(progress)
    coil_x = 390
    distance = abs(coil_x - magnet_x)
    flux = math.exp(-distance / 90)
    clean = Image.new("RGB", (WIDTH, HEIGHT), (231, 228, 214))
    draw = ImageDraw.Draw(clean, "RGBA")
    draw.rectangle((magnet_x - 54, 142, magnet_x + 54, 218), fill=(213, 213, 205, 255), outline=(62, 75, 79, 255), width=3)
    draw.rectangle((magnet_x - 54, 142, magnet_x, 218), fill=(190, 65, 64, 255))
    draw.rectangle((magnet_x, 142, magnet_x + 54, 218), fill=(64, 105, 177, 255))
    for offset in range(-34, 35, 11):
        draw.ellipse(
            (coil_x + offset - 22, 112, coil_x + offset + 22, 248),
            outline=(182, 104, 44, 255),
            width=4,
        )
    draw.line((424, 180, 500, 180, 500, 102), fill=(74, 83, 82, 255), width=3)
    draw.ellipse((480, 30, 602, 150), fill=(237, 240, 231, 255), outline=(57, 75, 78, 255), width=3)
    pointer_angle = -math.pi / 2 + current * 0.75
    pointer_end = (
        541 + 43 * math.cos(pointer_angle),
        91 + 43 * math.sin(pointer_angle),
    )
    draw.line((541, 91, *pointer_end), fill=(207, 73, 62, 255), width=4)
    draw.ellipse((536, 86, 546, 96), fill=(57, 75, 78, 255))

    magnet_region_image = Image.new("L", (WIDTH, HEIGHT), 0)
    magnet_region_draw = ImageDraw.Draw(magnet_region_image)
    magnet_region_draw.rectangle(
        (magnet_x - 54, 142, magnet_x + 54, 218),
        fill=255,
    )
    magnet_region = np.asarray(magnet_region_image, dtype=np.uint8)
    boundary = _line_mask(
        [
            [(magnet_x - 54, 142), (magnet_x + 54, 142), (magnet_x + 54, 218), (magnet_x - 54, 218), (magnet_x - 54, 142)],
            [(coil_x - 56, 112), (coil_x + 56, 112), (coil_x + 56, 248), (coil_x - 56, 248), (coil_x - 56, 112)],
        ],
        3,
    )
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    vector = np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32)
    vector[:, :, 0] = np.exp(-((x - magnet_x) / 150) ** 2) * np.sign(coil_x - magnet_x)
    objects = [
        {
            "object_id": "PHYS-02-magnet",
            "class_id": "bar_magnet",
            "geometry": {
                "kind": "polygon",
                "points": [
                    [magnet_x - 54, 142],
                    [magnet_x + 54, 142],
                    [magnet_x + 54, 218],
                    [magnet_x - 54, 218],
                ],
            },
        },
        {
            "object_id": "PHYS-02-coil",
            "class_id": "fixed_coil",
            "geometry": {
                "kind": "ellipse",
                "bbox_xyxy": [coil_x - 56, 112, coil_x + 56, 248],
            },
        },
    ]
    annotation = _annotation_layer(
        "PHYS-02",
        [
            ((76, 292), (magnet_x, 225), "磁铁只沿线圈轴线运动"),
            ((548, 168), pointer_end, "电流表方向"),
            ((335, 42), (coil_x, 120), "磁通变化而非磁通本身"),
        ],
    )
    program = _overlay_annotations(
        clean,
        annotation,
        [(16, 16, stage), (16, 36, f"flux={flux:.3f}"), (16, 56, f"current={current:+.2f}")],
    )
    state = {
        "case_id": "PHYS-02",
        "progress": round(progress, 6),
        "stage": stage,
        "magnet_x": round(magnet_x, 6),
        "coil_x": coil_x,
        "magnetic_flux": round(flux, 8),
        "induced_current": round(current, 8),
        "meter_pointer_angle": round(pointer_angle, 8),
    }
    layers = (
        LayerSample(
            "phys02_hard_boundary",
            "hard_boundary",
            "磁铁与线圈器材边界",
            "锁定磁铁尺寸和固定线圈位置。",
            boundary,
            _binary_preview(boundary, (255, 255, 255)),
            "allowed_by_route",
            "防止模型复制磁铁或改变线圈。",
        ),
        LayerSample(
            "phys02_magnetic_field",
            "vector_field",
            "磁场方向场",
            "程序根据磁铁位置计算的磁场方向示意。",
            vector,
            _vector_preview(vector),
            "allowed_by_route",
            "供程序验收和视频方向审计，不默认显示箭头。",
        ),
        LayerSample(
            "phys02_magnet_region",
            "region",
            "移动条形磁铁表面",
            "只包含程序磁铁的红蓝矩形内部，并随磁铁位置移动。",
            magnet_region,
            _binary_preview(magnet_region, (190, 65, 64)),
            "allowed_by_route",
            "限定哑光金属细节，轮廓、极性分色和位置仍由程序决定。",
        ),
        _object_layer(
            "phys02_object_identity",
            "磁铁与线圈身份",
            "一个移动磁铁和一个固定线圈的稳定 ID。",
            objects,
            "确保器材不复制且线圈保持固定。",
        ),
        annotation,
    )
    return ProgramSample(state, clean, program, layers)


def _validate_induction(samples: list[ProgramSample]) -> list[dict[str, Any]]:
    currents = [sample.state["induced_current"] for sample in samples]
    coil = [sample.state["coil_x"] for sample in samples]
    return [
        {
            "name": "approach_stop_withdraw_current_signs",
            "passed": currents[0] == 0
            and currents[1] > 0.9
            and currents[2] == 0
            and currents[3] < -0.9,
            "evidence": currents,
        },
        {
            "name": "nearby_stationary_magnet_has_zero_current",
            "passed": samples[2].state["magnetic_flux"] > 0.2
            and samples[2].state["induced_current"] == 0,
            "evidence": samples[2].state,
        },
        {
            "name": "coil_is_fixed",
            "passed": len(set(coil)) == 1,
            "evidence": coil,
        },
    ]


# ---------------------------------------------------------------------------
# CHEM-02: evaporation, supersaturation, nucleation, and growth


_CRYSTAL_CENTERS = ((235, 250), (310, 222), (395, 250), (315, 294))


def _sample_crystallization(progress: float) -> ProgramSample:
    solvent = 1.0 - 0.62 * progress
    concentration = 0.48 / solvent
    saturation = concentration / 0.82
    if progress < 0.55:
        crystal_count = 0
        growth = 0.0
        stage = "concentrating"
    else:
        growth = _smoothstep((progress - 0.55) / 0.45)
        crystal_count = min(4, 1 + int(growth * 3.99))
        stage = "nucleation" if progress < 0.72 else "crystal_growth"
    liquid_solute = 0.48 - 0.16 * growth
    crystal_solute = 0.16 * growth
    level_y = int(110 + (1 - solvent) * 120)

    clean = Image.new("RGB", (WIDTH, HEIGHT), (231, 226, 218))
    draw = ImageDraw.Draw(clean, "RGBA")
    draw.ellipse((145, 80, 495, 318), fill=(221, 235, 238, 95), outline=(70, 91, 98, 255), width=4)
    liquid_mask_image = Image.new("L", (WIDTH, HEIGHT), 0)
    liquid_draw = ImageDraw.Draw(liquid_mask_image)
    liquid_draw.ellipse((154, level_y - 18, 486, 309), fill=255)
    liquid = np.asarray(liquid_mask_image, dtype=np.uint8)
    liquid_rgba = Image.new("RGBA", (WIDTH, HEIGHT), (126, 190, 207, 135))
    clean_rgba = clean.convert("RGBA")
    clean_rgba.paste(liquid_rgba, mask=liquid_mask_image)
    clean = clean_rgba.convert("RGB")
    draw = ImageDraw.Draw(clean)
    draw.ellipse((154, level_y - 18, 486, level_y + 18), outline=(211, 245, 246), width=3)
    objects = []
    crystal_region_image = Image.new("L", (WIDTH, HEIGHT), 0)
    crystal_region_draw = ImageDraw.Draw(crystal_region_image)
    for index, (cx, cy) in enumerate(_CRYSTAL_CENTERS[:crystal_count]):
        size = 5 + growth * (19 + index * 2)
        points = [
            [cx, cy - size],
            [cx + size * 0.7, cy],
            [cx, cy + size],
            [cx - size * 0.7, cy],
        ]
        # These crystals occupy less than one percent of the full frame. A
        # diffusion texture donor cannot add readable facets at that scale
        # without risking the exact count and outline, so the program owns the
        # coarse material cues as four deterministic faces.
        center = [cx, cy]
        facet_colors = (
            (248, 252, 238),
            (221, 232, 216),
            (199, 216, 207),
            (238, 245, 230),
        )
        for facet_index, color in enumerate(facet_colors):
            draw.polygon(
                [
                    center,
                    points[facet_index],
                    points[(facet_index + 1) % len(points)],
                ],
                fill=color,
            )
        draw.line(
            [*points, points[0]],
            fill=(83, 107, 113),
            width=2,
            joint="curve",
        )
        draw.line(
            [points[0], center, points[3]],
            fill=(255, 255, 247),
            width=1,
        )
        crystal_region_draw.polygon(points, fill=255)
        objects.append(
            {
                "object_id": f"CHEM-02-crystal-{index + 1:02d}",
                "class_id": "salt_crystal",
                "geometry": {"kind": "polygon", "points": points},
                "mass_fraction": round(crystal_solute / max(crystal_count, 1), 8),
            }
        )
    crystal_region = np.asarray(
        crystal_region_image, dtype=np.uint8
    )
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    concentration_field = np.where(
        liquid > 0,
        concentration
        * (
            1
            + 0.04
            * np.sin(x / 23)
            * np.cos(y / 19)
        ),
        0,
    ).astype(np.float32)
    boundary = _edge(
        np.uint8(
            (
                ((x - 320) / 175) ** 2
                + ((y - 199) / 119) ** 2
            )
            <= 1
        )
        * 255
    )
    annotation = _annotation_layer(
        "CHEM-02",
        [
            ((92, 62), (190, level_y), "液面随蒸发下降"),
            ((530, 250), (382, 238), "达到阈值后先成核"),
            ((520, 300), (320, 262), "已有晶体继续生长"),
        ],
    )
    program = _overlay_annotations(
        clean,
        annotation,
        [
            (16, 16, stage),
            (16, 36, f"solvent={solvent:.2f}"),
            (16, 56, f"saturation={saturation:.2f}"),
        ],
    )
    state = {
        "case_id": "CHEM-02",
        "progress": round(progress, 6),
        "stage": stage,
        "solvent_volume": round(solvent, 8),
        "concentration": round(concentration, 8),
        "saturation_ratio": round(saturation, 8),
        "crystal_count": crystal_count,
        "liquid_solute_mass": round(liquid_solute, 8),
        "crystal_solute_mass": round(crystal_solute, 8),
        "total_solute_mass": round(liquid_solute + crystal_solute, 8),
    }
    layers = (
        LayerSample(
            "chem02_dish_boundary",
            "hard_boundary",
            "蒸发皿硬边界",
            "固定器皿外形和镜头。",
            boundary,
            _binary_preview(boundary, (255, 255, 255)),
            "allowed_by_route",
            "防止器皿随蒸发改变。",
        ),
        LayerSample(
            "chem02_liquid_region",
            "region",
            "当前溶液区域",
            "液面以下仍由溶液占据的像素。",
            liquid,
            _binary_preview(liquid, (85, 183, 200)),
            "allowed_by_route",
            "限定液体材质和液面位置。",
        ),
        LayerSample(
            "chem02_concentration",
            "scalar_field",
            "溶质浓度场",
            "由溶剂体积和守恒质量计算的连续浓度。",
            concentration_field,
            _scalar_preview(concentration_field),
            "allowed_by_route",
            "控制阈值与成核时刻，不直接变成密集线稿。",
        ),
        LayerSample(
            "chem02_crystal_region",
            "region",
            "晶体占用区",
            "每颗已成核晶体的程序轮廓并集，数量和位置来自稳定对象身份。",
            crystal_region,
            _binary_preview(crystal_region, (235, 241, 226)),
            "allowed_by_route",
            "限定晶面材质只能进入四颗程序晶体内部。",
        ),
        _object_layer(
            "chem02_crystal_identity",
            "晶体身份",
            "记录每个晶核的稳定 ID、位置与质量。",
            objects,
            "防止模型凭空增加或删除晶体。",
        ),
        annotation,
    )
    return ProgramSample(state, clean, program, layers)


def _validate_crystallization(samples: list[ProgramSample]) -> list[dict[str, Any]]:
    states = [sample.state for sample in samples]
    solvent = [state["solvent_volume"] for state in states]
    counts = [state["crystal_count"] for state in states]
    masses = [state["total_solute_mass"] for state in states]
    return [
        {
            "name": "solvent_decreases_and_concentration_increases",
            "passed": all(b < a for a, b in zip(solvent, solvent[1:]))
            and all(
                b["concentration"] > a["concentration"]
                for a, b in zip(states, states[1:])
            ),
            "evidence": {
                "solvent": solvent,
                "concentration": [state["concentration"] for state in states],
            },
        },
        {
            "name": "nuclei_appear_only_after_threshold",
            "passed": counts[:2] == [0, 0] and counts[2] > 0 and counts[3] >= counts[2],
            "evidence": counts,
        },
        {
            "name": "solute_mass_is_conserved",
            "passed": max(masses) - min(masses) < 1e-8,
            "evidence": masses,
        },
    ]


# ---------------------------------------------------------------------------
# BIO-02: guard-cell turgor opens and then closes a stomatal pore


def _sample_stoma(progress: float) -> ProgramSample:
    opening = math.sin(math.pi * progress) ** 2
    aperture = 10 + 52 * opening
    turgor = 0.25 + 0.7 * opening
    stage = "opening" if progress < 0.5 else "closing"
    left_center, right_center = 320 - aperture / 2 - 34, 320 + aperture / 2 + 34
    region_image = Image.new("L", (WIDTH, HEIGHT), 0)
    region_draw = ImageDraw.Draw(region_image)
    left_box = (left_center - 48, 82, left_center + 48, 278)
    right_box = (right_center - 48, 82, right_center + 48, 278)
    region_draw.ellipse(left_box, fill=255)
    region_draw.ellipse(right_box, fill=255)
    region = np.asarray(region_image, dtype=np.uint8)
    pore = np.uint8(
        (
            ((np.arange(WIDTH)[None, :] - 320) / max(aperture / 2, 1)) ** 2
            + ((np.arange(HEIGHT)[:, None] - 180) / 78) ** 2
            <= 1
        )
    ) * 255
    turgor_field = np.where(region > 0, turgor, 0).astype(np.float32)
    boundary = _edge(region)

    clean = Image.new("RGB", (WIDTH, HEIGHT), (210, 226, 190))
    draw = ImageDraw.Draw(clean, "RGBA")
    draw.ellipse(left_box, fill=(99, 165, 91, 225), outline=(43, 102, 61, 255), width=4)
    draw.ellipse(right_box, fill=(99, 165, 91, 225), outline=(43, 102, 61, 255), width=4)
    draw.ellipse(
        (
            320 - aperture / 2,
            102,
            320 + aperture / 2,
            258,
        ),
        fill=(232, 239, 220, 255),
        outline=(68, 104, 73, 255),
        width=2,
    )
    objects = [
        {
            "object_id": "BIO-02-guard-cell-left",
            "class_id": "guard_cell",
            "geometry": {"kind": "ellipse", "bbox_xyxy": list(left_box)},
        },
        {
            "object_id": "BIO-02-guard-cell-right",
            "class_id": "guard_cell",
            "geometry": {"kind": "ellipse", "bbox_xyxy": list(right_box)},
        },
    ]
    annotation = _annotation_layer(
        "BIO-02",
        [
            ((100, 70), (left_center, 120), "左保卫细胞"),
            ((540, 70), (right_center, 120), "右保卫细胞"),
            ((320, 325), (320, 180), "气孔开度由膨压决定"),
        ],
    )
    program = _overlay_annotations(
        clean,
        annotation,
        [(16, 16, stage), (16, 36, f"turgor={turgor:.2f}"), (16, 56, f"aperture={aperture:.1f}px")],
    )
    state = {
        "case_id": "BIO-02",
        "progress": round(progress, 6),
        "stage": stage,
        "guard_cell_count": 2,
        "turgor": round(turgor, 8),
        "aperture_px": round(aperture, 8),
        "pore_open": aperture > 20,
    }
    layers = (
        LayerSample(
            "bio02_guard_boundary",
            "hard_boundary",
            "两个保卫细胞边界",
            "分别保存左右两个保卫细胞的轮廓。",
            boundary,
            _binary_preview(boundary, (255, 255, 255)),
            "allowed_by_route",
            "防止模型合并、复制或交换细胞。",
        ),
        LayerSample(
            "bio02_guard_region",
            "region",
            "两个保卫细胞占用区",
            "只允许表面材质进入左右保卫细胞内部。",
            region,
            _binary_preview(region, (99, 165, 91)),
            "allowed_by_route",
            "限定细胞材质，中央气孔和背景保持不变。",
        ),
        LayerSample(
            "bio02_pore_region",
            "region",
            "气孔开口区域",
            "随膨压变化的中央孔隙。",
            pore,
            _binary_preview(pore, (232, 239, 220)),
            "allowed_by_route",
            "验证开闭幅度而不是只看细胞外观。",
        ),
        LayerSample(
            "bio02_turgor",
            "scalar_field",
            "保卫细胞膨压",
            "两个保卫细胞内部的连续隐变量。",
            turgor_field,
            _scalar_preview(turgor_field),
            "allowed_by_route",
            "驱动形变并用于机制验收。",
        ),
        _object_layer(
            "bio02_guard_identity",
            "保卫细胞身份",
            "左右两细胞稳定 ID。",
            objects,
            "确保始终只有两个保卫细胞。",
        ),
        annotation,
    )
    return ProgramSample(state, clean, program, layers)


def _validate_stoma(samples: list[ProgramSample]) -> list[dict[str, Any]]:
    states = [sample.state for sample in samples]
    apertures = [state["aperture_px"] for state in states]
    turgor = [state["turgor"] for state in states]
    return [
        {
            "name": "exactly_two_guard_cells",
            "passed": all(state["guard_cell_count"] == 2 for state in states),
            "evidence": [state["guard_cell_count"] for state in states],
        },
        {
            "name": "aperture_tracks_turgor",
            "passed": int(np.argmax(apertures)) in (1, 2)
            and int(np.argmax(turgor)) in (1, 2)
            and math.isclose(apertures[0], apertures[-1], abs_tol=1e-6),
            "evidence": {"aperture": apertures, "turgor": turgor},
        },
        {
            "name": "opening_then_closing_is_visible",
            "passed": apertures[1] > apertures[0]
            and apertures[2] > apertures[3],
            "evidence": apertures,
        },
    ]


# ---------------------------------------------------------------------------
# GEO-01: meander neck cutoff creates an isolated oxbow lake


def _meander_points() -> list[tuple[float, float]]:
    anchors = np.asarray(
        [
            (15, 225),
            (100, 220),
            (190, 235),
            (250, 250),
            (282, 190),
            (258, 125),
            (278, 68),
            (338, 52),
            (390, 90),
            (382, 145),
            (342, 190),
            (372, 246),
            (470, 238),
            (625, 220),
        ],
        dtype=float,
    )
    padded = np.vstack((anchors[0], anchors, anchors[-1]))
    points: list[tuple[float, float]] = []
    for index in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[index - 1 : index + 3]
        for amount in np.linspace(0, 1, 14, endpoint=False):
            value = (
                2 * p1
                + (-p0 + p2) * amount
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * amount**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * amount**3
            ) * 0.5
            points.append((float(value[0]), float(value[1])))
    points.append(tuple(float(value) for value in anchors[-1]))
    return points


def _draw_round_channel(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: int,
    width: int,
) -> None:
    """Seal every polyline join so wide raster channels have no pinholes."""
    radius = width / 2
    for first, second in zip(points, points[1:]):
        draw.line((first, second), fill=fill, width=width)
    for x, y in points:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
        )


def _water_topology(mask: np.ndarray) -> tuple[int, int, int]:
    """Measure raster connectivity instead of trusting declared state."""

    water = mask > 0
    height, width = water.shape
    visited = np.zeros_like(water, dtype=bool)
    component_count = 0
    spanning_count = 0
    for start_y, start_x in np.argwhere(water):
        y0, x0 = int(start_y), int(start_x)
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        stack = [(y0, x0)]
        area = 0
        touches_left = x0 == 0
        touches_right = x0 == width - 1
        while stack:
            y, x = stack.pop()
            area += 1
            for next_y, next_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if not (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and water[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    continue
                visited[next_y, next_x] = True
                stack.append((next_y, next_x))
                touches_left = touches_left or next_x == 0
                touches_right = (
                    touches_right or next_x == width - 1
                )
        # Ignore isolated one-pixel raster remnants at rounded joins.
        if area < 16:
            continue
        component_count += 1
        spanning_count += touches_left and touches_right
    isolated_count = component_count - spanning_count
    return component_count, spanning_count, isolated_count


def _sample_oxbow(progress: float) -> ProgramSample:
    cutoff = _smoothstep(progress)
    neck_width = 55 * (1 - cutoff) + 8 * cutoff
    cutoff_complete = progress >= 0.86
    river = Image.new("L", (WIDTH, HEIGHT), 0)
    draw_river = ImageDraw.Draw(river)
    points = _meander_points()
    _draw_round_channel(draw_river, points, fill=255, width=38)
    if cutoff > 0:
        draw_river.line(
            ((282, 190), (342, 190)),
            fill=int(255 * cutoff),
            width=max(3, int(36 * cutoff)),
        )
    if cutoff_complete:
        # Draw the new lower shortcut first, then place the two plugs on the
        # upward old-channel branches. Keeping the plugs slightly above the
        # shortcut leaves the main left-to-right route connected while the
        # abandoned loop becomes a distinct water component.
        draw_river.line(((272, 190), (352, 190)), fill=255, width=36)
        draw_river.ellipse((247, 135, 297, 185), fill=0)
        draw_river.ellipse((341, 135, 391, 185), fill=0)
    water = np.asarray(river, dtype=np.uint8)
    (
        water_component_count,
        main_channel_components,
        isolated_oxbow_count,
    ) = _water_topology(water)
    boundary = _edge(water)
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    vector = np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32)
    vector[:, :, 0] = np.where(water > 0, 1.0, 0)
    vector[:, :, 1] = np.where(
        water > 0,
        0.35 * np.cos((x - 20) / 600 * 2 * math.pi) * (1 - cutoff),
        0,
    )

    clean = Image.new("RGB", (WIDTH, HEIGHT), (185, 205, 153))
    water_rgba = Image.new("RGBA", (WIDTH, HEIGHT), (64, 155, 186, 225))
    clean_rgba = clean.convert("RGBA")
    clean_rgba.paste(water_rgba, mask=river)
    clean = clean_rgba.convert("RGB")
    draw = ImageDraw.Draw(clean, "RGBA")
    if cutoff > 0.05:
        alpha = int(210 * cutoff)
        draw.line(((282, 190), (342, 190)), fill=(55, 142, 178, alpha), width=max(3, int(34 * cutoff)))
    if cutoff_complete:
        draw.line(((272, 190), (352, 190)), fill=(55, 142, 178, 255), width=34)
        draw.ellipse((247, 135, 297, 185), fill=(191, 174, 119, 240))
        draw.ellipse((341, 135, 391, 185), fill=(191, 174, 119, 240))
    objects = [
        {
            "object_id": "GEO-01-main-channel",
            "class_id": "main_river",
            "geometry": {"kind": "polyline", "points": points},
        }
    ]
    if cutoff_complete:
        objects.append(
            {
                "object_id": "GEO-01-oxbow-waterbody",
                "class_id": "isolated_oxbow_lake",
                "geometry": {
                    "kind": "ellipse",
                    "bbox_xyxy": [248, 42, 400, 190],
                },
            }
        )
    annotation = _annotation_layer(
        "GEO-01",
        [
            ((470, 38), (342, 190), "两段河颈逐渐接近"),
            ((500, 315), (312, 190), "洪水切穿形成捷径"),
            ((125, 45), (320, 75), "两端封堵后旧河湾隔离"),
        ],
    )
    program = _overlay_annotations(
        clean,
        annotation,
        [(16, 16, "cutoff" if cutoff_complete else "neck_narrowing"), (16, 36, f"neck={neck_width:.1f}px")],
    )
    state = {
        "case_id": "GEO-01",
        "progress": round(progress, 6),
        "stage": "oxbow_isolated" if cutoff_complete else "neck_narrowing",
        "neck_width_px": round(neck_width, 8),
        "cutoff_fraction": round(cutoff, 8),
        "cutoff_complete": cutoff_complete,
        "water_component_count": water_component_count,
        "main_channel_components": main_channel_components,
        "isolated_oxbow_count": isolated_oxbow_count,
        "water_area_px": int((water > 0).sum()),
    }
    layers = (
        LayerSample(
            "geo01_water_boundary",
            "hard_boundary",
            "河道与牛轭湖水体边界",
            "直接从程序水体拓扑取边，不从成图反推。",
            boundary,
            _binary_preview(boundary, (255, 255, 255)),
            "allowed_by_route",
            "锁定主河道连通与旧河湾隔离关系。",
        ),
        LayerSample(
            "geo01_water_region",
            "region",
            "当前水体区域",
            "主河道与可能形成的牛轭湖占用区。",
            water,
            _binary_preview(water, (64, 155, 186)),
            "allowed_by_route",
            "限定水面材质和连通性。",
        ),
        LayerSample(
            "geo01_flow",
            "vector_field",
            "主河道流向",
            "切弯前沿曲流、切弯后沿捷径向右流动。",
            vector,
            _vector_preview(vector),
            "allowed_by_route",
            "用于动画方向与视频审计。",
        ),
        _object_layer(
            "geo01_waterbody_identity",
            "主河道与牛轭湖身份",
            "旧河湾隔离后获得独立稳定 ID。",
            objects,
            "防止视频把牛轭湖重新接回主河。",
        ),
        annotation,
    )
    return ProgramSample(state, clean, program, layers)


def _validate_oxbow(samples: list[ProgramSample]) -> list[dict[str, Any]]:
    states = [sample.state for sample in samples]
    neck = [state["neck_width_px"] for state in states]
    isolated = [state["isolated_oxbow_count"] for state in states]
    return [
        {
            "name": "meander_neck_narrows_monotonically",
            "passed": all(b <= a for a, b in zip(neck, neck[1:])),
            "evidence": neck,
        },
        {
            "name": "cutoff_occurs_after_narrowing",
            "passed": isolated[:3] == [0, 0, 0] and isolated[3] == 1,
            "evidence": isolated,
        },
        {
            "name": "main_channel_remains_connected",
            "passed": all(state["main_channel_components"] == 1 for state in states),
            "evidence": [state["main_channel_components"] for state in states],
        },
    ]


PROGRAMS: dict[str, SentinelProgram] = {
    "MATH-01": SentinelProgram(
        "MATH-01",
        "单位圆生成正弦曲线",
        "圆周点纵坐标与曲线当前值严格同步，轨迹只向前增长。",
        _sample_unit_circle,
        _validate_unit_circle,
    ),
    "PHYS-02": SentinelProgram(
        "PHYS-02",
        "磁通变化产生电磁感应",
        "靠近与撤离电流方向相反，磁铁停止时电流归零。",
        _sample_induction,
        _validate_induction,
    ),
    "CHEM-02": SentinelProgram(
        "CHEM-02",
        "蒸发、过饱和与晶体生长",
        "溶剂减少先提高浓度，达到阈值后才成核并守恒溶质质量。",
        _sample_crystallization,
        _validate_crystallization,
    ),
    "BIO-02": SentinelProgram(
        "BIO-02",
        "保卫细胞膨压控制气孔开闭",
        "两个稳定保卫细胞随膨压共同改变中央气孔开度。",
        _sample_stoma,
        _validate_stoma,
    ),
    "GEO-01": SentinelProgram(
        "GEO-01",
        "曲流裁弯形成牛轭湖",
        "曲流颈先变窄，切弯后主河连通而旧河湾成为独立水体。",
        _sample_oxbow,
        _validate_oxbow,
    ),
}
