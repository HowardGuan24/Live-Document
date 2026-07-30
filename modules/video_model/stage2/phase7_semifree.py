"""Run Phase 7 route A: semi-free SDXL scene generation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .framework.contracts import load_json, write_json
from .framework.image_experiment import generate_experiment


STAGE2_ROOT = Path(__file__).resolve().parent
MATRIX_PATH = STAGE2_ROOT / "phase7_semifree_matrix.json"
OUTPUT_ROOT = STAGE2_ROOT / "output/phase-7/route-a"


def _chemistry_control() -> Path:
    return (
        STAGE2_ROOT
        / "experiments/EXP-20260729-009/"
        "semantic_apparatus_line_art.png"
    )


def _layer_path(
    semantic_path: Path, layer_id: str
) -> Path:
    manifest = load_json(semantic_path)
    layer = next(
        item
        for item in manifest["layers"]
        if item["layer_id"] == layer_id
    )
    return semantic_path.parents[2] / layer["data"]["path"]


def _geography_control() -> Path:
    output = OUTPUT_ROOT / "controls/geography_semantic_landscape.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    semantic_path = (
        STAGE2_ROOT
        / "output/phase-2/GEO-02/keyframes/02_result/"
        "semantic_layers.json"
    )
    region = np.load(
        _layer_path(semantic_path, "geo02_terrain_region"),
        allow_pickle=False,
    )
    scalar = np.load(
        _layer_path(
            semantic_path, "geo02_humidity_cloud_rain"
        ),
        allow_pickle=False,
    )
    height, width = region.shape
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    outline = []
    for x in range(width):
        ys = np.flatnonzero(region[:, x] > 0)
        if len(ys):
            outline.append((x, int(ys[0])))
    draw.line(outline, fill=255, width=4, joint="curve")
    cloud = scalar >= 0.22
    # Draw only the outer cloud/rain envelope, not every scalar contour.
    for x in range(0, width, 3):
        ys = np.flatnonzero(cloud[:, x])
        if len(ys):
            draw.point((x, int(ys[0])), fill=255)
            draw.point((x, int(ys[-1])), fill=255)
    for x in range(210, 340, 15):
        ys = np.flatnonzero(scalar[:, x] >= 0.45)
        if len(ys):
            start = int(max(ys.min(), 115))
            draw.line(
                (x, start, x - 8, min(start + 55, 300)),
                fill=255,
                width=2,
            )
    image = image.resize((1024, 576), Image.Resampling.NEAREST)
    image.convert("RGB").save(output, optimize=False)
    return output


def _geography_terrain_control() -> Path:
    output = OUTPUT_ROOT / "controls/geography_terrain_only.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    semantic_path = (
        STAGE2_ROOT
        / "output/phase-2/GEO-02/keyframes/02_result/"
        "semantic_layers.json"
    )
    region = np.load(
        _layer_path(semantic_path, "geo02_terrain_region"),
        allow_pickle=False,
    )
    height, width = region.shape
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    outline = []
    for x in range(width):
        ys = np.flatnonzero(region[:, x] > 0)
        if len(ys):
            outline.append((x, int(ys[0])))
    draw.line(outline, fill=255, width=4, joint="curve")
    image.resize((1024, 576), Image.Resampling.NEAREST).convert(
        "RGB"
    ).save(output, optimize=False)
    return output


def _control_for(builder: str) -> Path:
    if builder == "chemistry_semantic_apparatus":
        return _chemistry_control()
    if builder == "geography_semantic_landscape":
        return _geography_control()
    if builder == "geography_terrain_only":
        return _geography_terrain_control()
    raise ValueError(f"unknown control builder: {builder}")


def _experiment_spec(
    case: dict[str, Any],
    state: dict[str, Any],
    matrix: dict[str, Any],
    control_path: Path,
) -> dict[str, Any]:
    case_id = case["case_id"]
    keyframe_id = state["keyframe_id"]
    slug = f"{case_id.lower()}-{keyframe_id}"
    suffix = case.get("experiment_suffix")
    experiment_id = (
        f"EXP-P7-A-{slug}-{suffix}"
        if suffix
        else f"EXP-P7-A-{slug}"
    )
    source_root = (
        STAGE2_ROOT
        / "output/phase-2"
        / case_id
        / "keyframes"
        / keyframe_id
    )
    configurations = [
        {
            "configuration_id": (
                "semantic_control_"
                + str(round(scale * 100)).zfill(3)
            ),
            "pipeline_mode": "controlnet_t2i",
            "control_route": "phase7_semantic_control",
            "controlnet_conditioning_scale": scale,
        }
        for scale in case["control_scales"]
    ]
    render = copy.deepcopy(matrix["render"])
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "case_id": case_id,
        "hypothesis_zh": (
            "语义完整的场景线稿允许 SDXL 从噪声半自由生成真实体积、"
            "光照和材质，同时保留本状态的主要结构。"
        ),
        "single_variable_zh": (
            "同一状态只改变 ControlNet 强度；提示词、控制图、模型、"
            "采样参数和三个复现编号保持一致。"
        ),
        "source": {
            "keyframe_id": keyframe_id,
            "clean_frame": str(source_root / "clean.png"),
            "semantic_layers": str(
                source_root / "semantic_layers.json"
            ),
        },
        "control_overrides": {
            "phase7_semantic_control": str(control_path)
        },
        "control_override_explanations": {
            "phase7_semantic_control": (
                "由程序语义层或确定性器材模板绘制，只表达需要固定的"
                "场景结构；没有颜色、材质、教学文字和 UI。"
            )
        },
        "prompt_parts": state["prompt_parts"],
        "negative_artifacts": case["negative_artifacts"],
        "render": render,
        "configurations": configurations,
        "blind_shuffle_seed": (
            2026073000
            + sum(ord(char) for char in experiment_id)
        ),
        "budget": {
            "maximum_new_image_candidates": len(
                configurations
            )
            * len(render["seeds"]),
            "actual_planned_image_candidates": len(
                configurations
            )
            * len(render["seeds"]),
            "planned_external_reuse": 0,
            "maximum_new_generation": len(configurations)
            * len(render["seeds"]),
            "maximum_video_trials": 0,
        },
    }


def run() -> dict[str, Any]:
    matrix = load_json(MATRIX_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for case in matrix["cases"]:
        control = _control_for(case["control_builder"])
        for state in case["states"]:
            spec = _experiment_spec(case, state, matrix, control)
            experiment_root = (
                OUTPUT_ROOT / "experiments" / spec["experiment_id"]
            )
            metadata = generate_experiment(spec, experiment_root)
            spec_path = experiment_root / "spec.json"
            write_json(spec_path, spec)
            records.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "case_id": spec["case_id"],
                    "keyframe_id": spec["source"]["keyframe_id"],
                    "candidate_count": len(metadata["candidates"]),
                    "generated": metadata["cache"]["generated"],
                    "reused": metadata["cache"]["reused"],
                    "root": str(
                        experiment_root.relative_to(STAGE2_ROOT)
                    ),
                }
            )
    result = {
        "schema_version": "1.0",
        "route_id": matrix["route_id"],
        "status": "generated",
        "model_runs": {
            "image": sum(item["generated"] for item in records),
            "video": 0,
        },
        "experiments": records,
    }
    write_json(OUTPUT_ROOT / "route-a-manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run()
    print(
        f"Phase 7 route A: {result['status']} · "
        f"{len(result['experiments'])} experiments · "
        f"{result['model_runs']['image']} new images"
    )


if __name__ == "__main__":
    main()
