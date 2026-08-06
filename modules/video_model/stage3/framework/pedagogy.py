"""Case-agnostic compiler for readable, deterministic teaching timelines.

The scientific program still owns every mechanism state.  This module adds a
second clock for an audience: it divides progress into named beats, allocates
more display frames where declared state metrics change quickly, and inserts a
short hold after every beat.  It never invents a scientific state.
"""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    write_json,
)


def load_program(case: dict[str, Any]) -> Any:
    module = importlib.import_module(case["provider_module"])
    program = module.PROGRAMS[case["case_id"]]
    if program.case_id != case["case_id"]:
        raise ValueError("provider Case ID mismatch")
    return program


def _state_number(state: dict[str, Any], path: str) -> float:
    value: Any = state
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"state metric is missing: {path}")
        value = value[key]
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        raise ValueError(f"state metric is not numeric: {path}")
    return float(value)


def _feature_matrix(
    program: Any,
    progresses: np.ndarray,
    metrics: list[str],
) -> np.ndarray:
    rows = []
    for progress in progresses:
        state = program.sample(float(progress)).state
        rows.append([_state_number(state, metric) for metric in metrics])
    return np.asarray(rows, dtype=np.float64)


def _normalize_features(
    features: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    lower = reference.min(axis=0)
    span = reference.max(axis=0) - lower
    span[span < 1e-12] = 1.0
    return (features - lower) / span


def adaptive_progresses(
    program: Any,
    beat: dict[str, Any],
    defaults: dict[str, Any],
    frame_count: int,
) -> tuple[list[float], dict[str, Any]]:
    """Reparameterize time without changing the program's state function."""

    start = float(beat["progress_start"])
    end = float(beat["progress_end"])
    dense_count = int(
        beat.get("analysis_samples", defaults["analysis_samples"])
    )
    dense_progress = np.linspace(start, end, dense_count)
    metrics = list(beat["attention_metrics"])
    dense_features = _feature_matrix(program, dense_progress, metrics)
    normalized = _normalize_features(dense_features, dense_features)
    changes = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
    baseline = np.diff(dense_progress) / max(end - start, 1e-12)
    if float(changes.sum()) > 1e-12:
        changes /= changes.sum()
    else:
        changes = baseline.copy()
    weight = float(beat.get("change_weight", defaults["change_weight"]))
    increments = (1.0 - weight) * baseline + weight * changes
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    cumulative[-1] = 1.0
    targets = np.linspace(0.0, 1.0, frame_count)
    chosen = np.interp(targets, cumulative, dense_progress)
    chosen[0], chosen[-1] = start, end

    uniform = np.linspace(start, end, frame_count)
    chosen_features = _normalize_features(
        _feature_matrix(program, chosen, metrics), dense_features
    )
    uniform_features = _normalize_features(
        _feature_matrix(program, uniform, metrics), dense_features
    )

    def maximum_step(values: np.ndarray) -> float:
        return float(
            np.linalg.norm(np.diff(values, axis=0), axis=1).max(initial=0)
        )

    diagnostic = {
        "attention_metrics": metrics,
        "analysis_samples": dense_count,
        "change_weight": weight,
        "total_normalized_state_variation": round(
            float(np.linalg.norm(np.diff(normalized, axis=0), axis=1).sum()),
            8,
        ),
        "uniform_maximum_normalized_state_step": round(
            maximum_step(uniform_features), 8
        ),
        "adaptive_maximum_normalized_state_step": round(
            maximum_step(chosen_features), 8
        ),
        "progress_minimum_step": round(
            float(np.diff(chosen).min(initial=0)), 10
        ),
        "progress_maximum_step": round(
            float(np.diff(chosen).max(initial=0)), 10
        ),
    }
    return [round(float(value), 10) for value in chosen], diagnostic


def compile_timeline(
    program: Any,
    case: dict[str, Any],
    defaults: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fps = int(defaults["fps"])
    beats = case["beats"]
    if not beats or float(beats[0]["progress_start"]) != 0.0:
        raise ValueError(f"{case['case_id']} timeline must start at zero")
    if float(beats[-1]["progress_end"]) != 1.0:
        raise ValueError(f"{case['case_id']} timeline must end at one")
    for left, right in zip(beats, beats[1:]):
        if abs(
            float(left["progress_end"])
            - float(right["progress_start"])
        ) > 1e-6:
            raise ValueError(f"{case['case_id']} beats are not contiguous")

    records: list[dict[str, Any]] = []
    diagnostics = []
    for beat_index, beat in enumerate(beats):
        dynamic_count = max(2, round(float(beat["dynamic_seconds"]) * fps))
        hold_count = max(1, round(float(beat["hold_seconds"]) * fps))
        progresses, diagnostic = adaptive_progresses(
            program, beat, defaults, dynamic_count
        )
        diagnostic.update(
            {
                "beat_id": beat["beat_id"],
                "dynamic_frame_count": dynamic_count,
                "hold_frame_count": hold_count,
            }
        )
        diagnostics.append(diagnostic)
        for local_index, progress in enumerate(progresses):
            records.append(
                {
                    "display_index": len(records),
                    "beat_index": beat_index,
                    "beat_id": beat["beat_id"],
                    "beat_title_zh": beat["title_zh"],
                    "caption_zh": beat["caption_zh"],
                    "progress": progress,
                    "is_hold": False,
                    "beat_local_index": local_index,
                }
            )
        for hold_index in range(hold_count):
            records.append(
                {
                    "display_index": len(records),
                    "beat_index": beat_index,
                    "beat_id": beat["beat_id"],
                    "beat_title_zh": beat["title_zh"],
                    "caption_zh": beat["caption_zh"],
                    "progress": round(float(beat["progress_end"]), 10),
                    "is_hold": True,
                    "beat_local_index": dynamic_count + hold_index,
                }
            )
    for record in records:
        record["time_seconds"] = round(record["display_index"] / fps, 4)
    duration = len(records) / fps
    return records, {
        "case_id": case["case_id"],
        "fps": fps,
        "frame_count": len(records),
        "duration_seconds": round(duration, 4),
        "beat_diagnostics": diagnostics,
    }


def required_layers(plan: dict[str, Any], sample: Any) -> set[str]:
    result = {
        layer.layer_id
        for layer in sample.layers
        if layer.layer_type == "object_identity"
    }
    for operator in plan["operators"]:
        for key, value in operator["config"].items():
            if key == "layer_id" or key.endswith("_layer_id"):
                result.add(value)
    return result


def _serialize_layer(
    layer: Any,
    frame_root: Path,
    timeline_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if isinstance(layer.data, np.ndarray):
        path = frame_root / f"{layer.layer_id}.npy"
        np.save(path, layer.data, allow_pickle=False)
        encoding = "npy"
    else:
        path = frame_root / f"{layer.layer_id}.json"
        write_json(path, layer.data)
        encoding = "json"
    return {
        "layer_id": layer.layer_id,
        "layer_type": layer.layer_type,
        "title_zh": layer.title_zh,
        "meaning_zh": layer.meaning_zh,
        "source_zh": "由确定性程序在该展示时刻直接计算，不从 RGB 图片识别。",
        "model_input_policy": layer.model_input_policy,
        "used_as_model_input": False,
        "final_role_zh": layer.final_role_zh,
        "data": {
            "encoding": encoding,
            "path": path.relative_to(timeline_root).as_posix(),
            "sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
        },
    }


def export_timeline_contract(
    *,
    program: Any,
    case: dict[str, Any],
    timeline: list[dict[str, Any]],
    plan: dict[str, Any],
    stage3_root: Path,
    repo_root: Path,
    case_root: Path,
) -> tuple[dict[str, Any], list[Any]]:
    timeline_root = case_root / "timeline-input"
    samples = [program.sample(float(item["progress"])) for item in timeline]
    needed = required_layers(plan, samples[0])
    keyframes = []
    for index, (record, sample) in enumerate(zip(timeline, samples)):
        frame_id = f"frame_{index:03d}"
        frame_root = timeline_root / frame_id
        frame_root.mkdir(parents=True, exist_ok=True)
        state_path = frame_root / "state.json"
        write_json(state_path, sample.state)
        available = {layer.layer_id: layer for layer in sample.layers}
        missing = sorted(needed - set(available))
        if missing:
            raise ValueError(f"{case['case_id']} is missing layers {missing}")
        layers = [
            _serialize_layer(
                available[layer_id], frame_root, timeline_root, repo_root
            )
            for layer_id in sorted(needed)
        ]
        semantic_path = frame_root / "semantic_layers.json"
        write_json(
            semantic_path,
            {
                "schema_version": "1.0",
                "case_id": case["case_id"],
                "state_id": frame_id,
                "canvas": {
                    "width": 640,
                    "height": 360,
                    "coordinate_system": "pixel_xy_top_left",
                },
                "layers": layers,
            },
        )
        record["keyframe_id"] = frame_id
        record["state"] = file_record(state_path, repo_root)
        record["semantic_layers"] = file_record(semantic_path, repo_root)
        record["state_evidence"] = {
            metric: _state_number(sample.state, metric)
            for metric in case["beats"][record["beat_index"]][
                "attention_metrics"
            ]
        }
        keyframes.append(
            {
                "keyframe_id": frame_id,
                "order": index,
                "progress": record["progress"],
                "state": record["state"],
                "semantic_layers": record["semantic_layers"],
            }
        )

    source_contract = load_json(
        stage3_root / f"contracts/{case['case_id']}.json"
    )
    provider_path = Path(importlib.import_module(case["provider_module"]).__file__)
    contract = {
        **source_contract,
        "program_source": {
            **source_contract["program_source"],
            "root": timeline_root.relative_to(repo_root).as_posix(),
            "timeline_materialization": {
                "provider": f"{case['provider_module']}.PROGRAMS",
                "provider_source": file_record(provider_path, repo_root),
                "sample_count": len(samples),
                "progress_rule": "compiled by pedagogy_contracts_v1.json",
                "exported_layer_ids": sorted(needed),
            },
        },
        "keyframes": keyframes,
    }
    contract_path = case_root / "timeline-contract.json"
    write_json(contract_path, contract)
    return contract, samples


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def compile_render_plan(
    source_plan: dict[str, Any],
    case: dict[str, Any],
    contract_path: Path,
    stage3_root: Path,
    frame_count: int,
) -> dict[str, Any]:
    plan = copy.deepcopy(source_plan)
    override = case.get("appearance_override", {})
    if "anchor" in override:
        plan["anchor"] = copy.deepcopy(override["anchor"])
    if "base_canvas" in override:
        plan["base_canvas"] = copy.deepcopy(override["base_canvas"])
    removed = set(override.get("remove_operators", []))
    plan["operators"] = [
        operator
        for operator in plan["operators"]
        if operator["operator_id"] not in removed
    ]
    by_id = {item["operator_id"]: item for item in plan["operators"]}
    for operator_id, change in override.get("operator_overrides", {}).items():
        if operator_id not in by_id:
            raise ValueError(f"unknown operator override: {operator_id}")
        operator = by_id[operator_id]
        if (
            "operator_type" in change
            and change["operator_type"] != operator["operator_type"]
        ):
            operator["operator_type"] = change["operator_type"]
            operator["config"] = copy.deepcopy(change["config"])
        else:
            _deep_update(operator, change)
    for operator in override.get("add_operators", []):
        operator_id = operator["operator_id"]
        if operator_id in by_id:
            raise ValueError(f"duplicate added operator: {operator_id}")
        copied = copy.deepcopy(operator)
        plan["operators"].append(copied)
        by_id[operator_id] = copied
    for operator in plan["operators"]:
        projection = operator["config"].get("projection")
        if projection and "initial_keyframe_id" in projection:
            projection["initial_keyframe_id"] = "frame_000"
    plan.update(
        {
            "schema_version": "1.0",
            "plan_id": f"S3.11-{case['case_id']}-PEDAGOGICAL-V1",
            "case_id": case["case_id"],
            "role": "deterministic_pedagogical_full_timeline",
            "contract": contract_path.relative_to(stage3_root).as_posix(),
            "keyframe_ids": [
                f"frame_{index:03d}" for index in range(frame_count)
            ],
        }
    )
    return plan


def find_cjk_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("/tmp/noto-cjk/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    path = next((value for value in candidates if value.is_file()), None)
    if path is None:
        raise FileNotFoundError("a CJK font is required for teaching captions")
    return ImageFont.truetype(str(path), size)


def compose_teaching_frame(
    scene: Image.Image,
    record: dict[str, Any],
    case: dict[str, Any],
    defaults: dict[str, Any],
) -> Image.Image:
    scene_size = tuple(int(value) for value in defaults["scene_size"])
    panel_height = int(defaults["panel_height"])
    fitted = ImageOps.fit(
        scene.convert("RGB"), scene_size, method=Image.Resampling.LANCZOS
    )
    output = Image.new(
        "RGB", (scene_size[0], scene_size[1] + panel_height), (10, 26, 29)
    )
    output.paste(fitted, (0, 0))
    draw = ImageDraw.Draw(output)
    title_font = find_cjk_font(20)
    caption_font = find_cjk_font(16)
    small_font = find_cjk_font(14)
    panel_y = scene_size[1]
    draw.rectangle(
        (0, panel_y, scene_size[0], output.height), fill=(10, 26, 29)
    )
    draw.text(
        (18, panel_y + 8), record["beat_title_zh"], font=title_font,
        fill=(244, 196, 96)
    )
    stage_text = f"{record['beat_index'] + 1} / {len(case['beats'])}"
    stage_box = draw.textbbox((0, 0), stage_text, font=small_font)
    draw.text(
        (scene_size[0] - (stage_box[2] - stage_box[0]) - 18, panel_y + 11),
        stage_text,
        font=small_font,
        fill=(190, 213, 207),
    )
    draw.text(
        (18, panel_y + 39), record["caption_zh"], font=caption_font,
        fill=(232, 240, 237)
    )
    bar_y = output.height - 5
    draw.rectangle((0, bar_y, scene_size[0], output.height), fill=(36, 62, 62))
    progress_width = round(scene_size[0] * float(record["progress"]))
    draw.rectangle((0, bar_y, progress_width, output.height), fill=(71, 183, 158))
    return output


def contact_sheet(
    frames: list[Path],
    timeline: list[dict[str, Any]],
    case: dict[str, Any],
    output: Path,
) -> Path:
    selected = []
    for beat_index in range(len(case["beats"])):
        candidates = [
            item for item in timeline
            if item["beat_index"] == beat_index and not item["is_hold"]
        ]
        selected.append(candidates[-1]["display_index"])
    width, height = 768, 512
    columns = 2
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), (10, 26, 29))
    for slot, index in enumerate(selected):
        image = Image.open(frames[index]).convert("RGB")
        canvas.paste(image, ((slot % columns) * width, (slot // columns) * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, subsampling=0)
    return output


def story_audit(
    *,
    case: dict[str, Any],
    defaults: dict[str, Any],
    timeline: list[dict[str, Any]],
    compile_info: dict[str, Any],
    scene_arrays: list[np.ndarray],
) -> dict[str, Any]:
    progresses = [float(item["progress"]) for item in timeline]
    differences = [
        float(
            np.abs(right.astype(np.int16) - left.astype(np.int16)).mean()
        )
        for left, right in zip(scene_arrays, scene_arrays[1:])
    ]
    beat_counts = {
        beat["beat_id"]: {
            "dynamic": sum(
                item["beat_id"] == beat["beat_id"] and not item["is_hold"]
                for item in timeline
            ),
            "hold": sum(
                item["beat_id"] == beat["beat_id"] and item["is_hold"]
                for item in timeline
            ),
        }
        for beat in case["beats"]
    }
    duration = float(compile_info["duration_seconds"])
    checks = [
        {
            "name": "duration_is_chosen_for_explanation_not_legacy_49_frames",
            "passed": float(defaults["minimum_total_seconds"])
            <= duration
            <= float(defaults["maximum_total_seconds"]),
            "evidence": {"duration_seconds": duration, "frame_count": len(timeline)},
        },
        {
            "name": "all_named_beats_have_motion_and_reading_hold",
            "passed": all(
                value["dynamic"] >= 2
                and value["hold"]
                >= round(float(defaults["minimum_hold_seconds"]) * int(defaults["fps"]))
                for value in beat_counts.values()
            ),
            "evidence": beat_counts,
        },
        {
            "name": "program_progress_is_monotonic_and_complete",
            "passed": progresses[0] == 0.0
            and progresses[-1] == 1.0
            and all(left <= right for left, right in zip(progresses, progresses[1:])),
            "evidence": {"first": progresses[0], "last": progresses[-1]},
        },
        {
            "name": "scene_has_no_large_single_frame_pixel_jump",
            "passed": max(differences, default=0.0) <= 24.0,
            "evidence": {
                "maximum_scene_mae_0_255": round(max(differences, default=0.0), 6),
                "mean_scene_mae_0_255": round(float(np.mean(differences)), 6),
                "threshold": 24.0,
            },
        },
        {
            "name": "adaptive_clock_does_not_worsen_declared_state_steps",
            "passed": all(
                item["adaptive_maximum_normalized_state_step"]
                <= item["uniform_maximum_normalized_state_step"] + 0.06
                for item in compile_info["beat_diagnostics"]
            ),
            "evidence": compile_info["beat_diagnostics"],
        },
    ]
    return {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
