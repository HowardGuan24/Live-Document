"""Upgrade the three legacy sentinel fallbacks to materialized Stage 3 timelines."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from modules.video_model.stage2.cases.sentinel_programs import PROGRAMS
from modules.video_model.stage3.framework.contracts import file_record, load_json, sha256_path, write_json
from modules.video_model.stage3.framework.motion import (
    audit_sparse_checkpoints,
    consecutive_metrics,
    decode_video,
    encode_video,
    endpoint_metrics,
)
from modules.video_model.stage3.framework.state_renderer import render_plan
from modules.video_model.stage3.phase9_scale_motion import _preview, _required_layers, _serialize_layer


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-10-release/sentinel-motion"
CASES = ("MATH-02", "PHYS-01", "CHEM-01")
KEYFRAME_IDS = ("00_start", "01_mechanism", "02_result", "03_end")
KEYFRAME_INDICES = (0, 16, 32, 48)
FRAME_COUNT = 49
FPS = 24


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _export(case_id: str, plan: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    program = PROGRAMS[case_id]
    root = OUTPUT / case_id
    timeline = root / "timeline-input"
    samples = [program.sample(index / 48) for index in range(FRAME_COUNT)]
    required = _required_layers(plan, samples[0])
    keyframes = []
    for index, sample in enumerate(samples):
        frame_id = f"frame_{index:03d}"
        frame_root = timeline / frame_id
        frame_root.mkdir(parents=True, exist_ok=True)
        state_path = frame_root / "state.json"
        write_json(state_path, sample.state)
        available = {layer.layer_id: layer for layer in sample.layers}
        missing = sorted(required - set(available))
        if missing:
            raise ValueError(f"{case_id} missing timeline layers: {missing}")
        layers = [_serialize_layer(available[layer_id], frame_root, timeline) for layer_id in sorted(required)]
        semantic_path = frame_root / "semantic_layers.json"
        write_json(semantic_path, {
            "schema_version": "1.0",
            "case_id": case_id,
            "state_id": frame_id,
            "canvas": {"width": 640, "height": 360, "coordinate_system": "pixel_xy_top_left"},
            "layers": layers,
        })
        keyframes.append({
            "keyframe_id": frame_id,
            "order": index,
            "progress": round(index / 48, 8),
            "state": file_record(state_path, REPO_ROOT),
            "semantic_layers": file_record(semantic_path, REPO_ROOT),
        })
    source = load_json(STAGE3 / f"contracts/{case_id}.json")
    provider = REPO_ROOT / "modules/video_model/stage2/cases/sentinel_programs.py"
    contract = {
        **source,
        "program_source": {
            **source["program_source"],
            "root": timeline.relative_to(REPO_ROOT).as_posix(),
            "timeline_materialization": {
                "provider": "modules.video_model.stage2.cases.sentinel_programs.PROGRAMS",
                "provider_source": file_record(provider, REPO_ROOT),
                "sample_count": FRAME_COUNT,
                "progress_rule": "frame_index / 48",
                "exported_layer_ids": sorted(required),
            },
        },
        "keyframes": keyframes,
    }
    path = root / "timeline-contract.json"
    write_json(path, contract)
    write_json(root / "timeline-export.json", {
        "schema_version": "1.0",
        "case_id": case_id,
        "frame_count": FRAME_COUNT,
        "contract": file_record(path, REPO_ROOT),
        "provider": contract["program_source"]["timeline_materialization"],
    })
    return contract, samples


def _program_checks(case_id: str, samples: list[Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    states = [sample.state for sample in samples]
    if case_id == "MATH-02":
        ids = [[item["object_id"] for item in state["objects"]] for state in states]
        areas = [[item["area_px2"] for item in state["objects"]] for state in states]
        return [
            _check("four_triangle_identities_persist", all(value == ids[0] and len(value) == 4 for value in ids), ids),
            _check("triangle_areas_are_preserved", all(value == areas[0] for value in areas), areas),
            _check("remaining_area_proves_a2_plus_b2_equals_c2", all(abs(state["remaining_area_px2"] - state["expected_c2_px2"]) < 1e-7 and abs(state["remaining_area_px2"] - state["expected_a2_plus_b2_px2"]) < 1e-7 for state in states), [state["remaining_area_px2"] for state in states]),
        ]
    if case_id == "PHYS-01":
        sources = [state["sources_xy"] for state in states]
        radius = [state["wavefront_radius_px"] for state in states]
        overlap = [state["overlap_pixel_count"] for state in states]
        return [
            _check("two_sources_remain_fixed", all(value == sources[0] and len(value) == 2 for value in sources), sources),
            _check("wavefront_radius_increases", all(left < right for left, right in zip(radius, radius[1:])), radius),
            _check("overlap_grows_and_nodes_appear", all(left <= right for left, right in zip(overlap, overlap[1:])) and states[-1]["node_pixel_count"] > 0 and states[-1]["antinode_pixel_count"] > 0, {"overlap": overlap, "final_nodes": states[-1]["node_pixel_count"], "final_antinodes": states[-1]["antinode_pixel_count"]}),
        ]
    volume = [state["base_volume_ml"] for state in states]
    level = [state["liquid_level_y"] for state in states]
    plume = [state["plume_strength"] for state in states]
    indicator = [state["indicator_mean_inside_liquid"] for state in states]
    return [
        _check("added_base_volume_never_decreases", all(left <= right for left, right in zip(volume, volume[1:])), volume),
        _check("liquid_level_tracks_added_volume", all(left >= right for left, right in zip(level, level[1:])), level),
        _check("local_plume_peaks_then_clears_before_endpoint", int(np.argmax(plume)) == 16 and max(plume) > 0.9 and plume[32] == 0 and indicator[-1] > 0.9, {"plume": plume, "indicator": indicator}),
    ]


def _video_gate(case_id: str, video: Path, frame_paths: list[Path]) -> dict[str, Any]:
    info, frames = decode_video(video)
    endpoints = endpoint_metrics(frames, frame_paths[0], frame_paths[-1])
    consecutive = consecutive_metrics(frames)
    refs = [STAGE3 / f"output/phase-4/{case_id}/frames/{keyframe_id}.png" for keyframe_id in KEYFRAME_IDS]
    sparse = audit_sparse_checkpoints(frames, list(zip(KEYFRAME_INDICES, refs)), maximum_mae_0_255=6)
    checks = [
        _check("video_has_49_frames", info["frame_count"] == FRAME_COUNT, info),
        _check("endpoints_follow_rendered_inputs", all(item["mean_absolute_pixel_error_0_255"] <= 6 for item in endpoints.values()), endpoints),
        _check("no_abrupt_pixel_jump", consecutive["maximum"] <= 12, consecutive),
        sparse,
    ]
    return {"video": info, "checks": checks, "passed": all(item["passed"] for item in checks)}


def _run_case(case_id: str, source_plan: dict[str, Any]) -> dict[str, Any]:
    root = OUTPUT / case_id
    contract, samples = _export(case_id, source_plan)
    source_plan = copy.deepcopy(source_plan)
    for operator in source_plan["operators"]:
        projection = operator["config"].get("projection", {})
        if projection.get("initial_keyframe_id") == "00_start":
            projection["initial_keyframe_id"] = "frame_000"
    plan = {
        **source_plan,
        "plan_id": f"S3.10-{case_id}-FULL-TIMELINE-V1",
        "role": "deterministic_full_program_timeline",
        "contract": (root / "timeline-contract.json").relative_to(STAGE3).as_posix(),
        "keyframe_ids": [item["keyframe_id"] for item in contract["keyframes"]],
    }
    write_json(root / "state-render-plan.json", plan)
    manifest = render_plan(plan, STAGE3, REPO_ROOT, root / "render")
    paths = [REPO_ROOT / item["output"]["path"] for item in manifest["records"]]
    arrays = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    video = root / "deterministic/transition.mp4"
    encode_video(arrays, video, fps=FPS)
    gate = _video_gate(case_id, video, paths)
    checks = _program_checks(case_id, samples, manifest)
    preview = _preview(decode_video(video)[1], root / "deterministic/generated-frames.jpg")
    result = {
        "schema_version": "1.0",
        "case_id": case_id,
        "classification": "materialized deterministic full-program-timeline fallback",
        "passed": gate["passed"] and all(item["passed"] for item in checks),
        "video_gate": gate,
        "program_mechanism_checks": checks,
        "video": file_record(video, REPO_ROOT),
        "preview": file_record(preview, REPO_ROOT),
        "render_manifest": file_record(root / "render/manifest.json", REPO_ROOT),
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
    }
    write_json(root / "g4.json", result)
    return result


def run() -> dict[str, Any]:
    plans = {item["case_id"]: item for item in load_json(STAGE3 / "state_render_plans.json")["plans"]}
    cases = {}
    for case_id in CASES:
        existing = OUTPUT / case_id / "g4.json"
        value = load_json(existing) if existing.is_file() else None
        cases[case_id] = value if value and value.get("passed", False) else _run_case(case_id, plans[case_id])
    result = {"schema_version": "1.0", "cases": {case_id: {"passed": value["passed"], "g4": file_record(OUTPUT / f"{case_id}/g4.json", REPO_ROOT)} for case_id, value in cases.items()}, "passed": all(value["passed"] for value in cases.values())}
    write_json(OUTPUT / "g4-machine.json", result)
    if not result["passed"]:
        raise RuntimeError("legacy sentinel materialization failed")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
