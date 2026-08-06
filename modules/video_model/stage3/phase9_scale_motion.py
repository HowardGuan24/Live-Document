"""Materialize and audit 49-frame deterministic motion for five scale cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage2.cases.remaining_programs import PROGRAMS
from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    write_json,
)
from modules.video_model.stage3.framework.motion import (
    audit_sparse_checkpoints,
    consecutive_metrics,
    decode_video,
    encode_video,
    endpoint_metrics,
)
from modules.video_model.stage3.framework.state_renderer import render_plan


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-9-scale-motion"
PLAN_PATH = STAGE3 / "scale_state_render_plans_v1.json"
CASES = ("MATH-01", "PHYS-02", "CHEM-02", "BIO-02", "GEO-01")
FRAME_COUNT = 49
FPS = 24
KEYFRAME_INDICES = (0, 16, 32, 48)
KEYFRAME_IDS = ("00_start", "01_mechanism", "02_result", "03_end")
THRESHOLDS = {
    "maximum_endpoint_mae_0_255": 6.0,
    "maximum_sparse_checkpoint_mae_0_255": 6.0,
    "maximum_consecutive_frame_mae_0_255": 12.0,
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _required_layers(plan: dict[str, Any], sample: Any) -> set[str]:
    result = {
        layer.layer_id for layer in sample.layers if layer.layer_type == "object_identity"
    }
    for operator in plan["operators"]:
        for key, value in operator["config"].items():
            if key == "layer_id" or key.endswith("_layer_id"):
                result.add(value)
    return result


def _serialize_layer(layer: Any, frame_root: Path, timeline_root: Path) -> dict[str, Any]:
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
        "source_zh": "由确定性程序 provider 在当前时间点直接计算，不从 RGB 视频反推。",
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


def _export_case(case_id: str, plan: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    program = PROGRAMS[case_id]
    case_root = OUTPUT / case_id
    timeline_root = case_root / "timeline-input"
    samples = [program.sample(index / (FRAME_COUNT - 1)) for index in range(FRAME_COUNT)]
    required = _required_layers(plan, samples[0])
    records = []
    for index, sample in enumerate(samples):
        frame_id = f"frame_{index:03d}"
        frame_root = timeline_root / frame_id
        frame_root.mkdir(parents=True, exist_ok=True)
        state_path = frame_root / "state.json"
        write_json(state_path, sample.state)
        available = {layer.layer_id: layer for layer in sample.layers}
        missing = sorted(required - set(available))
        if missing:
            raise ValueError(f"{case_id} timeline missing layers: {missing}")
        layers = [_serialize_layer(available[layer_id], frame_root, timeline_root) for layer_id in sorted(required)]
        semantic = {
            "schema_version": "1.0",
            "case_id": case_id,
            "state_id": frame_id,
            "canvas": {"width": 640, "height": 360, "coordinate_system": "pixel_xy_top_left"},
            "layers": layers,
        }
        semantic_path = frame_root / "semantic_layers.json"
        write_json(semantic_path, semantic)
        records.append({
            "keyframe_id": frame_id,
            "order": index,
            "progress": round(index / (FRAME_COUNT - 1), 8),
            "state": file_record(state_path, REPO_ROOT),
            "semantic_layers": file_record(semantic_path, REPO_ROOT),
        })

    source = load_json(STAGE3 / f"contracts/{case_id}.json")
    provider_path = REPO_ROOT / "modules/video_model/stage2/cases/remaining_programs.py"
    contract = {
        **source,
        "program_source": {
            **source["program_source"],
            "root": timeline_root.relative_to(REPO_ROOT).as_posix(),
            "timeline_materialization": {
                "provider": "modules.video_model.stage2.cases.remaining_programs.PROGRAMS",
                "provider_source": file_record(provider_path, REPO_ROOT),
                "sample_count": FRAME_COUNT,
                "progress_rule": "frame_index / 48",
                "exported_layer_ids": sorted(required),
            },
        },
        "keyframes": records,
    }
    contract_path = case_root / "timeline-contract.json"
    write_json(contract_path, contract)
    export = {
        "schema_version": "1.0",
        "case_id": case_id,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "contract": file_record(contract_path, REPO_ROOT),
        "provider": contract["program_source"]["timeline_materialization"],
        "note_zh": "49 帧逐一调用程序状态函数；没有从程序截图抠颜色，也没有在四张关键帧之间做像素插值。",
    }
    write_json(case_root / "timeline-export.json", export)
    return contract, samples


def _timeline_plan(source: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    case_id = source["case_id"]
    return {
        **source,
        "plan_id": f"S3.9-{case_id}-FULL-TIMELINE-V1",
        "role": "deterministic_full_program_timeline",
        "contract": (OUTPUT / case_id / "timeline-contract.json").relative_to(STAGE3).as_posix(),
        "keyframe_ids": [item["keyframe_id"] for item in contract["keyframes"]],
    }


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _program_checks(case_id: str, samples: list[Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    states = [sample.state for sample in samples]
    if case_id == "MATH-01":
        trace = [item["trace_fraction"] for item in states]
        sync = [abs(item["circle_normalized_y"] - item["curve_normalized_y"]) for item in states]
        point_counts = [next(record for record in row["operator_records"] if record["operator_id"] == "tracked_points")["object_count"] for row in manifest["records"]]
        return [
            _check("circle_and_curve_remain_synchronized", max(sync) < 1e-7, max(sync)),
            _check("trace_never_retracts", all(a <= b for a, b in zip(trace, trace[1:])), trace),
            _check("two_tracked_points_exist_in_every_frame", set(point_counts) == {2}, point_counts),
        ]
    if case_id == "PHYS-02":
        groups = {name: [item for item in states if item["stage"] == name] for name in ("approach", "stopped", "withdraw")}
        coil = [item["coil_x"] for item in states]
        current_ok = all(item["induced_current"] >= -1e-8 for item in groups["approach"]) and all(abs(item["induced_current"]) < 1e-8 for item in groups["stopped"]) and all(item["induced_current"] <= 1e-8 for item in groups["withdraw"])
        magnet = [item["magnet_x"] for item in states]
        return [
            _check("coil_remains_fixed", len(set(coil)) == 1, coil),
            _check("approach_stop_withdraw_current_signs", current_ok, [item["induced_current"] for item in states]),
            _check("magnet_approaches_holds_then_withdraws", all(a <= b for a, b in zip(magnet[:17], magnet[1:17])) and len(set(magnet[17:33])) == 1 and all(a >= b for a, b in zip(magnet[32:], magnet[33:])), magnet),
        ]
    if case_id == "CHEM-02":
        solvent = [item["solvent_volume"] for item in states]
        concentration = [item["concentration"] for item in states]
        counts = [item["crystal_count"] for item in states]
        mass = [item["total_solute_mass"] for item in states]
        return [
            _check("solvent_decreases_and_concentration_increases", all(a > b for a, b in zip(solvent, solvent[1:])) and all(a < b for a, b in zip(concentration, concentration[1:])), {"solvent": solvent, "concentration": concentration}),
            _check("crystals_only_nucleate_after_threshold_and_never_disappear", all(count == 0 for count, state in zip(counts, states) if state["progress"] < .55) and all(a <= b for a, b in zip(counts, counts[1:])), counts),
            _check("total_solute_mass_is_conserved", max(mass) - min(mass) < 1e-7, mass),
        ]
    if case_id == "BIO-02":
        aperture = [item["aperture_px"] for item in states]
        turgor = [item["turgor"] for item in states]
        counts = [item["guard_cell_count"] for item in states]
        peak_index = int(np.argmax(aperture))
        return [
            _check("two_guard_cells_persist", set(counts) == {2}, counts),
            _check(
                "pore_opens_to_midpoint_peak_then_closes_symmetrically",
                peak_index == 24
                and all(a <= b for a, b in zip(aperture[:25], aperture[1:25]))
                and all(a >= b for a, b in zip(aperture[24:], aperture[25:]))
                and max(abs(left - right) for left, right in zip(aperture, reversed(aperture))) < 1e-7,
                {"peak_index": peak_index, "aperture_px": aperture},
            ),
            _check("aperture_tracks_turgor", max(abs(a - (10 + (t - .25) / .525 * 39)) for a, t in zip(aperture, turgor)) < 1e-5, {"aperture": aperture, "turgor": turgor}),
        ]
    neck = [item["neck_width_px"] for item in states]
    cutoff_indices = [index for index, item in enumerate(states) if item["cutoff_complete"]]
    return [
        _check("neck_width_never_increases", all(a >= b for a, b in zip(neck, neck[1:])), neck),
        _check("main_channel_remains_connected", {item["main_channel_components"] for item in states} == {1}, [item["main_channel_components"] for item in states]),
        _check("oxbow_appears_only_after_cutoff", bool(cutoff_indices) and all(states[index]["isolated_oxbow_count"] == 0 for index in range(cutoff_indices[0])) and all(states[index]["isolated_oxbow_count"] == 1 for index in cutoff_indices), cutoff_indices),
    ]


def _preview(frames: list[np.ndarray], target: Path) -> Path:
    indices = [round(index * (len(frames) - 1) / 8) for index in range(9)]
    canvas = Image.new("RGB", (960, 618), (13, 29, 32))
    draw = ImageDraw.Draw(canvas)
    font = _font(15)
    for slot, frame_index in enumerate(indices):
        image = Image.fromarray(frames[frame_index]).convert("RGB")
        image.thumbnail((320, 176))
        x, y = slot % 3 * 320, slot // 3 * 206
        canvas.paste(image, (x + (320 - image.width) // 2, y))
        draw.text((x + 10, y + 182), f"frame {frame_index} / {frame_index / FPS:.2f}s", fill=(235, 247, 242), font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=92, subsampling=0)
    return target


def _accepted_keyframes(case_id: str) -> list[Path]:
    root = STAGE3 / f"output/phase-8-scale-image/{case_id}/candidate/frames"
    return [root / f"{keyframe_id}.png" for keyframe_id in KEYFRAME_IDS]


def _video_checks(case_id: str, video: Path, frame_paths: list[Path]) -> tuple[dict[str, Any], list[np.ndarray]]:
    info, decoded = decode_video(video)
    endpoints = endpoint_metrics(decoded, frame_paths[0], frame_paths[-1])
    consecutive = consecutive_metrics(decoded)
    sparse = audit_sparse_checkpoints(
        decoded,
        list(zip(KEYFRAME_INDICES, _accepted_keyframes(case_id))),
        maximum_mae_0_255=THRESHOLDS["maximum_sparse_checkpoint_mae_0_255"],
    )
    endpoint_passed = all(item["mean_absolute_pixel_error_0_255"] <= THRESHOLDS["maximum_endpoint_mae_0_255"] for item in endpoints.values())
    checks = [
        _check("encoded_video_has_all_49_frames", info["frame_count"] == FRAME_COUNT, info),
        _check("first_and_last_frames_follow_rendered_inputs", endpoint_passed, {"metrics": endpoints, "threshold": THRESHOLDS["maximum_endpoint_mae_0_255"]}),
        _check("no_abrupt_pixel_jump", consecutive["maximum"] <= THRESHOLDS["maximum_consecutive_frame_mae_0_255"], {**consecutive, "threshold": THRESHOLDS["maximum_consecutive_frame_mae_0_255"]}),
        sparse,
    ]
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "classification": "deterministic full-program-timeline State Renderer fallback",
        "thresholds_frozen_before_audit": THRESHOLDS,
        "video": info,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }, decoded


def _render_case(case_id: str, source_plan: dict[str, Any]) -> dict[str, Any]:
    case_root = OUTPUT / case_id
    contract, samples = _export_case(case_id, source_plan)
    plan = _timeline_plan(source_plan, contract)
    plan_path = case_root / "state-render-plan.json"
    write_json(plan_path, plan)
    manifest = render_plan(plan, STAGE3, REPO_ROOT, case_root / "render")
    frame_paths = [REPO_ROOT / item["output"]["path"] for item in manifest["records"]]
    frames = [np.asarray(Image.open(path).convert("RGB")) for path in frame_paths]
    video = case_root / "deterministic/transition.mp4"
    encode_video(frames, video, fps=FPS)
    video_gate, decoded = _video_checks(case_id, video, frame_paths)
    program_checks = _program_checks(case_id, samples, manifest)
    preview = _preview(decoded, case_root / "deterministic/generated-frames.jpg")
    result = {
        "schema_version": "1.0",
        "case_id": case_id,
        "classification": "deterministic full-program-timeline State Renderer fallback",
        "passed": video_gate["passed"] and all(item["passed"] for item in program_checks),
        "timeline_export": file_record(case_root / "timeline-export.json", REPO_ROOT),
        "timeline_contract": file_record(case_root / "timeline-contract.json", REPO_ROOT),
        "state_render_plan": file_record(plan_path, REPO_ROOT),
        "render_manifest": file_record(case_root / "render/manifest.json", REPO_ROOT),
        "video": file_record(video, REPO_ROOT),
        "preview": file_record(preview, REPO_ROOT),
        "video_gate": video_gate,
        "program_mechanism_checks": program_checks,
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
        "limitation_zh": "运动来自 49 个程序状态，外观来自已验收供体；这是机制正确的确定性视频基线，不冒充视频模型输出。",
    }
    write_json(case_root / "g4.json", result)
    return result


def _cross_case_regression() -> dict[str, Any]:
    accepted = load_json(STAGE3 / "baselines/accepted.json")
    watched_prefixes = (
        "VIDEO-PHYS-01-",
        "VIDEO-BIO-01-",
        "VIDEO-GEO-02-",
        "SEQUENCE-CHEM-01-",
        "SEQUENCE-MATH-02-",
    )
    records = [item for item in accepted["records"] if item["baseline_id"].startswith(watched_prefixes)]
    checks = []
    for record in records:
        path = REPO_ROOT / record["path"]
        checks.append(_check(record["baseline_id"], path.is_file() and sha256_path(path) == record["sha256"] and path.stat().st_size == record["size_bytes"], record["path"]))
    value = {"schema_version": "1.0", "checks": checks, "passed": bool(checks) and all(item["passed"] for item in checks)}
    write_json(OUTPUT / "cross-case-motion-regression.json", value)
    return value


def run() -> dict[str, Any]:
    previous = OUTPUT / "g4-machine.json"
    if previous.is_file() and not load_json(previous).get("passed", False):
        rejected = OUTPUT / "g4-machine-v1-rejected.json"
        if not rejected.is_file():
            write_json(rejected, load_json(previous))
    plan_map = {item["case_id"]: item for item in load_json(PLAN_PATH)["plans"]}
    results = {}
    for case_id in CASES:
        existing = OUTPUT / case_id / "g4.json"
        value = load_json(existing) if existing.is_file() else None
        results[case_id] = value if value and value.get("passed", False) else _render_case(case_id, plan_map[case_id])
    regression = _cross_case_regression()
    summary = {
        "schema_version": "1.0",
        "experiment_id": "EXP-S3-20260731-030",
        "classification": "five deterministic full-program-timeline baselines",
        "cases": {case_id: {"passed": value["passed"], "g4": file_record(OUTPUT / f"{case_id}/g4.json", REPO_ROOT)} for case_id, value in results.items()},
        "cross_case_motion_regression": regression,
        "passed": all(value["passed"] for value in results.values()) and regression["passed"],
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
    }
    write_json(OUTPUT / "g4-machine.json", summary)
    if not summary["passed"]:
        raise RuntimeError("at least one scale motion baseline failed G4")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
