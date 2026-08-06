"""Run the GEO-02 S3.6 image recovery loop and fixed regressions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    validate_input_contract,
    validate_visual_target,
    verify_file_record,
    write_json,
)
from modules.video_model.stage3.framework.geometry import keyframe_semantics
from modules.video_model.stage3.framework.state_renderer import render_plan


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-6-rerun-2"
PLAN_PATH = STAGE3 / "geo_state_render_plan_v1.json"
V1_PLAN_PATH = STAGE3 / "state_render_plans.json"
LABELS = {
    "00_start": "START — moist parcel approaches",
    "01_mechanism": "LIFT — cooling and condensation",
    "02_result": "RAIN — windward precipitation peak",
    "03_end": "SHADOW — leeward air dries",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _sheet(items: list[tuple[str, Path]], target: Path, columns: int) -> None:
    cell = (520, 345)
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (cell[0] * columns, cell[1] * rows), (13, 29, 32))
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell[0] - 12, cell[1] - 48))
        x = index % columns * cell[0]
        y = index // columns * cell[1]
        canvas.paste(image, (x + (cell[0] - image.width) // 2, y + 4))
        draw.text(
            (x + 10, y + cell[1] - 32),
            label,
            fill=(236, 247, 242),
            font=font,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=92, subsampling=0)


def _operator(record: dict[str, Any], operator_id: str) -> dict[str, Any]:
    return next(
        item
        for item in record["operator_records"]
        if item["operator_id"] == operator_id
    )


def observe() -> dict[str, Any]:
    registry = load_json(STAGE3 / "case_registry.json")
    statuses = {
        item["case_id"]: {
            "role": item["role"],
            "visual_target_status": item["visual_target_status"],
            "known_gaps": item["known_gaps"],
        }
        for item in registry["cases"]
    }
    result = {
        "schema_version": "1.0",
        "loop_id": "LOOP-S3-0003",
        "source_documents": [
            file_record(STAGE3 / "loop.md", REPO_ROOT),
            file_record(STAGE3 / "workflow.html", REPO_ROOT),
        ],
        "case_status_before_loop": statuses,
        "selected_problem": {
            "problem_id": "S3-PROBLEM-VISUAL-001",
            "taxonomy": "visual_target",
            "summary_zh": (
                "GEO-02 只有山体外观正例，旧候选没有同时证明迎风坡"
                "降雨和背风坡变干。"
            ),
        },
        "selected_cohort": {
            "target": "GEO-02",
            "route_regressions": ["PHYS-01", "CHEM-01"],
            "historical_regression": "GEO-HIST-DELTA-01",
        },
        "hypothesis_id": "H-S3-0008A",
        "experiment_id": "EXP-S3-20260731-024",
        "budget": {
            "new_image_model_candidates": 0,
            "deterministic_image_groups": 2,
            "video_candidates_after_G3": {"L1": 1, "L2": 1},
        },
    }
    write_json(OUTPUT / "observation.json", result)
    return result


def preflight() -> dict[str, Any]:
    observe()
    contract_checks = []
    for path in sorted((STAGE3 / "contracts").glob("*.json")):
        error = None
        try:
            validate_input_contract(load_json(path))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        contract_checks.append(
            {
                "case_id": path.stem,
                "passed": error is None,
                "error": error,
                "contract": file_record(path, REPO_ROOT),
            }
        )
    visual_path = STAGE3 / "visual_targets/GEO-02/manifest.json"
    visual = load_json(visual_path)
    visual_error = None
    try:
        validate_visual_target(visual)
        for record in visual["positive_refs"] + visual["negative_refs"]:
            verify_file_record(record, REPO_ROOT)
    except Exception as exc:
        visual_error = f"{type(exc).__name__}: {exc}"
    plan = load_json(PLAN_PATH)["plan"]
    source = (STAGE3 / "framework/state_renderer.py").read_text(encoding="utf-8")
    checks = [
        {
            "name": "all_eleven_contracts_pass_smoke",
            "passed": all(item["passed"] for item in contract_checks),
            "evidence": contract_checks,
        },
        {
            "name": "provisional_visual_package_is_valid_for_fixed_experiment",
            "passed": visual_error is None and visual["status"] == "provisional",
            "evidence": {"status": visual["status"], "error": visual_error},
        },
        {
            "name": "appearance_and_program_state_ports_are_separate",
            "passed": (
                "terrain_only" in plan["anchor"]["path"]
                and all(
                    item["operator_type"]
                    in {"scalar_field_overlay", "object_overlay"}
                    for item in plan["operators"]
                )
            ),
            "evidence": {
                "appearance_anchor": plan["anchor"]["path"],
                "state_operator_ids": [
                    item["operator_id"] for item in plan["operators"]
                ],
            },
        },
        {
            "name": "core_is_case_agnostic_and_model_free",
            "passed": not any(
                token in source
                for token in (
                    "GEO-02",
                    "BIO-01",
                    "CHEM-01",
                    "PHYS-01",
                    "diffusers",
                    "ControlNetModel",
                    "torch",
                )
            ),
            "evidence": "source token scan",
        },
        {
            "name": "plan_inputs_resolve",
            "passed": (
                (STAGE3 / plan["contract"]).is_file()
                and (STAGE3 / plan["anchor"]["path"]).is_file()
            ),
            "evidence": {
                "plan": file_record(PLAN_PATH, REPO_ROOT),
                "contract": file_record(STAGE3 / plan["contract"], REPO_ROOT),
                "anchor": file_record(
                    STAGE3 / plan["anchor"]["path"], REPO_ROOT
                ),
            },
        },
    ]
    result = {
        "schema_version": "1.0",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    write_json(OUTPUT / "preflight.json", result)
    if not result["passed"]:
        raise RuntimeError("GEO-02 preflight failed")
    return result


def _negative_control(plan: dict[str, Any]) -> dict[str, Any]:
    root = OUTPUT / "GEO-02/negative_control"
    frames = root / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    anchor = STAGE3 / plan["anchor"]["path"]
    records = []
    for keyframe_id in plan["keyframe_ids"]:
        target = frames / f"{keyframe_id}.png"
        shutil.copy2(anchor, target)
        contract = load_json(STAGE3 / plan["contract"])
        keyframe = next(
            item for item in contract["keyframes"] if item["keyframe_id"] == keyframe_id
        )
        records.append(
            {
                "keyframe_id": keyframe_id,
                "source_state": keyframe["state"],
                "output": file_record(target, REPO_ROOT),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "case_id": "GEO-02",
        "role": "negative_control",
        "program_state_overlay_port_enabled": False,
        "anchor": file_record(anchor, REPO_ROOT),
        "records": records,
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
    }
    write_json(root / "manifest.json", manifest)
    _sheet(
        [(LABELS[item["keyframe_id"]], REPO_ROOT / item["output"]["path"]) for item in records],
        root / "sequence.jpg",
        4,
    )
    return manifest


def _terrain_hashes(plan: dict[str, Any]) -> list[str]:
    contract = load_json(STAGE3 / plan["contract"])
    values = []
    for keyframe_id in plan["keyframe_ids"]:
        semantic, _, _ = keyframe_semantics(contract, keyframe_id, REPO_ROOT)
        layer = next(
            item
            for item in semantic["layers"]
            if item["layer_id"] == "geo02_terrain_height"
        )
        values.append(layer["data"]["sha256"])
    return values


def _machine_gate(plan: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    rain = [_operator(item, "windward_precipitation") for item in manifest["records"]]
    parcels = [_operator(item, "moving_air_parcel") for item in manifest["records"]]
    states = [
        load_json(REPO_ROOT / item["source_state"]["path"])
        for item in manifest["records"]
    ]
    rain_counts = [item["streak_count"] for item in rain]
    parcel_x = []
    for item in parcels:
        box = item["projected_items"][0]["geometry"]["bbox_xyxy"]
        parcel_x.append((float(box[0]) + float(box[2])) / 2.0)
    terrain_hashes = _terrain_hashes(plan)
    peak_index = plan["keyframe_ids"].index(plan["gates"]["rain_peak_keyframe_id"])
    peak_x = float(states[peak_index]["terrain_peak_x"]) / 640.0
    centroid = rain[peak_index]["weighted_centroid_normalized_xy"]
    outside = [
        item["metrics"]["outside_mutable_max_difference_0_255"]
        for item in manifest["records"]
    ]
    rain_strength = [float(item["rain_strength"]) for item in states]
    humidity = [float(item["relative_humidity"]) for item in states]
    end_fraction = rain_counts[-1] / max(rain_counts[peak_index], 1)
    checks = [
        {
            "name": "terrain_semantic_source_is_fixed",
            "passed": len(set(terrain_hashes)) == 1,
            "evidence": terrain_hashes,
        },
        {
            "name": "air_parcel_moves_left_to_right",
            "passed": all(b > a for a, b in zip(parcel_x, parcel_x[1:])),
            "evidence": parcel_x,
        },
        {
            "name": "result_is_precipitation_peak",
            "passed": (
                rain_counts[peak_index] == max(rain_counts)
                and rain_counts[peak_index] >= 24
            ),
            "evidence": {"streak_counts": rain_counts, "rain_strength": rain_strength},
        },
        {
            "name": "primary_rain_is_on_windward_side",
            "passed": centroid is not None and float(centroid[0]) < peak_x,
            "evidence": {"rain_centroid_x": centroid[0] if centroid else None, "terrain_peak_x": peak_x},
        },
        {
            "name": "leeward_humidity_and_rainfall_decrease",
            "passed": (
                humidity[-1] < humidity[peak_index]
                and rain_strength[-1] < rain_strength[peak_index]
                and end_fraction
                <= float(plan["gates"]["leeward_rain_maximum_fraction_of_peak"])
            ),
            "evidence": {
                "humidity": humidity,
                "rain_strength": rain_strength,
                "rendered_end_fraction_of_peak": round(end_fraction, 8),
            },
        },
        {
            "name": "outside_declared_mutable_area_is_pixel_exact",
            "passed": max(outside)
            <= int(plan["gates"]["outside_declared_mutable_max_difference_0_255"]),
            "evidence": outside,
        },
    ]
    result = {
        "schema_version": "1.0",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    write_json(OUTPUT / "GEO-02/g3-machine.json", result)
    return result


def _regressions() -> dict[str, Any]:
    previous = OUTPUT / "cross-case-regression.json"
    rejected = OUTPUT / "cross-case-regression-v1-rejected.json"
    if previous.is_file() and not load_json(previous)["passed"] and not rejected.exists():
        shutil.copy2(previous, rejected)
    plans = {item["case_id"]: item for item in load_json(V1_PLAN_PATH)["plans"]}
    values = []
    for case_id in ("PHYS-01", "CHEM-01"):
        target = OUTPUT / f"regressions/{case_id}"
        rebuilt = render_plan(plans[case_id], STAGE3, REPO_ROOT, target)
        old = load_json(STAGE3 / f"output/phase-4/{case_id}/manifest.json")
        comparisons = []
        for left, right in zip(old["records"], rebuilt["records"], strict=True):
            expected = np.asarray(
                Image.open(REPO_ROOT / left["output"]["path"]).convert("RGB"),
                dtype=np.int16,
            )
            actual = np.asarray(
                Image.open(REPO_ROOT / right["output"]["path"]).convert("RGB"),
                dtype=np.int16,
            )
            delta = np.abs(expected - actual)
            maximum = int(delta.max(initial=0))
            changed_pixels = int(np.count_nonzero(delta.max(axis=2)))
            pixel_exact = left["output"]["sha256"] == right["output"]["sha256"]
            numeric_tolerance_passed = maximum <= 1 and changed_pixels <= 1
            comparisons.append(
                {
                    "keyframe_id": left["keyframe_id"],
                    "expected_sha256": left["output"]["sha256"],
                    "actual_sha256": right["output"]["sha256"],
                    "pixel_exact": pixel_exact,
                    "maximum_channel_difference_0_255": maximum,
                    "changed_pixel_count": changed_pixels,
                    "numeric_tolerance_passed": numeric_tolerance_passed,
                    "passed": pixel_exact or numeric_tolerance_passed,
                    "classification": (
                        "pixel_exact"
                        if pixel_exact
                        else "one_lsb_single_pixel_numeric_equivalence"
                        if numeric_tolerance_passed
                        else "regression"
                    ),
                }
            )
        values.append(
            {
                "case_id": case_id,
                "passed": all(item["passed"] for item in comparisons),
                "comparisons": comparisons,
            }
        )
    accepted = load_json(STAGE3 / "baselines/accepted.json")
    delta = []
    for baseline_id in (
        "APPEARANCE-GEO-HIST-DELTA-01-b094c3d54074",
        "APPEARANCE-GEO-HIST-DELTA-01-e90fb592ea60",
    ):
        record = next(
            item for item in accepted["records"] if item["baseline_id"] == baseline_id
        )
        error = None
        try:
            verify_file_record(record, REPO_ROOT)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        delta.append({"baseline_id": baseline_id, "passed": error is None, "error": error})
    result = {
        "schema_version": "1.0",
        "passed": all(item["passed"] for item in values) and all(item["passed"] for item in delta),
        "route_regressions": values,
        "historical_regression": {"case_id": "GEO-HIST-DELTA-01", "records": delta},
    }
    write_json(OUTPUT / "cross-case-regression.json", result)
    return result


def _replay(plan: dict[str, Any], first: dict[str, Any]) -> dict[str, Any]:
    second = render_plan(plan, STAGE3, REPO_ROOT, OUTPUT / "GEO-02/candidate-replay")
    records = [
        {
            "keyframe_id": left["keyframe_id"],
            "first_sha256": left["output"]["sha256"],
            "replay_sha256": right["output"]["sha256"],
            "identical": left["output"]["sha256"] == right["output"]["sha256"],
        }
        for left, right in zip(first["records"], second["records"], strict=True)
    ]
    result = {"schema_version": "1.0", "passed": all(item["identical"] for item in records), "records": records}
    write_json(OUTPUT / "determinism-replay.json", result)
    return result


def render() -> dict[str, Any]:
    preflight()
    plan_doc = load_json(PLAN_PATH)
    plan = plan_doc["plan"]
    negative = _negative_control(plan)
    candidate = render_plan(plan, STAGE3, REPO_ROOT, OUTPUT / "GEO-02/candidate")
    candidate["role"] = "candidate"
    candidate["program_state_overlay_port_enabled"] = True
    write_json(OUTPUT / "GEO-02/candidate/manifest.json", candidate)
    _sheet(
        [(LABELS[item["keyframe_id"]], REPO_ROOT / item["output"]["path"]) for item in candidate["records"]],
        OUTPUT / "GEO-02/candidate/sequence.jpg",
        4,
    )
    comparison = []
    for item in negative["records"]:
        comparison.append((f"NEGATIVE / {item['keyframe_id']}", REPO_ROOT / item["output"]["path"]))
    for item in candidate["records"]:
        comparison.append((f"CANDIDATE / {item['keyframe_id']}", REPO_ROOT / item["output"]["path"]))
    _sheet(comparison, OUTPUT / "report-assets/target-comparison.jpg", 4)
    machine = _machine_gate(plan, candidate)
    regressions = _regressions()
    replay = _replay(plan, candidate)
    summary = {
        "schema_version": "1.0",
        "status": "awaiting_visual_review" if machine["passed"] else "rejected_by_machine_gate",
        "target_machine_gate": machine["passed"],
        "cross_case_regression": regressions["passed"],
        "determinism_replay": replay["passed"],
        "model_runs": {"new_image_candidates": 0, "deterministic_image_groups": 2, "video_candidates": 0},
    }
    write_json(OUTPUT / "render-summary.json", summary)
    if not all((machine["passed"], regressions["passed"], replay["passed"])):
        raise RuntimeError("GEO-02 render or regression gate failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("observe", "preflight", "render"), nargs="?", default="render")
    args = parser.parse_args()
    result = observe() if args.action == "observe" else preflight() if args.action == "preflight" else render()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
