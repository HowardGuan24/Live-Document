"""Rerun the incomplete S3.6 release gate with BIO-01 as the target.

This module preserves the original phase-6 output as historical evidence.  It
creates a new rerun directory, executes one deterministic negative control and
one candidate, then checks two cross-discipline State Renderer regressions and
the delta historical baseline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    validate_input_contract,
    validate_visual_target,
    verify_file_record,
    write_json,
)
from modules.video_model.stage3.framework.state_renderer import render_plan


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output" / "phase-6-rerun-1"
PLAN_PATH = STAGE3 / "state_render_plans_v2.json"
V1_PLAN_PATH = STAGE3 / "state_render_plans.json"
KEYFRAME_LABELS = {
    "00_start": "START",
    "01_mechanism": "MECHANISM",
    "02_result": "RESULT",
    "03_end": "END",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _operator(
    record: dict[str, Any], operator_type: str
) -> dict[str, Any]:
    return next(
        value
        for value in record["operator_records"]
        if value["operator_type"] == operator_type
    )


def _sheet(
    items: list[tuple[str, Path]],
    output: Path,
    *,
    columns: int,
    cell: tuple[int, int] = (480, 320),
) -> None:
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (cell[0] * columns, cell[1] * rows), (13, 29, 32)
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell[0] - 16, cell[1] - 48))
        x = (index % columns) * cell[0]
        y = (index // columns) * cell[1]
        canvas.paste(
            image,
            (
                x + (cell[0] - image.width) // 2,
                y + 8,
            ),
        )
        draw.rectangle(
            (x, y + cell[1] - 38, x + cell[0], y + cell[1]),
            fill=(13, 29, 32),
        )
        draw.text(
            (x + 10, y + cell[1] - 29),
            label,
            fill=(236, 247, 242),
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, subsampling=0)


def observe() -> dict[str, Any]:
    registry = load_json(STAGE3 / "case_registry.json")
    state = load_json(STAGE3 / "state.json")
    release = load_json(STAGE3 / "output/phase-6/release-audit.json")
    accepted = load_json(STAGE3 / "baselines/accepted.json")
    visual_status = {
        item["case_id"]: item["visual_target_status"]
        for item in registry["cases"]
        if item["case_id"] != "GEO-HIST-DELTA-01"
    }
    accepted_core_record = next(
        item
        for item in accepted["records"]
        if item["baseline_id"] == "CORE-STATE-RENDERER-B-V1"
    )
    observation = {
        "schema_version": "1.0",
        "loop_id": "LOOP-S3-0002",
        "source_documents": [
            file_record(STAGE3 / "loop.md", REPO_ROOT),
            file_record(STAGE3 / "workflow.html", REPO_ROOT),
        ],
        "previous_state": state,
        "previous_release_claim": {
            "path": "modules/video_model/stage3/output/phase-6/release-audit.json",
            "alpha_release_passed": release["alpha_release_passed"],
            "validated_back_half_cases": [
                "CHEM-01",
                "MATH-02",
                "PHYS-01",
            ],
            "missing_back_half_cases": ["BIO-01", "GEO-02"],
            "audit_verdict": "invalid_phase_exit",
            "reason_zh": (
                "loop.md 的 S3.6 要求数学、物理、化学、生物、地理代表案例都运行正式"
                "发布回归；旧审计只让 BIO/GEO 通过 G0/G1，却把 alpha 标为 passed。"
            ),
        },
        "current_best": {
            "state_renderer_baseline": accepted_core_record,
            "validated_state_sequences": [
                "CHEM-01",
                "MATH-02",
                "PHYS-01",
            ],
            "historical_bio_reference": file_record(
                REPO_ROOT
                / "modules/video_model/stage2/output/phase-7/route-b/"
                "BIO-01/variants/stable_material_plus_depth/02_result.png",
                REPO_ROOT,
            ),
        },
        "visual_target_status_by_case": visual_status,
        "prioritized_problems": [
            {
                "rank": 1,
                "problem_id": "S3-PROBLEM-RELEASE-CLAIM-001",
                "taxonomy": "gate_or_selector",
                "summary_zh": "S3.6 退出条件被错误缩减为五学科 G0/G1。",
            },
            {
                "rank": 2,
                "problem_id": "S3-PROBLEM-BACKHALF-BIO-001",
                "taxonomy": "state_renderer",
                "summary_zh": (
                    "BIO-01 有完整 Visual Target 和语义层，但 Stage 3 没有 G2/G3。"
                ),
            },
            {
                "rank": 3,
                "problem_id": "S3-PROBLEM-VISUAL-001",
                "taxonomy": "visual_target",
                "summary_zh": "GEO-02 Visual Target 仍是 provisional。",
            },
        ],
        "selected_problem_id": "S3-PROBLEM-BACKHALF-BIO-001",
        "selected_cohort": {
            "target": "BIO-01",
            "route_regressions": ["CHEM-01", "MATH-02"],
            "historical_regression": "GEO-HIST-DELTA-01",
            "selection_reason_zh": (
                "BIO-01 是离五学科发布出口最近的硬缺口，且已有 accepted Visual Target；"
                "CHEM/MATH 与它共享 region/object_identity 数据类型并属于不同学科。"
            ),
        },
        "budget": {
            "new_image_model_candidates": 0,
            "deterministic_image_candidates": 2,
            "video_model_candidates_after_image_gate": 2,
        },
    }
    write_json(OUTPUT / "observation.json", observation)
    return observation


def preflight() -> dict[str, Any]:
    observe()
    plans = load_json(PLAN_PATH)
    source = (STAGE3 / "framework/state_renderer.py").read_text(
        encoding="utf-8"
    )
    contract_smoke = []
    for path in sorted((STAGE3 / "contracts").glob("*.json")):
        error = None
        try:
            validate_input_contract(load_json(path))
        except Exception as exc:  # evidence is written before stopping
            error = f"{type(exc).__name__}: {exc}"
        contract_smoke.append(
            {
                "case_id": path.stem,
                "passed": error is None,
                "error": error,
                "contract": file_record(path, REPO_ROOT),
            }
        )
    target_contract = load_json(STAGE3 / "contracts/BIO-01.json")
    target_visual_path = STAGE3 / target_contract[
        "visual_target_package"
    ]["path"].replace("modules/video_model/stage3/", "")
    target_visual = load_json(target_visual_path)
    target_visual_error = None
    try:
        validate_visual_target(target_visual)
        for record in (
            target_visual["positive_refs"] + target_visual["negative_refs"]
        ):
            verify_file_record(record, REPO_ROOT)
    except Exception as exc:
        target_visual_error = f"{type(exc).__name__}: {exc}"
    plan_inputs = []
    for plan in plans["plans"]:
        anchor_path = STAGE3 / plan["anchor"]["path"]
        contract_path = STAGE3 / plan["contract"]
        plan_inputs.append(
            {
                "plan_id": plan["plan_id"],
                "anchor": file_record(anchor_path, REPO_ROOT),
                "contract": file_record(contract_path, REPO_ROOT),
                "keyframes_match_contract": plan["keyframe_ids"]
                == [
                    item["keyframe_id"]
                    for item in load_json(contract_path)["keyframes"]
                ],
            }
        )
    provenance = plans["appearance_anchor_provenance"]
    provenance_files = []
    for name in (
        "source_experiment",
        "positive_prompt",
        "negative_prompt",
        "control",
        "model_fingerprints",
    ):
        path = STAGE3 / provenance[name]
        provenance_files.append(
            {"role": name, **file_record(path, REPO_ROOT)}
        )
    checks = [
        {
            "name": "old_S3_6_claim_is_not_reused",
            "passed": True,
            "evidence": "output/phase-6 is retained as history; rerun writes a new directory",
        },
        {
            "name": "all_eleven_contracts_pass_smoke",
            "passed": all(item["passed"] for item in contract_smoke),
            "evidence": contract_smoke,
        },
        {
            "name": "target_visual_target_is_accepted_and_resolves",
            "passed": target_visual_error is None
            and target_visual["status"] == "accepted_project_baseline",
            "evidence": {
                "status": target_visual["status"],
                "error": target_visual_error,
            },
        },
        {
            "name": "target_has_exactly_one_negative_and_one_candidate",
            "passed": [item["role"] for item in plans["plans"]]
            == ["negative_control", "candidate"],
            "evidence": [item["role"] for item in plans["plans"]],
        },
        {
            "name": "single_variable_is_frozen",
            "passed": plans["single_changed_variable"]
            == "region_material.transfer_mode",
            "evidence": plans["single_changed_variable"],
        },
        {
            "name": "core_has_no_case_ids_or_model_runtime",
            "passed": not any(
                token in source
                for token in (
                    "BIO-01",
                    "CHEM-01",
                    "MATH-02",
                    "PHYS-01",
                    "diffusers",
                    "ControlNetModel",
                    "torch",
                )
            ),
            "evidence": "source token scan",
        },
        {
            "name": "all_plan_and_provenance_files_resolve",
            "passed": all(
                item["keyframes_match_contract"] for item in plan_inputs
            ),
            "evidence": {
                "plans": plan_inputs,
                "provenance": provenance_files,
            },
        },
    ]
    result = {
        "schema_version": "1.0",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "model_budget": {
            "new_image_model_candidates": 0,
            "deterministic_image_candidates": 2,
            "video_model_candidates_after_image_gate": 2,
        },
    }
    write_json(OUTPUT / "preflight.json", result)
    if not result["passed"]:
        raise RuntimeError("S3.6 rerun preflight failed")
    return result


def _render_target() -> dict[str, dict[str, Any]]:
    plans = load_json(PLAN_PATH)["plans"]
    result = {}
    for plan in plans:
        role = plan["role"]
        target = OUTPUT / "BIO-01" / role
        manifest = render_plan(plan, STAGE3, REPO_ROOT, target)
        manifest["role"] = role
        write_json(target / "manifest.json", manifest)
        result[role] = manifest
        _sheet(
            [
                (
                    KEYFRAME_LABELS[record["keyframe_id"]],
                    REPO_ROOT / record["output"]["path"],
                )
                for record in manifest["records"]
            ],
            target / "sequence.jpg",
            columns=4,
            cell=(420, 280),
        )
    _sheet(
        [
            (
                f"NEGATIVE / {KEYFRAME_LABELS[record['keyframe_id']]}",
                REPO_ROOT / record["output"]["path"],
            )
            for record in result["negative_control"]["records"]
        ]
        + [
            (
                f"CANDIDATE / {KEYFRAME_LABELS[record['keyframe_id']]}",
                REPO_ROOT / record["output"]["path"],
            )
            for record in result["candidate"]["records"]
        ],
        OUTPUT / "report-assets/target-comparison.jpg",
        columns=4,
        cell=(420, 280),
    )
    return result


def _machine_gate_target(
    manifests: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    plan = next(
        item
        for item in load_json(PLAN_PATH)["plans"]
        if item["role"] == "candidate"
    )
    gates = plan["gates"]
    candidate = manifests["candidate"]
    region_records = [
        _operator(record, "region_material")
        for record in candidate["records"]
    ]
    identity_records = [
        _operator(record, "identity_stroke")
        for record in candidate["records"]
    ]
    states = [
        load_json(REPO_ROOT / record["source_state"]["path"])
        for record in candidate["records"]
    ]
    components = [
        item["connected_component_count"] for item in region_records
    ]
    object_counts = [item["object_count"] for item in identity_records]
    lineage_units = [
        item["lineage_unit_count"] for item in identity_records
    ]
    final_parent_counts = [
        item["children_per_parent"] for item in identity_records[2:]
    ]
    final_destinations = [
        item["destination_counts"] for item in identity_records[2:]
    ]
    outside = [
        record["metrics"]["outside_mutable_max_difference_0_255"]
        for record in candidate["records"]
    ]
    checks = [
        {
            "name": "cell_topology_matches_program",
            "passed": components == gates["cell_connected_components"],
            "evidence": components,
        },
        {
            "name": "identity_object_counts_match_program",
            "passed": object_counts == gates["identity_object_counts"],
            "evidence": object_counts,
        },
        {
            "name": "no_chromosome_lineage_unit_is_created_or_lost",
            "passed": lineage_units == gates["lineage_unit_counts"],
            "evidence": lineage_units,
        },
        {
            "name": "six_parents_each_map_to_two_sisters",
            "passed": all(
                len(value) == 6
                and set(value.values())
                == {gates["sisters_per_parent_at_result_and_end"]}
                for value in final_parent_counts
            ),
            "evidence": final_parent_counts,
        },
        {
            "name": "sisters_split_six_left_six_right",
            "passed": all(
                [
                    value.get("left", 0),
                    value.get("right", 0),
                ]
                == gates[
                    "left_right_destination_counts_at_result_and_end"
                ]
                for value in final_destinations
            ),
            "evidence": final_destinations,
        },
        {
            "name": "mitosis_stage_order_is_preserved",
            "passed": [value["stage"] for value in states]
            == [
                "prophase_to_metaphase",
                "prophase_to_metaphase",
                "sister_separation",
                "anaphase_to_cytokinesis",
            ],
            "evidence": [value["stage"] for value in states],
        },
        {
            "name": "outside_declared_mutable_area_is_pixel_exact",
            "passed": max(outside)
            <= gates["outside_declared_mutable_max_difference_0_255"],
            "evidence": outside,
        },
    ]
    return {
        "schema_version": "1.0",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def _run_regressions() -> dict[str, Any]:
    v1_plans = {
        item["case_id"]: item
        for item in load_json(V1_PLAN_PATH)["plans"]
    }
    records = []
    for case_id in ("CHEM-01", "MATH-02"):
        output = OUTPUT / "regressions" / case_id
        rebuilt = render_plan(
            v1_plans[case_id], STAGE3, REPO_ROOT, output
        )
        old = load_json(
            STAGE3 / f"output/phase-4/{case_id}/manifest.json"
        )
        comparisons = []
        for old_record, new_record in zip(
            old["records"], rebuilt["records"], strict=True
        ):
            comparisons.append(
                {
                    "keyframe_id": old_record["keyframe_id"],
                    "old_sha256": old_record["output"]["sha256"],
                    "rebuilt_sha256": new_record["output"]["sha256"],
                    "pixel_exact": old_record["output"]["sha256"]
                    == new_record["output"]["sha256"],
                }
            )
        records.append(
            {
                "case_id": case_id,
                "passed": all(
                    item["pixel_exact"] for item in comparisons
                ),
                "comparisons": comparisons,
                "rebuilt_manifest": file_record(
                    output / "manifest.json", REPO_ROOT
                ),
            }
        )
    delta_records = []
    accepted = load_json(STAGE3 / "baselines/accepted.json")
    for baseline_id in (
        "APPEARANCE-GEO-HIST-DELTA-01-b094c3d54074",
        "APPEARANCE-GEO-HIST-DELTA-01-e90fb592ea60",
    ):
        record = next(
            item
            for item in accepted["records"]
            if item["baseline_id"] == baseline_id
        )
        error = None
        try:
            verify_file_record(record, REPO_ROOT)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        delta_records.append(
            {
                "baseline_id": baseline_id,
                "passed": error is None,
                "error": error,
                "record": record,
            }
        )
    result = {
        "schema_version": "1.0",
        "passed": all(item["passed"] for item in records)
        and all(item["passed"] for item in delta_records),
        "route_regressions": records,
        "historical_regression": {
            "case_id": "GEO-HIST-DELTA-01",
            "passed": all(item["passed"] for item in delta_records),
            "records": delta_records,
        },
    }
    write_json(OUTPUT / "cross-case-regression.json", result)
    return result


def _determinism_replay(
    candidate_plan: dict[str, Any],
    first: dict[str, Any],
) -> dict[str, Any]:
    replay_dir = OUTPUT / "BIO-01" / "candidate-replay"
    second = render_plan(
        candidate_plan, STAGE3, REPO_ROOT, replay_dir
    )
    records = []
    for a, b in zip(first["records"], second["records"], strict=True):
        records.append(
            {
                "keyframe_id": a["keyframe_id"],
                "first_sha256": a["output"]["sha256"],
                "replay_sha256": b["output"]["sha256"],
                "identical": a["output"]["sha256"]
                == b["output"]["sha256"],
            }
        )
    result = {
        "schema_version": "1.0",
        "passed": all(item["identical"] for item in records),
        "records": records,
        "seed_robustness": {
            "status": "not_applicable",
            "reason_zh": (
                "本轮没有重新运行图片模型；同一冻结外观供体进入确定性 State Renderer。"
                "因此检查的是逐文件重建一致性，不伪装成多 seed 结论。"
            ),
        },
    }
    write_json(OUTPUT / "determinism-replay.json", result)
    return result


def render_and_gate() -> dict[str, Any]:
    preflight()
    manifests = _render_target()
    machine = _machine_gate_target(manifests)
    write_json(OUTPUT / "BIO-01/g3-machine.json", machine)
    if not machine["passed"]:
        raise RuntimeError("BIO-01 machine gate failed")
    regressions = _run_regressions()
    candidate_plan = next(
        item
        for item in load_json(PLAN_PATH)["plans"]
        if item["role"] == "candidate"
    )
    replay = _determinism_replay(
        candidate_plan, manifests["candidate"]
    )
    summary = {
        "schema_version": "1.0",
        "status": "awaiting_blind_visual_review",
        "target_machine_gate": machine["passed"],
        "cross_case_regression": regressions["passed"],
        "determinism_replay": replay["passed"],
        "model_runs": {
            "new_image_candidates": 0,
            "deterministic_candidates": 2,
            "video_candidates": 0,
        },
        "next_action": (
            "Blind visual review against BIO-01 positive/negative references; "
            "only then decide whether video is allowed."
        ),
    }
    write_json(OUTPUT / "render-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("observe", "preflight", "render"),
        nargs="?",
        default="render",
    )
    args = parser.parse_args()
    if args.action == "observe":
        value = observe()
    elif args.action == "preflight":
        value = preflight()
    else:
        value = render_and_gate()
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
