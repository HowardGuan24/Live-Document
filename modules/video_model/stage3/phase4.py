"""Run S3.4 deterministic State Renderer B and its cross-case gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage3.framework.contracts import (
    load_json,
    sha256_path,
    verify_file_record,
    write_json,
)
from modules.video_model.stage3.framework.state_renderer import render_plan


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output" / "phase-4"
PLANS = STAGE3 / "state_render_plans.json"
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


def _plan_records() -> list[dict[str, Any]]:
    return load_json(PLANS)["plans"]


def preflight() -> dict[str, Any]:
    source = (STAGE3 / "framework/state_renderer.py").read_text(
        encoding="utf-8"
    )
    checks = [
        {
            "name": "plan_set_frozen",
            "passed": load_json(PLANS)["status"]
            == "frozen_before_S3.4_render",
        },
        {
            "name": "three_cross_case_plans",
            "passed": len(_plan_records()) == 3,
        },
        {
            "name": "operator_core_has_no_case_ids",
            "passed": not any(
                case_id in source
                for case_id in ("CHEM-01", "PHYS-01", "MATH-02")
            ),
        },
        {
            "name": "no_model_runtime_import",
            "passed": not any(
                token in source
                for token in ("diffusers", "ControlNetModel", "torch")
            ),
        },
    ]
    plan_checks = []
    for plan in _plan_records():
        contract_path = STAGE3 / plan["contract"]
        anchor_path = STAGE3 / plan["anchor"]["path"]
        contract = load_json(contract_path)
        ids = [item["keyframe_id"] for item in contract["keyframes"]]
        operator_ids = [
            item["operator_id"] for item in plan["operators"]
        ]
        plan_check = {
            "plan_id": plan["plan_id"],
            "passed": (
                anchor_path.is_file()
                and plan["keyframe_ids"] == ids
                and len(operator_ids) == len(set(operator_ids))
            ),
            "anchor_path": str(anchor_path.resolve()),
            "anchor_sha256": (
                sha256_path(anchor_path) if anchor_path.is_file() else None
            ),
            "keyframes_match_contract": plan["keyframe_ids"] == ids,
            "operator_ids_unique": len(operator_ids)
            == len(set(operator_ids)),
        }
        plan_checks.append(plan_check)
    checks.append(
        {
            "name": "all_plan_inputs_complete",
            "passed": all(item["passed"] for item in plan_checks),
        }
    )
    result = {
        "schema_version": "1.0",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "plans": plan_checks,
        "model_budget": {
            "image_candidates": 0,
            "video_candidates": 0,
        },
    }
    write_json(OUTPUT / "preflight.json", result)
    if not result["passed"]:
        raise RuntimeError("S3.4 preflight failed")
    return result


def _sequence_sheet(case_id: str, manifest: dict[str, Any]) -> Path:
    cells = []
    for record in manifest["records"]:
        cells.append(
            (
                KEYFRAME_LABELS[record["keyframe_id"]],
                REPO_ROOT / record["output"]["path"],
            )
        )
    cell = (512, 326)
    sheet = Image.new(
        "RGB", (cell[0] * len(cells), cell[1]), (13, 29, 32)
    )
    draw = ImageDraw.Draw(sheet)
    label_font = _font(18)
    for index, (label, path) in enumerate(cells):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell[0], 288))
        x = index * cell[0]
        sheet.paste(
            image,
            (x + (cell[0] - image.width) // 2, 0),
        )
        draw.text(
            (x + 14, 298),
            f"{index + 1}  {label}",
            fill=(235, 247, 242),
            font=label_font,
        )
    path = OUTPUT / case_id / "sequence.jpg"
    sheet.save(path, quality=92, subsampling=0)
    return path


def _mutable_sheet(case_id: str, manifest: dict[str, Any]) -> Path:
    cell = (384, 250)
    sheet = Image.new(
        "RGB",
        (cell[0] * len(manifest["records"]), cell[1]),
        (13, 29, 32),
    )
    draw = ImageDraw.Draw(sheet)
    label_font = _font(16)
    for index, record in enumerate(manifest["records"]):
        image = Image.open(
            REPO_ROOT / record["mutable_mask"]["path"]
        ).convert("RGB")
        image.thumbnail((cell[0], 216))
        x = index * cell[0]
        sheet.paste(
            image, (x + (cell[0] - image.width) // 2, 0)
        )
        draw.text(
            (x + 12, 224),
            KEYFRAME_LABELS[record["keyframe_id"]],
            fill=(235, 247, 242),
            font=label_font,
        )
    path = OUTPUT / case_id / "mutable-sequence.jpg"
    sheet.save(path, quality=92, subsampling=0)
    return path


def render() -> dict[str, Any]:
    preflight()
    manifests = {}
    for plan in _plan_records():
        case_id = plan["case_id"]
        manifest = render_plan(
            plan,
            STAGE3,
            REPO_ROOT,
            OUTPUT / case_id,
        )
        sequence = _sequence_sheet(case_id, manifest)
        mutable = _mutable_sheet(case_id, manifest)
        manifest["comparison_assets"] = {
            "sequence": {
                "path": sequence.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_path(sequence),
                "size_bytes": sequence.stat().st_size,
            },
            "mutable_sequence": {
                "path": mutable.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_path(mutable),
                "size_bytes": mutable.stat().st_size,
            },
        }
        write_json(OUTPUT / case_id / "manifest.json", manifest)
        manifests[case_id] = manifest
    summary = {
        "schema_version": "1.0",
        "status": "rendered",
        "cases": {
            case_id: {
                "manifest": (
                    OUTPUT / case_id / "manifest.json"
                ).relative_to(REPO_ROOT).as_posix(),
                "frame_count": len(value["records"]),
                "operator_ids": value["operator_ids"],
            }
            for case_id, value in manifests.items()
        },
        "model_runs": {
            "image_candidates": 0,
            "video_candidates": 0,
        },
    }
    write_json(OUTPUT / "render-summary.json", summary)
    return summary


def _object_metrics(
    record: dict[str, Any], operator_type: str
) -> dict[str, Any]:
    return next(
        item
        for item in record["operator_records"]
        if item["operator_type"] == operator_type
    )


def _correlation(a: list[float], b: list[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.std() < 1e-9 or bb.std() < 1e-9:
        return 1.0 if np.allclose(aa, bb) else 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def gate() -> dict[str, Any]:
    manifests = {
        plan["case_id"]: load_json(
            OUTPUT / plan["case_id"] / "manifest.json"
        )
        for plan in _plan_records()
    }
    plan_by_case = {
        plan["case_id"]: plan for plan in _plan_records()
    }

    chem = manifests["CHEM-01"]
    chem_plan = plan_by_case["CHEM-01"]
    anchor_pixels = np.asarray(
        Image.open(
            REPO_ROOT / chem["anchor"]["prepared"]["path"]
        ).convert("RGB")
    )
    start_pixels = np.asarray(
        Image.open(
            REPO_ROOT / chem["records"][0]["output"]["path"]
        ).convert("RGB")
    )
    chem_indicators = [
        _object_metrics(record, "scalar_transfer")[
            "indicator_mean_in_region"
        ]
        for record in chem["records"]
    ]
    chem_drop_counts = [
        _object_metrics(record, "object_overlay")[
            "object_counts_by_class"
        ].get("base_solution_drop", 0)
        for record in chem["records"]
    ]
    chem_checks = [
        {
            "name": "outside_declared_mutable_is_pixel_exact",
            "passed": all(
                record["metrics"][
                    "outside_mutable_max_difference_0_255"
                ]
                <= chem_plan["gates"][
                    "outside_declared_mutable_max_difference_0_255"
                ]
                for record in chem["records"]
            ),
            "evidence": [
                record["metrics"][
                    "outside_mutable_max_difference_0_255"
                ]
                for record in chem["records"]
            ],
        },
        {
            "name": "initial_frame_equals_frozen_anchor",
            "passed": bool(
                np.array_equal(anchor_pixels, start_pixels)
            ),
            "evidence": int(
                np.abs(
                    anchor_pixels.astype(np.int16)
                    - start_pixels.astype(np.int16)
                ).max()
            ),
        },
        {
            "name": "endpoint_indicator_increase",
            "passed": (
                chem_indicators[-1] - chem_indicators[0]
                >= chem_plan["gates"][
                    "indicator_end_must_exceed_start_by"
                ]
            ),
            "evidence": chem_indicators,
        },
        {
            "name": "dynamic_drop_count_matches_contract",
            "passed": chem_drop_counts
            == chem_plan["gates"]["object_count_by_class"][
                "base_solution_drop"
            ],
            "evidence": chem_drop_counts,
        },
    ]

    phys = manifests["PHYS-01"]
    phys_plan = plan_by_case["PHYS-01"]
    phys_correlations = [
        _object_metrics(record, "height_normal")[
            "program_shading_to_realized_luminance_correlation"
        ]
        for record in phys["records"]
    ]
    phys_objects = [
        _object_metrics(record, "object_overlay")
        for record in phys["records"]
    ]
    phys_counts = [
        item["object_counts_by_class"].get(
            "coherent_point_source", 0
        )
        for item in phys_objects
    ]
    phys_centers = []
    for item in phys_objects:
        centers = []
        for obj in item["projected_items"]:
            box = obj["geometry"]["bbox_xyxy"]
            centers.append(
                [
                    round((box[0] + box[2]) / 2, 4),
                    round((box[1] + box[3]) / 2, 4),
                ]
            )
        phys_centers.append(centers)
    phys_checks = [
        {
            "name": "height_drives_realized_luminance",
            "passed": all(
                value
                >= phys_plan["gates"][
                    "minimum_height_to_luminance_correlation"
                ]
                for value in phys_correlations
            ),
            "evidence": phys_correlations,
        },
        {
            "name": "source_count_stable",
            "passed": phys_counts
            == phys_plan["gates"]["object_count_by_class"][
                "coherent_point_source"
            ],
            "evidence": phys_counts,
        },
        {
            "name": "source_centers_stable",
            "passed": all(
                value == phys_centers[0] for value in phys_centers
            ),
            "evidence": phys_centers,
        },
    ]

    math = manifests["MATH-02"]
    math_plan = plan_by_case["MATH-02"]
    math_objects = [
        _object_metrics(record, "object_material")
        for record in math["records"]
    ]
    math_counts = [item["object_count"] for item in math_objects]
    math_overlap = [
        item["interior_overlap_area_px"] for item in math_objects
    ]
    math_area_error = [
        item["area_relative_error"] for item in math_objects
    ]
    object_ids = math_objects[0]["object_ids"]
    material_correlations = {}
    for object_id in object_ids:
        reference = math_objects[0][
            "object_local_luminance_samples"
        ][object_id]
        material_correlations[object_id] = [
            round(
                _correlation(
                    reference,
                    record["object_local_luminance_samples"][
                        object_id
                    ],
                ),
                8,
            )
            for record in math_objects
        ]
    minimum_correlation = min(
        value
        for records in material_correlations.values()
        for value in records
    )
    math_checks = [
        {
            "name": "piece_count_and_identity_stable",
            "passed": math_counts
            == math_plan["gates"]["object_count_by_class"][
                "congruent_right_triangle"
            ],
            "evidence": math_counts,
        },
        {
            "name": "piece_interiors_do_not_overlap",
            "passed": max(math_overlap)
            <= math_plan["gates"]["maximum_overlap_area_px"],
            "evidence": math_overlap,
        },
        {
            "name": "piece_area_preserved",
            "passed": max(math_area_error)
            <= math_plan["gates"]["area_relative_error_maximum"],
            "evidence": math_area_error,
        },
        {
            "name": "object_local_material_binding_stable",
            "passed": minimum_correlation
            >= math_plan["gates"][
                "minimum_object_local_material_correlation"
            ],
            "evidence": {
                "minimum": round(minimum_correlation, 8),
                "per_object": material_correlations,
            },
        },
    ]

    contract_smoke = []
    for contract_path in sorted((STAGE3 / "contracts").glob("*.json")):
        contract = load_json(contract_path)
        passed = True
        for keyframe in contract["keyframes"]:
            for field in (
                "state",
                "clean_program_frame",
                "semantic_layers",
            ):
                try:
                    verify_file_record(
                        keyframe[field], REPO_ROOT
                    )
                except (FileNotFoundError, ValueError):
                    passed = False
        contract_smoke.append(
            {
                "case_id": contract["case_id"],
                "passed": passed,
            }
        )
    cohorts = {
        "CHEM-01": chem_checks,
        "PHYS-01": phys_checks,
        "MATH-02": math_checks,
    }
    result = {
        "schema_version": "1.0",
        "passed": (
            all(
                check["passed"]
                for checks in cohorts.values()
                for check in checks
            )
            and all(item["passed"] for item in contract_smoke)
        ),
        "cohorts": {
            case_id: {
                "passed": all(
                    check["passed"] for check in checks
                ),
                "checks": checks,
            }
            for case_id, checks in cohorts.items()
        },
        "contract_smoke": contract_smoke,
        "model_runs": {
            "image_candidates": 0,
            "video_candidates": 0,
        },
    }
    write_json(OUTPUT / "g3.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["preflight", "render", "gate"]
    )
    args = parser.parse_args()
    if args.action == "preflight":
        result = preflight()
    elif args.action == "render":
        result = render()
    else:
        result = gate()
    if args.action != "gate":
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
