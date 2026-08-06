"""Render and machine-audit all five Stage 3 scale image cases."""

from __future__ import annotations

import json
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
OUTPUT = STAGE3 / "output/phase-8-scale-image"
PLAN_PATH = STAGE3 / "scale_state_render_plans_v1.json"
IDS = ("00_start", "01_mechanism", "02_result", "03_end")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _operator(record: dict[str, Any], operator_id: str) -> dict[str, Any]:
    return next(item for item in record["operator_records"] if item["operator_id"] == operator_id)


def _states(case_id: str) -> list[dict[str, Any]]:
    contract = load_json(STAGE3 / "contracts" / f"{case_id}.json")
    return [load_json(REPO_ROOT / item["state"]["path"]) for item in contract["keyframes"]]


def _identity_items(case_id: str) -> list[list[dict[str, Any]]]:
    contract = load_json(STAGE3 / "contracts" / f"{case_id}.json")
    result = []
    for keyframe in contract["keyframes"]:
        semantic = load_json(REPO_ROOT / keyframe["semantic_layers"]["path"])
        layer = next(item for item in semantic["layers"] if item["layer_type"] == "object_identity")
        root = REPO_ROOT / contract["program_source"]["root"]
        result.append(load_json(root / layer["data"]["path"])["items"])
    return result


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _math_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    states = _states("MATH-01")
    records = [_operator(item, "tracked_points") for item in manifest["records"]]
    errors = []
    for state, record in zip(states, records):
        centers = {
            item["class_id"]: [
                (item["geometry"]["bbox_xyxy"][0] + item["geometry"]["bbox_xyxy"][2]) / 2,
                (item["geometry"]["bbox_xyxy"][1] + item["geometry"]["bbox_xyxy"][3]) / 2,
            ]
            for item in record["projected_items"]
        }
        errors.append({
            "circle_point_px": round(float(np.linalg.norm(np.asarray(centers["rotating_point"]) - np.asarray(state["circle_point_xy"]))), 6),
            "curve_head_px": round(float(np.linalg.norm(np.asarray(centers["sine_trace_head"]) - np.asarray(state["curve_head_xy"]))), 6),
        })
    fractions = [state["trace_fraction"] for state in states]
    checks = [
        _check("exactly_two_tracked_identities", all(record["object_count"] == 2 for record in records), [record["object_count"] for record in records]),
        _check("tracked_points_match_state", all(max(item.values()) <= 1.5 for item in errors), errors),
        _check("trace_fraction_is_monotonic", all(a <= b for a, b in zip(fractions, fractions[1:])), fractions),
        _check("exact_geometry_comes_from_typed_raster", all(_operator(item, "exact_ink_geometry")["source_layer_type"] == "hard_boundary" for item in manifest["records"]), "math01_hard_boundary"),
    ]
    return {"checks": checks, "passed": all(item["passed"] for item in checks)}


def _phys_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    states = _states("PHYS-02")
    identities = _identity_items("PHYS-02")
    magnet_centers = []
    coil_boxes = []
    counts = []
    for items in identities:
        counts.append({class_id: sum(item["class_id"] == class_id for item in items) for class_id in ("bar_magnet", "fixed_coil")})
        magnet = next(item for item in items if item["class_id"] == "bar_magnet")
        xs = [point[0] for point in magnet["geometry"]["points"]]
        magnet_centers.append((min(xs) + max(xs)) / 2)
        coil_boxes.append(next(item for item in items if item["class_id"] == "fixed_coil")["geometry"]["bbox_xyxy"])
    pointer = [state["meter_pointer_angle"] for state in states]
    checks = [
        _check("one_magnet_and_one_coil", all(value == {"bar_magnet": 1, "fixed_coil": 1} for value in counts), counts),
        _check("fixed_coil_geometry_is_constant", all(box == coil_boxes[0] for box in coil_boxes), coil_boxes),
        _check("magnet_center_matches_state", all(abs(center - state["magnet_x"]) <= 1.5 for center, state in zip(magnet_centers, states)), {"rendered_centers": magnet_centers, "state": [item["magnet_x"] for item in states]}),
        _check("meter_state_sequence_is_signed_and_stopped_is_zero", abs(pointer[0] + np.pi / 2) < 1e-7 and abs(pointer[2] + np.pi / 2) < 1e-7 and pointer[1] > pointer[2] and pointer[3] < pointer[2], pointer),
        _check("render_uses_typed_hard_boundary", all(_operator(item, "instrument_geometry")["source_layer_type"] == "hard_boundary" for item in manifest["records"]), True),
    ]
    return {"checks": checks, "passed": all(item["passed"] for item in checks)}


def _chem_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    states = _states("CHEM-02")
    counts = [_operator(item, "crystal_identity")["object_count"] for item in manifest["records"]]
    liquid_areas = [_operator(item, "solution_material")["region_area_px"] for item in manifest["records"]]
    volumes = [state["solvent_volume"] for state in states]
    masses = [round(state["liquid_solute_mass"] + state["crystal_solute_mass"], 8) for state in states]
    checks = [
        _check("crystal_identity_counts_match_contract", counts == [0, 0, 1, 4], counts),
        _check("solvent_volume_and_rendered_liquid_area_decrease", all(a > b for a, b in zip(volumes, volumes[1:])) and all(a > b for a, b in zip(liquid_areas, liquid_areas[1:])), {"volumes": volumes, "areas_px": liquid_areas}),
        _check("total_solute_mass_is_conserved", all(abs(value - states[0]["total_solute_mass"]) < 1e-7 for value in masses), masses),
        _check("dish_boundary_is_typed_and_present", all(_operator(item, "fixed_glass_boundary")["source_pixel_count"] > 0 for item in manifest["records"]), [_operator(item, "fixed_glass_boundary")["source_pixel_count"] for item in manifest["records"]]),
    ]
    return {"checks": checks, "passed": all(item["passed"] for item in checks)}


def _bio_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    states = _states("BIO-02")
    identities = _identity_items("BIO-02")
    ids = [[item["object_id"] for item in values] for values in identities]
    components = [_operator(item, "guard_cell_material")["connected_component_count"] for item in manifest["records"]]
    pore_areas = [_operator(item, "pore_void")["region_area_px"] for item in manifest["records"]]
    checks = [
        _check("two_guard_cells_keep_same_ids", all(len(value) == 2 and value == ids[0] for value in ids), ids),
        _check("guard_regions_touch_only_when_pore_is_closed", components == [1, 2, 2, 1], components),
        _check("aperture_state_sequence", [state["aperture_px"] for state in states] == [10.0, 49.0, 49.0, 10.0], [state["aperture_px"] for state in states]),
        _check("pore_render_opens_then_closes", pore_areas[1] > pore_areas[0] and pore_areas[2] == pore_areas[1] and pore_areas[3] == pore_areas[0], pore_areas),
    ]
    return {"checks": checks, "passed": all(item["passed"] for item in checks)}


def _geo_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    states = _states("GEO-01")
    components = [_operator(item, "river_water_material")["connected_component_count"] for item in manifest["records"]]
    neck = [state["neck_width_px"] for state in states]
    checks = [
        _check("water_topology_matches_program", components == [1, 1, 1, 2], components),
        _check("neck_width_strictly_decreases", all(a > b for a, b in zip(neck, neck[1:])), neck),
        _check("oxbow_appears_only_at_end", [state["isolated_oxbow_count"] for state in states] == [0, 0, 0, 1], [state["isolated_oxbow_count"] for state in states]),
        _check("main_channel_remains_connected", [state["main_channel_components"] for state in states] == [1, 1, 1, 1], [state["main_channel_components"] for state in states]),
    ]
    return {"checks": checks, "passed": all(item["passed"] for item in checks)}


GATES = {"MATH-01": _math_gate, "PHYS-02": _phys_gate, "CHEM-02": _chem_gate, "BIO-02": _bio_gate, "GEO-01": _geo_gate}


def preflight() -> dict[str, Any]:
    source = (STAGE3 / "framework/state_renderer.py").read_text(encoding="utf-8")
    records = []
    for path in sorted((STAGE3 / "contracts").glob("*.json")):
        error = None
        try:
            contract = load_json(path)
            validate_input_contract(contract)
            visual = contract["visual_target_package"]
            if path.stem != "GEO-HIST-DELTA-01":
                manifest = load_json(REPO_ROOT / visual["path"])
                validate_visual_target(manifest)
                for item in manifest["positive_refs"] + manifest["negative_refs"]:
                    verify_file_record(item, REPO_ROOT)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        records.append({"case_id": path.stem, "passed": error is None, "error": error})
    checks = [
        _check("all_eleven_contracts_and_visual_refs_resolve", all(item["passed"] for item in records), records),
        _check("core_has_no_case_ids_or_model_runtime", not any(token in source for token in ["MATH-01", "MATH-02", "PHYS-01", "PHYS-02", "CHEM-01", "CHEM-02", "BIO-01", "BIO-02", "GEO-01", "GEO-02", "diffusers", "ControlNetModel", "torch"]), "source token scan"),
        _check("five_scale_plans_are_frozen", load_json(PLAN_PATH)["status"] == "frozen_before_EXP-S3-20260731-029", file_record(PLAN_PATH, REPO_ROOT)),
    ]
    result = {"schema_version": "1.0", "checks": checks, "passed": all(item["passed"] for item in checks), "model_budget": {"image_candidates": 0, "video_candidates": 0}}
    write_json(OUTPUT / "preflight.json", result)
    if not result["passed"]:
        raise RuntimeError("scale image preflight failed")
    return result


def _negative_math(plan: dict[str, Any]) -> dict[str, Any]:
    negative = json.loads(json.dumps(plan))
    negative["plan_id"] = "S3.8-MATH-01-DONOR-UNDERLAY-NEGATIVE"
    negative["role"] = "negative_control"
    negative.pop("base_canvas", None)
    return negative


def _render() -> dict[str, dict[str, Any]]:
    plans = load_json(PLAN_PATH)["plans"]
    rendered = {}
    for plan in plans:
        case_id = plan["case_id"]
        manifest = render_plan(plan, STAGE3, REPO_ROOT, OUTPUT / case_id / "candidate")
        rendered[case_id] = manifest
    negative = _negative_math(plans[0])
    render_plan(negative, STAGE3, REPO_ROOT, OUTPUT / "MATH-01/negative_control")
    return rendered


def _replay(plans: list[dict[str, Any]], rendered: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cases = []
    for plan in plans:
        case_id = plan["case_id"]
        replay = render_plan(plan, STAGE3, REPO_ROOT, OUTPUT / case_id / "candidate-replay")
        left = [item["output"]["sha256"] for item in rendered[case_id]["records"]]
        right = [item["output"]["sha256"] for item in replay["records"]]
        cases.append({"case_id": case_id, "passed": left == right, "hashes": left})
    value = {"schema_version": "1.0", "cases": cases, "passed": all(item["passed"] for item in cases)}
    write_json(OUTPUT / "determinism-replay.json", value)
    return value


def _sentinel_regression() -> dict[str, Any]:
    plans = {item["case_id"]: item for item in load_json(STAGE3 / "state_render_plans.json")["plans"]}
    plans["BIO-01"] = next(item for item in load_json(STAGE3 / "state_render_plans_v2.json")["plans"] if item["role"] == "candidate")
    plans["GEO-02"] = load_json(STAGE3 / "geo_state_render_plan_v1.json")["plan"]
    baselines = {
        "MATH-02": STAGE3 / "output/phase-4/MATH-02/frames",
        "PHYS-01": STAGE3 / "output/phase-4/PHYS-01/frames",
        "CHEM-01": STAGE3 / "output/phase-4/CHEM-01/frames",
        "BIO-01": STAGE3 / "output/phase-6-rerun-1/BIO-01/candidate/frames",
        "GEO-02": STAGE3 / "output/phase-6-rerun-2/GEO-02/candidate/frames",
    }
    cases = []
    for case_id, plan in plans.items():
        manifest = render_plan(plan, STAGE3, REPO_ROOT, OUTPUT / "sentinel-regressions" / case_id)
        comparisons = []
        for record in manifest["records"]:
            frame_id = record["keyframe_id"]
            expected = baselines[case_id] / f"{frame_id}.png"
            actual = REPO_ROOT / record["output"]["path"]
            comparisons.append({"keyframe_id": frame_id, "passed": sha256_path(expected) == sha256_path(actual), "expected_sha256": sha256_path(expected), "actual_sha256": sha256_path(actual)})
        cases.append({"case_id": case_id, "comparisons": comparisons, "passed": all(item["passed"] for item in comparisons)})
    accepted = load_json(STAGE3 / "baselines/accepted.json")
    delta = [record for record in accepted["records"] if record["baseline_id"].startswith("APPEARANCE-GEO-HIST-DELTA-01")]
    delta_checks = []
    for record in delta:
        error = None
        try:
            verify_file_record(record, REPO_ROOT)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        delta_checks.append({"baseline_id": record["baseline_id"], "passed": error is None, "error": error})
    result = {"schema_version": "1.0", "sentinel_cases": cases, "historical_delta": delta_checks, "passed": all(item["passed"] for item in cases + delta_checks)}
    write_json(OUTPUT / "cross-case-regression.json", result)
    return result


def _sheet(rendered: dict[str, dict[str, Any]]) -> Path:
    cell = (320, 220)
    canvas = Image.new("RGB", (cell[0] * 4, cell[1] * 5), (12, 29, 31))
    draw = ImageDraw.Draw(canvas)
    font = _font(15)
    for row, (case_id, manifest) in enumerate(rendered.items()):
        for column, record in enumerate(manifest["records"]):
            image = Image.open(REPO_ROOT / record["output"]["path"]).convert("RGB")
            image.thumbnail((cell[0], cell[1] - 32))
            x, y = column * cell[0], row * cell[1]
            canvas.paste(image, (x + (cell[0] - image.width) // 2, y))
            draw.text((x + 8, y + cell[1] - 25), f"{case_id} / {record['keyframe_id']}", fill=(235, 244, 238), font=font)
    target = OUTPUT / "report-assets/all-scale-candidates.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=92, subsampling=0)
    return target


def run() -> dict[str, Any]:
    previous = OUTPUT / "g3-machine.json"
    rejected = OUTPUT / "g3-machine-v1-rejected.json"
    if previous.is_file() and not rejected.is_file():
        value = load_json(previous)
        if not value.get("passed", False):
            write_json(rejected, value)
    preflight()
    plans = load_json(PLAN_PATH)["plans"]
    rendered = _render()
    gates = {case_id: GATES[case_id](manifest) for case_id, manifest in rendered.items()}
    replay = _replay(plans, rendered)
    regressions = _sentinel_regression()
    sheet = _sheet(rendered)
    result = {
        "schema_version": "1.0",
        "experiment_id": "EXP-S3-20260731-029",
        "case_gates": gates,
        "determinism_replay": replay,
        "cross_case_regression": regressions,
        "contact_sheet": file_record(sheet, REPO_ROOT),
        "passed": all(item["passed"] for item in gates.values()) and replay["passed"] and regressions["passed"],
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
    }
    write_json(OUTPUT / "g3-machine.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
