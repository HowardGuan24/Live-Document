"""Build and audit ten readable, near-ten-second Stage 3 teaching videos."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    write_json,
)
from modules.video_model.stage3.framework.motion import decode_video, encode_video
from modules.video_model.stage3.framework.pedagogy import (
    compile_render_plan,
    compile_timeline,
    compose_teaching_frame,
    contact_sheet,
    export_timeline_contract,
    find_cjk_font,
    load_program,
    story_audit,
)
from modules.video_model.stage3.framework.state_renderer import render_plan


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-11-pedagogy"
CONFIG = STAGE3 / "pedagogy_contracts_v1.json"


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _specific_gate(
    case_id: str,
    program: Any,
    samples: list[Any],
) -> dict[str, Any]:
    key_samples = [program.sample(value) for value in (0.0, 1 / 3, 2 / 3, 1.0)]
    checks = list(program.validate(key_samples))
    states = [sample.state for sample in samples]
    if case_id == "CHEM-01":
        indicator = [state["indicator_mean_inside_liquid"] for state in states]
        transition_indices = [
            index for index, value in enumerate(indicator) if 0.05 <= value <= 0.9
        ]
        maximum_step = max(
            (abs(right - left) for left, right in zip(indicator, indicator[1:])),
            default=0.0,
        )
        checks.extend(
            [
                _check(
                    "persistent_endpoint_color_has_multiple_transition_frames",
                    len(transition_indices) >= 6,
                    {"transition_frame_indices": transition_indices, "minimum": 6},
                ),
                _check(
                    "indicator_does_not_change_in_one_frame",
                    maximum_step <= 0.22,
                    {"maximum_indicator_step": round(maximum_step, 8), "threshold": 0.22},
                ),
            ]
        )
    elif case_id == "CHEM-02":
        counts = [state["crystal_count"] for state in states]
        masses = [state["total_solute_mass"] for state in states]
        checks.extend(
            [
                _check(
                    "no_crystal_before_nucleation_threshold",
                    all(
                        state["crystal_count"] == 0
                        for state in states
                        if state["progress"] < 0.55
                    ),
                    "all progress < 0.55 states have zero crystals",
                ),
                _check(
                    "crystal_identity_count_is_monotonic_and_ends_at_four",
                    all(left <= right for left, right in zip(counts, counts[1:]))
                    and counts[-1] == 4,
                    counts,
                ),
                _check(
                    "solute_mass_is_conserved_on_display_clock",
                    max(masses) - min(masses) < 1e-7,
                    {"minimum": min(masses), "maximum": max(masses)},
                ),
            ]
        )
    elif case_id == "GEO-01":
        stage_order = {
            "meander_migration": 0,
            "shortcut_breach": 1,
            "entrance_sealing": 2,
            "oxbow_isolated": 3,
        }
        orders = [stage_order[state["stage"]] for state in states]
        first_isolated = next(
            (index for index, state in enumerate(states) if state["cutoff_complete"]),
            None,
        )
        plug_areas = [state["sediment_plug_area_px"] for state in states]
        checks.extend(
            [
                _check(
                    "four_oxbow_mechanisms_occur_in_order",
                    all(left <= right for left, right in zip(orders, orders[1:]))
                    and set(orders) == {0, 1, 2, 3},
                    [state["stage"] for state in states],
                ),
                _check(
                    "isolation_requires_breach_and_most_of_both_plugs",
                    first_isolated is not None
                    and states[first_isolated]["breach_fraction"] == 1.0
                    and states[first_isolated]["entrance_sealing_fraction"] > 0.75,
                    states[first_isolated] if first_isolated is not None else None,
                ),
                _check(
                    "main_channel_stays_connected_and_exactly_one_oxbow_remains",
                    all(state["main_channel_components"] == 1 for state in states)
                    and states[-1]["isolated_oxbow_count"] == 1,
                    {
                        "main_channel_components": sorted(
                            {state["main_channel_components"] for state in states}
                        ),
                        "final_isolated_oxbow_count": states[-1]["isolated_oxbow_count"],
                    },
                ),
                _check(
                    "sediment_plugs_are_absent_before_sealing_and_grow_monotonically",
                    all(
                        state["sediment_plug_area_px"] == 0
                        for state in states
                        if state["entrance_sealing_fraction"] == 0.0
                    )
                    and all(
                        left <= right
                        for left, right in zip(plug_areas, plug_areas[1:])
                    )
                    and plug_areas[-1] > 0,
                    {
                        "first_nonzero_frame": next(
                            (
                                index for index, area in enumerate(plug_areas)
                                if area > 0
                            ),
                            None,
                        ),
                        "final_area_px": plug_areas[-1],
                    },
                ),
            ]
        )
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }


def _render_case(case: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    case_id = case["case_id"]
    root = OUTPUT / "cases" / case_id
    root.mkdir(parents=True, exist_ok=True)
    program = load_program(case)
    timeline, compile_info = compile_timeline(program, case, defaults)
    source_plan = load_json(STAGE3 / case["source_render_plan"])
    provisional = compile_render_plan(
        source_plan,
        case,
        root / "timeline-contract.json",
        STAGE3,
        len(timeline),
    )
    contract, samples = export_timeline_contract(
        program=program,
        case=case,
        timeline=timeline,
        plan=provisional,
        stage3_root=STAGE3,
        repo_root=REPO_ROOT,
        case_root=root,
    )
    contract_path = root / "timeline-contract.json"
    plan = compile_render_plan(
        source_plan, case, contract_path, STAGE3, len(timeline)
    )
    plan_path = root / "state-render-plan.json"
    write_json(plan_path, plan)
    manifest = render_plan(plan, STAGE3, REPO_ROOT, root / "render")
    rendered_paths = [REPO_ROOT / item["output"]["path"] for item in manifest["records"]]

    final_root = root / "final-frames"
    final_root.mkdir(parents=True, exist_ok=True)
    final_paths: list[Path] = []
    final_arrays: list[np.ndarray] = []
    scene_arrays: list[np.ndarray] = []
    for index, (record, rendered_path) in enumerate(zip(timeline, rendered_paths)):
        scene = Image.open(rendered_path).convert("RGB")
        scene_arrays.append(np.asarray(scene))
        final = compose_teaching_frame(scene, record, case, defaults)
        final_path = final_root / f"frame_{index:03d}.png"
        final.save(final_path, optimize=False)
        final_paths.append(final_path)
        final_arrays.append(np.asarray(final))
        record["rendered_scene"] = file_record(rendered_path, REPO_ROOT)
        record["teaching_frame"] = file_record(final_path, REPO_ROOT)

    video_path = root / "teaching-video.mp4"
    encode_video(final_arrays, video_path, fps=float(defaults["fps"]))
    sheet_path = contact_sheet(
        final_paths, timeline, case, root / "stage-contact-sheet.jpg"
    )
    write_json(root / "timeline.json", timeline)
    write_json(root / "timeline-compile.json", compile_info)
    story = story_audit(
        case=case,
        defaults=defaults,
        timeline=timeline,
        compile_info=compile_info,
        scene_arrays=scene_arrays,
    )
    mechanism = _specific_gate(case_id, program, samples)
    write_json(root / "story-audit.json", story)
    write_json(root / "mechanism-audit.json", mechanism)
    info, decoded = decode_video(video_path)
    video_check = _check(
        "encoded_video_matches_compiled_timeline",
        info["frame_count"] == len(timeline)
        and abs(info["fps"] - float(defaults["fps"])) < 1e-6,
        info,
    )
    result = {
        "schema_version": "1.0",
        "case_id": case_id,
        "title_zh": case["title_zh"],
        "passed": story["passed"] and mechanism["passed"] and video_check["passed"],
        "timeline_compile": compile_info,
        "story_audit": file_record(root / "story-audit.json", REPO_ROOT),
        "mechanism_audit": file_record(root / "mechanism-audit.json", REPO_ROOT),
        "video_check": video_check,
        "video": file_record(video_path, REPO_ROOT),
        "stage_contact_sheet": file_record(sheet_path, REPO_ROOT),
        "timeline": file_record(root / "timeline.json", REPO_ROOT),
        "state_render_plan": file_record(plan_path, REPO_ROOT),
        "render_manifest": file_record(root / "render/manifest.json", REPO_ROOT),
        "appearance_anchor": file_record(
            STAGE3 / plan["anchor"]["path"], REPO_ROOT
        ),
        "model_runs_this_step": {"image_candidates": 0, "video_candidates": 0},
        "appearance_model_provenance": (
            "frozen SDXL candidate selected before timeline render"
            if case.get("appearance_override")
            else "previously accepted frozen Stage 3 appearance anchor"
        ),
    }
    write_json(root / "result.json", result)
    return result


def run(selected_cases: set[str] | None = None) -> dict[str, Any]:
    config = load_json(CONFIG)
    cases = [
        item
        for item in config["cases"]
        if selected_cases is None or item["case_id"] in selected_cases
    ]
    if selected_cases and {item["case_id"] for item in cases} != selected_cases:
        raise ValueError("one or more selected Case IDs are unknown")
    results = {item["case_id"]: _render_case(item, config["defaults"]) for item in cases}
    # A selected rerun is an incremental build.  Keep already rendered cases
    # in the phase manifest so a one-case repair cannot make the ten-case
    # release look incomplete.  The finalizer independently verifies every
    # retained result's recorded artifact hashes before publishing a release.
    for item in config["cases"]:
        case_id = item["case_id"]
        result_path = OUTPUT / f"cases/{case_id}/result.json"
        if case_id not in results and result_path.is_file():
            results[case_id] = load_json(result_path)
    summary = {
        "schema_version": "1.0",
        "phase": "S3.11",
        "contract": file_record(CONFIG, REPO_ROOT),
        "case_count": len(results),
        "cases": {
            case_id: {
                "passed": value["passed"],
                "result": file_record(
                    OUTPUT / f"cases/{case_id}/result.json", REPO_ROOT
                ),
            }
            for case_id, value in results.items()
        },
        "passed": bool(results) and all(value["passed"] for value in results.values()),
    }
    write_json(OUTPUT / "phase11-manifest.json", summary)
    if not summary["passed"]:
        failed = [case_id for case_id, value in results.items() if not value["passed"]]
        raise RuntimeError(f"S3.11 failed: {failed}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    selection = set(args.cases) if args.cases else None
    print(json.dumps(run(selection), ensure_ascii=False, indent=2))
