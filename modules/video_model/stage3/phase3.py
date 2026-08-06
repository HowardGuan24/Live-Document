"""Stage 3 S3.3 traceable prompt experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from modules.video_model.stage2.framework.image_experiment import (
    _token_preflight,
    generate_experiment,
)
from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    validate_loop_state,
    write_json,
)
from modules.video_model.stage3.framework.prompt import compile_prompt
from modules.video_model.stage3.framework.selector import (
    evaluate_and_select,
)


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output" / "phase-3"
MATRIX_PATH = STAGE3 / "prompt_candidate_matrix.json"
LEXICON_PATH = STAGE3 / "prompt_lexicon.json"
SELECTOR_PATH = STAGE3 / "selector_v2.json"
EXPERIMENT_ROOT = (
    OUTPUT / "experiments" / "EXP-S3-20260730-005"
)
SELECTION_ROOT = OUTPUT / "selection"
GEOMETRY_GATE = (
    STAGE3
    / "output/phase-1/controls/CHEM-01/00_start/g1.json"
)


def compile_all() -> dict[str, Any]:
    results = {}
    base = Path(
        "/workspace/ai-concept-animator/.cache/models/sdxl-base-1.0"
    )
    for case_id in ("CHEM-01", "MATH-02", "PHYS-01"):
        result = compile_prompt(
            STAGE3 / "contracts" / f"{case_id}.json",
            STAGE3 / "visual_targets" / case_id / "manifest.json",
            LEXICON_PATH,
            OUTPUT / "prompts" / case_id,
        )
        tokens = _token_preflight(
            base,
            result["positive_prompt"],
            result["negative_prompt"],
        )
        result["token_preflight"] = tokens
        write_json(
            OUTPUT / "prompts" / case_id / "prompt_manifest.json",
            result,
        )
        results[case_id] = result
    summary = {
        "schema_version": "1.0",
        "compiler_id": "stage3_prompt_compiler_v1",
        "cases": {
            case_id: {
                "appearance_profile": value["appearance_profile"],
                "positive_tokens": value["token_preflight"][
                    "positive"
                ]["counts_including_special_tokens"],
                "negative_tokens": value["token_preflight"][
                    "negative"
                ]["counts_including_special_tokens"],
                "provenance_complete": set(value["provenance"])
                == {
                    "scene_identity",
                    "camera",
                    "material_and_light",
                    "state",
                    "must_preserve",
                    "negative",
                },
            }
            for case_id, value in results.items()
        },
    }
    summary["passed"] = all(
        item["provenance_complete"]
        for item in summary["cases"].values()
    )
    write_json(OUTPUT / "prompt-compile-summary.json", summary)
    return results


def build_spec() -> dict[str, Any]:
    compiled = compile_all()["CHEM-01"]
    matrix = load_json(MATRIX_PATH)
    slots = compiled["slots"]
    source = (
        REPO_ROOT
        / "modules/video_model/stage2/output/phase-2/CHEM-01/"
        "keyframes/00_start"
    )
    return {
        "schema_version": "1.0",
        "experiment_id": "EXP-S3-20260730-005",
        "case_id": "CHEM-01",
        "hypothesis_zh": (
            "可追溯 prompt compiler 在控制、seed、模型和选择器不变时，"
            "能提高或保持外观质量，同时通过对象分离几何门禁。"
        ),
        "single_variable_zh": (
            "相对 S3.2 只改变 prompt；九组控制强度/seed 完全相同。"
        ),
        "source": {
            "keyframe_id": "00_start",
            "clean_frame": str(source / "clean.png"),
            "semantic_layers": str(source / "semantic_layers.json"),
        },
        "control_overrides": {
            "stage3_auto_control": str(
                REPO_ROOT / matrix["geometry_control"]["path"]
            )
        },
        "control_override_explanations": {
            "stage3_auto_control": (
                "S3.1 accepted automatic geometry control; unchanged from S3.2"
            )
        },
        "prompt_parts": {
            "scene_identity": (
                slots["scene_identity"] + ", " + slots["camera"]
            ),
            "material_goal": slots["material_and_light"],
            "state_delta": slots["state"],
            "must_preserve": slots["must_preserve"],
        },
        "negative_artifacts": compiled["negative_prompt"],
        "render": {
            key: value
            for key, value in matrix["render"].items()
            if key != "scheduler_expected"
        },
        "configurations": [
            {
                **item,
                "pipeline_mode": matrix["pipeline_mode"],
                "control_route": "stage3_auto_control",
            }
            for item in matrix["configurations"]
        ],
        "blind_shuffle_seed": 2026073005,
        "budget": {
            "maximum_new_image_candidates": 9,
            "actual_planned_image_candidates": 9,
            "planned_external_reuse": 0,
            "maximum_new_generation": 9,
            "maximum_video_trials": 0,
        },
    }


def preflight() -> dict[str, Any]:
    matrix = load_json(MATRIX_PATH)
    baseline_matrix = load_json(STAGE3 / "candidate_matrix.json")
    spec = build_spec()
    same_render = {
        key: value
        for key, value in matrix["render"].items()
        if key != "scheduler_expected"
    } == {
        key: value
        for key, value in baseline_matrix["render"].items()
        if key != "scheduler_expected"
    }
    checks = [
        {
            "name": "matrix_frozen_before_generation",
            "passed": matrix["status"] == "frozen_before_generation",
        },
        {
            "name": "control_path_matches_S3_2",
            "passed": matrix["geometry_control"]["path"]
            == baseline_matrix["geometry_control"]["path"],
        },
        {
            "name": "render_seeds_and_scales_match_S3_2",
            "passed": same_render
            and matrix["configurations"]
            == baseline_matrix["configurations"],
        },
        {
            "name": "three_cross_case_prompts_compile_with_provenance",
            "passed": load_json(
                OUTPUT / "prompt-compile-summary.json"
            )["passed"],
        },
        {
            "name": "candidate_budget_is_nine",
            "passed": len(spec["configurations"])
            * len(spec["render"]["seeds"])
            == 9,
        },
    ]
    result = {
        "schema_version": "1.0",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "spec": spec,
    }
    write_json(OUTPUT / "preflight.json", result)
    write_json(EXPERIMENT_ROOT / "spec.json", spec)
    if not result["passed"]:
        raise RuntimeError("S3.3 preflight failed")
    return result


def generate() -> dict[str, Any]:
    result = preflight()
    metadata = generate_experiment(result["spec"], EXPERIMENT_ROOT)
    if len(metadata["candidates"]) != 9:
        raise RuntimeError("S3.3 candidate count changed")
    return metadata


def select() -> dict[str, Any]:
    result = evaluate_and_select(
        MATRIX_PATH,
        EXPERIMENT_ROOT / "_work" / "generate.json",
        GEOMETRY_GATE,
        SELECTION_ROOT,
        REPO_ROOT,
        selector_policy_path=SELECTOR_PATH,
    )
    replay = evaluate_and_select(
        MATRIX_PATH,
        EXPERIMENT_ROOT / "_work" / "generate.json",
        GEOMETRY_GATE,
        OUTPUT / "_selection_replay",
        REPO_ROOT,
        selector_policy_path=SELECTOR_PATH,
    )
    check = {
        "selected": result["selected_candidate_id"],
        "replay": replay["selected_candidate_id"],
        "passed": result["selected_candidate_id"]
        == replay["selected_candidate_id"],
    }
    write_json(OUTPUT / "selection-replay.json", check)
    if not check["passed"]:
        raise RuntimeError("S3.3 selection replay failed")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["preflight", "generate", "select"]
    )
    args = parser.parse_args()
    if args.action == "preflight":
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
    elif args.action == "generate":
        print(json.dumps(generate(), ensure_ascii=False, indent=2))
    else:
        select()


if __name__ == "__main__":
    main()
