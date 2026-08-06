"""Run a data-defined S3.3 prompt trial against the frozen candidate grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from modules.video_model.stage2.framework.image_experiment import (
    _token_preflight,
    generate_experiment,
)
from modules.video_model.stage3.framework.contracts import load_json, write_json
from modules.video_model.stage3.framework.prompt import compile_prompt
from modules.video_model.stage3.framework.selector import evaluate_and_select


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
MODEL_ROOT = Path(
    "/workspace/ai-concept-animator/.cache/models/sdxl-base-1.0"
)


def _stage_path(value: str) -> Path:
    return STAGE3 / value


def build_spec(plan_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_json(plan_path)
    matrix = load_json(_stage_path(plan["matrix"]))
    prompt = compile_prompt(
        _stage_path(plan["contract"]),
        _stage_path(plan["visual_target"]),
        _stage_path(plan["lexicon"]),
        _stage_path(plan["prompt_output"]),
    )
    tokens = _token_preflight(
        MODEL_ROOT,
        prompt["positive_prompt"],
        prompt["negative_prompt"],
    )
    prompt["token_preflight"] = tokens
    write_json(
        _stage_path(plan["prompt_output"]) / "prompt_manifest.json",
        prompt,
    )
    forbidden = [
        word
        for word in plan.get("forbidden_positive_words", [])
        if word.lower() in prompt["positive_prompt"].lower()
    ]
    if forbidden:
        raise ValueError(
            "forbidden term leaked into positive conditioning: "
            + ", ".join(forbidden)
        )
    if any(
        record["would_truncate"]
        for record in tokens.values()
    ):
        raise ValueError("prompt would be truncated")

    case_id = plan["case_id"]
    keyframe_id = plan["keyframe_id"]
    contract = load_json(_stage_path(plan["contract"]))
    keyframe = next(
        item
        for item in contract["keyframes"]
        if item["keyframe_id"] == keyframe_id
    )
    slots = prompt["slots"]
    spec = {
        "schema_version": "1.0",
        "experiment_id": plan["experiment_id"],
        "case_id": case_id,
        "hypothesis_zh": plan["hypothesis_zh"],
        "single_variable_zh": plan["single_variable_zh"],
        "source": {
            "keyframe_id": keyframe_id,
            "clean_frame": str(
                REPO_ROOT / keyframe["clean_program_frame"]["path"]
            ),
            "semantic_layers": str(
                REPO_ROOT / keyframe["semantic_layers"]["path"]
            ),
        },
        "control_overrides": {
            "stage3_auto_control": str(
                REPO_ROOT / matrix["geometry_control"]["path"]
            )
        },
        "control_override_explanations": {
            "stage3_auto_control": (
                "frozen geometry control referenced by the trial matrix"
            )
        },
        "prompt_parts": {
            "scene_identity": slots["scene_identity"] + ", " + slots["camera"],
            "material_goal": slots["material_and_light"],
            "state_delta": slots["state"],
            "must_preserve": slots["must_preserve"],
        },
        "negative_artifacts": prompt["negative_prompt"],
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
        "blind_shuffle_seed": plan["blind_shuffle_seed"],
        "budget": {
            "maximum_new_image_candidates": plan["budget"][
                "maximum_new_image_candidates"
            ],
            "actual_planned_image_candidates": (
                len(matrix["configurations"])
                * len(matrix["render"]["seeds"])
            ),
            "planned_external_reuse": 0,
            "maximum_new_generation": plan["budget"][
                "maximum_new_image_candidates"
            ],
            "maximum_video_trials": plan["budget"][
                "maximum_video_candidates"
            ],
        },
    }
    if (
        spec["budget"]["actual_planned_image_candidates"]
        > spec["budget"]["maximum_new_image_candidates"]
    ):
        raise ValueError("trial exceeds frozen candidate budget")
    experiment = _stage_path(plan["experiment_output"])
    write_json(experiment / "spec.json", spec)
    write_json(
        experiment / "preflight.json",
        {
            "schema_version": "1.0",
            "passed": True,
            "plan": str(plan_path),
            "positive_forbidden_words": forbidden,
            "token_preflight": tokens,
            "candidate_budget": spec["budget"],
            "selector_frozen_before_generation": plan["selector"],
        },
    )
    return plan, spec


def generate(plan_path: Path) -> dict[str, Any]:
    plan, spec = build_spec(plan_path)
    output = _stage_path(plan["experiment_output"])
    metadata = generate_experiment(spec, output)
    expected = spec["budget"]["actual_planned_image_candidates"]
    if len(metadata["candidates"]) != expected:
        raise RuntimeError("candidate count changed")
    return metadata


def select(plan_path: Path) -> dict[str, Any]:
    plan = load_json(plan_path)
    experiment = _stage_path(plan["experiment_output"])
    result = evaluate_and_select(
        _stage_path(plan["matrix"]),
        experiment / "_work/generate.json",
        _stage_path(plan["geometry_gate"]),
        _stage_path(plan["selection_output"]),
        REPO_ROOT,
        selector_policy_path=_stage_path(plan["selector"]),
    )
    write_json(experiment / "selection-summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "generate", "select"])
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "preflight":
        plan, spec = build_spec(args.plan)
        result = {
            "experiment_id": plan["experiment_id"],
            "spec": spec,
        }
    elif args.action == "generate":
        result = generate(args.plan)
    else:
        result = select(args.plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
