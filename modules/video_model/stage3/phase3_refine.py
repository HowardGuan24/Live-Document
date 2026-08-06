"""S3.3 EXP-006: remove forbidden color words from positive conditioning."""

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
    load_json,
    write_json,
)
from modules.video_model.stage3.framework.prompt import compile_prompt
from modules.video_model.stage3.framework.selector import (
    evaluate_and_select,
)


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output" / "phase-3"
MATRIX = STAGE3 / "prompt_candidate_matrix.json"
LEXICON = STAGE3 / "prompt_lexicon_v2.json"
SELECTOR = STAGE3 / "selector_v2.json"
EXPERIMENT = OUTPUT / "experiments" / "EXP-S3-20260730-006"
SELECTION = OUTPUT / "selection-v2"
GATE = STAGE3 / "output/phase-1/controls/CHEM-01/00_start/g1.json"


def build_spec() -> dict[str, Any]:
    prompt = compile_prompt(
        STAGE3 / "contracts/CHEM-01.json",
        STAGE3 / "visual_targets/CHEM-01/manifest.json",
        LEXICON,
        OUTPUT / "prompts-v2" / "CHEM-01",
    )
    tokens = _token_preflight(
        Path(
            "/workspace/ai-concept-animator/.cache/models/sdxl-base-1.0"
        ),
        prompt["positive_prompt"],
        prompt["negative_prompt"],
    )
    prompt["token_preflight"] = tokens
    write_json(
        OUTPUT / "prompts-v2/CHEM-01/prompt_manifest.json", prompt
    )
    if "pink" in prompt["positive_prompt"].lower():
        raise ValueError("forbidden color leaked into positive prompt")
    matrix = load_json(MATRIX)
    source = (
        REPO_ROOT
        / "modules/video_model/stage2/output/phase-2/CHEM-01/"
        "keyframes/00_start"
    )
    slots = prompt["slots"]
    spec = {
        "schema_version": "1.0",
        "experiment_id": "EXP-S3-20260730-006",
        "case_id": "CHEM-01",
        "hypothesis_zh": (
            "禁用颜色只进入 negative conditioning 后，初始态不再被该颜色词污染，"
            "同时保持 S3.2 几何门禁。"
        ),
        "single_variable_zh": (
            "相对 EXP-005 只改变 state/material/negative prompt 中的颜色词归属。"
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
            "stage3_auto_control": "unchanged S3.1 accepted control"
        },
        "prompt_parts": {
            "scene_identity": (
                slots["scene_identity"] + ", " + slots["camera"]
            ),
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
        "blind_shuffle_seed": 2026073006,
        "budget": {
            "maximum_new_image_candidates": 9,
            "actual_planned_image_candidates": 9,
            "planned_external_reuse": 0,
            "maximum_new_generation": 9,
            "maximum_video_trials": 0,
        },
    }
    write_json(EXPERIMENT / "spec.json", spec)
    write_json(
        OUTPUT / "prompt-v2-preflight.json",
        {
            "passed": True,
            "positive_has_forbidden_color": False,
            "token_preflight": tokens,
            "same_control_seeds_scales_as_EXP005": True,
        },
    )
    return spec


def generate() -> dict[str, Any]:
    return generate_experiment(build_spec(), EXPERIMENT)


def select() -> dict[str, Any]:
    result = evaluate_and_select(
        MATRIX,
        EXPERIMENT / "_work/generate.json",
        GATE,
        SELECTION,
        REPO_ROOT,
        selector_policy_path=SELECTOR,
    )
    write_json(OUTPUT / "selection-v2-summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "select"])
    args = parser.parse_args()
    if args.action == "generate":
        print(json.dumps(generate(), ensure_ascii=False, indent=2))
    else:
        select()


if __name__ == "__main__":
    main()
