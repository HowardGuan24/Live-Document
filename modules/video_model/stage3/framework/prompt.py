"""Traceable Stage 3 prompt compiler with no free-form generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modules.video_model.stage3.framework.contracts import (
    load_json,
    sha256_path,
    write_json,
)


def _profile(case_id: str, classes: list[str]) -> str:
    if {"glass_beaker", "glass_burette"} <= set(classes):
        return "transparent_lab"
    if "congruent_right_triangle" in classes:
        return "wood_geometry"
    if "coherent_point_source" in classes:
        return "water_field"
    raise ValueError(f"no appearance profile for {case_id}/{classes}")


def compile_prompt(
    contract_path: Path,
    visual_manifest_path: Path,
    lexicon_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    contract = load_json(contract_path)
    visual = load_json(visual_manifest_path)
    lexicon = load_json(lexicon_path)
    requirements = contract["semantic_exports"].get(
        "control_object_requirements", []
    )
    classes = [item["class_id"] for item in requirements]
    if not classes:
        classes = contract["semantic_exports"]["object_classes"]
    phrases = lexicon["object_phrases"]
    objects = [
        phrases.get(class_id, phrases["untyped_object"])
        for class_id in classes
    ]
    # Preserve order while removing repeated phrases.
    objects = list(dict.fromkeys(objects))
    visual_profiles = lexicon.get("visual_package_profiles", {})
    profile = visual_profiles.get(visual["package_id"])
    compiler_id = "stage3_prompt_compiler_v2"
    if profile is None:
        # Compatibility route for the already frozen v1-v3 experiments.
        profile = _profile(contract["case_id"], classes)
        compiler_id = "stage3_prompt_compiler_v1"
    policy = contract["geometry_policy"]
    keyframe_id = contract["keyframes"][0]["keyframe_id"]
    state_key = f"{contract['case_id']}:{keyframe_id}"
    preservation = lexicon.get("preservation_profiles", {})
    must_preserve = preservation.get(state_key)
    if must_preserve is None:
        must_preserve = preservation.get("policy_defaults", {}).get(
            policy
        )
    if must_preserve is None:
        # Compatibility route for the already frozen v1-v3 experiments.
        must_preserve = (
            "air gap between burette tip and beaker rim, one of each, no text"
            if contract["case_id"] == "CHEM-01"
            else "preserve object count, identity, topology and locked camera"
        )
    slots = {
        "scene_identity": ", ".join(objects),
        "camera": lexicon["policy_camera"][policy],
        "material_and_light": lexicon["material_profiles"][profile],
        "state": lexicon.get("state_profiles", {}).get(
            state_key,
            (
                "initial solution remains colorless, no pink"
                if contract["case_id"] == "CHEM-01"
                else "show only the contracted program state"
            ),
        ),
        "must_preserve": must_preserve,
    }
    positive = ", ".join(slots.values())
    negative = lexicon["negative_profiles"][policy]
    provenance = {
        "scene_identity": [
            "input_contract.semantic_exports.control_object_requirements",
            f"prompt_lexicon.object_phrases@{lexicon['lexicon_id']}",
        ],
        "camera": [
            "input_contract.geometry_policy",
            f"prompt_lexicon.policy_camera@{lexicon['lexicon_id']}",
        ],
        "material_and_light": [
            f"visual_target:{visual['package_id']}",
            f"prompt_lexicon.material_profiles.{profile}",
        ],
        "state": ["input_contract.keyframes[00_start]", "case hard gates"],
        "must_preserve": [
            "input_contract.hard_gates",
            "S3.2 selector failure pattern: object separation",
        ],
        "negative": [
            f"visual_target:{visual['package_id']}.negative_refs",
            f"prompt_lexicon.negative_profiles.{policy}",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_path = output_dir / "positive_prompt.txt"
    negative_path = output_dir / "negative_prompt.txt"
    positive_path.write_text(positive + "\n", encoding="utf-8")
    negative_path.write_text(negative + "\n", encoding="utf-8")
    result = {
        "schema_version": "1.0",
        "compiler_id": compiler_id,
        "case_id": contract["case_id"],
        "visual_target_status": visual["status"],
        "appearance_profile": profile,
        "slots": slots,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "provenance": provenance,
        "input_signatures": {
            "contract_sha256": sha256_path(contract_path),
            "visual_target_sha256": sha256_path(visual_manifest_path),
            "lexicon_sha256": sha256_path(lexicon_path),
        },
    }
    write_json(output_dir / "prompt_manifest.json", result)
    return result
