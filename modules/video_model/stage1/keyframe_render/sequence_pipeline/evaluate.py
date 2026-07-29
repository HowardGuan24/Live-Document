"""Evaluate generic invariants, then delegate case rules to an evaluator."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .evaluators import get_case_evaluator
from .utils import sha256, stable_hash, write_json


def evaluate_sequence(
    spec: dict[str, Any],
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    prepare_path = output_root / "_work" / "manifests" / "prepare.json"
    compose_path = output_root / "_work" / "manifests" / "compose.json"
    if not prepare_path.is_file() or not compose_path.is_file():
        raise FileNotFoundError("run --prepare and --compose before --evaluate")
    prepared = json.loads(prepare_path.read_text(encoding="utf-8"))
    composed = json.loads(compose_path.read_text(encoding="utf-8"))
    case_name = spec.get("case_evaluator")
    case_evaluator = (
        get_case_evaluator(case_name) if case_name else None
    )
    signature = stable_hash(
        {
            "prepare_input_signature": prepared.get("input_signature"),
            "compose_input_signature": composed.get("input_signature"),
            "case_rules": spec.get("case_evaluation"),
            "generic_code_sha256": sha256(Path(__file__)),
            "case_code_sha256": (
                sha256(Path(inspect.getsourcefile(case_evaluator)))
                if case_evaluator
                else None
            ),
        }
    )
    evaluate_path = (
        output_root / "_work" / "manifests" / "evaluate.json"
    )
    if (
        evaluate_path.is_file()
        and (output_root / "video_handoff.json").is_file()
        and not force
    ):
        previous = json.loads(
            evaluate_path.read_text(encoding="utf-8")
        )
        if previous.get("input_signature") == signature:
            previous["cache"] = {"reused": True}
            write_json(evaluate_path, previous)
            return previous
    generate_path = output_root / "_work" / "manifests" / "generate.json"
    generated = (
        json.loads(generate_path.read_text(encoding="utf-8"))
        if generate_path.is_file()
        else None
    )
    checks: list[dict[str, Any]] = []

    expected_size = (
        int(spec["canvas"]["width"]),
        int(spec["canvas"]["height"]),
    )
    final_paths = [
        Path(prepared["anchor"]["final"]["path"]),
        *[
            Path(composed["keyframes"][item["id"]]["final"]["path"])
            for item in spec["keyframes"]
        ],
    ]
    file_evidence: dict[str, Any] = {}
    dimensions_ok = True
    modes_ok = True
    hashes_ok = True
    for path in final_paths:
        with Image.open(path) as image:
            dimensions_ok &= image.size == expected_size
            modes_ok &= image.mode == "RGB"
            file_evidence[path.name] = {
                "size": list(image.size),
                "mode": image.mode,
                "sha256": sha256(path),
            }
    expected_records = [
        prepared["anchor"]["final"],
        *[
            composed["keyframes"][item["id"]]["final"]
            for item in spec["keyframes"]
        ],
    ]
    hashes_ok = all(
        record["sha256"] == sha256(Path(record["path"]))
        for record in expected_records
    )
    checks.append(
        {
            "name": "final_dimensions",
            "scope": "generic",
            "passed": dimensions_ok,
            "evidence": file_evidence,
        }
    )
    checks.append(
        {
            "name": "final_rgb_and_hashes",
            "scope": "generic",
            "passed": modes_ok and hashes_ok,
            "evidence": {
                "all_rgb": modes_ok,
                "manifest_hashes_match": hashes_ok,
            },
        }
    )
    controls_ok = True
    controls_sparse = True
    prompt_ok = True
    static_ok = True
    control_evidence: dict[str, Any] = {}
    for item in spec["keyframes"]:
        entry = prepared["keyframes"][item["id"]]
        control_image = Image.open(entry["control"]["canny"]["path"])
        control_array = np.asarray(control_image)
        values = sorted(
            int(value)
            for value in np.unique(control_array)
        )
        edge_fraction = float((control_array > 0).mean())
        control_size_ok = control_image.size == expected_size
        controls_ok &= values == [0, 255] and control_size_ok
        controls_sparse &= 0.001 <= edge_fraction <= 0.05
        control_evidence[item["id"]] = {
            "unique_values": values,
            "size": list(control_image.size),
            "edge_fraction": round(edge_fraction, 6),
        }
        prompt_ok &= all(
            counts["positive"] <= counts["limit"]
            and counts["negative"] <= counts["limit"]
            for counts in entry["prompt"]["token_counts"].values()
        )
        static_ok &= (
            composed["keyframes"][item["id"]][
                "static_region_max_difference_0_255"
            ]
            == 0
        )
    checks.extend(
        (
            {
                "name": "binary_canny",
                "scope": "generic",
                "passed": controls_ok,
                "evidence": control_evidence,
            },
            {
                "name": "sparse_canny",
                "scope": "generic",
                "passed": controls_sparse,
                "evidence": "every control uses 0.1%–5% edge pixels",
            },
            {
                "name": "prompt_token_limits",
                "scope": "generic",
                "passed": prompt_ok,
                "evidence": "both SDXL tokenizers stay at or below 77",
            },
            {
                "name": "fixed_pixels_outside_allowed_region",
                "scope": "generic",
                "passed": static_ok,
                "evidence": "maximum difference is 0 outside every allowed region",
            },
        )
    )

    raw_separate = generated is not None and all(
        record["classification"] == "raw SDXL ControlNet output"
        and Path(record["path"]).parent.name == record["keyframe_id"]
        and Path(record["path"])
        != Path(composed["keyframes"][record["keyframe_id"]]["final"]["path"])
        for record in generated["candidates"]
    )
    checks.append(
        {
            "name": "raw_and_composite_are_separate",
            "scope": "generic",
            "passed": raw_separate,
            "evidence": (
                f"{len(generated['candidates'])} raw files kept under review/raw"
                if generated
                else "generate manifest missing"
            ),
        }
    )

    if case_evaluator:
        checks.extend(case_evaluator(spec, prepared, composed))
    passed = all(check["passed"] for check in checks)
    result = {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "input_signature": signature,
        "cache": {"reused": False},
        "sequence_id": spec["sequence_id"],
        "checks": checks,
        "review": {
            "verdict": (
                "Mechanism constraints and static-scene invariants pass. "
                "Final acceptance still requires visual review of texture "
                "quality and stage readability."
                if passed
                else "At least one mechanism or file invariant failed."
            ),
            "classification": composed["classification"],
            "known_limitations": [
                "Suspended sediment and underwater deposits are deterministic color layers, not raw SDXL outputs.",
                "Wet-sand texture is synthesized from the anchor and may still look smoother than natural sediment.",
                "Raw full-frame SDXL candidates are evidence and texture references only; using them directly would move the fixed scene.",
            ],
        },
    }
    write_json(evaluate_path, result)
    write_json(output_root / "_work" / "review.json", result["review"])
    write_video_handoff(spec, output_root, prepared, composed)
    return result


def write_video_handoff(
    spec: dict[str, Any],
    output_root: Path,
    prepared: dict[str, Any],
    composed: dict[str, Any],
) -> None:
    ordered = [
        {
            "id": spec["anchor"]["id"],
            "path": prepared["anchor"]["final"]["path"],
            "meaning": spec["anchor"]["meaning"],
        },
        *[
            {
                "id": item["id"],
                "path": composed["keyframes"][item["id"]]["final"]["path"],
                "meaning": item["meaning"],
            }
            for item in spec["keyframes"]
        ],
    ]
    handoff_settings = spec["video_handoff"]
    transitions = []
    for index, (first, last, keyframe) in enumerate(
        zip(ordered, ordered[1:], spec["keyframes"])
    ):
        transition = keyframe["video_transition"]
        transitions.append(
            {
                "index": index,
                "first": first,
                "last": last,
                "only_major_change": transition["only_major_change"],
                "must_remain_fixed": handoff_settings[
                    "must_remain_fixed"
                ],
                "forbidden": transition["forbidden"],
                "suggested_duration_seconds": handoff_settings[
                    "suggested_duration_seconds"
                ],
                "fps": handoff_settings["fps"],
                "model_route": handoff_settings["model_route"],
            }
        )
    write_json(
        output_root / "video_handoff.json",
        {
            "sequence_id": spec["sequence_id"],
            "ordered_keyframes": ordered,
            "transitions": transitions,
        },
    )
