"""Version MATH-01's contract after exporting the missing trace layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.video_model.stage2.cases.remaining_programs import PROGRAMS
from modules.video_model.stage2.framework.program_runner import build_program
from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    validate_input_contract,
    write_json,
)


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
CASE_ID = "MATH-01"
PROGRAM_ROOT = REPO_ROOT / "modules/video_model/stage2/output/phase-4/programs/MATH-01"


def run() -> dict[str, Any]:
    contract_path = STAGE3 / "contracts/MATH-01.json"
    accepted_path = STAGE3 / "baselines/accepted.json"
    accepted = load_json(accepted_path)
    by_id = {item["baseline_id"]: item for item in accepted["records"]}
    v2_record = by_id["CONTRACT-MATH-01-V2"]
    archive = STAGE3 / "baselines/contracts/MATH-01-v2.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        archive.write_bytes(contract_path.read_bytes())
    if sha256_path(archive) != v2_record["sha256"]:
        raise RuntimeError("MATH-01 V2 archive does not match accepted hash")
    v2_record["path"] = archive.relative_to(REPO_ROOT).as_posix()

    manifest = build_program(PROGRAMS[CASE_ID], PROGRAM_ROOT, phase=4)
    visual_path = STAGE3 / "visual_targets/MATH-01/manifest.json"
    visual = load_json(visual_path)
    program_reference = next(
        item
        for item in visual["positive_refs"]
        if item["role"] == "accepted_mechanism_reference"
    )
    program_reference.update(
        file_record(PROGRAM_ROOT / "keyframe-contact-sheet.jpg", REPO_ROOT)
    )
    write_json(visual_path, visual)
    contract = load_json(contract_path)
    contract["program_source"]["program_manifest"] = file_record(
        PROGRAM_ROOT / "program_manifest.json", REPO_ROOT
    )
    for keyframe in contract["keyframes"]:
        root = PROGRAM_ROOT / "keyframes" / keyframe["keyframe_id"]
        keyframe["annotated_program_frame"] = file_record(root / "program.png", REPO_ROOT)
        keyframe["clean_program_frame"] = file_record(root / "clean.png", REPO_ROOT)
        keyframe["state"] = file_record(root / "state.json", REPO_ROOT)
        keyframe["semantic_layers"] = file_record(root / "semantic_layers.json", REPO_ROOT)
    layer_ids = set(contract["semantic_exports"]["layer_ids"])
    layer_ids.add("math01_trace_boundary")
    contract["semantic_exports"]["layer_ids"] = sorted(layer_ids)
    validate_input_contract(contract)
    write_json(contract_path, contract)
    by_id["CONTRACT-MATH-01-V3"] = {
        "baseline_id": "CONTRACT-MATH-01-V3",
        "kind": "input_contract",
        **file_record(contract_path, REPO_ROOT),
    }
    accepted["records"] = list(by_id.values())
    write_json(accepted_path, accepted)
    validation = load_json(PROGRAM_ROOT / manifest["validation"]["path"])
    return {
        "schema_version": "1.0",
        "case_id": CASE_ID,
        "old_contract": file_record(archive, REPO_ROOT),
        "new_contract": file_record(contract_path, REPO_ROOT),
        "new_layer_id": "math01_trace_boundary",
        "program_manifest": file_record(PROGRAM_ROOT / "program_manifest.json", REPO_ROOT),
        "program_checks_passed": all(
            item["passed"]
            for item in validation["mechanism_checks"]
            + validation["common_checks"]
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
