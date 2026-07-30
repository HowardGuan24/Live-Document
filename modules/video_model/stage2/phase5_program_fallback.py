"""Render one declared deterministic program-motion fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cases.remaining_programs import PROGRAMS as REMAINING_PROGRAMS
from .cases.sentinel_programs import PROGRAMS as SENTINEL_PROGRAMS
from .framework.contracts import load_json
from .framework.program_video import render_program_video


STAGE2_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = STAGE2_ROOT / "phase5_experiments"
OUTPUT_ROOT = STAGE2_ROOT / "output" / "phase-5" / "experiments"
PROGRAMS = {**SENTINEL_PROGRAMS, **REMAINING_PROGRAMS}


def resolve_spec(experiment_id: str) -> dict[str, Any]:
    path = SOURCE_ROOT / experiment_id / "spec.json"
    spec = load_json(path)
    if spec["experiment_id"] != experiment_id:
        raise ValueError("experiment ID does not match directory")
    resolved = json.loads(json.dumps(spec))
    resolved["_spec_path"] = str(path.resolve())
    for key in ("first_frame", "last_frame"):
        resolved["source"][key] = str(
            (STAGE2_ROOT / spec["source"][key]).resolve()
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    spec = resolve_spec(args.experiment)
    case_id = spec["program_case_id"]
    if case_id not in PROGRAMS:
        raise ValueError(f"unknown program plugin: {case_id}")
    result = render_program_video(
        spec,
        PROGRAMS[case_id],
        OUTPUT_ROOT / args.experiment,
    )
    passed = sum(item["passed"] for item in result["hard_checks"])
    print(
        f"{args.experiment}: deterministic fallback · "
        f"{passed}/{len(result['hard_checks'])} hard checks · "
        "video_model_runs=0"
    )


if __name__ == "__main__":
    main()
