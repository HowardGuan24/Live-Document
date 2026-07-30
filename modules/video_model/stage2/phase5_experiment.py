"""Prepare or run one spec-driven Phase 5 FLF2V experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .framework.contracts import load_json
from .framework.ltx_flf import (
    prepare_video_experiment,
    reaudit_video_experiment,
    run_video_experiment,
)


STAGE2_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = STAGE2_ROOT / "phase5_experiments"
OUTPUT_ROOT = STAGE2_ROOT / "output" / "phase-5" / "experiments"


def resolve_spec(experiment_id: str) -> dict[str, Any]:
    spec_path = SOURCE_ROOT / experiment_id / "spec.json"
    spec = load_json(spec_path)
    if spec["experiment_id"] != experiment_id:
        raise ValueError("experiment ID does not match directory")
    resolved = json.loads(json.dumps(spec))
    resolved["_spec_path"] = str(spec_path.resolve())
    for key in ("first_frame", "last_frame"):
        resolved["source"][key] = str(
            (STAGE2_ROOT / spec["source"][key]).resolve()
        )
    identity_audit = resolved.get("audit", {}).get(
        "color_identity_reference_sequence"
    )
    if identity_audit:
        identity_audit["reference_frame_directory"] = str(
            (
                STAGE2_ROOT
                / identity_audit["reference_frame_directory"]
            ).resolve()
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    parser.add_argument(
        "--server", default="http://127.0.0.1:8188"
    )
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if not (args.prepare or args.generate or args.audit_existing):
        parser.error("choose --prepare, --generate, or --audit-existing")
    spec = resolve_spec(args.experiment)
    root = OUTPUT_ROOT / args.experiment
    if args.prepare:
        prepared = prepare_video_experiment(spec, root)
        print(
            f"{args.experiment}: prepared · "
            f"video_runs={prepared['model_runs']['video']}"
        )
    if args.generate:
        result = run_video_experiment(
            spec,
            root,
            server=args.server,
            timeout_seconds=args.timeout,
        )
        passed = sum(item["passed"] for item in result["hard_checks"])
        print(
            f"{args.experiment}: generated · "
            f"{passed}/{len(result['hard_checks'])} hard checks"
        )
    if args.audit_existing:
        result = reaudit_video_experiment(spec, root)
        passed = sum(item["passed"] for item in result["hard_checks"])
        print(
            f"{args.experiment}: audited existing video · "
            f"{passed}/{len(result['hard_checks'])} hard checks · "
            "video_runs_added=0"
        )


if __name__ == "__main__":
    main()
