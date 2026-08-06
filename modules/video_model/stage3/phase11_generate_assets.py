"""Generate the two frozen, mechanism-free appearance anchors for S3.11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from modules.video_model.stage2.framework.image_experiment import (
    generate_experiment,
    prepare_experiment,
)
from modules.video_model.stage3.framework.contracts import load_json, write_json


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-11-pedagogy/appearance"
EXPERIMENTS = (
    "EXP-S3-20260731-035",
    "EXP-S3-20260731-036",
    "EXP-S3-20260731-037",
    "EXP-S3-20260731-038",
)


def _spec(experiment_id: str) -> dict[str, Any]:
    source = load_json(STAGE3 / f"experiments/{experiment_id}/spec.json")
    value = json.loads(json.dumps(source))
    for field in ("clean_frame", "semantic_layers"):
        value["source"][field] = str(
            (REPO_ROOT / value["source"][field]).resolve()
        )
    return value


def run(*, generate: bool, force: bool = False) -> dict[str, Any]:
    records = {}
    for experiment_id in EXPERIMENTS:
        spec = _spec(experiment_id)
        root = OUTPUT / spec["case_id"] / experiment_id
        result = (
            generate_experiment(spec, root, force=force)
            if generate
            else prepare_experiment(spec, root)
        )
        records[experiment_id] = {
            "case_id": spec["case_id"],
            "root": root.relative_to(REPO_ROOT).as_posix(),
            "status": result.get("status", "prepared"),
        }
    summary = {
        "schema_version": "1.0",
        "experiments": records,
        "generated": generate,
    }
    write_json(OUTPUT / "appearance-generation.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(generate=args.generate, force=args.force), ensure_ascii=False, indent=2))
