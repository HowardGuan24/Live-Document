"""Run all reproducible Stage 1 work and report the optional model-stage blocker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .causal_delta.config import MECHANISM_ROOT, OUTPUT_ROOT
from .causal_delta.export import export
from .causal_delta.simulate import run_simulation
from .causal_delta.validate import validate
from .keyframe_render.enhance import probe_environment
from .keyframe_render.evaluate import evaluate
from .keyframe_render.prepare import KEYFRAME_OUTPUT_ROOT, prepare


def run_all() -> dict[str, Any]:
    simulation = run_simulation(output_root=MECHANISM_ROOT)
    validation = validate(MECHANISM_ROOT / "states.jsonl", MECHANISM_ROOT)
    if not validation["passed"]:
        raise RuntimeError("Track A mechanism gate failed; rendering was not started")
    media = export(MECHANISM_ROOT, OUTPUT_ROOT)
    keyframes = prepare(MECHANISM_ROOT, OUTPUT_ROOT, KEYFRAME_OUTPUT_ROOT)
    model_status = probe_environment(KEYFRAME_OUTPUT_ROOT)
    keyframe_evaluation = evaluate(KEYFRAME_OUTPUT_ROOT)
    if keyframe_evaluation["status"] == "selected_pair_ready":
        status = "stage1_complete_selected_pair_ready"
    elif keyframe_evaluation["status"] == "candidates_ready_for_style_review":
        status = "track_a_complete_track_b_candidates_ready"
    elif model_status["status"].startswith("blocked"):
        status = "track_a_complete_track_b_prepared_model_blocked"
    else:
        status = "ready_for_track_b_candidate_generation"
    result = {
        "status": status,
        "track_a": {
            "simulation": simulation,
            "validation_passed": validation["passed"],
            "media": {
                "mp4": media["artifacts"]["mp4"],
                "gif": media["artifacts"]["gif"],
                "duration_seconds": media["duration_seconds"],
            },
        },
        "track_b": {
            "prepare_status": keyframes["status"],
            "model_status": model_status["status"],
            "missing_models": model_status["missing_models"],
            "missing_packages": model_status["missing_packages"],
            "evaluation_status": keyframe_evaluation["status"],
        },
    }
    summary_path = OUTPUT_ROOT.parent / "stage1_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run_all(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
