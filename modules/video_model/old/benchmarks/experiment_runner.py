"""Run one-variable Module B experiments against a fixed case and seed."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from modules.video_model.benchmarks.runner import _result_from_metadata, load_case  # type: ignore
    from modules.video_model.generate_explainer_video import run  # type: ignore
else:
    from .runner import _result_from_metadata, load_case
    from ..generate_explainer_video import run


BENCHMARK_ROOT = Path(__file__).resolve().parent
EXPERIMENT_CONFIG = BENCHMARK_ROOT / "configs" / "experiments.json"
ALLOWED_VARIANTS = {
    "model_reuse": {"model_reuse"},
    "inference_steps": {"inference_steps"},
    "resolution": {"width", "height"},
}


def load_experiments(path: str | Path = EXPERIMENT_CONFIG) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    experiments: dict[str, dict[str, Any]] = {}
    for experiment in data.get("experiments", []):
        identifier = experiment.get("experiment_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("every experiment needs a non-empty experiment_id")
        if identifier in experiments:
            raise ValueError(f"duplicate experiment_id: {identifier}")
        changed = experiment.get("changed_variable")
        if changed not in ALLOWED_VARIANTS:
            raise ValueError(f"unsupported changed_variable in {identifier}: {changed}")
        variant_keys = set(experiment.get("variant", {}))
        if variant_keys != ALLOWED_VARIANTS[changed]:
            raise ValueError(
                f"{identifier} must change exactly {sorted(ALLOWED_VARIANTS[changed])}"
            )
        if not experiment.get("case_id") or not experiment.get("baseline"):
            raise ValueError(f"{identifier} needs a fixed case_id and baseline")
        if experiment.get("backend") not in {"ltx", "wan"}:
            raise ValueError(f"{identifier} needs an ltx or wan backend")
        experiments[identifier] = experiment
    return experiments


def _run_args(
    cli: argparse.Namespace,
    case: dict[str, Any],
    experiment: dict[str, Any],
    label: str,
    timestamp: str,
) -> argparse.Namespace:
    baseline = experiment["baseline"]
    values: dict[str, Any] = {
        "input": str(
            BENCHMARK_ROOT / "speed_cases" / f"{experiment['case_id']}.json"
        ),
        "output_dir": cli.output_dir,
        "backend": cli.backend,
        "profile": baseline["profile"],
        "profile_dir": None,
        "model_id": None,
        "ltx_model_id": None,
        "wan_model_id": None,
        "model_revision": "main",
        "ltx_checkpoint": "ltxv-2b-0.9.8-distilled.safetensors",
        "duration": None,
        "fps": None,
        "width": None,
        "height": None,
        "num_frames": None,
        "seed": case["seed"],
        "inference_steps": None,
        "guidance_scale": None if cli.backend == "ltx" else 5.0,
        "cpu_offload": False,
        "vae_tiling": False,
        "vae_slicing": False,
        "use_multiscale": None,
        "first_frame": None,
        "prompt_template_version": case["prompt_version"],
        "loop_mode": "none",
        "no_gif": True,
        "run_name": f"experiment-{timestamp}-{experiment['experiment_id']}-{label}",
        "batch_id": f"experiment-{timestamp}-{experiment['experiment_id']}",
        "num_candidates": 1,
        "overwrite": cli.overwrite,
        "failure_note": None,
    }
    if label == "variant":
        for key, value in experiment["variant"].items():
            if key != "model_reuse":
                values[key] = value
    return argparse.Namespace(**values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a fixed, one-variable video experiment.")
    parser.add_argument("experiment_id")
    parser.add_argument("--backend", choices=("ltx", "wan"), default="ltx")
    parser.add_argument("--output-dir", default=str(BENCHMARK_ROOT.parent / "outputs"))
    parser.add_argument("--results-dir", default=str(BENCHMARK_ROOT / "results"))
    parser.add_argument("--quality-evidence")
    parser.add_argument("--conclusion")
    parser.add_argument("--decision", choices=("pending", "accepted", "rejected"), default="pending")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.decision != "pending" and (not args.quality_evidence or not args.conclusion):
        parser.error("accepted/rejected decisions require --quality-evidence and --conclusion")

    experiments = load_experiments()
    if args.experiment_id not in experiments:
        parser.error(f"unknown experiment_id: {args.experiment_id}")
    experiment = experiments[args.experiment_id]
    if args.backend != experiment["backend"]:
        parser.error(
            f"{args.experiment_id} is controlled for backend={experiment['backend']}; "
            f"received {args.backend}"
        )
    case_path = BENCHMARK_ROOT / "speed_cases" / f"{experiment['case_id']}.json"
    case = load_case(case_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    session: dict[str, Any] = {}
    code_a, metadata_a = run(
        _run_args(args, case, experiment, "baseline", timestamp),
        session=session,
        batch_id=f"experiment-{timestamp}-{args.experiment_id}",
    )
    # Reusing this exact session is the controlled variant for exp_model_reuse;
    # it also prevents repeated loading from contaminating other warm comparisons.
    code_b, metadata_b = run(
        _run_args(args, case, experiment, "variant", timestamp),
        session=session,
        batch_id=f"experiment-{timestamp}-{args.experiment_id}",
    )
    record = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "backend": args.backend,
        "case_id": case["case_id"],
        "seed": case["seed"],
        "prompt_version": case["prompt_version"],
        "baseline_configuration": experiment["baseline_configuration"],
        "changed_variable": experiment["changed_variable"],
        "variant": experiment["variant"],
        "expected_effect": experiment["expected_effect"],
        "measured_speed": {
            "baseline": _result_from_metadata(metadata_a),
            "variant": _result_from_metadata(metadata_b),
        },
        "measured_quality": args.quality_evidence,
        "conclusion": args.conclusion,
        "status": args.decision,
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{timestamp}-{args.experiment_id}.json"
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result_path)
    return max(code_a, code_b)


if __name__ == "__main__":
    raise SystemExit(main())
