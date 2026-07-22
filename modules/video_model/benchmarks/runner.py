"""Benchmark case validation, execution, and result aggregation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from modules.video_model.generate_explainer_video import run  # type: ignore
    from modules.video_model.prompting import validate_input  # type: ignore
else:
    from ..generate_explainer_video import run
    from ..prompting import validate_input


BENCHMARK_ROOT = Path(__file__).resolve().parent


def load_case(path: str | Path) -> dict[str, Any]:
    case_path = Path(path)
    try:
        data = json.loads(case_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load benchmark case {case_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"benchmark case {case_path} must be a JSON object")
    for field in ("case_id", "seed", "prompt_version", "expected_content_type"):
        if field not in data:
            raise ValueError(f"benchmark case {case_path} is missing '{field}'")
    if not isinstance(data["case_id"], str) or not data["case_id"].strip():
        raise ValueError("case_id must be a non-empty string")
    if not isinstance(data["seed"], int):
        raise ValueError("seed must be an integer")
    spec = validate_input(data)
    if spec.content_type != data["expected_content_type"]:
        raise ValueError(
            f"case {data['case_id']} normalized to {spec.content_type}, "
            f"expected {data['expected_content_type']}"
        )
    return data


def load_cases(directory: str | Path) -> list[dict[str, Any]]:
    paths = sorted(Path(directory).glob("*.json"))
    cases = [load_case(path) for path in paths]
    identifiers = [case["case_id"] for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate case_id in {directory}")
    return cases


def _result_from_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    timing = data["timing_seconds"]
    params = data["generation_parameters"]
    duration = params["duration_seconds"]
    outputs = data.get("outputs", {})
    return {
        "case_id": data.get("benchmark_case_id"),
        "run_id": data["run_id"],
        "metadata_path": str(path.resolve()),
        "status": data["status"],
        "backend": data.get("actual_backend"),
        "model_id": params.get("model_id"),
        "model_revision": data.get("model", {}).get("resolved_revision"),
        "profile": data.get("generation_profile"),
        "cold_start_seconds": timing.get("model_load"),
        "warm_inference_seconds": timing.get("inference"),
        "total_seconds": timing.get("total"),
        "seconds_per_generated_second": (
            round(timing["total"] / duration, 4) if timing.get("total") is not None and duration else None
        ),
        "resolution": f"{params['width']}x{params['height']}",
        "frame_count": data.get("chosen_workflow", {}).get("num_frames", params["num_frames"]),
        "fps": params["fps"],
        "inference_steps": data.get("chosen_workflow", {}).get(
            "effective_inference_steps", params["inference_steps"]
        ),
        "peak_gpu_memory_gib": data.get("chosen_workflow", {}).get("peak_gpu_memory_gib"),
        "output_sizes_bytes": {name: item["size_bytes"] for name, item in outputs.items()},
        "fallback": data.get("fallback"),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault((str(result.get("backend")), str(result.get("profile"))), []).append(result)
    summaries = []
    for (backend, profile), items in sorted(groups.items()):
        model_successes = [item for item in items if item.get("status") == "success_model"]
        fallback_successes = [item for item in items if item.get("status") == "success_fallback"]
        completed = model_successes + fallback_successes
        totals = [float(item["total_seconds"]) for item in completed if item.get("total_seconds") is not None]
        cold_loads = [
            float(item["cold_start_seconds"])
            for item in completed
            if item.get("cold_start_seconds") is not None
        ]
        per_generated_second = [
            float(item["seconds_per_generated_second"])
            for item in completed
            if item.get("seconds_per_generated_second") is not None
        ]
        inferences = [
            float(item["warm_inference_seconds"])
            for item in model_successes
            if item.get("warm_inference_seconds") is not None
        ]
        peaks = [
            float(item["peak_gpu_memory_gib"])
            for item in model_successes
            if item.get("peak_gpu_memory_gib") is not None
        ]
        summaries.append(
            {
                "backend": backend,
                "profile": profile,
                "runs": len(items),
                "successes": len(model_successes),
                "model_successes": len(model_successes),
                "fallback_successes": len(fallback_successes),
                "failures": len(items) - len(completed),
                "cold_start_seconds": round(max(cold_loads), 3) if cold_loads else None,
                "mean_total_seconds": round(statistics.mean(totals), 3) if totals else None,
                "mean_warm_inference_seconds": round(statistics.mean(inferences), 3) if inferences else None,
                "mean_seconds_per_generated_second": (
                    round(statistics.mean(per_generated_second), 4)
                    if per_generated_second
                    else None
                ),
                "max_peak_gpu_memory_gib": round(max(peaks), 3) if peaks else None,
            }
        )
    return {"runs": len(results), "groups": summaries}


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Benchmark summary",
        "",
        "| Backend | Profile | Model success | Fallback success | Failed | Cold load (s) | Mean warm inference (s) | Mean total (s) | s/generated-s | Peak VRAM (GiB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["groups"]:
        values = [
            group["backend"],
            group["profile"],
            str(group["model_successes"]),
            str(group["fallback_successes"]),
            str(group["failures"]),
            str(group["cold_start_seconds"] if group["cold_start_seconds"] is not None else "—"),
            str(group["mean_warm_inference_seconds"] or "—"),
            str(group["mean_total_seconds"] or "—"),
            str(group["mean_seconds_per_generated_second"] or "—"),
            str(group["max_peak_gpu_memory_gib"] or "—"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "> Timings are machine-specific. Quality must be reviewed separately with the fixed rubric.",
            "",
        ]
    )
    return "\n".join(lines)


def build_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed Module B V2 benchmarks with model reuse.")
    parser.add_argument("--backend", action="append", choices=("ltx", "wan", "procedural"), required=True)
    parser.add_argument("--profile", action="append", choices=("fast", "balanced", "quality"), required=True)
    parser.add_argument("--case-set", choices=("speed", "quality"), default="speed")
    parser.add_argument("--case-id", action="append", help="run only selected stable case IDs")
    parser.add_argument("--output-dir", default=str(BENCHMARK_ROOT.parent / "outputs"))
    parser.add_argument("--results-dir", default=str(BENCHMARK_ROOT / "results"))
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)
    case_dir = BENCHMARK_ROOT / ("speed_cases" if args.case_set == "speed" else "quality_cases")
    cases = load_cases(case_dir)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in selected]
        missing = selected - {case["case_id"] for case in cases}
        if missing:
            raise SystemExit(f"unknown case IDs: {', '.join(sorted(missing))}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    exit_code = 0
    for backend in args.backend:
        session: dict[str, Any] = {}
        for profile in args.profile:
            for case in cases:
                case_path = case_dir / f"{case['case_id']}.json"
                run_name = f"bench-{timestamp}-{backend}-{profile}-{case['case_id']}"
                run_args = argparse.Namespace(
                    input=str(case_path),
                    output_dir=args.output_dir,
                    backend=backend,
                    profile=profile,
                    profile_dir=None,
                    model_id=None,
                    ltx_model_id=None,
                    wan_model_id=None,
                    model_revision="main",
                    ltx_checkpoint="ltxv-2b-0.9.8-distilled.safetensors",
                    duration=None,
                    fps=None,
                    width=None,
                    height=None,
                    num_frames=None,
                    seed=case["seed"],
                    inference_steps=None,
                    guidance_scale=5.0 if backend == "wan" else None,
                    cpu_offload=False,
                    vae_tiling=False,
                    vae_slicing=False,
                    use_multiscale=None,
                    first_frame=None,
                    prompt_template_version=case["prompt_version"],
                    loop_mode="none",
                    no_gif=args.no_gif,
                    run_name=run_name,
                    batch_id=f"bench-{timestamp}",
                    num_candidates=1,
                    overwrite=args.overwrite,
                    failure_note=None,
                )
                code, metadata_path = run(run_args, session=session, batch_id=f"bench-{timestamp}")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["benchmark_case_id"] = case["case_id"]
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                records.append(_result_from_metadata(metadata_path))
                exit_code = max(exit_code, code)

    jsonl_path = result_dir / f"{timestamp}-{args.case_set}.jsonl"
    jsonl_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    summary = aggregate_results(records)
    summary.update(
        {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "case_set": args.case_set,
            "result_jsonl": str(jsonl_path.resolve()),
        }
    )
    summary_path = result_dir / f"{timestamp}-{args.case_set}-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path = result_dir / f"{timestamp}-{args.case_set}-summary.md"
    markdown_path.write_text(markdown_summary(summary), encoding="utf-8")
    print(markdown_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
