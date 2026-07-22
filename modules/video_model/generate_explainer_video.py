#!/usr/bin/env python3
"""Generate a short explanation video plus WebM, GIF, and run metadata."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Support `python modules/video_model/generate_explainer_video.py ...`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modules.video_model.backends import (  # type: ignore
        GenerationSettings,
        generate_procedural,
        generate_wan,
        wan_environment,
    )
    from modules.video_model.postprocess import (  # type: ignore
        export_gif,
        export_webm,
        normalize_video,
        probe_media,
    )
    from modules.video_model.prompting import build_prompts, load_input  # type: ignore
else:
    from .backends import GenerationSettings, generate_procedural, generate_wan, wan_environment
    from .postprocess import export_gif, export_webm, normalize_video, probe_media
    from .prompting import build_prompts, load_input


DEFAULT_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FAILURE_CATEGORIES = [
    "object_drift",
    "incoherent_motion",
    "bad_loop",
    "semantically_unclear",
    "prompt_ignored",
    "noisy_or_cluttered",
]


def _safe_run_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not safe:
        raise ValueError("run name must contain at least one letter or number")
    return safe[:100]


def _default_run_name(input_path: Path, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _safe_run_name(f"{input_path.stem}-{timestamp}-s{seed}")


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _output_paths(root: Path, run_name: str) -> dict[str, Path]:
    return {
        "video": root / "video" / f"{run_name}.mp4",
        "webm": root / "webm" / f"{run_name}.webm",
        "gif": root / "gif" / f"{run_name}.gif",
        "meta": root / "meta" / f"{run_name}.json",
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not 3 <= args.duration <= 8:
        raise ValueError("--duration must be between 3 and 8 seconds")
    if not 4 <= args.fps <= 30:
        raise ValueError("--fps must be between 4 and 30")
    if args.width < 256 or args.height < 256 or args.width % 16 or args.height % 16:
        raise ValueError("--width and --height must be at least 256 and divisible by 16")
    if args.inference_steps < 1:
        raise ValueError("--inference-steps must be positive")
    if args.guidance_scale <= 0:
        raise ValueError("--guidance-scale must be positive")


def run(args: argparse.Namespace) -> tuple[int, Path]:
    started_at = datetime.now(timezone.utc)
    started_timer = time.perf_counter()
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2**31)
    input_path = Path(args.input).resolve()
    output_root = Path(args.output_dir).resolve()
    run_name = _safe_run_name(args.run_name) if args.run_name else _default_run_name(input_path, seed)
    paths = _output_paths(output_root, run_name)
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_name,
        "status": "failure",
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "input_file": str(input_path),
        "input": None,
        "requested_backend": args.backend,
        "chosen_workflow": None,
        "generation_parameters": {
            "model_id": args.model_id,
            "duration_seconds": args.duration,
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "seed": seed,
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "cpu_offload": args.cpu_offload,
            "loop_mode": args.loop_mode,
        },
        "prepared_prompts": None,
        "environment": {"wan": wan_environment()},
        "runtime_seconds": {},
        "outputs": {},
        "fallback": {"used": False, "reason": None},
        "failure": None,
        "quality_review": {
            "automated_checks": {},
            "manual_review_required": FAILURE_CATEGORIES,
            "reported_issues": list(args.failure_note or []),
        },
        "notes": [],
    }

    existing_paths = [path for path in paths.values() if path.exists()]
    if existing_paths and not args.overwrite:
        raise ValueError(f"run '{run_name}' already exists; choose another --run-name or pass --overwrite")
    if args.overwrite:
        # All targets are exact paths derived from the sanitized run name.
        # Removing stale media avoids reporting old output after a failed rerun.
        for key in ("video", "webm", "gif"):
            if paths[key].is_file():
                paths[key].unlink()

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _validate_args(args)
        spec = load_input(input_path)
        positive_prompt, negative_prompt = build_prompts(spec)
        metadata["input"] = spec.to_dict()
        metadata["prepared_prompts"] = {
            "positive": positive_prompt,
            "negative": negative_prompt,
        }

        settings = GenerationSettings(
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            seed=seed,
            inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            model_id=args.model_id,
            cpu_offload=args.cpu_offload,
        )

        requested_backend = args.backend
        chosen_backend = requested_backend
        if requested_backend == "auto":
            if metadata["environment"]["wan"]["available"]:
                chosen_backend = "wan"
            else:
                chosen_backend = "procedural"
                metadata["fallback"] = {
                    "used": True,
                    "reason": metadata["environment"]["wan"]["reason"],
                }

        generation_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"{run_name}-", dir=output_root) as work_dir:
            raw_path = Path(work_dir) / "raw.mp4"
            if chosen_backend == "wan":
                try:
                    backend_info = generate_wan(positive_prompt, negative_prompt, raw_path, settings)
                except Exception as exc:
                    if requested_backend != "auto":
                        raise
                    metadata["fallback"] = {
                        "used": True,
                        "reason": f"Wan generation failed: {type(exc).__name__}: {exc}",
                    }
                    metadata["notes"].append("The model error was preserved and procedural fallback completed the run.")
                    chosen_backend = "procedural"
                    backend_info = generate_procedural(spec, raw_path, settings)
            else:
                backend_info = generate_procedural(spec, raw_path, settings)
            metadata["runtime_seconds"]["generation"] = round(time.perf_counter() - generation_started, 3)
            metadata["chosen_workflow"] = backend_info

            post_started = time.perf_counter()
            normalize_video(
                raw_path,
                paths["video"],
                duration=args.duration,
                fps=args.fps,
                width=args.width,
                height=args.height,
                loop_mode=args.loop_mode,
            )
            export_webm(paths["video"], paths["webm"])
            if not args.no_gif:
                export_gif(paths["video"], paths["gif"])
            metadata["runtime_seconds"]["post_processing"] = round(time.perf_counter() - post_started, 3)

        output_keys = ["video", "webm"] + ([] if args.no_gif else ["gif"])
        metadata["outputs"] = {key: probe_media(paths[key]) for key in output_keys}
        video_probe = metadata["outputs"]["video"]
        actual_duration = video_probe["duration_seconds"]
        metadata["quality_review"]["automated_checks"] = {
            "duration_in_range": actual_duration is not None and 2.95 <= actual_duration <= 8.05,
            "duration_matches_request": actual_duration is not None and abs(actual_duration - args.duration) <= 0.1,
            "dimensions_match": video_probe["width"] == args.width and video_probe["height"] == args.height,
            "all_outputs_nonempty": all(paths[key].stat().st_size > 0 for key in output_keys),
        }
        metadata["notes"].append(
            "Automated checks cover file integrity only; explanation clarity and model artifacts require human review."
        )
        if chosen_backend == "procedural":
            metadata["notes"].append(
                "Procedural output is a runnable baseline/fallback, not evidence that the probabilistic model succeeded."
            )
        metadata["status"] = "success"
        return_code = 0
    except Exception as exc:
        metadata["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        metadata["notes"].append("The metadata file was written even though generation failed.")
        return_code = 1
    finally:
        finished_at = datetime.now(timezone.utc)
        metadata["finished_at"] = finished_at.isoformat()
        metadata["runtime_seconds"]["total"] = round(time.perf_counter() - started_timer, 3)
        _write_metadata(paths["meta"], metadata)

    return return_code, paths["meta"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an explanation-oriented MP4, WebM, GIF, and metadata JSON.",
    )
    parser.add_argument("input", help="document-module JSON input")
    parser.add_argument("--output-dir", default=str(Path(__file__).with_name("outputs")))
    parser.add_argument("--backend", choices=("auto", "wan", "procedural"), default="auto")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--loop-mode", choices=("none", "pingpong"), default="none")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--run-name", help="stable output basename (a timestamp is used by default)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--failure-note",
        action="append",
        choices=FAILURE_CATEGORIES,
        help="record a manually observed issue; may be supplied more than once",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code, metadata_path = run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if code == 0:
        print(f"Generation complete. Metadata: {metadata_path}")
    else:
        print(f"Generation failed. See metadata: {metadata_path}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
