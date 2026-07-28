#!/usr/bin/env python3
"""Module B V2 CLI: explanation video generation, artifacts, and metadata."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modules.video_model.backends import (  # type: ignore
        GenerationSettings,
        LTX_CHECKPOINT,
        LTX_REPO_ID,
        ltx_environment,
        make_runner,
        release_runner,
        wan_environment,
    )
    from modules.video_model.config import load_profile  # type: ignore
    from modules.video_model.postprocess import (  # type: ignore
        export_gif,
        export_webm,
        normalize_video,
        probe_media,
        sha256_file,
    )
    from modules.video_model.prompting import build_prompts, load_input  # type: ignore
else:
    from .backends import (
        GenerationSettings,
        LTX_CHECKPOINT,
        LTX_REPO_ID,
        ltx_environment,
        make_runner,
        release_runner,
        wan_environment,
    )
    from .config import load_profile
    from .postprocess import export_gif, export_webm, normalize_video, probe_media, sha256_file
    from .prompting import build_prompts, load_input


DEFAULT_WAN_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_LTX_MODEL = LTX_REPO_ID
# Backward-compatible name imported by V1 callers and tests.
DEFAULT_MODEL = DEFAULT_WAN_MODEL

FAILURE_CATEGORIES = [
    "object_drift",
    "object_duplication",
    "object_disappearance",
    "semantic_error",
    "motion_unclear",
    "camera_drift",
    "visual_clutter",
    "bad_loop",
    "prompt_ignored",
    "unreadable_generated_text",
    "temporal_flicker",
]

TIMING_KEYS = [
    "model_load",
    "prompt_preparation",
    "inference",
    "vae_decode",
    "postprocess",
    "encoding_mp4",
    "encoding_webm",
    "encoding_gif",
    "probe_validation",
    "total",
]


def _arg(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _safe_run_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not safe:
        raise ValueError("run name must contain at least one letter or number")
    return safe[:100]


def _default_run_name(input_path: Path, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _safe_run_name(f"{input_path.stem}-{timestamp}-s{seed}")


def candidate_run_name(batch_id: str, candidate_index: int, seed: int) -> str:
    if candidate_index < 1:
        raise ValueError("candidate_index must start at 1")
    safe_batch = _safe_run_name(batch_id)
    suffix = f"-c{candidate_index:02d}-s{seed}"
    return f"{safe_batch[: 100 - len(suffix)]}{suffix}"


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _output_paths(root: Path, run_name: str) -> dict[str, Path]:
    return {
        "raw": root / "raw" / f"{run_name}.mp4",
        "video": root / "video" / f"{run_name}.mp4",
        "webm": root / "webm" / f"{run_name}.webm",
        "gif": root / "gif" / f"{run_name}.gif",
        "meta": root / "meta" / f"{run_name}.json",
    }


def _resolved_parameters(args: argparse.Namespace) -> dict[str, Any]:
    profile_name = _arg(args, "profile", "balanced")
    profile_dir = _arg(args, "profile_dir", None)
    profile = load_profile(profile_name, profile_dir) if profile_dir else load_profile(profile_name)
    values = {
        "profile": profile,
        "duration": _arg(args, "duration") if _arg(args, "duration") is not None else profile.duration_seconds,
        "fps": _arg(args, "fps") if _arg(args, "fps") is not None else profile.fps,
        "width": _arg(args, "width") if _arg(args, "width") is not None else profile.width,
        "height": _arg(args, "height") if _arg(args, "height") is not None else profile.height,
        "num_frames": _arg(args, "num_frames") if _arg(args, "num_frames") is not None else profile.num_frames,
        "inference_steps": (
            _arg(args, "inference_steps")
            if _arg(args, "inference_steps") is not None
            else profile.inference_steps
        ),
        "guidance_scale": (
            _arg(args, "guidance_scale")
            if _arg(args, "guidance_scale") is not None
            else profile.guidance_scale
        ),
        "cpu_offload": bool(_arg(args, "cpu_offload", False) or profile.cpu_offload),
        "vae_tiling": bool(_arg(args, "vae_tiling", False) or profile.vae_tiling),
        "vae_slicing": bool(_arg(args, "vae_slicing", False) or profile.vae_slicing),
        "use_multiscale": (
            _arg(args, "use_multiscale")
            if _arg(args, "use_multiscale") is not None
            else profile.use_multiscale
        ),
        "prompt_template_version": (
            _arg(args, "prompt_template_version") or profile.prompt_template_version
        ),
    }
    return values


def _validate_parameters(values: dict[str, Any]) -> None:
    if not 3 <= values["duration"] <= 8:
        raise ValueError("--duration must be between 3 and 8 seconds")
    if not 4 <= values["fps"] <= 30:
        raise ValueError("--fps must be between 4 and 30")
    if values["width"] < 256 or values["height"] < 256:
        raise ValueError("--width and --height must be at least 256")
    if values["width"] % 16 or values["height"] % 16:
        raise ValueError("--width and --height must be divisible by 16")
    if values["num_frames"] < 2:
        raise ValueError("--num-frames must be at least 2")
    if values["inference_steps"] < 1:
        raise ValueError("--inference-steps must be positive")
    if values["guidance_scale"] <= 0:
        raise ValueError("--guidance-scale must be positive")


def _settings_for_backend(
    args: argparse.Namespace,
    values: dict[str, Any],
    backend: str,
    seed: int,
) -> GenerationSettings:
    generic_model = _arg(args, "model_id")
    if backend == "ltx":
        model_id = _arg(args, "ltx_model_id") or generic_model or DEFAULT_LTX_MODEL
    elif backend == "wan":
        model_id = _arg(args, "wan_model_id") or generic_model or DEFAULT_WAN_MODEL
    else:
        model_id = None
    first_frame = _arg(args, "first_frame")
    if first_frame and backend != "ltx":
        raise ValueError("--first-frame is supported only by the experimental LTX I2V path")
    explicit_steps = _arg(args, "inference_steps")
    explicit_guidance = _arg(args, "guidance_scale")
    settings = GenerationSettings(
        duration=float(values["duration"]),
        fps=int(values["fps"]),
        width=int(values["width"]),
        height=int(values["height"]),
        seed=seed,
        inference_steps=int(
            explicit_steps
            if explicit_steps is not None
            else (30 if backend == "wan" else values["inference_steps"])
        ),
        guidance_scale=float(
            explicit_guidance
            if explicit_guidance is not None
            else (5.0 if backend == "wan" else values["guidance_scale"])
        ),
        model_id=model_id,
        cpu_offload=bool(values["cpu_offload"]),
        num_frames=int(values["num_frames"]),
        model_revision=_arg(args, "model_revision", "main"),
        ltx_checkpoint=_arg(args, "ltx_checkpoint", LTX_CHECKPOINT),
        use_multiscale=bool(values["use_multiscale"]) if backend == "ltx" else False,
        vae_tiling=bool(values["vae_tiling"]) if backend == "ltx" else False,
        vae_slicing=bool(values["vae_slicing"]) if backend == "ltx" else False,
        first_frame=first_frame,
    )
    if backend == "ltx":
        if settings.width % 32 or settings.height % 32:
            raise ValueError("LTX --width and --height must be divisible by 32")
        if settings.num_frames is None or (settings.num_frames - 1) % 8:
            raise ValueError("LTX --num-frames must follow the 8*k+1 constraint")
        if not 1 <= settings.inference_steps <= 7:
            raise ValueError("LTX 0.9.8 distilled supports 1-7 first-pass steps")
        if settings.guidance_scale != 1.0:
            raise ValueError("LTX 0.9.8 distilled requires --guidance-scale 1.0")
        if settings.first_frame and not Path(settings.first_frame).is_file():
            raise ValueError(f"LTX first-frame image does not exist: {settings.first_frame}")
    return settings


def _sync_settings_metadata(metadata: dict[str, Any], settings: GenerationSettings) -> None:
    parameters = metadata["generation_parameters"]
    parameters.update(
        {
            "model_id": settings.model_id,
            "model_revision": settings.model_revision,
            "inference_steps": settings.inference_steps,
            "guidance_scale": settings.guidance_scale,
            "cpu_offload": settings.cpu_offload,
            "vae_tiling": settings.vae_tiling,
            "vae_slicing": settings.vae_slicing,
            "use_multiscale": settings.use_multiscale,
            "first_frame": (
                str(Path(settings.first_frame).resolve()) if settings.first_frame else None
            ),
        }
    )


def _success_status(actual_backend: str, fallback_used: bool) -> str:
    """Keep every backend substitution explicit in the top-level status."""

    if fallback_used or actual_backend == "procedural":
        return "success_fallback"
    return "success_model"


def _select_and_load_runner(
    args: argparse.Namespace,
    values: dict[str, Any],
    seed: int,
    session: dict[str, Any],
) -> tuple[Any, GenerationSettings, float, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    requested = _arg(args, "backend", "auto")
    if session.get("runner") is not None:
        runner = session["runner"]
        settings = _settings_for_backend(args, values, runner.backend, seed)
        load_seconds, load_info = runner.load(settings)
        return runner, settings, load_seconds, load_info, [], dict(session.get("fallback", {}))

    order = [requested] if requested != "auto" else ["ltx", "wan", "procedural"]
    attempts: list[dict[str, Any]] = []
    total_load = 0.0
    last_error: Exception | None = None
    for backend in order:
        runner = make_runner(backend)
        started = time.perf_counter()
        try:
            settings = _settings_for_backend(args, values, backend, seed)
            load_seconds, load_info = runner.load(settings)
            total_load += time.perf_counter() - started
            attempts.append({"backend": backend, "status": "loaded", "seconds": round(load_seconds, 3)})
            fallback = {
                "used": requested == "auto" and backend != "ltx",
                "reason": "; ".join(
                    f"{attempt['backend']}: {attempt['error']}"
                    for attempt in attempts
                    if attempt.get("error")
                )
                or None,
            }
            session.update({"runner": runner, "backend": backend, "fallback": fallback})
            return runner, settings, total_load, load_info, attempts, fallback
        except Exception as exc:
            elapsed = time.perf_counter() - started
            total_load += elapsed
            attempts.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "seconds": round(elapsed, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            last_error = exc
            if requested != "auto":
                raise
    raise RuntimeError(f"no backend could be loaded: {last_error}")


def _generate_with_runner(
    runner: Any,
    spec: Any,
    positive_prompt: str,
    negative_prompt: str,
    raw_path: Path,
    settings: GenerationSettings,
) -> dict[str, Any]:
    if runner.backend == "procedural":
        return runner.generate(spec, raw_path, settings)
    return runner.generate(positive_prompt, negative_prompt, raw_path, settings)


def _generate_with_auto_fallback(
    args: argparse.Namespace,
    values: dict[str, Any],
    seed: int,
    session: dict[str, Any],
    runner: Any,
    settings: GenerationSettings,
    spec: Any,
    positive_prompt: str,
    negative_prompt: str,
    raw_path: Path,
    attempts: list[dict[str, Any]],
) -> tuple[
    Any,
    GenerationSettings,
    dict[str, Any],
    float,
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    """Generate once and, in auto mode only, continue after runtime failures."""

    requested = _arg(args, "backend", "auto")
    try:
        result = _generate_with_runner(
            runner, spec, positive_prompt, negative_prompt, raw_path, settings
        )
        return runner, settings, result, 0.0, {"used": False, "reason": None}, attempts, None
    except Exception as exc:
        attempts.append(
            {
                "backend": runner.backend,
                "status": "generation_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        if requested != "auto" or runner.backend == "procedural":
            raise

    backend_order = ["ltx", "wan", "procedural"]
    remaining = backend_order[backend_order.index(runner.backend) + 1 :]
    release_runner(runner)
    session.clear()
    additional_load = 0.0
    last_error: Exception | None = None
    for backend in remaining:
        fallback_args = argparse.Namespace(**vars(args))
        fallback_args.backend = backend
        fallback_session: dict[str, Any] = {}
        try:
            next_runner, next_settings, load_seconds, next_load_info, load_attempts, _ = _select_and_load_runner(
                fallback_args, values, seed, fallback_session
            )
            additional_load += load_seconds
            attempts.extend(load_attempts)
            result = _generate_with_runner(
                next_runner,
                spec,
                positive_prompt,
                negative_prompt,
                raw_path,
                next_settings,
            )
            reasons = "; ".join(
                f"{attempt['backend']}: {attempt['error']}"
                for attempt in attempts
                if attempt.get("error")
            )
            session.update(fallback_session)
            session["fallback"] = {"used": True, "reason": reasons}
            return (
                next_runner,
                next_settings,
                result,
                additional_load,
                dict(session["fallback"]),
                attempts,
                next_load_info,
            )
        except Exception as exc:
            last_error = exc
            attempts.append(
                {
                    "backend": backend,
                    "status": "generation_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            candidate = fallback_session.get("runner")
            if candidate is not None:
                release_runner(candidate)
    raise RuntimeError(f"all auto backends failed during generation: {last_error}")


def run(
    args: argparse.Namespace,
    *,
    session: dict[str, Any] | None = None,
    batch_id: str | None = None,
    candidate_index: int | None = None,
) -> tuple[int, Path]:
    started_at = datetime.now(timezone.utc)
    started_timer = time.perf_counter()
    session = session if session is not None else {}
    seed = _arg(args, "seed") if _arg(args, "seed") is not None else random.SystemRandom().randrange(0, 2**31)
    input_path = Path(args.input).resolve()
    output_root = Path(_arg(args, "output_dir", Path(__file__).with_name("outputs"))).resolve()
    run_name = _safe_run_name(_arg(args, "run_name")) if _arg(args, "run_name") else _default_run_name(input_path, seed)
    paths = _output_paths(output_root, run_name)
    values = _resolved_parameters(args)
    timing = {key: None for key in TIMING_KEYS}
    metadata: dict[str, Any] = {
        "schema_version": "2.0",
        "run_id": run_name,
        "batch_id": batch_id,
        "candidate_index": candidate_index,
        "status": "failed",
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "input_file": str(input_path),
        "input": None,
        "requested_backend": _arg(args, "backend", "auto"),
        "actual_backend": None,
        "backend_used": None,
        "backend_attempts": [],
        "chosen_workflow": None,
        "model": {},
        "environment": {"ltx": ltx_environment(), "wan": wan_environment()},
        "generation_profile": values["profile"].name,
        "generation_parameters": {
            "duration_seconds": values["duration"],
            "fps": values["fps"],
            "width": values["width"],
            "height": values["height"],
            "num_frames": values["num_frames"],
            "seed": seed,
            "inference_steps": values["inference_steps"],
            "guidance_scale": values["guidance_scale"],
            "cpu_offload": values["cpu_offload"],
            "vae_tiling": values["vae_tiling"],
            "vae_slicing": values["vae_slicing"],
            "use_multiscale": values["use_multiscale"],
            "loop_mode": _arg(args, "loop_mode", "none"),
            "first_frame": str(Path(_arg(args, "first_frame")).resolve()) if _arg(args, "first_frame") else None,
        },
        "prompt_template_version": values["prompt_template_version"],
        "prepared_prompts": None,
        "conditioning": None,
        "timing_seconds": timing,
        "runtime_seconds": {},
        "ffmpeg_commands": [],
        "outputs": {},
        "fallback": {"used": False, "reason": None},
        "failure": None,
        "quality_review": {
            "automated_checks": {},
            "ai_review": None,
            "human_review": None,
            "manual_review_required": FAILURE_CATEGORIES,
            "reported_issues": list(_arg(args, "failure_note", None) or []),
        },
        "evaluation_references": {
            "ai": _arg(args, "ai_evaluation_ref"),
            "human": _arg(args, "human_evaluation_ref"),
        },
        "notes": [],
    }

    existing_paths = [path for path in paths.values() if path.exists()]
    if existing_paths and not _arg(args, "overwrite", False):
        raise ValueError(f"run '{run_name}' already exists; choose another --run-name or pass --overwrite")
    if _arg(args, "overwrite", False):
        for key in ("raw", "video", "webm", "gif"):
            if paths[key].is_file():
                paths[key].unlink()
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _validate_parameters(values)
        prompt_started = time.perf_counter()
        spec = load_input(input_path)
        positive_prompt, negative_prompt = build_prompts(spec, values["prompt_template_version"])
        timing["prompt_preparation"] = round(time.perf_counter() - prompt_started, 3)
        metadata["input"] = spec.to_dict()
        metadata["prepared_prompts"] = {"positive": positive_prompt, "negative": negative_prompt}
        if _arg(args, "first_frame"):
            first_frame = Path(_arg(args, "first_frame")).resolve()
            if not first_frame.is_file():
                raise ValueError(f"first-frame image does not exist: {first_frame}")
            metadata["conditioning"] = {
                "mode": "first_frame_i2v",
                "path": str(first_frame),
                "size_bytes": first_frame.stat().st_size,
                "sha256": sha256_file(first_frame),
            }
        if spec.content_type == "data_flow":
            metadata["notes"].append(
                "Probabilistic video may be unsuitable when this data flow requires exact paths, symbols, or labels."
            )

        runner, settings, load_seconds, load_info, attempts, fallback = _select_and_load_runner(
            args, values, seed, session
        )
        timing["model_load"] = round(load_seconds, 3)
        metadata["backend_attempts"] = attempts
        metadata["fallback"] = fallback or {"used": False, "reason": None}
        metadata["actual_backend"] = runner.backend
        metadata["backend_used"] = runner.backend
        metadata["model"] = load_info
        _sync_settings_metadata(metadata, settings)

        (
            runner,
            settings,
            backend_info,
            extra_fallback_load,
            runtime_fallback,
            attempts,
            runtime_load_info,
        ) = (
            _generate_with_auto_fallback(
                args,
                values,
                seed,
                session,
                runner,
                settings,
                spec,
                positive_prompt,
                negative_prompt,
                paths["raw"],
                attempts,
            )
        )
        timing["model_load"] = round((timing["model_load"] or 0) + extra_fallback_load, 3)
        metadata["backend_attempts"] = attempts
        if runtime_fallback["used"]:
            fallback = runtime_fallback
            metadata["fallback"] = fallback
        if runtime_load_info is not None:
            metadata["model"] = runtime_load_info
        metadata["actual_backend"] = runner.backend
        metadata["backend_used"] = runner.backend
        _sync_settings_metadata(metadata, settings)
        metadata["chosen_workflow"] = backend_info
        backend_timing = backend_info.get("timing_seconds", {})
        timing["model_load"] = round((timing["model_load"] or 0) + backend_timing.get("additional_model_load", 0), 3)
        timing["inference"] = backend_timing.get("inference")
        timing["vae_decode"] = backend_timing.get("vae_decode")
        timing["encoding_mp4"] = backend_timing.get("encoding_mp4")
        metadata["notes"].extend(backend_info.get("timing_limitations", []))

        post_started = time.perf_counter()
        normalize_command = normalize_video(
            paths["raw"],
            paths["video"],
            duration=values["duration"],
            fps=values["fps"],
            width=values["width"],
            height=values["height"],
            loop_mode=_arg(args, "loop_mode", "none"),
        )
        timing["postprocess"] = round(time.perf_counter() - post_started, 3)
        metadata["ffmpeg_commands"].append(normalize_command)

        webm_started = time.perf_counter()
        metadata["ffmpeg_commands"].append(export_webm(paths["video"], paths["webm"]))
        timing["encoding_webm"] = round(time.perf_counter() - webm_started, 3)
        if not _arg(args, "no_gif", False):
            gif_started = time.perf_counter()
            metadata["ffmpeg_commands"].append(export_gif(paths["video"], paths["gif"]))
            timing["encoding_gif"] = round(time.perf_counter() - gif_started, 3)

        output_keys = ["raw", "video", "webm"] + ([] if _arg(args, "no_gif", False) else ["gif"])
        probe_started = time.perf_counter()
        metadata["outputs"] = {key: probe_media(paths[key]) for key in output_keys}
        timing["probe_validation"] = round(time.perf_counter() - probe_started, 3)
        video_probe = metadata["outputs"]["video"]
        actual_duration = video_probe["duration_seconds"]
        metadata["quality_review"]["automated_checks"] = {
            "duration_in_range": actual_duration is not None and 2.95 <= actual_duration <= 8.05,
            "duration_matches_request": actual_duration is not None and abs(actual_duration - values["duration"]) <= 0.1,
            "dimensions_match": video_probe["width"] == values["width"] and video_probe["height"] == values["height"],
            "all_outputs_nonempty": all(paths[key].stat().st_size > 0 for key in output_keys),
        }
        metadata["notes"].append(
            "Automated checks cover artifact integrity only; semantic correctness requires AI or human review."
        )
        metadata["status"] = _success_status(
            runner.backend, bool(metadata["fallback"].get("used"))
        )
        return_code = 0
    except Exception as exc:
        metadata["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        metadata["notes"].append("Metadata was written even though generation failed.")
        return_code = 1
    finally:
        timing["total"] = round(time.perf_counter() - started_timer, 3)
        metadata["runtime_seconds"] = {
            "generation": timing["inference"],
            "post_processing": round(
                sum(value or 0 for key, value in timing.items() if key in {"postprocess", "encoding_webm", "encoding_gif", "probe_validation"}),
                3,
            ),
            "total": timing["total"],
        }
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_metadata(paths["meta"], metadata)
    return return_code, paths["meta"]


def run_candidates(args: argparse.Namespace) -> list[tuple[int, Path]]:
    count = _arg(args, "num_candidates", 1)
    if not 1 <= count <= 20:
        raise ValueError("--num-candidates must be between 1 and 20")
    if count == 1:
        return [run(args)]

    base_seed = _arg(args, "seed") if _arg(args, "seed") is not None else random.SystemRandom().randrange(0, 2**31)
    input_path = Path(args.input)
    batch = _safe_run_name(_arg(args, "batch_id") or _arg(args, "run_name") or _default_run_name(input_path, base_seed))
    session: dict[str, Any] = {}
    results = []
    for index in range(1, count + 1):
        candidate_args = argparse.Namespace(**vars(args))
        candidate_args.seed = base_seed + index - 1
        candidate_args.run_name = candidate_run_name(batch, index, candidate_args.seed)
        candidate_args.num_candidates = 1
        results.append(
            run(candidate_args, session=session, batch_id=batch, candidate_index=index)
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate explanation-oriented MP4, WebM, GIF, raw output, and V2 metadata.",
    )
    parser.add_argument("input", help="document-module JSON input")
    parser.add_argument("--output-dir", default=str(Path(__file__).with_name("outputs")))
    parser.add_argument("--backend", choices=("auto", "ltx", "wan", "procedural"), default="auto")
    parser.add_argument("--profile", choices=("fast", "balanced", "quality"), default="balanced")
    parser.add_argument("--profile-dir", help=argparse.SUPPRESS)
    parser.add_argument("--model-id", help="override the selected backend model repository")
    parser.add_argument("--ltx-model-id", default=DEFAULT_LTX_MODEL)
    parser.add_argument("--wan-model-id", default=DEFAULT_WAN_MODEL)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--ltx-checkpoint", default=LTX_CHECKPOINT)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--use-multiscale", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--first-frame", help="experimental LTX I2V first-frame image")
    parser.add_argument("--prompt-template-version")
    parser.add_argument("--loop-mode", choices=("none", "pingpong"), default="none")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--run-name", help="stable output basename")
    parser.add_argument("--batch-id", help="parent identifier for multiple candidates")
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--ai-evaluation-ref", help="path or ID of an imported advisory AI review")
    parser.add_argument("--human-evaluation-ref", help="path or ID of a human pairwise review")
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
        results = run_candidates(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    failed = [path for code, path in results if code]
    if failed:
        for path in failed:
            print(f"Generation failed. See metadata: {path}", file=sys.stderr)
        return 1
    for _, path in results:
        print(f"Generation complete. Metadata: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
