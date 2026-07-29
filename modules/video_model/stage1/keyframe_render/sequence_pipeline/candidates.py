"""Generate auditable raw SDXL Canny candidates for sequence keyframes."""

from __future__ import annotations

import gc
import importlib
import json
import random
import time
from pathlib import Path
from typing import Any

from PIL import Image

from ..enhance import (
    MODEL_IDS,
    _diffusers_runtime,
    fingerprint_models,
    model_paths,
)
from ..first_frame_test import _contact_sheet
from .utils import image_record, stable_hash, write_json


def _inference_settings(render: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "width",
        "height",
        "steps",
        "guidance_scale",
        "controlnet_conditioning_scale",
        "dtype",
    )
    return {key: render[key] for key in keys}


def generate_candidates(
    spec: dict[str, Any],
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    prepare_path = output_root / "_work" / "manifests" / "prepare.json"
    if not prepare_path.is_file():
        raise FileNotFoundError("run --prepare before --generate")
    prepare_manifest = json.loads(
        prepare_path.read_text(encoding="utf-8")
    )
    paths = model_paths()
    missing = [name for name, path in paths.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing local model directories: {missing}")

    fingerprints = fingerprint_models(output_root, paths)
    metadata_path = (
        output_root / "_work" / "manifests" / "generate.json"
    )
    previous = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    previous_candidates = {
        (record["keyframe_id"], int(record["seed"])): record
        for record in previous.get("candidates", [])
    }
    models = {
        name: {
            "model_id": MODEL_IDS[name],
            "path": str(path.resolve()),
            "variant": "fp16",
        }
        for name, path in paths.items()
    }
    metadata: dict[str, Any] = {
        "status": "generating",
        "sequence_id": spec["sequence_id"],
        "settings": spec["render"],
        "inference_settings": _inference_settings(spec["render"]),
        "prepare_spec_sha256": prepare_manifest["spec_sha256"],
        "models": models,
        "model_fingerprints": fingerprints,
        "candidates": [],
    }
    jobs = []
    seeds = [int(seed) for seed in spec["render"]["seeds"]]
    for seed in seeds:
        for keyframe in spec["keyframes"]:
            keyframe_id = keyframe["id"]
            prepared = prepare_manifest["keyframes"][keyframe_id]
            control_path = Path(
                prepared["control"]["canny"]["path"]
            )
            target = (
                output_root
                / "review"
                / "raw"
                / keyframe_id
                / f"{keyframe_id}_s{seed}.png"
            )
            signature_payload = {
                "prepare_spec_sha256": prepare_manifest["spec_sha256"],
                "keyframe_id": keyframe_id,
                "seed": seed,
                "positive": prepared["prompt"]["positive_combined"],
                "negative": prepared["prompt"]["negative_combined"],
                "control_sha256": prepared["control"]["canny"]["sha256"],
                "inference_settings": _inference_settings(spec["render"]),
                "models": fingerprints["models"],
            }
            signature = stable_hash(signature_payload)
            old = previous_candidates.get((keyframe_id, seed))
            legacy_match = bool(
                old
                and _inference_settings(previous.get("settings", {}))
                == _inference_settings(spec["render"])
                and previous.get("models") == models
                and old.get("prompt_parts", {}).get("positive_combined")
                == prepared["prompt"]["positive_combined"]
                and old.get("prompt_parts", {}).get("negative_combined")
                == prepared["prompt"]["negative_combined"]
                and old.get("control", {}).get("sha256")
                == prepared["control"]["canny"]["sha256"]
            )
            reused = bool(
                target.is_file()
                and not force
                and old
                and (
                    old.get("input_signature") == signature
                    or legacy_match
                )
            )
            jobs.append(
                {
                    "seed": seed,
                    "keyframe": keyframe,
                    "prepared": prepared,
                    "control_path": control_path,
                    "target": target,
                    "signature": signature,
                    "reused": reused,
                    "old": old,
                }
            )

    to_generate = [job for job in jobs if not job["reused"]]
    pipeline = None
    controlnet = None
    torch = None
    invocation_started = time.time()
    if to_generate:
        torch, ControlNetModel, _ = _diffusers_runtime()
        Pipeline = getattr(
            importlib.import_module("diffusers"),
            "StableDiffusionXLControlNetPipeline",
        )
        metadata["runtime"] = {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0),
        }
        load_started = time.perf_counter()
        controlnet = ControlNetModel.from_pretrained(
            str(paths["controlnet_canny"]),
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
        )
        pipeline = Pipeline.from_pretrained(
            str(paths["sdxl_base"]),
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
            use_safetensors=True,
        ).to("cuda")
        pipeline.set_progress_bar_config(disable=True)
        metadata["model_load_seconds"] = round(
            time.perf_counter() - load_started, 3
        )
        metadata["scheduler"] = type(pipeline.scheduler).__name__
    else:
        metadata["runtime"] = previous.get("runtime", {})
        metadata["model_load_seconds"] = previous.get(
            "model_load_seconds", 0
        )
        metadata["scheduler"] = previous.get("scheduler")
    metadata["model_loaded_this_invocation"] = bool(to_generate)
    write_json(metadata_path, metadata)

    for job in jobs:
        seed = job["seed"]
        keyframe = job["keyframe"]
        keyframe_id = keyframe["id"]
        prepared = job["prepared"]
        target = job["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        reused = job["reused"]
        peak_memory: int | None = None
        started = time.perf_counter()
        if not reused:
            assert torch is not None and pipeline is not None
            control = Image.open(job["control_path"]).convert("RGB")
            torch.cuda.reset_peak_memory_stats()
            generator = torch.Generator(device="cuda").manual_seed(seed)
            image = pipeline(
                prompt=prepared["prompt"]["positive_combined"],
                negative_prompt=prepared["prompt"]["negative_combined"],
                image=control,
                width=int(spec["render"]["width"]),
                height=int(spec["render"]["height"]),
                num_inference_steps=int(spec["render"]["steps"]),
                guidance_scale=float(
                    spec["render"]["guidance_scale"]
                ),
                controlnet_conditioning_scale=float(
                    spec["render"]["controlnet_conditioning_scale"]
                ),
                generator=generator,
            ).images[0]
            image.save(target)
            torch.cuda.synchronize()
            peak_memory = int(torch.cuda.max_memory_allocated())
        metadata["candidates"].append(
            {
                "keyframe_id": keyframe_id,
                "display_frame": keyframe["display_frame"],
                "state_frame": keyframe["state_frame"],
                "seed": seed,
                **image_record(target),
                "prompt_parts": prepared["prompt"],
                "control": prepared["control"]["canny"],
                "inference_seconds": round(
                    time.perf_counter() - started, 3
                ),
                "peak_gpu_memory_bytes": peak_memory,
                "reused": reused,
                "input_signature": job["signature"],
                "classification": "raw SDXL ControlNet output",
            }
        )
        write_json(metadata_path, metadata)

    invocation_completed = time.time()
    metadata["cache"] = {
        "policy": (
            "reuse only when prompt, Canny hash, render settings, model "
            "fingerprints, state specification and seed all match"
        ),
        "reused": sum(bool(job["reused"]) for job in jobs),
        "generated": len(to_generate),
    }
    metadata["invocation"] = {
        "started_at_unix": invocation_started,
        "completed_at_unix": invocation_completed,
        "wall_seconds": round(
            invocation_completed - invocation_started, 3
        ),
        "force": force,
    }
    metadata["total_generation_seconds"] = (
        previous.get("total_generation_seconds", 0)
        if not to_generate
        else round(
            sum(
                record["inference_seconds"]
                for record in metadata["candidates"]
                if not record["reused"]
            ),
            3,
        )
    )
    metadata["status"] = "generated"
    write_json(metadata_path, metadata)
    if pipeline is not None:
        del pipeline, controlnet
        gc.collect()
        assert torch is not None
        torch.cuda.empty_cache()
    build_candidate_sheets(spec, output_root, metadata)
    return metadata


def build_candidate_sheets(
    spec: dict[str, Any],
    output_root: Path,
    metadata: dict[str, Any],
) -> None:
    records = metadata["candidates"]
    for keyframe in spec["keyframes"]:
        keyframe_id = keyframe["id"]
        selected = [
            record
            for record in records
            if record["keyframe_id"] == keyframe_id
        ]
        _contact_sheet(
            [
                (f"{keyframe_id} | seed {record['seed']}", Path(record["path"]))
                for record in selected
            ],
            output_root
            / "review"
            / "raw"
            / keyframe_id
            / "contact-sheet.jpg",
            columns=2,
        )
    sequence_order = {
        (record["seed"], record["keyframe_id"]): record
        for record in records
    }
    _contact_sheet(
        [
            (
                f"{keyframe['id']} | seed {seed}",
                Path(sequence_order[(seed, keyframe["id"])]["path"]),
            )
            for seed in spec["render"]["seeds"]
            for keyframe in spec["keyframes"]
        ],
        output_root / "raw-candidates-by-seed.jpg",
        columns=len(spec["keyframes"]),
    )
    blind = list(records)
    blind_seed = int(spec["render"]["blind_shuffle_seed"])
    random.Random(blind_seed).shuffle(blind)
    blind_records = []
    for index, record in enumerate(blind, start=1):
        blind_records.append(
            {
                "candidate": f"C{index:02d}",
                "keyframe_id": record["keyframe_id"],
                "seed": record["seed"],
                "path": record["path"],
                "sha256": record["sha256"],
            }
        )
    _contact_sheet(
        [
            (record["candidate"], Path(record["path"]))
            for record in blind_records
        ],
        output_root / "blind-review.jpg",
        columns=4,
    )
    write_json(
        output_root / "_work" / "blind_order.json",
        {
            "shuffle_seed": blind_seed,
            "purpose": (
                "Inspect candidates without stage or diffusion-seed labels; "
                "the mapping is disclosed here after review."
            ),
            "order": blind_records,
        },
    )
