"""Generate two consistent sparse-Canny keyframes for sediment transport."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..causal_delta.config import (
    MECHANISM_ROOT,
    OUTPUT_ROOT as CAUSAL_OUTPUT_ROOT,
)
from ..causal_delta.validate import load_states
from .enhance import (
    MODEL_IDS,
    _diffusers_runtime,
    fingerprint_models,
    model_paths,
)
from .first_frame_test import _contact_sheet, _package_versions
from .transport_pair_report import render_visual_report


STAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = STAGE_ROOT / "output" / "keyframe_render" / "transport_pair"
PROMPT_ROOT = Path(__file__).with_name("transport_pair_prompts")

STAGES = {
    "in_channel": {
        "display_frame": 10,
        "state_frame": 21,
        "prompt": "in_channel.txt",
        "negative": "in_channel_negative.txt",
        "meaning": "sediment moves downstream while remaining inside the river",
    },
    "at_outlet": {
        "display_frame": 13,
        "state_frame": 27,
        "prompt": "at_outlet.txt",
        "negative": "at_outlet_negative.txt",
        "meaning": "sediment front reaches the river outlet without entering the sea",
    },
}

REVISED_PROMPTS = {
    "in_channel": {
        "prompt": "in_channel_v2.txt",
        "negative": "sediment_emphasis_negative.txt",
    },
    "at_outlet": {
        "prompt": "at_outlet_v2.txt",
        "negative": "sediment_emphasis_negative.txt",
    },
}

SETTINGS = {
    "width": 1344,
    "height": 768,
    "steps": 36,
    "guidance_scale": 6.5,
    "controlnet_conditioning_scale": 0.60,
    "seeds": [3101, 3102, 3103, 3104],
    "pipeline": "StableDiffusionXLControlNetPipeline",
    "dtype": "float16",
    "img2img_initial_image": None,
    "strength": None,
    "mask_projection": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_prompt(filename: str) -> str:
    return (PROMPT_ROOT / filename).read_text(encoding="utf-8").strip()


def _curve_points(
    xs: np.ndarray,
    ys: np.ndarray,
    scale: int,
) -> list[tuple[int, int]]:
    return [
        (int(round(x * scale)), int(round(y * scale)))
        for x, y in zip(xs, ys)
    ]


def _natural_river_geometry(
    size: tuple[int, int],
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    width, height = size
    center_y = height / 2
    mouth_x = width * 0.43
    river_x = np.linspace(0, mouth_x, 260)
    progress = river_x / mouth_x
    center = (
        center_y
        + 5 * np.sin(river_x / 115 + 0.3)
        + 2 * np.sin(river_x / 39)
    )
    half_width = 52 + 20 * progress**1.8
    return (
        mouth_x,
        river_x,
        center,
        center - half_width,
        center + half_width,
    )


def natural_sparse_canny(
    size: tuple[int, int] = (SETTINGS["width"], SETTINGS["height"]),
) -> Image.Image:
    """Draw only the naturalized coast and two river banks.

    No sediment, plume, bathymetry, contour, texture, annotation, or mask
    boundary is encoded in this ControlNet input.
    """

    width, height = size
    scale = 2
    canvas = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(canvas)
    center_y = height / 2
    line_width = 3 * scale
    mouth_x, river_x, _, upper_bank, lower_bank = (
        _natural_river_geometry(size)
    )
    draw.line(
        _curve_points(river_x, upper_bank, scale),
        fill=255,
        width=line_width,
        joint="curve",
    )
    draw.line(
        _curve_points(river_x, lower_bank, scale),
        fill=255,
        width=line_width,
        joint="curve",
    )

    def coast_x(ys: np.ndarray) -> np.ndarray:
        mouth_relief = 12 * np.exp(-((ys - center_y) / 150) ** 2)
        return (
            mouth_x
            + mouth_relief
            + 10 * np.sin(ys / 91 + 0.2)
            + 4 * np.sin(ys / 29)
        )

    upper_y = np.linspace(0, float(upper_bank[-1]), 180)
    lower_y = np.linspace(float(lower_bank[-1]), height - 1, 180)
    draw.line(
        _curve_points(coast_x(upper_y), upper_y, scale),
        fill=255,
        width=line_width,
        joint="curve",
    )
    draw.line(
        _curve_points(coast_x(lower_y), lower_y, scale),
        fill=255,
        width=line_width,
        joint="curve",
    )

    downsampled = canvas.resize(size, Image.Resampling.LANCZOS)
    array = np.asarray(downsampled)
    return Image.fromarray(np.uint8(array >= 42) * 255, mode="L")


def _soft_sediment_density(
    state: dict[str, Any],
    config: dict[str, Any],
    size: tuple[int, int],
) -> np.ndarray:
    """Map audited particles into the naturalized river as a soft density."""

    import cv2

    width, height = size
    mouth_x, river_x, center, upper_bank, lower_bank = (
        _natural_river_geometry(size)
    )
    density = np.zeros((height, width), dtype=np.float32)
    coast_x = float(config["coastline_x"])
    grid_center = float(config["river_center_y"])
    grid_half_width = float(config["river_half_width"])
    for particle in state["particles"]:
        px = float(np.clip(particle["x"] / coast_x * mouth_x, 0, mouth_x))
        local_center = float(np.interp(px, river_x, center))
        local_half_width = float(
            np.interp(px, river_x, (lower_bank - upper_bank) / 2)
        )
        lateral = np.clip(
            (float(particle["y"]) - grid_center) / grid_half_width,
            -1.0,
            1.0,
        )
        py = local_center + lateral * local_half_width * 0.78
        x0 = int(np.clip(round(px), 0, width - 1))
        y0 = int(np.clip(round(py), 0, height - 1))
        density[y0, x0] += 1.0

    density = cv2.GaussianBlur(
        density,
        (0, 0),
        sigmaX=20.0,
        sigmaY=30.0,
        borderType=cv2.BORDER_REFLECT,
    )
    positive = density[density > 0]
    normalization = (
        float(np.quantile(positive, 0.99))
        if positive.size
        else 1.0
    )
    density = np.clip(density / max(normalization, 1e-8), 0.0, 1.0)
    density = np.power(density, 0.72)
    return density * _river_corridor_mask(size)


def _river_corridor_mask(
    size: tuple[int, int],
    *,
    extend_to_sea: bool = False,
) -> np.ndarray:
    """Return the softly feathered water region between the two banks."""

    _, river_x, _, upper_bank, lower_bank = _natural_river_geometry(size)
    blur_radius = 2.5
    if extend_to_sea:
        extension_x = river_x[-1] + np.array([18.0, 36.0, 54.0])
        extension_center = np.full(3, (upper_bank[-1] + lower_bank[-1]) / 2)
        extension_half_width = (
            (lower_bank[-1] - upper_bank[-1]) / 2
            + np.array([8.0, 17.0, 28.0])
        )
        river_x = np.concatenate((river_x, extension_x))
        upper_bank = np.concatenate(
            (upper_bank, extension_center - extension_half_width)
        )
        lower_bank = np.concatenate(
            (lower_bank, extension_center + extension_half_width)
        )
        blur_radius = 8.0
    corridor = Image.new("L", size, 0)
    draw = ImageDraw.Draw(corridor)
    polygon = [
        *(tuple(point) for point in zip(river_x, upper_bank)),
        *(tuple(point) for point in zip(river_x[::-1], lower_bank[::-1])),
    ]
    draw.polygon(polygon, fill=255)
    corridor_array = np.asarray(
        corridor.filter(ImageFilter.GaussianBlur(blur_radius)),
        dtype=np.float32,
    ) / 255.0
    return corridor_array


def _apply_soft_sediment(
    base: Image.Image,
    density: np.ndarray,
) -> tuple[Image.Image, np.ndarray]:
    """Normalize the river as water, then add audited sediment color."""

    rgb = np.asarray(base.convert("RGB"), dtype=np.float32)
    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    luminance_factor = np.clip(luminance / 128.0, 0.62, 1.25)
    corridor = _river_corridor_mask(base.size)
    mouth_x = base.size[0] * 0.43
    xx = np.arange(base.size[0], dtype=np.float32)
    outlet_fade = np.clip((mouth_x - xx) / 70.0, 0.0, 1.0)
    corridor = corridor * outlet_fade[None, :]
    water_color = np.array([54.0, 123.0, 132.0], dtype=np.float32)
    water_rgb = water_color * luminance_factor[..., None]
    water_alpha = (corridor * 0.62)[..., None]
    normalized = rgb * (1.0 - water_alpha) + water_rgb * water_alpha

    sediment_color = np.array([150.0, 76.0, 35.0], dtype=np.float32)
    sediment_rgb = sediment_color * luminance_factor[..., None]
    alpha = np.clip(density * 0.72, 0.0, 0.72)[..., None]
    result = normalized * (1.0 - alpha) + sediment_rgb * alpha
    return (
        Image.fromarray(np.uint8(np.clip(result, 0, 255)), mode="RGB"),
        corridor,
    )


def _prompt_preflight(base_path: Path) -> dict[str, Any]:
    from transformers import CLIPTokenizer

    result: dict[str, Any] = {}
    violations: list[str] = []
    filenames = {
        stage[field]
        for stage in STAGES.values()
        for field in ("prompt", "negative")
    }
    filenames.update(
        stage[field]
        for stage in REVISED_PROMPTS.values()
        for field in ("prompt", "negative")
    )
    for filename in sorted(filenames):
        text = _load_prompt(filename)
        counts: dict[str, int] = {}
        limits: dict[str, int] = {}
        for subfolder in ("tokenizer", "tokenizer_2"):
            tokenizer = CLIPTokenizer.from_pretrained(
                str(base_path),
                subfolder=subfolder,
                local_files_only=True,
            )
            count = len(
                tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
            limit = int(tokenizer.model_max_length)
            counts[subfolder] = count
            limits[subfolder] = limit
            if count > limit:
                violations.append(
                    f"{filename}/{subfolder}: {count} > {limit}"
                )
        result[filename] = {
            "counts_including_special_tokens": counts,
            "limits": limits,
            "would_truncate": any(
                counts[name] > limits[name] for name in counts
            ),
        }
    if violations:
        raise ValueError(
            "SDXL prompt would be truncated: " + "; ".join(violations)
        )
    return result


def _select_mechanism_states() -> dict[str, dict[str, Any]]:
    states = load_states(MECHANISM_ROOT / "states.jsonl")
    timeline = json.loads(
        (CAUSAL_OUTPUT_ROOT / "timeline.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (MECHANISM_ROOT / "simulation_config.json").read_text(encoding="utf-8")
    )
    coast_x = int(config["coastline_x"])
    selections: dict[str, dict[str, Any]] = {}
    for name, spec in STAGES.items():
        entry = next(
            row
            for row in timeline
            if row["display_frame"] == spec["display_frame"]
            and row["state_frame"] == spec["state_frame"]
        )
        state = states[spec["state_frame"]]
        xs = [float(particle["x"]) for particle in state["particles"]]
        selections[name] = {
            "display_frame": entry["display_frame"],
            "state_frame": entry["state_frame"],
            "beat_id": entry["beat_id"],
            "caption": entry["caption"],
            "rendered_file": entry["rendered_file"],
            "state_stats": entry["state_stats"],
            "particle_count_from_state": len(xs),
            "particle_min_x": round(min(xs), 4),
            "particle_max_x": round(max(xs), 4),
            "particle_mean_x": round(float(np.mean(xs)), 4),
            "particles_at_or_beyond_coast": sum(x >= coast_x for x in xs),
            "coastline_x": coast_x,
            "distance_from_front_to_coast": round(
                coast_x - max(xs),
                4,
            ),
        }
    if selections["in_channel"]["particles_at_or_beyond_coast"] != 0:
        raise ValueError("first keyframe sediment must remain inside river")
    if selections["at_outlet"]["particles_at_or_beyond_coast"] != 0:
        raise ValueError("second keyframe sediment must not enter sea")
    if not (
        selections["in_channel"]["particle_max_x"]
        < selections["at_outlet"]["particle_max_x"]
        < coast_x
    ):
        raise ValueError("mechanism keyframes are not ordered toward outlet")
    return selections


def prepare(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    source_root = output_root / "_work" / "source"
    prompt_root = output_root / "_work" / "prompts"
    source_root.mkdir(parents=True, exist_ok=True)
    prompt_root.mkdir(parents=True, exist_ok=True)

    selections = _select_mechanism_states()
    audit_frames: dict[str, dict[str, Any]] = {}
    for name, selection in selections.items():
        source = CAUSAL_OUTPUT_ROOT / selection["rendered_file"]
        target = source_root / f"{name}_mechanism_audit.png"
        shutil.copyfile(source, target)
        audit_frames[name] = {
            "path": str(target.resolve()),
            "sha256": _sha256(target),
            "size": list(Image.open(target).size),
            "model_input": False,
        }

    control_path = source_root / "natural_sparse_canny.png"
    control = natural_sparse_canny()
    control.save(control_path)
    control_array = np.asarray(control)
    control_record = {
        "path": str(control_path.resolve()),
        "sha256": _sha256(control_path),
        "size": list(control.size),
        "mode": control.mode,
        "edge_pixels": int((control_array > 0).sum()),
        "edge_fraction": round(float((control_array > 0).mean()), 6),
        "contents": [
            "one naturalized continuous coastline",
            "two gently curved river banks with a slightly widening outlet",
        ],
        "excluded": [
            "sediment and plume edges",
            "bathymetry and contour lines",
            "particles, arrows, labels, text, and masks",
        ],
    }

    prompts: dict[str, dict[str, Any]] = {}
    prompt_filenames = {
        spec[field]
        for group in (STAGES, REVISED_PROMPTS)
        for spec in group.values()
        for field in ("prompt", "negative")
    }
    for filename in sorted(prompt_filenames):
        source = PROMPT_ROOT / filename
        target = prompt_root / filename
        shutil.copyfile(source, target)
        prompts[filename] = {
            "path": str(target.resolve()),
            "sha256": _sha256(target),
            "text": target.read_text(encoding="utf-8").strip(),
        }

    _contact_sheet(
        [
            (
                "Mechanism audit only | display 10 / state 21",
                Path(audit_frames["in_channel"]["path"]),
            ),
            (
                "Mechanism audit only | display 13 / state 27",
                Path(audit_frames["at_outlet"]["path"]),
            ),
            (
                "Shared model input | natural sparse Canny",
                control_path,
            ),
        ],
        output_root / "source-comparison.jpg",
        columns=3,
    )
    result = {
        "status": "prepared",
        "experiment": "Stage 1.2 sediment-transport two-keyframe pair",
        "mechanism_sources": {
            "timeline": str(
                (CAUSAL_OUTPUT_ROOT / "timeline.json").resolve()
            ),
            "states": str((MECHANISM_ROOT / "states.jsonl").resolve()),
            "simulation_config": str(
                (MECHANISM_ROOT / "simulation_config.json").resolve()
            ),
        },
        "selections": selections,
        "audit_frames": audit_frames,
        "control": control_record,
        "prompts": prompts,
        "settings": SETTINGS,
        "experiment_design": {
            "shared_between_keyframes": [
                "SDXL and ControlNet weights",
                "natural sparse Canny",
                "seed within each pair",
                "resolution, steps, guidance, and control scale",
                "camera, geography, material, and negative style constraints",
            ],
            "changed_between_keyframes": (
                "prompt clauses describing sediment-front position"
            ),
            "img2img": False,
            "mask_projection": False,
        },
    }
    _write_json(output_root / "_work" / "prepare_manifest.json", result)
    return result


def generate(
    output_root: Path = OUTPUT_ROOT,
    *,
    force: bool = False,
) -> dict[str, Any]:
    manifest = prepare(output_root)
    paths = model_paths()
    missing = [name for name, path in paths.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing local model directories: {missing}")

    prompt_preflight = _prompt_preflight(paths["sdxl_base"])
    fingerprints = fingerprint_models(output_root, paths)
    torch, ControlNetModel, _ = _diffusers_runtime()
    diffusers_module = importlib.import_module("diffusers")
    Pipeline = getattr(
        diffusers_module,
        "StableDiffusionXLControlNetPipeline",
    )
    metadata: dict[str, Any] = {
        "status": "generating",
        "experiment": manifest["experiment"],
        "settings": SETTINGS,
        "models": {
            name: {
                "model_id": MODEL_IDS[name],
                "path": str(path.resolve()),
                "variant": "fp16",
            }
            for name, path in paths.items()
        },
        "package_versions": _package_versions(),
        "prompt_token_preflight": prompt_preflight,
        "model_fingerprints": fingerprints,
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
        },
        "candidates": [],
    }
    metadata_path = output_root / "_work" / "metadata.json"
    _write_json(metadata_path, metadata)

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
        time.perf_counter() - load_started,
        3,
    )
    metadata["scheduler"] = {
        "class": type(pipeline.scheduler).__name__,
        "config": dict(pipeline.scheduler.config),
    }
    control = Image.open(manifest["control"]["path"]).convert("RGB")
    metadata["started_at_unix"] = time.time()

    for seed in SETTINGS["seeds"]:
        for stage_name, stage in STAGES.items():
            candidate_root = output_root / "review" / stage_name
            candidate_root.mkdir(parents=True, exist_ok=True)
            path = candidate_root / f"{stage_name}_s{seed}.png"
            started = time.perf_counter()
            reused = path.is_file() and not force
            peak_memory: int | None = None
            if not reused:
                torch.cuda.reset_peak_memory_stats()
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipeline(
                    prompt=_load_prompt(stage["prompt"]),
                    negative_prompt=_load_prompt(stage["negative"]),
                    image=control,
                    width=SETTINGS["width"],
                    height=SETTINGS["height"],
                    num_inference_steps=SETTINGS["steps"],
                    guidance_scale=SETTINGS["guidance_scale"],
                    controlnet_conditioning_scale=SETTINGS[
                        "controlnet_conditioning_scale"
                    ],
                    generator=generator,
                ).images[0]
                image.save(path)
                torch.cuda.synchronize()
                peak_memory = int(torch.cuda.max_memory_allocated())
            metadata["candidates"].append(
                {
                    "name": path.stem,
                    "revision": "baseline",
                    "stage": stage_name,
                    "meaning": stage["meaning"],
                    "display_frame": stage["display_frame"],
                    "state_frame": stage["state_frame"],
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "size": list(Image.open(path).size),
                    "inference_seconds": round(
                        time.perf_counter() - started,
                        3,
                    ),
                    "peak_gpu_memory_bytes": peak_memory,
                    "reused": reused,
                    "prompt_path": manifest["prompts"][
                        stage["prompt"]
                    ]["path"],
                    "negative_prompt_path": manifest["prompts"][
                        stage["negative"]
                    ]["path"],
                    "control_path": manifest["control"]["path"],
                }
            )
            _write_json(metadata_path, metadata)

    metadata["completed_at_unix"] = time.time()
    metadata["total_generation_seconds"] = round(
        metadata["completed_at_unix"] - metadata["started_at_unix"],
        3,
    )
    metadata["status"] = "generated"
    _write_json(metadata_path, metadata)
    del pipeline, controlnet
    gc.collect()
    torch.cuda.empty_cache()
    build_sheets(output_root)
    return metadata


def generate_revision(
    output_root: Path = OUTPUT_ROOT,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run the evidence-triggered sediment-wording revision only."""

    manifest = prepare(output_root)
    metadata_path = output_root / "_work" / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "baseline metadata is missing; run --generate first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    baseline = [
        record
        for record in metadata.get("candidates", [])
        if record.get("revision", "baseline") == "baseline"
    ]
    if len(baseline) != len(STAGES) * len(SETTINGS["seeds"]):
        raise ValueError(
            "the complete 8-image baseline is required before revision"
        )
    for record in baseline:
        record["revision"] = "baseline"

    paths = model_paths()
    missing = [name for name, path in paths.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing local model directories: {missing}")
    metadata["prompt_token_preflight"] = _prompt_preflight(
        paths["sdxl_base"]
    )
    torch, ControlNetModel, _ = _diffusers_runtime()
    diffusers_module = importlib.import_module("diffusers")
    Pipeline = getattr(
        diffusers_module,
        "StableDiffusionXLControlNetPipeline",
    )
    metadata["status"] = "generating_revision"
    metadata["candidates"] = baseline
    _write_json(metadata_path, metadata)

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
    revision_model_load_seconds = round(
        time.perf_counter() - load_started,
        3,
    )
    revision_started = time.time()
    control = Image.open(manifest["control"]["path"]).convert("RGB")

    for seed in SETTINGS["seeds"]:
        for stage_name, stage in STAGES.items():
            prompt_spec = REVISED_PROMPTS[stage_name]
            candidate_root = (
                output_root
                / "review"
                / "sediment_emphasis"
                / stage_name
            )
            candidate_root.mkdir(parents=True, exist_ok=True)
            path = candidate_root / f"{stage_name}_v2_s{seed}.png"
            started = time.perf_counter()
            reused = path.is_file() and not force
            peak_memory: int | None = None
            if not reused:
                torch.cuda.reset_peak_memory_stats()
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipeline(
                    prompt=_load_prompt(prompt_spec["prompt"]),
                    negative_prompt=_load_prompt(prompt_spec["negative"]),
                    image=control,
                    width=SETTINGS["width"],
                    height=SETTINGS["height"],
                    num_inference_steps=SETTINGS["steps"],
                    guidance_scale=SETTINGS["guidance_scale"],
                    controlnet_conditioning_scale=SETTINGS[
                        "controlnet_conditioning_scale"
                    ],
                    generator=generator,
                ).images[0]
                image.save(path)
                torch.cuda.synchronize()
                peak_memory = int(torch.cuda.max_memory_allocated())
            metadata["candidates"].append(
                {
                    "name": path.stem,
                    "revision": "sediment_emphasis",
                    "stage": stage_name,
                    "meaning": stage["meaning"],
                    "display_frame": stage["display_frame"],
                    "state_frame": stage["state_frame"],
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "size": list(Image.open(path).size),
                    "inference_seconds": round(
                        time.perf_counter() - started,
                        3,
                    ),
                    "peak_gpu_memory_bytes": peak_memory,
                    "reused": reused,
                    "prompt_path": manifest["prompts"][
                        prompt_spec["prompt"]
                    ]["path"],
                    "negative_prompt_path": manifest["prompts"][
                        prompt_spec["negative"]
                    ]["path"],
                    "control_path": manifest["control"]["path"],
                }
            )
            _write_json(metadata_path, metadata)

    revision_completed = time.time()
    metadata["revision_run"] = {
        "name": "sediment_emphasis",
        "reason": (
            "baseline pairs were consistent but 8/8 lacked a visible "
            "ochre sediment front"
        ),
        "changed_variable": "sediment wording in prompt and negative prompt",
        "unchanged": [
            "model weights",
            "control image",
            "pipeline",
            "resolution, steps, CFG, and ControlNet scale",
            "four paired seeds",
        ],
        "model_load_seconds": revision_model_load_seconds,
        "started_at_unix": revision_started,
        "completed_at_unix": revision_completed,
        "generation_seconds": round(
            revision_completed - revision_started,
            3,
        ),
    }
    metadata["status"] = "generated_with_revision"
    _write_json(metadata_path, metadata)
    del pipeline, controlnet
    gc.collect()
    torch.cuda.empty_cache()
    build_sheets(output_root)
    return metadata


def build_sheets(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    metadata = json.loads(
        (output_root / "_work" / "metadata.json").read_text(encoding="utf-8")
    )
    candidates = metadata["candidates"]
    baseline = [
        record
        for record in candidates
        if record.get("revision", "baseline") == "baseline"
    ]
    for stage_name in STAGES:
        records = [
            record
            for record in baseline
            if record["stage"] == stage_name
        ]
        _contact_sheet(
            [
                (
                    f"{stage_name} | seed {record['seed']}",
                    Path(record["path"]),
                )
                for record in records
            ],
            output_root / "review" / stage_name / "contact-sheet.jpg",
            columns=2,
        )

    by_key = {
        (record["stage"], record["seed"]): record
        for record in baseline
    }
    paired_entries: list[tuple[str, Path]] = []
    for seed in SETTINGS["seeds"]:
        for stage_name in STAGES:
            record = by_key[(stage_name, seed)]
            paired_entries.append(
                (
                    f"{stage_name} | seed {seed}",
                    Path(record["path"]),
                )
            )
    _contact_sheet(
        paired_entries,
        output_root / "pairs-labeled.jpg",
        columns=2,
    )

    rng = np.random.default_rng(1912)
    blind_records = [
        baseline[index]
        for index in rng.permutation(len(baseline))
    ]
    _contact_sheet(
        [
            (record["name"], Path(record["path"]))
            for record in blind_records
        ],
        output_root / "comparison-blind.jpg",
        columns=2,
        blind=True,
    )
    mapping = {
        f"Candidate {index + 1:02d}": {
            "name": record["name"],
            "stage": record["stage"],
            "seed": record["seed"],
        }
        for index, record in enumerate(blind_records)
    }
    _write_json(output_root / "_work" / "blind_order.json", mapping)

    revised = [
        record
        for record in candidates
        if record.get("revision") == "sediment_emphasis"
    ]
    revised_mapping: dict[str, Any] = {}
    if revised:
        for stage_name in STAGES:
            records = [
                record
                for record in revised
                if record["stage"] == stage_name
            ]
            _contact_sheet(
                [
                    (
                        f"sediment_emphasis {stage_name} | "
                        f"seed {record['seed']}",
                        Path(record["path"]),
                    )
                    for record in records
                ],
                (
                    output_root
                    / "review"
                    / "sediment_emphasis"
                    / stage_name
                    / "contact-sheet.jpg"
                ),
                columns=2,
            )
        revised_by_key = {
            (record["stage"], record["seed"]): record
            for record in revised
        }
        revised_pairs: list[tuple[str, Path]] = []
        for seed in SETTINGS["seeds"]:
            for stage_name in STAGES:
                record = revised_by_key[(stage_name, seed)]
                revised_pairs.append(
                    (
                        f"sediment_emphasis {stage_name} | seed {seed}",
                        Path(record["path"]),
                    )
                )
        _contact_sheet(
            revised_pairs,
            output_root / "pairs-revised.jpg",
            columns=2,
        )
        revised_blind = [
            revised[index]
            for index in rng.permutation(len(revised))
        ]
        _contact_sheet(
            [
                (record["name"], Path(record["path"]))
                for record in revised_blind
            ],
            output_root / "comparison-revised-blind.jpg",
            columns=2,
            blind=True,
        )
        revised_mapping = {
            f"Candidate {index + 1:02d}": {
                "name": record["name"],
                "stage": record["stage"],
                "seed": record["seed"],
            }
            for index, record in enumerate(revised_blind)
        }
        _write_json(
            output_root / "_work" / "blind_order_revised.json",
            revised_mapping,
        )
    return {
        "baseline": mapping,
        "sediment_emphasis": revised_mapping,
    }


def build_soft_sediment_pair(
    output_root: Path = OUTPUT_ROOT,
    *,
    base_seed: int = 3102,
) -> dict[str, Any]:
    """Add audited soft sediment density to one consistent raw model pair."""

    if base_seed not in SETTINGS["seeds"]:
        raise ValueError(f"unsupported base seed: {base_seed}")
    states = load_states(MECHANISM_ROOT / "states.jsonl")
    config = json.loads(
        (MECHANISM_ROOT / "simulation_config.json").read_text(
            encoding="utf-8"
        )
    )
    work_root = output_root / "_work" / "soft_sediment"
    review_root = output_root / "review" / "soft_sediment"
    work_root.mkdir(parents=True, exist_ok=True)
    review_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for stage_name, stage in STAGES.items():
        base_path = (
            output_root
            / "review"
            / stage_name
            / f"{stage_name}_s{base_seed}.png"
        )
        if not base_path.is_file():
            raise FileNotFoundError(
                f"baseline candidate is missing: {base_path}"
            )
        state = states[stage["state_frame"]]
        density = _soft_sediment_density(
            state,
            config,
            (SETTINGS["width"], SETTINGS["height"]),
        )
        density_path = work_root / f"{stage_name}_density.png"
        alpha_path = work_root / f"{stage_name}_alpha.png"
        corridor_path = work_root / "river_corridor.png"
        Image.fromarray(
            np.uint8(np.clip(density, 0.0, 1.0) * 255),
            mode="L",
        ).save(density_path)
        output_path = review_root / f"{stage_name}_s{base_seed}.png"
        result, corridor = _apply_soft_sediment(
            Image.open(base_path),
            density,
        )
        alpha = np.clip(density * 0.72, 0.0, 0.72)
        Image.fromarray(
            np.uint8(np.clip(alpha, 0.0, 1.0) * 255),
            mode="L",
        ).save(alpha_path)
        Image.fromarray(
            np.uint8(np.clip(corridor, 0.0, 1.0) * 255),
            mode="L",
        ).save(corridor_path)
        result.save(output_path)
        records[stage_name] = {
            "display_frame": stage["display_frame"],
            "state_frame": stage["state_frame"],
            "particle_count": len(state["particles"]),
            "base_model_output": {
                "path": str(base_path.resolve()),
                "sha256": _sha256(base_path),
            },
            "density": {
                "path": str(density_path.resolve()),
                "sha256": _sha256(density_path),
                "nonzero_pixels": int((density > 0).sum()),
                "pixels_above_0_05": int((density > 0.05).sum()),
                "maximum": round(float(density.max()), 6),
            },
            "alpha": {
                "path": str(alpha_path.resolve()),
                "sha256": _sha256(alpha_path),
                "maximum_opacity": 0.72,
                "actual_maximum": round(float(alpha.max()), 6),
            },
            "river_corridor": {
                "path": str(corridor_path.resolve()),
                "sha256": _sha256(corridor_path),
                "purpose": (
                    "The controlled region between the two river banks is "
                    "mechanistically water; normalize only that region "
                    "before applying sediment."
                ),
            },
            "result": {
                "path": str(output_path.resolve()),
                "sha256": _sha256(output_path),
                "size": list(result.size),
            },
        }

    _contact_sheet(
        [
            (
                "1 | audited sediment inside river",
                Path(records["in_channel"]["result"]["path"]),
            ),
            (
                "2 | audited sediment reaches outlet",
                Path(records["at_outlet"]["result"]["path"]),
            ),
        ],
        output_root / "pairs-soft-sediment.jpg",
        columns=2,
    )
    result = {
        "status": "built",
        "base_seed": base_seed,
        "classification": (
            "hybrid: raw SDXL ControlNet base plus deterministic audited "
            "soft-sediment color layer"
        ),
        "reason": (
            "baseline and prompt revision both preserved scene geometry but "
            "16/16 raw outputs failed to show an ochre sediment front"
        ),
        "method": {
            "coordinate_mapping": (
                "Map every particle x from mechanism [0, coastline_x=38] "
                "to the naturalized river [0, mouth_x=0.43*1344]; map y "
                "relative to river_center_y=31.5 and river_half_width=4 "
                "into the local curved river corridor."
            ),
            "density": (
                "Nearest-pixel particle splat, OpenCV GaussianBlur "
                "sigmaX=20/sigmaY=30 pixels, positive 99th-percentile "
                "normalization, gamma=0.72, multiplied by a 2.5-pixel "
                "soft river-corridor boundary."
            ),
            "color": (
                "First normalize the softly feathered river corridor toward "
                "RGB(54,123,132) at alpha 0.62 while retaining raw-model "
                "luminance, fading that normalization over the last 70 "
                "pixels before the outlet; then blend particle density toward "
                "RGB(150,76,35) with maximum alpha 0.72."
            ),
            "geometry_change": False,
            "model_input": False,
            "mask_projection": (
                "No model texture projection. One explicit post-generation "
                "river-corridor color constraint is used and recorded."
            ),
        },
        "records": records,
    }
    _write_json(
        output_root / "_work" / "soft_sediment_manifest.json",
        result,
    )
    return result


def _review_text(output_root: Path) -> str:
    review_path = output_root / "_work" / "review.json"
    if not review_path.is_file():
        return "尚未完成审图，不能选择最终图对。"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    pair_lines = "\n".join(
        f"- seed {seed}：{text}"
        for seed, text in review["pair_reviews"].items()
    )
    selected = (
        "最终采用“ControlNet 地貌底图 + 机制泥沙层”的组合方案。"
        f"底图使用随机种子 {review['selected_seed']}；随机种子只是让模型"
        "从同一组随机噪声开始，便于两张图保持相似构图。"
        if review["selected_seed"] is not None
        else "没有图对通过本阶段门槛，因此未写入 `final/`。"
    )
    return f"""{review["verdict"]}

- 盲评：{review["blind_review"]}
- 河道内阶段：{review["in_channel"]}
- 到达河口阶段：{review["at_outlet"]}
- 一致性：{review["consistency"]}
- 首轮 raw 结果：{review["baseline_result"]}
- 提示词修订结果：{review["revision_result"]}
- 软悬沙层结果：{review["soft_sediment_result"]}

{pair_lines}

{selected}

下一步：{review["next_step"]}"""


def _materialize_final(output_root: Path) -> None:
    review_path = output_root / "_work" / "review.json"
    if not review_path.is_file():
        return
    review = json.loads(review_path.read_text(encoding="utf-8"))
    seed = review.get("selected_seed")
    if seed is None:
        return
    variant = review["selected_variant"]
    final_root = output_root / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    final_records: dict[str, Any] = {}
    for stage_name in STAGES:
        if variant == "soft_sediment":
            source = (
                output_root
                / "review"
                / "soft_sediment"
                / f"{stage_name}_s{seed}.png"
            )
        elif variant == "sediment_emphasis":
            source = (
                output_root
                / "review"
                / "sediment_emphasis"
                / stage_name
                / f"{stage_name}_v2_s{seed}.png"
            )
        else:
            source = (
                output_root
                / "review"
                / stage_name
                / f"{stage_name}_s{seed}.png"
            )
        target = final_root / f"{stage_name}.png"
        shutil.copyfile(source, target)
        final_records[stage_name] = {
            "source": str(source.resolve()),
            "path": str(target.resolve()),
            "sha256": _sha256(target),
            "size": list(Image.open(target).size),
        }
    _contact_sheet(
        [
            ("1 | sediment moving inside river", final_root / "in_channel.png"),
            ("2 | sediment reaches river outlet", final_root / "at_outlet.png"),
        ],
        final_root / "selected-pair.jpg",
        columns=2,
    )
    _write_json(
        final_root / "selection.json",
        {
            "selected_variant": variant,
            "selected_seed": seed,
            "classification": (
                "hybrid model base plus audited soft-sediment layer"
                if variant == "soft_sediment"
                else "raw model output"
            ),
            "records": final_records,
            "pair_preview": str(
                (final_root / "selected-pair.jpg").resolve()
            ),
        },
    )


def write_report(output_root: Path = OUTPUT_ROOT) -> Path:
    manifest = json.loads(
        (output_root / "_work" / "prepare_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = json.loads(
        (output_root / "_work" / "metadata.json").read_text(encoding="utf-8")
    )
    _materialize_final(output_root)
    rows = [
        "| 文件 | 版本 | 阶段 | display/state | seed | 秒 | SHA-256 |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for record in metadata["candidates"]:
        rows.append(
            f"| `{Path(record['path']).name}` | "
            f"{record.get('revision', 'baseline')} | {record['stage']} | "
            f"{record['display_frame']}/{record['state_frame']} | "
            f"{record['seed']} | {record['inference_seconds']:.3f} | "
            f"`{record['sha256']}` |"
        )
    token_lines = []
    for filename, record in metadata["prompt_token_preflight"].items():
        counts = record["counts_including_special_tokens"]
        limits = record["limits"]
        token_lines.append(
            f"- `{filename}`：tokenizer "
            f"{counts['tokenizer']}/{limits['tokenizer']}，tokenizer_2 "
            f"{counts['tokenizer_2']}/{limits['tokenizer_2']}"
        )
    weight_lines = []
    for model_id, files in metadata["model_fingerprints"]["models"].items():
        weight_lines.append(f"- `{model_id}`")
        for filename, digest in files.items():
            weight_lines.append(f"  - `{filename}`：`{digest}`")

    first = manifest["selections"]["in_channel"]
    second = manifest["selections"]["at_outlet"]
    prompts = manifest["prompts"]
    soft_manifest_path = (
        output_root / "_work" / "soft_sediment_manifest.json"
    )
    soft = (
        json.loads(soft_manifest_path.read_text(encoding="utf-8"))
        if soft_manifest_path.is_file()
        else None
    )
    soft_text = (
        "尚未构建软悬沙图对。"
        if soft is None
        else f"""选择 raw baseline seed {soft["base_seed"]} 作为同场景底图。对
state 21/27 的全部 {soft["records"]["in_channel"]["particle_count"]}/
{soft["records"]["at_outlet"]["particle_count"]} 个粒子执行：

1. 将每个粒子的 x 从机制河道 `[0, coast=38]` 映射到自然化河道
   `[0, mouth=0.43×1344]`；y 相对机制河心 31.5、半宽 4 映射到当地弯曲河道。
2. 粒子落点后使用 OpenCV 高斯扩散 `sigmaX=20 / sigmaY=30` 像素，按正值第 99
   百分位归一化、`gamma=0.72`，再乘以边缘羽化 2.5 像素的河道区域。
3. 保留 raw 底图明暗，将河道水色以 alpha 0.62 向 `RGB(54,123,132)` 归一化，并在
   出口前 70 像素渐隐；随后按粒子密度以最大 alpha 0.72 混合
   `RGB(150,76,35)` 的悬沙色。

中间文件不是隐藏的：

- `river_corridor.png`：两条 Canny 河岸之间的软水域约束；
- `in_channel_density.png` / `at_outlet_density.png`：机制粒子密度；
- `in_channel_alpha.png` / `at_outlet_alpha.png`：最终悬沙透明度；
- `_work/soft_sediment_manifest.json`：全部输入、参数和 SHA-256。

这层不输入扩散模型，也不改变岸线。它先修正 raw 模型把部分河道内部画成沙色的问题，
再只在河道水域内改变颜色。最终结果必须称为“模型底图 + 机制软悬沙层”的混合图，不能
冒充原始 SDXL 输出。"""
    )
    report = f"""# Stage 1.2 输沙两关键帧生成报告

## 结论

{_review_text(output_root)}

## 1. 实际做了什么

1. 从机制时间线选择 display 10 / state 21 和 display 13 / state 27。
2. 核对全部粒子坐标，保证第一张泥沙仍在河道上游，第二张刚到海岸左侧，两张都没有
   沉降和新生陆地。
3. 生成一张两帧共用的 `natural_sparse_canny.png`，只画自然海岸和两条河岸。
4. 使用纯 `StableDiffusionXLControlNetPipeline`，按相同 seed 生成首轮 8 张 raw 候选。
5. 首轮 8/8 没有赭色悬沙后，只修改泥沙措辞，再生成 `sediment_emphasis` 8 张 raw
   候选；模型、控制图、参数和 seed 不变。
6. 修订版仍失败后，选择地理关系最清楚的 baseline seed 3102，用 state 21/27 的真实
   粒子生成可审计的软悬沙密度层，得到当前两张混合关键帧。
7. raw 推理没有 img2img、strength、mask projection 或后期合成；最终混合关键帧额外
   使用了明确记录的河道水域约束和软颜色层，不能称为 raw 模型输出。
8. 两轮 raw 结果都先盲看单图，再解盲按 seed 成对审查。

## 2. 为什么选择这两个机制时刻

### 第一张：河道内输送

- display/state：{first["display_frame"]}/{first["state_frame"]}
- 悬浮颗粒：{first["particle_count_from_state"]}
- 粒子前缘 x：{first["particle_max_x"]}
- 海岸 x：{first["coastline_x"]}
- 前缘距海岸：{first["distance_from_front_to_coast"]} 个机制网格
- 到达或越过海岸的颗粒：{first["particles_at_or_beyond_coast"]}
- 水下沉积网格：{first["state_stats"]["underwater_deposit_cells"]}
- 新生陆地网格：{first["state_stats"]["new_land_cells"]}

### 第二张：到达河口

- display/state：{second["display_frame"]}/{second["state_frame"]}
- 悬浮颗粒：{second["particle_count_from_state"]}
- 粒子前缘 x：{second["particle_max_x"]}
- 海岸 x：{second["coastline_x"]}
- 前缘距海岸：{second["distance_from_front_to_coast"]} 个机制网格
- 到达或越过海岸的颗粒：{second["particles_at_or_beyond_coast"]}
- 水下沉积网格：{second["state_stats"]["underwater_deposit_cells"]}
- 新生陆地网格：{second["state_stats"]["new_land_cells"]}

`in_channel_mechanism_audit.png` 和 `at_outlet_mechanism_audit.png` 是对应程序帧，
只用于审计状态，没有输入模型。

## 3. sparse Canny 如何强化

共用控制图：`{manifest["control"]["path"]}`

- edge pixels：{manifest["control"]["edge_pixels"]}
- edge fraction：{manifest["control"]["edge_fraction"]}
- SHA-256：`{manifest["control"]["sha256"]}`
- 包含：轻微自然弯曲的连续海岸、两条河岸、出口处小幅扩宽；
- 不包含：泥沙、羽流、水深线、等高线、颗粒、箭头、文字和 mask。

这次没有用详细线条控制泥沙。Canny 只回答“岸在哪里”，提示词回答“泥沙前缘走到
哪里”，避免把柔软的悬沙密度误生成硬沟槽。

## 4. 实际提示词

### 第一张正向

```text
{prompts["in_channel.txt"]["text"]}
```

### 第一张负向

```text
{prompts["in_channel_negative.txt"]["text"]}
```

### 第二张正向

```text
{prompts["at_outlet.txt"]["text"]}
```

### 第二张负向

```text
{prompts["at_outlet_negative.txt"]["text"]}
```

### 有失败证据后的 `sediment_emphasis`

第一张：

```text
{prompts["in_channel_v2.txt"]["text"]}
```

第二张：

```text
{prompts["at_outlet_v2.txt"]["text"]}
```

共用负向：

```text
{prompts["sediment_emphasis_negative.txt"]["text"]}
```

推理前 token 检查：

{chr(10).join(token_lines)}

## 5. 固定模型和参数

- SDXL：`{MODEL_IDS["sdxl_base"]}`，FP16
- ControlNet：`{MODEL_IDS["controlnet_canny"]}`，FP16
- pipeline：`{SETTINGS["pipeline"]}`
- size：{SETTINGS["width"]}×{SETTINGS["height"]}
- steps：{SETTINGS["steps"]}
- CFG：{SETTINGS["guidance_scale"]}
- ControlNet scale：{SETTINGS["controlnet_conditioning_scale"]}
- seeds：{SETTINGS["seeds"]}
- scheduler：`{metadata["scheduler"]["class"]}`
- model load：{metadata["model_load_seconds"]:.3f} 秒
- 首轮 generation：{metadata["total_generation_seconds"]:.3f} 秒
- 修订版 model load：{metadata["revision_run"]["model_load_seconds"]:.3f} 秒
- 修订版 generation：{metadata["revision_run"]["generation_seconds"]:.3f} 秒
- GPU：`{metadata["runtime"]["gpu"]}`
- HIP：`{metadata["runtime"]["hip"]}`
- raw 生成的 img2img / strength / mask projection：无 / 无 / 无

## 6. 全部候选

{chr(10).join(rows)}

查看顺序：

- `source-comparison.jpg`：两张机制审计帧和共用 Canny；
- `comparison-blind.jpg`：隐藏阶段与 seed 的单图盲评；
- `pairs-labeled.jpg`：每行一个 seed，左侧河道内、右侧到达河口；
- `comparison-revised-blind.jpg` / `pairs-revised.jpg`：提示词修订版；
- `pairs-soft-sediment.jpg`：机制软悬沙图对；
- `review/*/contact-sheet.jpg`：各阶段内部对比；
- `final/selected-pair.jpg`：只有图对通过门槛时才存在。

## 7. 为什么最终增加软悬沙层

{soft_text}

## 8. 模型权重

{chr(10).join(weight_lines)}

## 9. 从仓库根目录复现

```bash
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test \\
  --prepare
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test \\
  --generate --force
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test \\
  --generate-revision --force
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test \\
  --build-soft-sediment --base-seed 3102
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test \\
  --report
```

`_work/prepare_manifest.json` 保存状态选择、输入图与 prompt；`metadata.json` 保存每张
raw 候选的参数、耗时与哈希；`soft_sediment_manifest.json` 保存最终软层的完整方法；
`review.json` 保存人工审图结论。
"""
    report_path = output_root / "report.md"
    report_path.write_text(report, encoding="utf-8")
    (output_root / "report.html").write_text(
        render_visual_report(output_root, manifest, metadata, soft),
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--generate-revision", action="store_true")
    parser.add_argument("--build-soft-sediment", action="store_true")
    parser.add_argument("--base-seed", type=int, default=3102)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not (
        args.prepare
        or args.generate
        or args.generate_revision
        or args.build_soft_sediment
        or args.report
    ):
        args.prepare = args.generate = args.report = True
    result: Any = None
    if args.prepare:
        result = prepare(args.output)
    if args.generate:
        result = generate(args.output, force=args.force)
    if args.generate_revision:
        result = generate_revision(args.output, force=args.force)
    if args.build_soft_sediment:
        result = build_soft_sediment_pair(
            args.output,
            base_seed=args.base_seed,
        )
    if args.report:
        result = {"report": str(write_report(args.output).resolve())}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
