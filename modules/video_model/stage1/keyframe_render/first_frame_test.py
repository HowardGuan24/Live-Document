"""Run the Stage 1.1 first-frame generation experiment.

The experiment deliberately separates three different questions:

* can the local SDXL installation make a normal high-quality image?
* can unconstrained or structurally controlled top-down generation depict the
  early river-mouth state?
* is a scene-first underwater shot a better visual translation of that state?

No semantic-mask texture projection is used in this module.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import importlib
import importlib.metadata
import json
import shutil
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..causal_delta.config import (
    MECHANISM_ROOT,
    OUTPUT_ROOT as CAUSAL_OUTPUT_ROOT,
    SimulationConfig,
    original_land,
    water_depth,
)
from ..causal_delta.validate import load_states
from .enhance import (
    MODEL_IDS,
    _diffusers_runtime,
    fingerprint_models,
    model_paths,
)


STAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = STAGE_ROOT / "output" / "keyframe_render" / "first_frame_test"
PROMPT_ROOT = Path(__file__).with_name("first_frame_prompts")
SOURCE_STATE_FRAME = 36
SOURCE_DISPLAY_FRAME = 25

SETTINGS = {
    "width": 1344,
    "height": 768,
    "steps": 36,
    "guidance_scale": 6.5,
    "dtype": "float16",
    "scheduler": "model default",
    "mask_projection": False,
}

PROMPT_FILES = {
    "sanity": "sanity.txt",
    "sanity_negative": "sanity_negative.txt",
    "topdown": "topdown.txt",
    "topdown_controlled": "topdown_controlled.txt",
    "topdown_negative": "topdown_negative.txt",
    "underwater": "underwater.txt",
    "underwater_negative": "underwater_negative.txt",
    "underwater_revised": "underwater_revised.txt",
    "underwater_revised_negative": "underwater_revised_negative.txt",
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    route: str
    seed: int
    prompt_key: str
    negative_key: str
    strength: float | None = None
    control_scale: float | None = None


TEXT_CANDIDATES = (
    CandidateSpec(
        "sanity_s3100",
        "model_sanity",
        3100,
        "sanity",
        "sanity_negative",
    ),
    *(
        CandidateSpec(
            f"free_topdown_s{seed}",
            "free_topdown",
            seed,
            "topdown",
            "topdown_negative",
        )
        for seed in (3101, 3102, 3103, 3104)
    ),
    *(
        CandidateSpec(
            f"underwater_scene_s{seed}",
            "underwater_scene",
            seed,
            "underwater",
            "underwater_negative",
        )
        for seed in (3101, 3102, 3103, 3104)
    ),
    *(
        CandidateSpec(
            f"underwater_revised_s{seed}",
            "underwater_revised",
            seed,
            "underwater_revised",
            "underwater_revised_negative",
        )
        for seed in (3101, 3102, 3103, 3104)
    ),
)

CONTROLLED_CANDIDATES = (
    CandidateSpec(
        "controlled_str055_ctrl050_s3102",
        "controlled_topdown",
        3102,
        "topdown_controlled",
        "topdown_negative",
        0.55,
        0.50,
    ),
    CandidateSpec(
        "controlled_str070_ctrl050_s3102",
        "controlled_topdown",
        3102,
        "topdown_controlled",
        "topdown_negative",
        0.70,
        0.50,
    ),
    CandidateSpec(
        "controlled_str085_ctrl050_s3102",
        "controlled_topdown",
        3102,
        "topdown_controlled",
        "topdown_negative",
        0.85,
        0.50,
    ),
    CandidateSpec(
        "controlled_str070_ctrl035_s3102",
        "controlled_topdown",
        3102,
        "topdown_controlled",
        "topdown_negative",
        0.70,
        0.35,
    ),
    CandidateSpec(
        "controlled_str070_ctrl065_s3102",
        "controlled_topdown",
        3102,
        "topdown_controlled",
        "topdown_negative",
        0.70,
        0.65,
    ),
    CandidateSpec(
        "controlled_str070_ctrl050_s3101",
        "controlled_topdown",
        3101,
        "topdown_controlled",
        "topdown_negative",
        0.70,
        0.50,
    ),
    CandidateSpec(
        "controlled_str070_ctrl050_s3103",
        "controlled_topdown",
        3103,
        "topdown_controlled",
        "topdown_negative",
        0.70,
        0.50,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_prompt(key: str) -> str:
    return (PROMPT_ROOT / PROMPT_FILES[key]).read_text(encoding="utf-8").strip()


def _load_source_state() -> tuple[dict[str, Any], dict[str, Any], SimulationConfig]:
    states = load_states(MECHANISM_ROOT / "states.jsonl")
    state = states[SOURCE_STATE_FRAME]
    timeline = json.loads(
        (CAUSAL_OUTPUT_ROOT / "timeline.json").read_text(encoding="utf-8")
    )
    selection = next(
        entry
        for entry in timeline
        if entry["display_frame"] == SOURCE_DISPLAY_FRAME
        and entry["state_frame"] == SOURCE_STATE_FRAME
    )
    config_data = json.loads(
        (MECHANISM_ROOT / "simulation_config.json").read_text(encoding="utf-8")
    )
    config = SimulationConfig(**config_data)
    return state, selection, config


def _resize_float(field: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(field, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BICUBIC), dtype=np.float64)


def _plume_density(
    state: dict[str, Any],
    config: SimulationConfig,
) -> np.ndarray:
    """Turn discrete audited particle positions into a continuous density field."""

    density = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)
    for particle in state["particles"]:
        x = float(np.clip(particle["x"], 0, config.grid_width - 1.001))
        y = float(np.clip(particle["y"], 0, config.grid_height - 1.001))
        x0, y0 = int(x), int(y)
        dx, dy = x - x0, y - y0
        for yy, wy in ((y0, 1.0 - dy), (min(y0 + 1, config.grid_height - 1), dy)):
            for xx, wx in ((x0, 1.0 - dx), (min(x0 + 1, config.grid_width - 1), dx)):
                density[yy, xx] += wx * wy
    try:
        import cv2

        density = cv2.GaussianBlur(
            density,
            (0, 0),
            sigmaX=2.25,
            sigmaY=2.60,
            borderType=cv2.BORDER_REFLECT,
        )
    except ImportError:
        maximum = float(density.max()) or 1.0
        blurred = Image.fromarray(np.uint8(density / maximum * 255), mode="L")
        density = np.asarray(
            blurred.filter(ImageFilter.GaussianBlur(2.4)),
            dtype=np.float32,
        )
    positive = density[density > 0]
    normalization = float(np.quantile(positive, 0.99)) if positive.size else 1.0
    density = np.clip(density / max(normalization, 1e-8), 0.0, 1.0)
    return np.power(density, 0.62)


def _make_smooth_base(
    state: dict[str, Any],
    config: SimulationConfig,
    density_grid: np.ndarray,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Create continuous img2img input, plume map, and geometry-only Canny."""

    size = (SETTINGS["width"], SETTINGS["height"])
    base_land = original_land(config)
    depth = water_depth(config)
    finite_water = (~base_land) & np.isfinite(depth)
    water_depth_values = depth[finite_water]
    low = float(np.quantile(water_depth_values, 0.02))
    high = float(np.quantile(water_depth_values, 0.98))
    normalized_depth = np.clip((depth - low) / max(high - low, 1e-8), 0.0, 1.0)
    normalized_depth[base_land] = 0.0

    depth_high = _resize_float(normalized_depth, size)
    yy, xx = np.indices((SETTINGS["height"], SETTINGS["width"]), dtype=np.float64)
    subtle_water = 0.025 * np.sin(xx / 43.0) + 0.018 * np.cos(yy / 31.0)
    deep = np.array([18.0, 78.0, 107.0])
    shallow = np.array([78.0, 151.0, 153.0])
    shallow_weight = np.clip(1.0 - depth_high + subtle_water, 0.0, 1.0)[..., None]
    water_rgb = deep * (1.0 - shallow_weight) + shallow * shallow_weight

    land_grid = np.uint8(base_land) * 255
    land_hard = np.asarray(
        Image.fromarray(land_grid, mode="L").resize(size, Image.Resampling.NEAREST)
    )
    land_alpha = np.asarray(
        Image.fromarray(land_grid, mode="L")
        .resize(size, Image.Resampling.LANCZOS)
        .filter(ImageFilter.GaussianBlur(0.7)),
        dtype=np.float64,
    ) / 255.0
    land_variation = (
        0.04 * np.sin(xx / 67.0)
        + 0.025 * np.cos(yy / 49.0)
        + 0.018 * np.sin((xx + yy) / 33.0)
    )[..., None]
    land_rgb = np.array([112.0, 117.0, 76.0]) * (1.0 + land_variation)
    rgb = water_rgb * (1.0 - land_alpha[..., None]) + land_rgb * land_alpha[..., None]

    thickness = np.asarray(state["thick"], dtype=np.float64)
    if float(thickness.max()) > 0:
        deposit_grid = np.clip(thickness / float(thickness.max()), 0.0, 1.0)
        deposit_high = _resize_float(deposit_grid, size)
        deposit_alpha = (
            np.clip(deposit_high, 0.0, 1.0) ** 0.7
            * (1.0 - land_alpha)
            * 0.22
        )
        deposit_color = np.array([159.0, 112.0, 62.0])
        rgb = (
            rgb * (1.0 - deposit_alpha[..., None])
            + deposit_color * deposit_alpha[..., None]
        )

    plume_high = np.clip(_resize_float(density_grid, size), 0.0, 1.0)
    plume_alpha = plume_high * (1.0 - 0.72 * land_alpha) * 0.66
    plume_color = np.array([174.0, 111.0, 48.0])
    rgb = rgb * (1.0 - plume_alpha[..., None]) + plume_color * plume_alpha[..., None]
    smooth_base = Image.fromarray(np.uint8(np.clip(rgb, 0, 255)), mode="RGB")
    plume_image = Image.fromarray(np.uint8(plume_high * 255), mode="L")

    try:
        import cv2

        canny_array = cv2.Canny(land_hard, 100, 200)
    except ImportError:
        canny_array = np.asarray(
            Image.fromarray(land_hard, mode="L").filter(ImageFilter.FIND_EDGES)
        )
    canny = Image.fromarray(canny_array, mode="L")
    return smooth_base, plume_image, canny


def prepare_sources(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    state, selection, config = _load_source_state()
    source_root = output_root / "_work" / "source"
    prompt_work_root = output_root / "_work" / "prompts"
    source_root.mkdir(parents=True, exist_ok=True)
    prompt_work_root.mkdir(parents=True, exist_ok=True)

    original_source = CAUSAL_OUTPUT_ROOT / selection["rendered_file"]
    original_target = source_root / "original_frame.png"
    shutil.copyfile(original_source, original_target)

    density_grid = _plume_density(state, config)
    smooth_base, plume_image, canny = _make_smooth_base(state, config, density_grid)
    source_paths = {
        "original_frame": original_target,
        "smooth_base": source_root / "smooth_base.png",
        "plume_density": source_root / "plume_density.png",
        "coastline_canny": source_root / "coastline_canny.png",
    }
    smooth_base.save(source_paths["smooth_base"])
    plume_image.save(source_paths["plume_density"])
    canny.save(source_paths["coastline_canny"])

    prompts: dict[str, dict[str, Any]] = {}
    for key, filename in PROMPT_FILES.items():
        source = PROMPT_ROOT / filename
        target = prompt_work_root / filename
        shutil.copyfile(source, target)
        prompts[key] = {
            "path": str(target.resolve()),
            "sha256": _sha256(target),
            "text": target.read_text(encoding="utf-8").strip(),
        }

    particle_count = len(state["particles"])
    if particle_count != selection["state_stats"]["suspended_particles"]:
        raise ValueError(
            f"particle count {particle_count} does not match timeline "
            f"{selection['state_stats']['suspended_particles']}"
        )
    result = {
        "status": "prepared",
        "source_selection": {
            "display_frame": selection["display_frame"],
            "state_frame": selection["state_frame"],
            "beat_id": selection["beat_id"],
            "caption": selection["caption"],
            "state_stats": selection["state_stats"],
            "timeline_path": str(
                (CAUSAL_OUTPUT_ROOT / "timeline.json").resolve()
            ),
            "states_path": str((MECHANISM_ROOT / "states.jsonl").resolve()),
        },
        "source_method": {
            "original_frame": "Exact rendered display frame 25; audit reference only.",
            "plume_density": (
                "Bilinear splat of all 516 suspended particle coordinates onto "
                "the 96x64 grid, Gaussian sigma 2.25x2.60 cells, 99th-percentile "
                "normalization, gamma 0.62, bicubic resize."
            ),
            "smooth_base": (
                "Continuous bathymetric water colors plus antialiased original "
                "land/river geometry, faint audited deposit thickness, and the "
                "plume density mixed at maximum opacity 0.66."
            ),
            "coastline_canny": (
                "OpenCV Canny 100/200 from the binary original-land mask only; "
                "particles, plume, deposit spots, labels, arrows, and panels "
                "are intentionally excluded."
            ),
            "mask_projection": "Not used.",
        },
        "sources": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size": list(Image.open(path).size),
                "mode": Image.open(path).mode,
            }
            for name, path in source_paths.items()
        },
        "prompts": prompts,
        "settings": SETTINGS,
    }
    _write_json(output_root / "_work" / "prepare_manifest.json", result)
    return result


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in (
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "opencv-python-headless",
        "Pillow",
        "numpy",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _validate_prompt_lengths(base_path: Path) -> dict[str, Any]:
    """Fail before inference if either SDXL CLIP tokenizer would truncate."""

    from transformers import CLIPTokenizer

    result: dict[str, Any] = {}
    violations: list[str] = []
    for key in PROMPT_FILES:
        prompt = _load_prompt(key)
        counts: dict[str, int] = {}
        limits: dict[str, int] = {}
        for subfolder in ("tokenizer", "tokenizer_2"):
            tokenizer = CLIPTokenizer.from_pretrained(
                str(base_path),
                subfolder=subfolder,
                local_files_only=True,
            )
            token_count = len(
                tokenizer(
                    prompt,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
            counts[subfolder] = token_count
            limits[subfolder] = int(tokenizer.model_max_length)
            if token_count > tokenizer.model_max_length:
                violations.append(
                    f"{key}/{subfolder}: {token_count} > "
                    f"{tokenizer.model_max_length}"
                )
        result[key] = {
            "counts_including_special_tokens": counts,
            "limits": limits,
            "would_truncate": any(
                counts[name] > limits[name] for name in counts
            ),
        }
    if violations:
        raise ValueError(
            "SDXL prompt preflight failed; prompt would be truncated: "
            + "; ".join(violations)
        )
    return result


def _candidate_path(output_root: Path, spec: CandidateSpec) -> Path:
    return output_root / "review" / spec.route / f"{spec.name}.png"


def _base_metadata(output_root: Path) -> dict[str, Any]:
    prepare_manifest_path = output_root / "_work" / "prepare_manifest.json"
    if not prepare_manifest_path.is_file():
        prepare_sources(output_root)
    prepare_manifest = json.loads(prepare_manifest_path.read_text(encoding="utf-8"))
    return {
        "status": "generating",
        "experiment": "Stage 1.1 first-frame route comparison",
        "settings": SETTINGS,
        "models": {
            "sdxl_base": {
                "model_id": MODEL_IDS["sdxl_base"],
                "path": str(model_paths()["sdxl_base"].resolve()),
                "variant": "fp16",
            },
            "controlnet_canny": {
                "model_id": MODEL_IDS["controlnet_canny"],
                "path": str(model_paths()["controlnet_canny"].resolve()),
                "variant": "fp16",
            },
            "refiner": None,
        },
        "package_versions": _package_versions(),
        "prepare_manifest": str(prepare_manifest_path.resolve()),
        "source_selection": prepare_manifest["source_selection"],
        "mask_projection": {
            "used": False,
            "reason": (
                "This experiment measures raw generation quality and must not "
                "project model texture back into semantic masks."
            ),
        },
        "candidates": [],
    }


def _record_candidate(
    metadata: dict[str, Any],
    output_root: Path,
    spec: CandidateSpec,
    path: Path,
    *,
    pipeline: str,
    seconds: float,
    peak_gpu_memory_bytes: int | None,
    reused: bool,
) -> None:
    metadata["candidates"].append(
        {
            **asdict(spec),
            "pipeline": pipeline,
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size": list(Image.open(path).size),
            "inference_seconds": round(seconds, 3),
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "reused": reused,
            "prompt_path": str(
                (output_root / "_work" / "prompts" / PROMPT_FILES[spec.prompt_key]).resolve()
            ),
            "negative_prompt_path": str(
                (
                    output_root
                    / "_work"
                    / "prompts"
                    / PROMPT_FILES[spec.negative_key]
                ).resolve()
            ),
        }
    )
    _write_json(output_root / "_work" / "metadata.json", metadata)


def generate_candidates(
    output_root: Path = OUTPUT_ROOT,
    *,
    force: bool = False,
) -> dict[str, Any]:
    prepare_sources(output_root)
    paths = model_paths()
    missing = [name for name, path in paths.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing required local models: {missing}")

    metadata = _base_metadata(output_root)
    metadata["prompt_token_preflight"] = _validate_prompt_lengths(
        paths["sdxl_base"]
    )
    fingerprints = fingerprint_models(output_root, paths)
    metadata["model_fingerprints"] = fingerprints
    torch, ControlNetModel, ControlPipeline = _diffusers_runtime()
    diffusers_module = importlib.import_module("diffusers")
    TextPipeline = getattr(diffusers_module, "StableDiffusionXLPipeline")
    metadata["runtime"] = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    metadata["started_at_unix"] = time.time()
    _write_json(output_root / "_work" / "metadata.json", metadata)

    load_started = time.perf_counter()
    text_pipeline = TextPipeline.from_pretrained(
        str(paths["sdxl_base"]),
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
        use_safetensors=True,
    ).to("cuda")
    text_pipeline.set_progress_bar_config(disable=True)
    metadata["text_pipeline_load_seconds"] = round(
        time.perf_counter() - load_started, 3
    )
    metadata["text_scheduler"] = {
        "class": type(text_pipeline.scheduler).__name__,
        "config": dict(text_pipeline.scheduler.config),
    }
    for spec in TEXT_CANDIDATES:
        path = _candidate_path(output_root, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        reused = path.is_file() and not force
        peak_memory: int | None = None
        if not reused:
            torch.cuda.reset_peak_memory_stats()
            generator = torch.Generator(device="cuda").manual_seed(spec.seed)
            image = text_pipeline(
                prompt=_load_prompt(spec.prompt_key),
                negative_prompt=_load_prompt(spec.negative_key),
                width=SETTINGS["width"],
                height=SETTINGS["height"],
                num_inference_steps=SETTINGS["steps"],
                guidance_scale=SETTINGS["guidance_scale"],
                generator=generator,
            ).images[0]
            image.save(path)
            torch.cuda.synchronize()
            peak_memory = int(torch.cuda.max_memory_allocated())
        _record_candidate(
            metadata,
            output_root,
            spec,
            path,
            pipeline="StableDiffusionXLPipeline (text-to-image)",
            seconds=time.perf_counter() - started,
            peak_gpu_memory_bytes=peak_memory,
            reused=reused,
        )

    del text_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    load_started = time.perf_counter()
    controlnet = ControlNetModel.from_pretrained(
        str(paths["controlnet_canny"]),
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    control_pipeline = ControlPipeline.from_pretrained(
        str(paths["sdxl_base"]),
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
        use_safetensors=True,
    ).to("cuda")
    control_pipeline.set_progress_bar_config(disable=True)
    metadata["control_pipeline_load_seconds"] = round(
        time.perf_counter() - load_started, 3
    )
    metadata["control_scheduler"] = {
        "class": type(control_pipeline.scheduler).__name__,
        "config": dict(control_pipeline.scheduler.config),
    }
    smooth_base = Image.open(
        output_root / "_work" / "source" / "smooth_base.png"
    ).convert("RGB")
    canny = Image.open(
        output_root / "_work" / "source" / "coastline_canny.png"
    ).convert("RGB")
    for spec in CONTROLLED_CANDIDATES:
        path = _candidate_path(output_root, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        reused = path.is_file() and not force
        peak_memory = None
        if not reused:
            torch.cuda.reset_peak_memory_stats()
            generator = torch.Generator(device="cuda").manual_seed(spec.seed)
            image = control_pipeline(
                prompt=_load_prompt(spec.prompt_key),
                negative_prompt=_load_prompt(spec.negative_key),
                image=smooth_base,
                control_image=canny,
                num_inference_steps=SETTINGS["steps"],
                strength=spec.strength,
                guidance_scale=SETTINGS["guidance_scale"],
                controlnet_conditioning_scale=spec.control_scale,
                generator=generator,
            ).images[0]
            image.save(path)
            torch.cuda.synchronize()
            peak_memory = int(torch.cuda.max_memory_allocated())
        _record_candidate(
            metadata,
            output_root,
            spec,
            path,
            pipeline="StableDiffusionXLControlNetImg2ImgPipeline",
            seconds=time.perf_counter() - started,
            peak_gpu_memory_bytes=peak_memory,
            reused=reused,
        )

    metadata["status"] = "generated"
    metadata["completed_at_unix"] = time.time()
    metadata["total_generation_seconds"] = round(
        metadata["completed_at_unix"] - metadata["started_at_unix"], 3
    )
    _write_json(output_root / "_work" / "metadata.json", metadata)
    del control_pipeline, controlnet
    gc.collect()
    torch.cuda.empty_cache()
    build_contact_sheets(output_root)
    return metadata


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    )
    path = next((Path(value) for value in candidates if Path(value).is_file()), None)
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size)


def _contact_sheet(
    entries: list[tuple[str, Path]],
    output_path: Path,
    *,
    columns: int = 3,
    blind: bool = False,
) -> None:
    thumb_size = (448, 256)
    label_height = 64
    rows = (len(entries) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * thumb_size[0], rows * (thumb_size[1] + label_height)),
        (19, 25, 29),
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(17)
    for index, (label, path) in enumerate(entries):
        column, row = index % columns, index // columns
        x, y = column * thumb_size[0], row * (thumb_size[1] + label_height)
        image = Image.open(path).convert("RGB").resize(
            thumb_size, Image.Resampling.LANCZOS
        )
        canvas.paste(image, (x, y))
        shown_label = f"Candidate {index + 1:02d}" if blind else label
        lines = (
            [shown_label]
            if blind
            else textwrap.wrap(shown_label, width=47, max_lines=2)
        )
        for line_index, line in enumerate(lines):
            draw.text(
                (x + 10, y + thumb_size[1] + 7 + line_index * 23),
                line,
                font=font,
                fill="white",
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, subsampling=0)


def build_contact_sheets(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    metadata_path = output_root / "_work" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidates = metadata["candidates"]
    route_labels = {
        "model_sanity": "Route 0 model sanity",
        "free_topdown": "Route A free top-down",
        "controlled_topdown": "Route B controlled top-down",
        "underwater_scene": "Route C underwater scene",
        "underwater_revised": "Route C2 revised underwater",
    }
    for route, route_label in route_labels.items():
        records = [record for record in candidates if record["route"] == route]
        entries = []
        for record in records:
            detail = f"{route_label} | seed {record['seed']}"
            if record["strength"] is not None:
                detail += (
                    f" | strength {record['strength']:.2f}"
                    f" | control {record['control_scale']:.2f}"
                )
            entries.append((detail, Path(record["path"])))
        _contact_sheet(
            entries,
            output_root / "review" / route / "contact-sheet.jpg",
            columns=3 if len(entries) > 4 else 2,
        )

    main_records = [
        record
        for record in candidates
        if record["route"] not in {"model_sanity", "underwater_revised"}
    ]
    rng = np.random.default_rng(1909)
    order = list(rng.permutation(len(main_records)))
    blind_records = [main_records[index] for index in order]
    blind_entries = [
        (record["name"], Path(record["path"])) for record in blind_records
    ]
    _contact_sheet(
        blind_entries,
        output_root / "comparison-blind.jpg",
        columns=3,
        blind=True,
    )
    _contact_sheet(
        [
            (
                (
                    f"{record['route']} | seed {record['seed']}"
                    + (
                        ""
                        if record["strength"] is None
                        else (
                            f" | str {record['strength']:.2f}"
                            f" | ctrl {record['control_scale']:.2f}"
                        )
                    )
                ),
                Path(record["path"]),
            )
            for record in candidates
        ],
        output_root / "comparison-labeled.jpg",
        columns=3,
    )
    blind_order = {
        f"Candidate {index + 1:02d}": {
            "name": record["name"],
            "route": record["route"],
            "path": record["path"],
        }
        for index, record in enumerate(blind_records)
    }
    _write_json(output_root / "_work" / "blind_order.json", blind_order)
    return {
        "comparison_blind": str((output_root / "comparison-blind.jpg").resolve()),
        "comparison_labeled": str(
            (output_root / "comparison-labeled.jpg").resolve()
        ),
        "blind_order": blind_order,
    }


def _candidate_table(metadata: dict[str, Any]) -> str:
    rows = [
        "| 文件 | 路线 | seed | strength | control | 秒 | SHA-256 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for record in metadata["candidates"]:
        strength = "—" if record["strength"] is None else f"{record['strength']:.2f}"
        control = (
            "—" if record["control_scale"] is None else f"{record['control_scale']:.2f}"
        )
        rows.append(
            f"| `{Path(record['path']).name}` | {record['route']} | "
            f"{record['seed']} | {strength} | {control} | "
            f"{record['inference_seconds']:.3f} | `{record['sha256']}` |"
        )
    return "\n".join(rows)


def _review_section(output_root: Path) -> str:
    review_path = output_root / "_work" / "review.json"
    if not review_path.is_file():
        return (
            "尚未写入人工审图结论。先查看 `comparison-blind.jpg`，再创建 "
            "`_work/review.json` 并重新运行 `--report`。"
        )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    blind = review["blind_review"]
    sections = [
        "### 盲评记录\n\n"
        f"{blind['summary']}\n\n"
        + "\n".join(f"- {item}" for item in blind["observations"])
    ]
    for route in (
        "model_sanity",
        "free_topdown",
        "controlled_topdown",
        "underwater_scene",
        "underwater_revised",
    ):
        item = review["routes"][route]
        observations = "\n".join(
            f"- {observation}" for observation in item.get("observations", [])
        )
        sections.append(
            f"### {item['title']}\n\n"
            f"- 最佳候选：`{item['selected']}`\n"
            f"- 是否摆脱像素风：{item['pixel_style']}\n"
            f"- 塑料 / 模糊问题：{item['surface_quality']}\n"
            f"- 语义与物理可信度：{item['semantic_quality']}\n"
            f"- 结论：{item['verdict']}"
            + (f"\n{observations}" if observations else "")
        )
    answers = "\n".join(
        f"{index}. **{item['question']}** {item['answer']}"
        for index, item in enumerate(review["overall"]["answers"], start=1)
    )
    sections.append(
        "### 总结\n\n"
        f"{review['overall']['conclusion']}\n\n"
        f"{answers}\n\n"
        f"下一步：{review['overall']['next_step']}"
    )
    return "\n\n".join(sections)


def write_report(output_root: Path = OUTPUT_ROOT) -> Path:
    metadata = json.loads(
        (output_root / "_work" / "metadata.json").read_text(encoding="utf-8")
    )
    prepare_manifest = json.loads(
        (output_root / "_work" / "prepare_manifest.json").read_text(encoding="utf-8")
    )
    state = prepare_manifest["source_selection"]
    sources = prepare_manifest["sources"]
    fingerprints = metadata["model_fingerprints"]
    weight_lines = []
    for model_id, records in fingerprints["models"].items():
        weight_lines.append(f"- `{model_id}`")
        for relative, digest in records.items():
            weight_lines.append(f"  - `{relative}`: `{digest}`")
    package_lines = [
        f"- {name}: `{version}`"
        for name, version in metadata["package_versions"].items()
    ]
    candidate_counts = {
        route: sum(
            record["route"] == route for record in metadata["candidates"]
        )
        for route in (
            "model_sanity",
            "free_topdown",
            "controlled_topdown",
            "underwater_scene",
            "underwater_revised",
        )
    }
    prompt_token_lines = []
    for key, record in metadata["prompt_token_preflight"].items():
        counts = record["counts_including_special_tokens"]
        limits = record["limits"]
        prompt_token_lines.append(
            f"- `{PROMPT_FILES[key]}`：tokenizer "
            f"{counts['tokenizer']}/{limits['tokenizer']}，tokenizer_2 "
            f"{counts['tokenizer_2']}/{limits['tokenizer_2']}"
        )
    prompt_blocks = [
        f"### `{filename}`\n\n```text\n{_load_prompt(key)}\n```"
        for key, filename in PROMPT_FILES.items()
    ]
    report = f"""# Stage 1.1 第一张图实验报告

## 先看结论

{_review_section(output_root)}

本报告记录的是实际运行结果，不是设计草案。共生成 {len(metadata["candidates"])} 张：
模型自检 {candidate_counts["model_sanity"]} 张、自由俯视
{candidate_counts["free_topdown"]} 张、受控俯视
{candidate_counts["controlled_topdown"]} 张、水下场景
第一版 {candidate_counts["underwater_scene"]} 张、水下场景修订版
{candidate_counts["underwater_revised"]} 张。

建议先看：

- `comparison-blind.jpg`：隐藏路线名称的第一轮 15 张主候选；
- `comparison-labeled.jpg`：显示路线和参数的全部候选；
- `review/*/contact-sheet.jpg`：每条路线内部对比。

## 1. 我实际做了什么

1. 从 `output/causal_delta/timeline.json` 精确选择 display frame
   {state["display_frame"]} / state frame {state["state_frame"]}。
2. 保留原程序帧作审计参考，但没有直接把它放进模型。
3. 从机制状态的全部 {state["state_stats"]["suspended_particles"]} 个悬浮颗粒坐标
   计算连续羽流密度图。
4. 用连续水深、原始陆地 / 河道、微量水下沉积和羽流密度合成
   `smooth_base.png`。
5. 只从原始陆地边界生成 Canny；颗粒、沉积斑点、文字、箭头、图例均不进入边缘图。
6. 用相同的 SDXL Base 分别跑模型自检、自由俯视、水下场景；用 SDXL Base +
   Canny ControlNet 跑受控俯视单变量参数实验。
7. 生成盲评图、带标签总览图和各路线 contact sheet，再基于实际输出写结论。
8. 第一版水下结果 4/4 变成水面以上的河滩后，只修改镜头约束，补跑 C2；
   C2 明确写入“camera fully submerged, no sky or horizon”，负向增加
   above-water / beach / shoreline / sandbar。

**这次没有使用旧阶段的 semantic mask projection。** 因此候选图就是模型的原始输出，
不是把模型纹理重新投影进程序图后的合成图。

第一次 GPU 试运行在模型自检后发现原俯视 prompt 是 143 tokens，SDXL 的 CLIP 上限为
77，后半段构图约束会被截断。我停止该轮，将正向和负向 prompt 都压缩并加入推理前
硬校验，然后使用 `--force` 重新生成；本报告列出的最终候选不包含被截断 prompt 的结果。

## 2. 为什么选择这一帧

数据来源：

- timeline：`{state["timeline_path"]}`
- states：`{state["states_path"]}`
- beat：`{state["beat_id"]}`（{state["caption"]}）

精确状态：

- injected particles：{state["state_stats"]["injected_particles"]}
- suspended particles：{state["state_stats"]["suspended_particles"]}
- settled particles：{state["state_stats"]["settled_particles"]}
- underwater deposit cells：{state["state_stats"]["underwater_deposit_cells"]}
- new land cells：{state["state_stats"]["new_land_cells"]}

这时泥沙已经从河口进入海域，足以表达“水流携沙”；同时 new land cells 为 0，
不会把第一张图误画成已经形成三角洲。

## 3. 四个输入文件是什么

### `original_frame.png`

路径：`{sources["original_frame"]["path"]}`

它是旧机制动画的原始第 25 个显示帧，保留文字、箭头、颗粒等，只用于核对所选状态。
它没有输入 SDXL。SHA-256：`{sources["original_frame"]["sha256"]}`。

### `plume_density.png`

路径：`{sources["plume_density"]["path"]}`

把 state 36 的全部 516 个粒子以双线性权重落到 96×64 网格，使用
`sigmaX=2.25 / sigmaY=2.60` 网格的高斯模糊，再按正值第 99 百分位归一化、
做 `gamma=0.62`，最后双三次放大到 1344×768。白色代表羽流更密。
它只用于生成下一项连续底图。SHA-256：`{sources["plume_density"]["sha256"]}`。

### `smooth_base.png`

路径：`{sources["smooth_base"]["path"]}`

由连续水深色、抗锯齿陆地 / 河道、state 36 的水下沉积厚度和羽流密度组成。
羽流以最大 0.66 的透明度混合；没有颗粒点、文字、箭头、图例或面板。
它只输入 Route B 的 img2img。SHA-256：`{sources["smooth_base"]["sha256"]}`。

### `coastline_canny.png`

路径：`{sources["coastline_canny"]["path"]}`

只对 `original_land(config)` 的二值陆地掩码执行 OpenCV Canny 100/200，所以边缘只表示
连续海岸和河道边界，不包含泥沙羽流与沉积斑点。它只输入 Route B 的 ControlNet。
SHA-256：`{sources["coastline_canny"]["sha256"]}`。

## 4. 路线、提示词和参数

完整提示词原文保存在 `_work/prompts/`，并在 `prepare_manifest.json` 中记录文本和哈希。
下面是本轮推理前实际检查的 token 数（包含首尾特殊 token）：

{chr(10).join(prompt_token_lines)}

- Route 0：纯 SDXL 文生图，seed 3100，验证模型能否正常生成自然材质；
- Route A：纯 SDXL 文生图，seeds 3101–3104，提示词要求正交俯视早期河口；
- Route B：SDXL ControlNet img2img。固定 control=0.50 扫
  strength=0.55/0.70/0.85；固定 strength=0.70 扫 control=0.35/0.50/0.65；
  再用基线 0.70/0.50 跑 seeds 3101/3103；
- Route C：纯 SDXL 文生图，seeds 3101–3104，把同一机制语义改写为真实水下镜头。
- Route C2：Route C 首轮全部变成水面以上场景后的单变量提示词修订；仍用相同
  seeds，只强化完全浸没镜头并禁止天空、地平线、海滩和沙洲。

本次实际送入模型的完整提示词如下：

{(chr(10) * 2).join(prompt_blocks)}

共同参数：1344×768、36 steps、guidance scale 6.5、FP16、模型默认 scheduler。

{_candidate_table(metadata)}

## 5. 模型、运行时与时间

- SDXL Base：`{MODEL_IDS["sdxl_base"]}`
  - 本地路径：`{metadata["models"]["sdxl_base"]["path"]}`
- Canny ControlNet：`{MODEL_IDS["controlnet_canny"]}`
  - 本地路径：`{metadata["models"]["controlnet_canny"]["path"]}`
- Refiner：未使用
- GPU：`{metadata["runtime"]["gpu"]}`
- HIP：`{metadata["runtime"]["hip"]}`
- 总运行时间（含两次模型加载和 20 张推理）：{metadata["total_generation_seconds"]:.3f} 秒
- 文生图 pipeline 加载：{metadata["text_pipeline_load_seconds"]:.3f} 秒
- ControlNet pipeline 加载：{metadata["control_pipeline_load_seconds"]:.3f} 秒
- 文生图 scheduler：`{metadata["text_scheduler"]["class"]}`
- ControlNet scheduler：`{metadata["control_scheduler"]["class"]}`

软件版本：

{chr(10).join(package_lines)}

实际 FP16 权重 SHA-256：

{chr(10).join(weight_lines)}

## 6. 从零复现

从仓库根目录运行：

```bash
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.first_frame_test \\
  --prepare

/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.first_frame_test \\
  --generate --force

/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.first_frame_test \\
  --report
```

如果模型不在默认位置，先设置：

```bash
export SDXL_BASE_PATH=/absolute/path/to/sdxl-base-1.0
export SDXL_CANNY_CONTROLNET_PATH=/absolute/path/to/controlnet-canny-sdxl-1.0
```

不带 `--force` 会复用同名 PNG；带 `--force` 才会重新推理。固定 prompt、参数、seed、
权重和软件栈后结果可复现。ROCm / PyTorch 或 Diffusers 版本变化仍可能产生细微数值差异。

## 7. 输出文件怎么读

- `report.md` / `report.html`：过程、参数、审图结论和复现方法；
- `comparison-blind.jpg`：最初三条主路线的 15 张盲评总览，不显示路线名；
- `comparison-labeled.jpg`：全部候选以及真实路线 / 参数；
- `review/model_sanity/`：模型健康检查，不参与路线优胜；
- `review/free_topdown/`：Route A 原始模型输出；
- `review/controlled_topdown/`：Route B 原始模型输出；
- `review/underwater_scene/`：Route C 原始模型输出；
- `review/underwater_revised/`：Route C2 单变量提示词修订后的原始模型输出；
- `_work/source/`：可审计、可重建的输入，不是最终图；
- `_work/prompts/`：本次实际送入模型的提示词；
- `_work/prepare_manifest.json`：状态选择、输入生成方法与输入哈希；
- `_work/metadata.json`：每张图的模型参数、耗时与哈希；
- `_work/model_fingerprints.json`：实际使用的 FP16 权重哈希；
- `_work/blind_order.json`：盲评编号与真实文件的映射；
- `_work/review.json`：人工审图记录。
"""
    report_path = output_root / "report.md"
    report_path.write_text(report, encoding="utf-8")
    html_path = output_root / "report.html"
    html_path.write_text(
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Stage 1.1 第一张图实验报告</title>
<style>
body{max-width:1100px;margin:32px auto;padding:0 24px;background:#f8fafb;
color:#182229;font:16px/1.65 system-ui,sans-serif}
pre{white-space:pre-wrap;background:white;padding:28px;border-radius:12px;
box-shadow:0 2px 16px #0001}
</style></head><body><pre>"""
        + html.escape(report)
        + "</pre></body></html>\n",
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not (args.prepare or args.generate or args.report):
        args.prepare = args.generate = args.report = True
    result: Any = None
    if args.prepare:
        result = prepare_sources(args.output)
    if args.generate:
        result = generate_candidates(args.output, force=args.force)
    if args.report:
        result = {"report": str(write_report(args.output).resolve())}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
