"""Generate SDXL style proposals and project their texture into semantic masks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .prepare import KEYFRAME_OUTPUT_ROOT, WORK_DIR_NAME


SETTINGS = {
    "width": 768,
    "height": 512,
    "steps": 36,
    "strength": 0.50,
    "guidance_scale": 6.5,
    "controlnet_scale": 1.35,
    "seeds": [3101, 3102],
    "dtype": "float16",
}

STYLE_FILES = {
    "physical_geography": "physical_geography.txt",
    "museum_infographic": "museum_infographic.txt",
    "remote_sensing": "remote_sensing.txt",
}

MODEL_IDS = {
    "sdxl_base": "stabilityai/stable-diffusion-xl-base-1.0",
    "controlnet_canny": "diffusers/controlnet-canny-sdxl-1.0",
}

MODEL_WEIGHT_FILES = {
    "sdxl_base": (
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/model.fp16.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ),
    "controlnet_canny": ("diffusion_pytorch_model.fp16.safetensors",),
}


def _diffusers_runtime() -> tuple[Any, Any, Any]:
    """Import Diffusers while ignoring a broken optional FlashAttention stub.

    The shared ROCm environment advertises ``flash_attn`` package metadata but
    does not contain its CUDA extension. Diffusers otherwise treats it as
    usable and fails during import. The project uses PyTorch SDPA, so hiding
    this optional package only during Diffusers import is both sufficient and
    non-destructive.
    """

    import torch

    original_find_spec = importlib.util.find_spec

    def filtered_find_spec(name: str, package: str | None = None) -> Any:
        if name in {"flash_attn", "flash_attn_3"}:
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = filtered_find_spec
    try:
        diffusers_module = importlib.import_module("diffusers")
        controlnet_class = getattr(diffusers_module, "ControlNetModel")
        pipeline_class = getattr(
            diffusers_module, "StableDiffusionXLControlNetImg2ImgPipeline"
        )
    finally:
        importlib.util.find_spec = original_find_spec
    return torch, controlnet_class, pipeline_class


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def model_paths() -> dict[str, Path]:
    base_candidates = [
        os.environ.get("SDXL_BASE_PATH"),
        "/workspace/ai-concept-animator/.cache/models/sdxl-base-1.0",
        str(
            Path.home()
            / ".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0"
        ),
    ]
    control_candidates = [
        os.environ.get("SDXL_CANNY_CONTROLNET_PATH"),
        "/workspace/ai-concept-animator/.cache/models/controlnet-canny-sdxl-1.0",
        str(
            Path.home()
            / ".cache/huggingface/hub/models--diffusers--controlnet-canny-sdxl-1.0"
        ),
    ]

    def choose(candidates: list[str | None]) -> Path:
        paths = [Path(value).expanduser() for value in candidates if value]
        return next((path for path in paths if path.is_dir()), paths[0])

    return {"sdxl_base": choose(base_candidates), "controlnet_canny": choose(control_candidates)}


def probe_environment(output_root: Path = KEYFRAME_OUTPUT_ROOT) -> dict[str, Any]:
    paths = model_paths()
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "diffusers", "transformers")
    }
    package_versions = {}
    for name in (
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "opencv-python-headless",
    ):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    models = {
        name: {
            "model_id": MODEL_IDS[name],
            "path": str(path),
            "available": path.is_dir(),
        }
        for name, path in paths.items()
    }
    missing_models = [name for name, value in models.items() if not value["available"]]
    missing_packages = [name for name, available in packages.items() if not available]
    result = {
        "status": (
            "ready"
            if not missing_models and not missing_packages
            else "blocked_missing_local_model_or_runtime"
        ),
        "required_models": models,
        "required_packages": packages,
        "package_versions": package_versions,
        "missing_models": missing_models,
        "missing_packages": missing_packages,
        "substitution_used": False,
        "settings": SETTINGS,
        "message": (
            "Local SDXL Base 1.0 FP16 and SDXL Canny ControlNet FP16 are required; "
            "no fallback model will be selected."
        ),
    }
    work_root = output_root / WORK_DIR_NAME
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "model_status.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def fingerprint_models(
    output_root: Path,
    paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Record exact hashes of the FP16 files used by an inference run."""

    paths = paths or model_paths()
    models: dict[str, dict[str, str]] = {}
    model_roots: dict[str, str] = {}
    for model_name, relative_paths in MODEL_WEIGHT_FILES.items():
        root = paths[model_name]
        model_roots[MODEL_IDS[model_name]] = str(root.resolve())
        records = {}
        for relative_path in relative_paths:
            weight_path = root / relative_path
            if not weight_path.is_file():
                raise FileNotFoundError(
                    f"required FP16 weight is missing: {weight_path}"
                )
            records[relative_path] = _sha256(weight_path)
        models[MODEL_IDS[model_name]] = records
    result = {
        "algorithm": "sha256",
        "purpose": (
            "Identify the exact local FP16 weight files used for this Stage 1 run."
        ),
        "model_roots": model_roots,
        "models": models,
    }
    work_root = output_root / WORK_DIR_NAME
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "model_fingerprints.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    current = mask.copy()
    for _ in range(iterations):
        padded = np.pad(current, 1, mode="constant")
        current = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return current


def _load_masks(
    prepare_manifest: dict[str, Any],
    keyframe: str,
) -> dict[str, np.ndarray]:
    records = prepare_manifest["keyframes"][keyframe]["masks"]
    return {
        name: np.asarray(Image.open(record["path"]).convert("L")) > 127
        for name, record in records.items()
    }


def project_texture(
    clean_base: Image.Image,
    proposal: Image.Image,
    masks: dict[str, np.ndarray],
) -> tuple[Image.Image, dict[str, Any]]:
    """Keep mechanism geometry exact while borrowing in-region texture."""

    base = np.asarray(clean_base.convert("RGB"), dtype=np.float64)
    proposal_rgb = np.asarray(
        proposal.convert("RGB").resize(clean_base.size, Image.Resampling.LANCZOS),
        dtype=np.float64,
    )
    # Retain medium/high frequency texture, not the proposal's region layout.
    blurred = np.asarray(
        Image.fromarray(np.uint8(proposal_rgb))
        .filter(ImageFilter.GaussianBlur(7.0)),
        dtype=np.float64,
    )
    texture_residual = proposal_rgb - blurred
    proposal_luminance = (
        0.2126 * proposal_rgb[..., 0]
        + 0.7152 * proposal_rgb[..., 1]
        + 0.0722 * proposal_rgb[..., 2]
    )
    blurred_luminance = (
        0.2126 * blurred[..., 0]
        + 0.7152 * blurred[..., 1]
        + 0.0722 * blurred[..., 2]
    )
    output = base.copy()
    mask_records: dict[str, Any] = {}
    coverage = np.zeros(base.shape[:2], dtype=np.uint8)
    for name, mask in masks.items():
        if mask.shape != base.shape[:2]:
            raise ValueError(f"mask {name} shape {mask.shape} != image shape {base.shape[:2]}")
        coverage += mask.astype(np.uint8)
        interior = _erode(mask, 4)
        boundary = mask & ~interior
        weights = np.zeros(mask.shape, dtype=np.float64)
        weights[interior] = 0.62
        weights[boundary] = 0.14
        # Proposal color is intentionally discarded. The mechanism renderer
        # anchors category color; only detail and bounded local lighting are
        # admitted from the same semantic region.
        region_luminance = float(np.median(proposal_luminance[mask])) if mask.any() else 1.0
        shading = np.clip(
            blurred_luminance / max(region_luminance, 1.0),
            0.82,
            1.18,
        )
        textured = np.clip(
            base * shading[..., None] + 0.72 * texture_residual,
            0.0,
            255.0,
        )
        weight_rgb = weights[..., None]
        mixed = base * (1.0 - weight_rgb) + textured * weight_rgb
        output[mask] = mixed[mask]
        mask_records[name] = {
            "pixels": int(mask.sum()),
            "interior_pixels": int(interior.sum()),
            "boundary_pixels": int(boundary.sum()),
            "interior_model_weight": 0.62,
            "boundary_model_weight": 0.14,
        }
    if not np.all(coverage == 1):
        values, counts = np.unique(coverage, return_counts=True)
        raise ValueError(f"projection masks overlap or leave gaps: {dict(zip(values, counts))}")
    return Image.fromarray(np.uint8(np.clip(output, 0, 255))), {
        "geometry_source": "mechanism semantic masks",
        "proposal_contribution": "in-mask medium/high-frequency texture only",
        "proposal_color_layout_used": False,
        "exclusive_exhaustive_masks": True,
        "categories": mask_records,
    }


def project_file(
    proposal_path: Path,
    *,
    keyframe: str,
    candidate_name: str,
    output_root: Path = KEYFRAME_OUTPUT_ROOT,
) -> dict[str, Any]:
    work_root = output_root / WORK_DIR_NAME
    prepare_manifest = json.loads(
        (work_root / "prepare_manifest.json").read_text(encoding="utf-8")
    )
    keyframe_record = prepare_manifest["keyframes"][keyframe]
    base = Image.open(keyframe_record["clean_base"]).convert("RGB")
    proposal = Image.open(proposal_path).convert("RGB")
    masks = _load_masks(prepare_manifest, keyframe)
    projected, projection = project_texture(base, proposal, masks)
    projected_dir = work_root / "constrained_candidates"
    projected_dir.mkdir(parents=True, exist_ok=True)
    output_path = projected_dir / f"{candidate_name}_{keyframe}.png"
    projected.save(output_path)
    result = {
        "status": "projected",
        "candidate": candidate_name,
        "keyframe": keyframe,
        "proposal": str(proposal_path.resolve()),
        "clean_base": keyframe_record["clean_base"],
        "output": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "mask_sha256": {
            name: record["sha256"] for name, record in keyframe_record["masks"].items()
        },
        "projection": projection,
    }
    manifest_path = projected_dir / f"{candidate_name}_{keyframe}.json"
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def generate_candidates(
    output_root: Path = KEYFRAME_OUTPUT_ROOT,
    *,
    force: bool = False,
) -> dict[str, Any]:
    status = probe_environment(output_root)
    work_root = output_root / WORK_DIR_NAME
    candidates_dir = work_root / "raw_proposals"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": status["status"],
        "settings": SETTINGS,
        "generation_order": "last keyframe first; first keyframe only after style selection",
        "styles": list(STYLE_FILES),
        "candidates": [],
        "blocked_by": {
            "models": status["missing_models"],
            "packages": status["missing_packages"],
        },
    }
    if status["status"] != "ready":
        (candidates_dir / "candidate_manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    # Heavy imports remain behind the explicit readiness gate.
    torch, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline = (
        _diffusers_runtime()
    )

    prepare_manifest = json.loads(
        (work_root / "prepare_manifest.json").read_text(encoding="utf-8")
    )
    last = prepare_manifest["keyframes"]["last"]
    clean_base = Image.open(last["clean_base"]).convert("RGB")
    control_image = Image.open(last["conditioning_image"]).convert("RGB")
    paths = model_paths()
    result["model_fingerprints"] = fingerprint_models(output_root, paths)
    load_started = time.perf_counter()
    controlnet = ControlNetModel.from_pretrained(
        str(paths["controlnet_canny"]),
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        str(paths["sdxl_base"]),
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
        use_safetensors=True,
    ).to("cuda")
    result["model_load_seconds"] = round(time.perf_counter() - load_started, 3)
    result["runtime"] = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    prompt_root = Path(__file__).with_name("prompts")
    negative = (prompt_root / "negative.txt").read_text(encoding="utf-8").strip()
    for style, prompt_file in STYLE_FILES.items():
        prompt = (prompt_root / prompt_file).read_text(encoding="utf-8").strip()
        for seed in SETTINGS["seeds"]:
            candidate_name = f"{style}_s{seed}"
            proposal_path = candidates_dir / f"{candidate_name}_proposal.png"
            inference_started = time.perf_counter()
            if proposal_path.is_file() and not force:
                reused = True
            else:
                reused = False
                torch.cuda.reset_peak_memory_stats()
                generator = torch.Generator(device="cuda").manual_seed(seed)
                proposal = pipeline(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=clean_base,
                    control_image=control_image,
                    num_inference_steps=SETTINGS["steps"],
                    strength=SETTINGS["strength"],
                    guidance_scale=SETTINGS["guidance_scale"],
                    controlnet_conditioning_scale=SETTINGS["controlnet_scale"],
                    generator=generator,
                ).images[0]
                proposal.save(proposal_path)
                torch.cuda.synchronize()
            candidate_record = project_file(
                proposal_path,
                keyframe="last",
                candidate_name=candidate_name,
                output_root=output_root,
            )
            candidate_record.update(
                {
                    "style": style,
                    "seed": seed,
                    "prompt_file": str((prompt_root / prompt_file).resolve()),
                    "inference_seconds": round(
                        time.perf_counter() - inference_started, 3
                    ),
                    "proposal_reused": reused,
                    "peak_gpu_memory_bytes": (
                        None
                        if reused
                        else int(torch.cuda.max_memory_allocated())
                    ),
                }
            )
            result["candidates"].append(candidate_record)
            (candidates_dir / "candidate_manifest.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    result["status"] = "generated"
    (candidates_dir / "candidate_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def generate_selected_pair(
    output_root: Path,
    style: str,
    seed: int,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if style not in STYLE_FILES:
        raise ValueError(f"unknown style {style!r}; choose from {sorted(STYLE_FILES)}")
    status = probe_environment(output_root)
    result: dict[str, Any] = {
        "status": status["status"],
        "style": style,
        "seed": seed,
        "settings": SETTINGS,
        "pair": [],
        "blocked_by": {
            "models": status["missing_models"],
            "packages": status["missing_packages"],
        },
    }
    if status["status"] != "ready":
        return result

    torch, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline = (
        _diffusers_runtime()
    )

    prepare_manifest = json.loads(
        (output_root / WORK_DIR_NAME / "prepare_manifest.json").read_text(encoding="utf-8")
    )
    paths = model_paths()
    result["model_fingerprints"] = fingerprint_models(output_root, paths)
    controlnet = ControlNetModel.from_pretrained(
        str(paths["controlnet_canny"]),
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        str(paths["sdxl_base"]),
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
        use_safetensors=True,
    ).to("cuda")
    prompt_root = Path(__file__).with_name("prompts")
    prompt = (prompt_root / STYLE_FILES[style]).read_text(encoding="utf-8").strip()
    negative = (prompt_root / "negative.txt").read_text(encoding="utf-8").strip()
    pair_dir = output_root / WORK_DIR_NAME / "selected_pair"
    pair_dir.mkdir(parents=True, exist_ok=True)
    candidate_name = f"{style}_s{seed}"
    for keyframe in ("last", "first"):
        record = prepare_manifest["keyframes"][keyframe]
        clean_base = Image.open(record["clean_base"]).convert("RGB")
        control_image = Image.open(record["conditioning_image"]).convert("RGB")
        proposal_path = pair_dir / f"{candidate_name}_{keyframe}_proposal.png"
        if force or not proposal_path.is_file():
            generator = torch.Generator(device="cuda").manual_seed(seed)
            proposal = pipeline(
                prompt=prompt,
                negative_prompt=negative,
                image=clean_base,
                control_image=control_image,
                num_inference_steps=SETTINGS["steps"],
                strength=SETTINGS["strength"],
                guidance_scale=SETTINGS["guidance_scale"],
                controlnet_conditioning_scale=SETTINGS["controlnet_scale"],
                generator=generator,
            ).images[0]
            proposal.save(proposal_path)
        result["pair"].append(
            project_file(
                proposal_path,
                keyframe=keyframe,
                candidate_name=candidate_name,
                output_root=output_root,
            )
        )
    result["status"] = "generated_selected_pair"
    (pair_dir / f"{candidate_name}_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=KEYFRAME_OUTPUT_ROOT)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--generate-candidates", action="store_true")
    parser.add_argument("--generate-pair", action="store_true")
    parser.add_argument("--selected-style", choices=tuple(STYLE_FILES))
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--force", action="store_true", help="regenerate existing proposal files")
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--keyframe", choices=("first", "last"), default="last")
    parser.add_argument("--candidate-name", default="external-proposal")
    args = parser.parse_args()
    if args.proposal:
        result = project_file(
            args.proposal,
            keyframe=args.keyframe,
            candidate_name=args.candidate_name,
            output_root=args.output,
        )
    elif args.generate_pair:
        if not args.selected_style:
            parser.error("--generate-pair requires --selected-style")
        result = generate_selected_pair(
            args.output,
            style=args.selected_style,
            seed=args.seed,
            force=args.force,
        )
    elif args.generate_candidates:
        result = generate_candidates(args.output, force=args.force)
    else:
        result = probe_environment(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if (args.generate_candidates or args.generate_pair) and result["status"].startswith("blocked"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
