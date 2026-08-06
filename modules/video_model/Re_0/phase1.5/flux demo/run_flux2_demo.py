#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path


SERVER = "http://127.0.0.1:8188"
COMFY_INPUT = Path("/workspace/persistent/ComfyUI/input")
MODEL = "flux2_dev_fp8mixed.safetensors"
TEXT_ENCODER = "mistral_3_small_flux2_fp4_mixed.safetensors"
VAE = "flux2-vae.safetensors"
DEFAULT_WIDTH = 1536
DEFAULT_HEIGHT = 864
DEFAULT_STEPS = 20
GUIDANCE = 4.0
SAMPLER = "euler"
VAE_TILE_SIZE = 512
VAE_TILE_OVERLAP = 64

SEEDS = {
    "delta-formation": 420260801,
    "wave-interference": 420260802,
    "mitosis": 420260803,
    "endocytosis-exocytosis": 420260804,
}


def request_json(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{SERVER}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def workflow(
    image_name: str,
    prompt: str,
    seed: int,
    prefix: str,
    width: int,
    height: int,
    steps: int,
) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        # Keeping the 12 GB text encoder on CPU prevents it from competing with
        # the 35 GB diffusion model for the 48 GB GPU.
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "flux2", "device": "cpu"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": GUIDANCE}},
        "7": {
            "class_type": "VAEEncodeTiled",
            "inputs": {
                "pixels": ["4", 0],
                "vae": ["3", 0],
                "tile_size": VAE_TILE_SIZE,
                "overlap": VAE_TILE_OVERLAP,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
        },
        "8": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["7", 0]}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["8", 0]}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}},
        "13": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["10", 0],
                "guider": ["9", 0],
                "sampler": ["12", 0],
                "sigmas": ["11", 0],
                "latent_image": ["13", 0],
            },
        },
        "15": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["14", 0],
                "vae": ["3", 0],
                "tile_size": VAE_TILE_SIZE,
                "overlap": VAE_TILE_OVERLAP,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
        },
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    }


def download_image(image: dict, destination: Path) -> None:
    query = urllib.parse.urlencode(
        {
            "filename": image["filename"],
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        }
    )
    with urllib.request.urlopen(f"{SERVER}/view?{query}", timeout=120) as response:
        destination.write_bytes(response.read())


def append_log(log_path: Path, message: str) -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    line = f"{timestamp} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_case(
    case_dir: Path,
    width: int,
    height: int,
    steps: int,
    log_path: Path,
    skip_existing: bool,
) -> None:
    case_id = case_dir.name
    if case_id not in SEEDS:
        raise ValueError(f"No seed configured for {case_id}")

    source = case_dir / "input-first-frame.png"
    model_input = case_dir / "model-input-1536x864.png"
    prompt_path = case_dir / "prompt.txt"
    if not source.exists() or not model_input.exists() or not prompt_path.exists():
        raise FileNotFoundError(f"Missing source, model input, or prompt in {case_dir}")

    output = case_dir / "flux-output.png"
    metadata_path = case_dir / "generation.json"
    if skip_existing and output.exists() and metadata_path.exists():
        append_log(log_path, f"{case_id}: already complete; skipped")
        return

    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    comfy_name = f"flux-demo-{case_id}.png"
    shutil.copy2(model_input, COMFY_INPUT / comfy_name)

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    graph = workflow(
        comfy_name,
        prompt,
        SEEDS[case_id],
        f"flux_demo/{case_id}/realistic",
        width,
        height,
        steps,
    )
    (case_dir / "workflow-api.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    queued = request_json("/prompt", {"prompt": graph, "client_id": "stage1.5-flux-demo"})
    prompt_id = queued["prompt_id"]
    append_log(log_path, f"{case_id}: queued {prompt_id} ({width}x{height}, {steps} steps)")

    started = time.monotonic()
    while True:
        history = request_json(f"/history/{prompt_id}")
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, ensure_ascii=False, indent=2))
            outputs = record.get("outputs", {})
            images = outputs.get("16", {}).get("images", [])
            if images:
                download_image(images[0], output)
                elapsed = round(time.monotonic() - started, 2)
                metadata = {
                    "engine": "local ComfyUI",
                    "server": SERVER,
                    "blueprint_basis": "Image Edit (Flux.2 Dev)",
                    "model": MODEL,
                    "text_encoder": TEXT_ENCODER,
                    "vae": VAE,
                    "steps": steps,
                    "guidance": GUIDANCE,
                    "sampler": SAMPLER,
                    "seed": SEEDS[case_id],
                    "width": width,
                    "height": height,
                    "prompt_id": prompt_id,
                    "elapsed_seconds": elapsed,
                    "memory_profile": {
                        "text_encoder_device": "cpu",
                        "vae_device": "cpu (ComfyUI --cpu-vae)",
                        "vae_tiled": True,
                        "vae_tile_size": VAE_TILE_SIZE,
                        "vae_tile_overlap": VAE_TILE_OVERLAP,
                        "unet_loading": os.environ.get("FLUX_MEMORY_PROFILE", "external ComfyUI server"),
                    },
                    "source_frame": "input-first-frame.png",
                    "model_input": "model-input-1536x864.png",
                    "output": "flux-output.png",
                }
                metadata_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                append_log(log_path, f"{case_id}: saved {output} in {elapsed}s")
                return
        if time.monotonic() - started > 3600:
            raise TimeoutError(f"Timed out waiting for {case_id}")
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--log-file", type=Path, default=Path("logs/generation.log"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("case_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32:
        parser.error("width and height must be positive multiples of 32")
    if args.steps <= 0:
        parser.error("steps must be positive")

    log_path = args.log_file.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    append_log(log_path, f"run started: {args.width}x{args.height}, {args.steps} steps")
    try:
        for case_dir in args.case_dirs:
            run_case(
                case_dir.resolve(),
                args.width,
                args.height,
                args.steps,
                log_path,
                args.skip_existing,
            )
    except Exception:
        append_log(log_path, "run failed:\n" + traceback.format_exc())
        raise
    append_log(log_path, "run completed")


if __name__ == "__main__":
    main()
