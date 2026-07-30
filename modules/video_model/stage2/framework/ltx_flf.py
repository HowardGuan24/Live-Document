"""Reusable local LTX-2.3 first/last-frame video experiment runner."""

from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .contracts import artifact_record, load_json, sha256_path, write_json


COMFY_ROOT = Path("/persistent/ComfyUI")
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_OUTPUT = COMFY_ROOT / "output"
MODEL_FILES = {
    "checkpoint": COMFY_ROOT
    / "models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors",
    "text_encoder": COMFY_ROOT
    / "models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "distilled_lora": COMFY_ROOT
    / (
        "models/loras/"
        "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
    ),
}
GLOBAL_MODEL_FINGERPRINTS = (
    Path(__file__).resolve().parents[1]
    / "output/phase-5/_work/model_fingerprints.json"
)


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def build_workflow(
    spec: dict[str, Any],
    first_name: str,
    last_name: str,
) -> dict[str, Any]:
    """Build the same official FLF2V node structure for any case spec."""

    settings = spec["settings"]
    model = spec["model"]
    width = int(settings["width"])
    height = int(settings["height"])
    fps = float(settings["fps"])
    frame_count = int(settings["frame_count"])
    if frame_count % 8 != 1:
        raise ValueError("LTX frame_count must equal 8n+1")
    return {
        "1": _node(
            "CheckpointLoaderSimple",
            ckpt_name=MODEL_FILES["checkpoint"].name,
        ),
        "2": _node(
            "LTXAVTextEncoderLoader",
            text_encoder=MODEL_FILES["text_encoder"].name,
            ckpt_name=MODEL_FILES["checkpoint"].name,
            device="default",
        ),
        "3": _node(
            "CLIPTextEncode",
            text=spec["prompt"]["positive"],
            clip=["2", 0],
        ),
        "4": _node(
            "CLIPTextEncode",
            text=spec["prompt"]["negative"],
            clip=["2", 0],
        ),
        "5": _node(
            "LTXVConditioning",
            positive=["3", 0],
            negative=["4", 0],
            frame_rate=fps,
        ),
        "6": _node(
            "LoraLoaderModelOnly",
            model=["1", 0],
            lora_name=MODEL_FILES["distilled_lora"].name,
            strength_model=float(model["lora_strength"]),
        ),
        "7": _node(
            "LTXVAudioVAELoader",
            ckpt_name=MODEL_FILES["checkpoint"].name,
        ),
        "8": _node("LoadImage", image=first_name),
        "9": _node("LoadImage", image=last_name),
        "10": _node(
            "ImageScale",
            image=["8", 0],
            upscale_method="bicubic",
            width=width,
            height=height,
            crop="center",
        ),
        "11": _node(
            "ImageScale",
            image=["9", 0],
            upscale_method="bicubic",
            width=width,
            height=height,
            crop="center",
        ),
        "12": _node(
            "LTXVPreprocess",
            image=["10", 0],
            img_compression=int(settings["image_compression"]),
        ),
        "13": _node(
            "LTXVPreprocess",
            image=["11", 0],
            img_compression=int(settings["image_compression"]),
        ),
        "14": _node(
            "EmptyLTXVLatentVideo",
            width=width,
            height=height,
            length=frame_count,
            batch_size=1,
        ),
        "15": _node(
            "LTXVEmptyLatentAudio",
            frames_number=frame_count,
            frame_rate=fps,
            batch_size=1,
            audio_vae=["7", 0],
        ),
        "16": _node(
            "LTXVAddGuide",
            positive=["5", 0],
            negative=["5", 1],
            vae=["1", 2],
            latent=["14", 0],
            image=["12", 0],
            frame_idx=0,
            strength=float(settings["guide_strength"]),
        ),
        "17": _node(
            "LTXVAddGuide",
            positive=["16", 0],
            negative=["16", 1],
            vae=["1", 2],
            latent=["16", 2],
            image=["13", 0],
            frame_idx=-1,
            strength=float(settings["guide_strength"]),
        ),
        "18": _node(
            "LTXVConcatAVLatent",
            video_latent=["17", 2],
            audio_latent=["15", 0],
        ),
        "19": _node("RandomNoise", noise_seed=int(settings["noise_seed"])),
        "20": _node(
            "CFGGuider",
            model=["6", 0],
            positive=["17", 0],
            negative=["17", 1],
            cfg=float(settings["cfg"]),
        ),
        "21": _node("SamplerEulerAncestral", eta=0.0, s_noise=1.0),
        "22": _node(
            "ManualSigmas",
            sigmas=", ".join(str(value) for value in settings["sigmas"]),
        ),
        "23": _node(
            "SamplerCustomAdvanced",
            noise=["19", 0],
            guider=["20", 0],
            sampler=["21", 0],
            sigmas=["22", 0],
            latent_image=["18", 0],
        ),
        "24": _node("LTXVSeparateAVLatent", av_latent=["23", 1]),
        "25": _node(
            "LTXVCropGuides",
            positive=["17", 0],
            negative=["17", 1],
            latent=["24", 0],
        ),
        "26": _node(
            "VAEDecodeTiled",
            samples=["25", 2],
            vae=["1", 2],
            tile_size=768,
            overlap=64,
            temporal_size=64,
            temporal_overlap=4,
        ),
        "27": _node(
            "LTXVAudioVAEDecode",
            samples=["24", 1],
            audio_vae=["7", 0],
        ),
        "28": _node(
            "CreateVideo",
            images=["26", 0],
            audio=["27", 0],
            fps=fps,
            bit_depth=8,
        ),
        "29": _node(
            "SaveVideo",
            video=["28", 0],
            filename_prefix=spec["output_prefix"],
            format="mp4",
            codec="h264",
        ),
    }


def _model_fingerprints(path: Path) -> dict[str, Any]:
    current = {
        key: {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "mtime_ns": model_path.stat().st_mtime_ns,
        }
        for key, model_path in MODEL_FILES.items()
    }
    for cache_path in (GLOBAL_MODEL_FINGERPRINTS, path):
        if not cache_path.is_file():
            continue
        cached = load_json(cache_path)
        cached_identity = {
            key: {
                field: value[field]
                for field in ("path", "size_bytes", "mtime_ns")
            }
            for key, value in cached["files"].items()
        }
        if cached_identity == current:
            GLOBAL_MODEL_FINGERPRINTS.parent.mkdir(
                parents=True, exist_ok=True
            )
            write_json(GLOBAL_MODEL_FINGERPRINTS, cached)
            if path != GLOBAL_MODEL_FINGERPRINTS:
                write_json(path, cached)
            return cached
    fingerprints = {
        "schema_version": "1.0",
        "algorithm": "sha256",
        "files": {
            key: {
                **record,
                "sha256": sha256_path(Path(record["path"])),
            }
            for key, record in current.items()
        },
    }
    GLOBAL_MODEL_FINGERPRINTS.parent.mkdir(parents=True, exist_ok=True)
    write_json(GLOBAL_MODEL_FINGERPRINTS, fingerprints)
    if path != GLOBAL_MODEL_FINGERPRINTS:
        write_json(path, fingerprints)
    return fingerprints


def prepare_video_experiment(
    spec: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    first_source = Path(spec["source"]["first_frame"])
    last_source = Path(spec["source"]["last_frame"])
    for path in (first_source, last_source, *MODEL_FILES.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = output_root / "inputs"
    work = output_root / "_work"
    inputs.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)

    safe_id = spec["experiment_id"].lower().replace("-", "_")
    first_comfy = COMFY_INPUT / f"{safe_id}_first.png"
    last_comfy = COMFY_INPUT / f"{safe_id}_last.png"
    first_input = inputs / "first.png"
    last_input = inputs / "last.png"
    for source, comfy_target, local_target in (
        (first_source, first_comfy, first_input),
        (last_source, last_comfy, last_input),
    ):
        shutil.copy2(source, comfy_target)
        shutil.copy2(source, local_target)

    workflow = build_workflow(spec, first_comfy.name, last_comfy.name)
    workflow_path = work / "workflow_api.json"
    write_json(workflow_path, workflow)
    prompt_path = inputs / "prompt.txt"
    prompt_path.write_text(
        "Positive prompt:\n\n"
        + spec["prompt"]["positive"]
        + "\n\nNegative prompt:\n\n"
        + spec["prompt"]["negative"]
        + "\n",
        encoding="utf-8",
    )
    fingerprints_path = work / "model_fingerprints.json"
    _model_fingerprints(fingerprints_path)
    prepared = {
        "schema_version": "1.0",
        "experiment_id": spec["experiment_id"],
        "case_id": spec["case_id"],
        "motion_class": spec["motion_class"],
        "classification": "prepared LTX-2.3 FLF2V experiment; no video call yet",
        "experiment_spec": {
            "path": str(Path(spec["_spec_path"])),
            "sha256": sha256_path(Path(spec["_spec_path"])),
            "size_bytes": Path(spec["_spec_path"]).stat().st_size,
        },
        "inputs": {
            "first": artifact_record(first_input, output_root),
            "last": artifact_record(last_input, output_root),
        },
        "source": {
            "first": {
                "path": str(first_source),
                "sha256": sha256_path(first_source),
            },
            "last": {
                "path": str(last_source),
                "sha256": sha256_path(last_source),
            },
        },
        "workflow": artifact_record(workflow_path, output_root),
        "prompt": artifact_record(prompt_path, output_root),
        "model_fingerprints": artifact_record(
            fingerprints_path, output_root
        ),
        "prompt_preflight": {
            "positive_characters": len(spec["prompt"]["positive"]),
            "negative_characters": len(spec["prompt"]["negative"]),
            "note": (
                "ComfyUI's LTXAVTextEncoderLoader does not expose tokenizer "
                "counts through the API. Prompts are deliberately short; this "
                "smoke does not yet claim the token-integrity release gate."
            ),
        },
        "model_runs": {"image": 0, "video": 0},
    }
    write_json(work / "prepare.json", prepared)
    return prepared


def _request_json(
    url: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI HTTP {error.code}: {body}") from error


def _find_video_record(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        filename = value.get("filename")
        if isinstance(filename, str) and filename.lower().endswith(".mp4"):
            return value
        for child in value.values():
            found = _find_video_record(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_video_record(child)
            if found:
                return found
    return None


def _wait_for_result(
    server: str,
    prompt_id: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    next_update = 0.0
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Generation {prompt_id} exceeded {timeout_seconds} seconds"
            )
        history = _request_json(f"{server}/history/{prompt_id}")
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(
                    "ComfyUI generation failed: "
                    + json.dumps(status, ensure_ascii=False)
                )
            if _find_video_record(record.get("outputs", {})):
                return record, elapsed
        if elapsed >= next_update:
            print(
                f"LTX-2.3 is generating: {elapsed / 60:.1f} minutes elapsed",
                flush=True,
            )
            next_update = elapsed + 15
        time.sleep(5)


def _decode_video(path: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = [
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(video=0)
        ]
        fps = float(stream.average_rate)
        info = {
            "codec": stream.codec_context.name,
            "width": stream.width,
            "height": stream.height,
            "fps": fps,
            "frame_count": len(frames),
            "duration_seconds": len(frames) / fps,
            "has_audio": bool(container.streams.audio),
        }
    return info, frames


def _fitted_reference(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(
        ImageOps.fit(
            image,
            size,
            method=Image.Resampling.BICUBIC,
            centering=(0.5, 0.5),
        )
    )


def _comparison_metrics(
    generated: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    delta = generated.astype(np.float32) - reference.astype(np.float32)
    mse = float(np.mean(delta * delta))
    return {
        "mean_absolute_pixel_error_0_255": round(
            float(np.mean(np.abs(delta))), 4
        ),
        "psnr_db": round(
            float(20 * np.log10(255.0 / np.sqrt(max(mse, 1e-12)))),
            4,
        ),
    }


def _anchor_audit(
    frames: list[np.ndarray],
    audit: dict[str, Any],
) -> dict[str, Any]:
    height, width = frames[0].shape[:2]
    color = audit["color_range"]
    records = []
    maximum_drift = 0.0
    all_visible = True
    for frame_index, frame in enumerate(frames):
        frame_records = []
        for anchor in audit["anchors_normalized_xy"]:
            expected_x = float(anchor[0]) * width
            expected_y = float(anchor[1]) * height
            radius = int(audit["search_radius_px"])
            left = max(0, int(round(expected_x)) - radius)
            right = min(width, int(round(expected_x)) + radius + 1)
            top = max(0, int(round(expected_y)) - radius)
            bottom = min(height, int(round(expected_y)) + radius + 1)
            crop = frame[top:bottom, left:right]
            mask = (
                (crop[:, :, 0] >= int(color["r_min"]))
                & (crop[:, :, 1] >= int(color["g_min"]))
                & (crop[:, :, 1] <= int(color["g_max"]))
                & (crop[:, :, 2] <= int(color["b_max"]))
            )
            ys, xs = np.nonzero(mask)
            visible = len(xs) >= int(audit["minimum_pixels"])
            all_visible = all_visible and visible
            if visible:
                center_x = float(xs.mean() + left)
                center_y = float(ys.mean() + top)
                drift = float(
                    np.hypot(center_x - expected_x, center_y - expected_y)
                )
                maximum_drift = max(maximum_drift, drift)
            else:
                center_x = center_y = drift = None
            frame_records.append(
                {
                    "expected_xy": [
                        round(expected_x, 3),
                        round(expected_y, 3),
                    ],
                    "detected_xy": (
                        [round(center_x, 3), round(center_y, 3)]
                        if visible
                        else None
                    ),
                    "pixel_count": int(len(xs)),
                    "drift_px": round(drift, 4) if visible else None,
                }
            )
        records.append(
            {"frame": frame_index, "anchors": frame_records}
        )
    threshold = float(audit["maximum_drift_px"])
    return {
        "passed": all_visible and maximum_drift <= threshold,
        "all_anchors_visible": all_visible,
        "maximum_drift_px": round(maximum_drift, 4),
        "threshold_px": threshold,
        "frames": records,
    }


def _radial_propagation_audit(
    frames: list[np.ndarray],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Measure outward movement without naming a particular case."""

    height, width = frames[0].shape[:2]
    y, x = np.mgrid[:height, :width]
    centers = [
        (float(anchor[0]) * width, float(anchor[1]) * height)
        for anchor in audit["centers_normalized_xy"]
    ]
    sample_indices = np.linspace(
        0, len(frames) - 1, int(audit["sample_count"]), dtype=int
    ).tolist()
    radius_min = int(audit["radius_min_px"])
    radius_max = int(audit["radius_max_px"])
    threshold = float(audit["minimum_annulus_deviation_0_255"])
    records = []
    extents_by_center: list[list[int]] = [
        [] for _ in centers
    ]
    for frame_index in sample_indices:
        gray = frames[frame_index].astype(np.float32).mean(axis=2)
        background_mask = (
            (gray >= float(audit["background_gray_min"]))
            & (gray <= float(audit["background_gray_max"]))
        )
        background = float(np.median(gray[background_mask]))
        frame_extents = []
        for center_index, (center_x, center_y) in enumerate(centers):
            radius = np.hypot(x - center_x, y - center_y)
            qualifying = []
            for value in range(radius_min, radius_max + 1):
                annulus = (
                    (radius >= value - 0.5)
                    & (radius < value + 0.5)
                )
                deviation = float(
                    np.mean(np.abs(gray[annulus] - background))
                )
                if deviation >= threshold:
                    qualifying.append(value)
            extent = max(qualifying) if qualifying else radius_min
            extents_by_center[center_index].append(extent)
            frame_extents.append(extent)
        records.append(
            {"frame": frame_index, "radial_extent_px": frame_extents}
        )
    tolerance = float(audit["allowed_backtrack_px"])
    monotonic = [
        all(
            right + tolerance >= left
            for left, right in zip(values, values[1:])
        )
        for values in extents_by_center
    ]
    progress = [
        values[-1] - values[0] for values in extents_by_center
    ]
    minimum_progress = float(audit["minimum_progress_px"])
    return {
        "passed": all(monotonic)
        and all(value >= minimum_progress for value in progress),
        "sample_indices": sample_indices,
        "records": records,
        "extent_by_center_px": extents_by_center,
        "progress_by_center_px": progress,
        "monotonic_with_tolerance": monotonic,
        "allowed_backtrack_px": tolerance,
        "minimum_progress_px": minimum_progress,
        "measurement_zh": (
            "在规格给出的固定中心周围逐像素扫描圆环，取相对背景亮度"
            "偏差超过阈值的最外半径；只用于验证总体向外传播。"
        ),
    }


def _color_stats(
    image: np.ndarray,
    target_rgb: list[int],
    tolerance: float,
) -> dict[str, Any] | None:
    delta = image.astype(np.float32) - np.asarray(
        target_rgb, dtype=np.float32
    )
    mask = np.linalg.norm(delta, axis=2) <= tolerance
    y, x = np.nonzero(mask)
    if len(x) < 3:
        return None
    coordinates = np.stack([x, y], axis=1).astype(np.float32)
    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return {
        "area_px": int(len(x)),
        "centroid_xy": [float(x.mean()), float(y.mean())],
        "covariance_eigenvalues": [
            float(value) for value in eigenvalues
        ],
    }


def _color_identity_reference_audit(
    frames: list[np.ndarray],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Compare colored object identity against a program reference sequence."""

    height, width = frames[0].shape[:2]
    start = int(audit["reference_start_index"])
    end = int(audit["reference_end_index"])
    reference_indices = np.rint(
        np.linspace(start, end, len(frames))
    ).astype(int).tolist()
    root = Path(audit["reference_frame_directory"])
    pattern = audit["reference_filename_pattern"]
    tolerance = float(audit["color_distance_tolerance"])
    area_min = float(audit["minimum_area_ratio"])
    area_max = float(audit["maximum_area_ratio"])
    shape_min = float(audit["minimum_shape_eigenvalue_ratio"])
    shape_max = float(audit["maximum_shape_eigenvalue_ratio"])
    centroid_limit = float(audit["maximum_centroid_error_px"])
    records = []
    all_present = True
    area_ratios: list[float] = []
    shape_ratios: list[float] = []
    centroid_errors: list[float] = []
    for frame_index, reference_index in enumerate(reference_indices):
        reference = _fitted_reference(
            root / pattern.format(index=reference_index),
            (width, height),
        )
        identities = []
        for identity in audit["identities"]:
            generated_stats = _color_stats(
                frames[frame_index],
                identity["rgb"],
                tolerance,
            )
            reference_stats = _color_stats(
                reference,
                identity["rgb"],
                tolerance,
            )
            present = (
                generated_stats is not None
                and reference_stats is not None
            )
            all_present = all_present and present
            if present:
                area_ratio = (
                    generated_stats["area_px"]
                    / reference_stats["area_px"]
                )
                centroid_error = float(
                    np.linalg.norm(
                        np.asarray(generated_stats["centroid_xy"])
                        - np.asarray(reference_stats["centroid_xy"])
                    )
                )
                eigen_generated = np.asarray(
                    generated_stats["covariance_eigenvalues"]
                )
                eigen_reference = np.asarray(
                    reference_stats["covariance_eigenvalues"]
                )
                identity_shape_ratios = (
                    eigen_generated / np.maximum(eigen_reference, 1e-6)
                ).tolist()
                area_ratios.append(float(area_ratio))
                centroid_errors.append(centroid_error)
                shape_ratios.extend(
                    float(value) for value in identity_shape_ratios
                )
            else:
                area_ratio = centroid_error = None
                identity_shape_ratios = None
            identities.append(
                {
                    "identity_id": identity["identity_id"],
                    "present": present,
                    "area_ratio_vs_program": (
                        round(float(area_ratio), 4)
                        if area_ratio is not None
                        else None
                    ),
                    "centroid_error_px": (
                        round(float(centroid_error), 4)
                        if centroid_error is not None
                        else None
                    ),
                    "shape_eigenvalue_ratios_vs_program": (
                        [
                            round(float(value), 4)
                            for value in identity_shape_ratios
                        ]
                        if identity_shape_ratios is not None
                        else None
                    ),
                }
            )
        records.append(
            {
                "frame": frame_index,
                "program_reference_frame": reference_index,
                "identities": identities,
            }
        )
    passed = (
        all_present
        and min(area_ratios, default=0) >= area_min
        and max(area_ratios, default=float("inf")) <= area_max
        and min(shape_ratios, default=0) >= shape_min
        and max(shape_ratios, default=float("inf")) <= shape_max
        and max(centroid_errors, default=float("inf")) <= centroid_limit
    )
    return {
        "passed": passed,
        "all_identities_present": all_present,
        "minimum_area_ratio": round(min(area_ratios, default=0), 4),
        "maximum_area_ratio": round(max(area_ratios, default=0), 4),
        "minimum_shape_eigenvalue_ratio": round(
            min(shape_ratios, default=0), 4
        ),
        "maximum_shape_eigenvalue_ratio": round(
            max(shape_ratios, default=0), 4
        ),
        "maximum_centroid_error_px": round(
            max(centroid_errors, default=0), 4
        ),
        "thresholds": {
            "area_ratio": [area_min, area_max],
            "shape_eigenvalue_ratio": [shape_min, shape_max],
            "maximum_centroid_error_px": centroid_limit,
        },
        "records": records,
        "measurement_zh": (
            "按每种程序颜色追踪稳定对象，将生成帧与同一时刻的程序参考帧"
            "比较面积、质心和协方差特征值；用于区分刚体搬运与形变、消失或复制。"
        ),
    }


def _color_components(
    image: np.ndarray,
    target_rgb: list[int],
    tolerance: float,
    minimum_pixels: int,
) -> list[dict[str, Any]]:
    """Find small colored teaching objects without an OpenCV dependency."""

    delta = image.astype(np.float32) - np.asarray(
        target_rgb, dtype=np.float32
    )
    mask = np.linalg.norm(delta, axis=2) <= tolerance
    height, width = mask.shape
    remaining = {
        int(y) * width + int(x)
        for y, x in np.argwhere(mask)
    }
    components: list[dict[str, Any]] = []
    neighbors = (
        -width - 1,
        -width,
        -width + 1,
        -1,
        1,
        width - 1,
        width,
        width + 1,
    )
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        members = [seed]
        while stack:
            current = stack.pop()
            current_y, current_x = divmod(current, width)
            for offset in neighbors:
                candidate = current + offset
                if candidate not in remaining:
                    continue
                candidate_y, candidate_x = divmod(candidate, width)
                if (
                    abs(candidate_x - current_x) > 1
                    or abs(candidate_y - current_y) > 1
                    or not (0 <= candidate_y < height)
                ):
                    continue
                remaining.remove(candidate)
                stack.append(candidate)
                members.append(candidate)
        if len(members) < minimum_pixels:
            continue
        ys = np.asarray(
            [value // width for value in members], dtype=np.float32
        )
        xs = np.asarray(
            [value % width for value in members], dtype=np.float32
        )
        components.append(
            {
                "pixel_count": len(members),
                "centroid_xy": [
                    round(float(xs.mean()), 3),
                    round(float(ys.mean()), 3),
                ],
                "bbox_xyxy": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                ],
            }
        )
    return sorted(
        components,
        key=lambda item: (
            item["centroid_xy"][0],
            item["centroid_xy"][1],
        ),
    )


def _color_component_division_audit(
    frames: list[np.ndarray],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Check one colored parent population becoming two balanced groups."""

    target = audit["target_rgb"]
    tolerance = float(audit["color_distance_tolerance"])
    minimum_pixels = int(audit["minimum_component_pixels"])
    records = []
    for frame_index, frame in enumerate(frames):
        components = _color_components(
            frame, target, tolerance, minimum_pixels
        )
        records.append(
            {
                "frame": frame_index,
                "component_count": len(components),
                "components": components,
            }
        )
    counts = [record["component_count"] for record in records]
    initial_range = [
        int(value) for value in audit["initial_component_count_range"]
    ]
    final_range = [
        int(value) for value in audit["final_component_count_range"]
    ]
    split_x = float(audit.get("split_x_normalized", 0.5))
    final_components = records[-1]["components"]
    width = frames[-1].shape[1]
    left_count = sum(
        item["centroid_xy"][0] < split_x * width
        for item in final_components
    )
    right_count = len(final_components) - left_count
    minimum_per_side = int(audit["minimum_final_components_per_side"])
    minimum_growth = int(audit["minimum_component_count_growth"])
    passed = (
        initial_range[0] <= counts[0] <= initial_range[1]
        and final_range[0] <= counts[-1] <= final_range[1]
        and counts[-1] - counts[0] >= minimum_growth
        and left_count >= minimum_per_side
        and right_count >= minimum_per_side
    )
    return {
        "passed": passed,
        "component_counts": counts,
        "initial_component_count": counts[0],
        "final_component_count": counts[-1],
        "maximum_component_count": max(counts),
        "final_left_component_count": left_count,
        "final_right_component_count": right_count,
        "thresholds": {
            "initial_component_count_range": initial_range,
            "final_component_count_range": final_range,
            "minimum_component_count_growth": minimum_growth,
            "minimum_final_components_per_side": minimum_per_side,
        },
        "records": records,
        "measurement_zh": (
            "按规格颜色提取连通组件，验证单组父对象在尾帧形成数量增加且"
            "左右均衡的两个子组；它检测复制、吞并和分配失衡，不宣称仅凭"
            "像素颜色恢复了生物学对象 ID。"
        ),
    }


def _color_region_topology_audit(
    frames: list[np.ndarray],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Check a colored region changes connectivity without breaking its trunk."""

    tolerance = float(audit["color_distance_tolerance"])
    minimum_pixels = int(audit["minimum_component_pixels"])
    edge_margin = int(audit.get("edge_margin_px", 3))
    records = []
    for frame_index, frame in enumerate(frames):
        components = _color_components(
            frame,
            audit["target_rgb"],
            tolerance,
            minimum_pixels,
        )
        width = frame.shape[1]
        spanning = [
            item
            for item in components
            if item["bbox_xyxy"][0] <= edge_margin
            and item["bbox_xyxy"][2] >= width - 1 - edge_margin
        ]
        isolated = [
            item for item in components if item not in spanning
        ]
        records.append(
            {
                "frame": frame_index,
                "component_count": len(components),
                "spanning_component_count": len(spanning),
                "isolated_component_count": len(isolated),
                "largest_isolated_area_px": max(
                    (
                        item["pixel_count"]
                        for item in isolated
                    ),
                    default=0,
                ),
                "components": components,
            }
        )
    counts = [record["component_count"] for record in records]
    spanning_counts = [
        record["spanning_component_count"] for record in records
    ]
    initial_range = [
        int(value) for value in audit["initial_component_count_range"]
    ]
    final_range = [
        int(value) for value in audit["final_component_count_range"]
    ]
    minimum_isolated = int(
        audit["minimum_final_isolated_components"]
    )
    minimum_isolated_area = int(
        audit["minimum_final_isolated_area_px"]
    )
    maximum_components = int(audit["maximum_component_count"])
    final = records[-1]
    passed = (
        initial_range[0] <= counts[0] <= initial_range[1]
        and final_range[0] <= counts[-1] <= final_range[1]
        and all(value == 1 for value in spanning_counts)
        and final["isolated_component_count"] >= minimum_isolated
        and final["largest_isolated_area_px"] >= minimum_isolated_area
        and max(counts) <= maximum_components
    )
    return {
        "passed": passed,
        "component_counts": counts,
        "spanning_component_counts": spanning_counts,
        "initial_component_count": counts[0],
        "final_component_count": counts[-1],
        "final_isolated_component_count": final[
            "isolated_component_count"
        ],
        "final_largest_isolated_area_px": final[
            "largest_isolated_area_px"
        ],
        "thresholds": {
            "initial_component_count_range": initial_range,
            "final_component_count_range": final_range,
            "minimum_final_isolated_components": minimum_isolated,
            "minimum_final_isolated_area_px": minimum_isolated_area,
            "maximum_component_count": maximum_components,
        },
        "records": records,
        "measurement_zh": (
            "按规格水色提取区域连通组件；每一帧必须恰有一个同时触及左右"
            "边界的主通道，尾帧还必须出现面积足够的独立水体。"
        ),
    }


def _reference_color_mask_stability(
    frames: list[np.ndarray],
    reference_path: Path,
    audit: dict[str, Any],
) -> dict[str, Any]:
    height, width = frames[0].shape[:2]
    reference = _fitted_reference(reference_path, (width, height))
    target = np.asarray(audit["target_rgb"], dtype=np.float32)
    reference_distance = np.linalg.norm(
        reference.astype(np.float32) - target, axis=2
    )
    mask = reference_distance <= float(audit["reference_tolerance"])
    if not mask.any():
        raise ValueError("reference color stability mask is empty")
    records = []
    coverage_values = []
    difference_values = []
    for index, frame in enumerate(frames):
        current_distance = np.linalg.norm(
            frame.astype(np.float32) - target, axis=2
        )
        coverage = float(
            (current_distance[mask] <= float(audit["current_tolerance"])).mean()
        )
        difference = float(
            np.mean(
                np.abs(
                    frame.astype(np.float32)[mask]
                    - reference.astype(np.float32)[mask]
                )
            )
        )
        coverage_values.append(coverage)
        difference_values.append(difference)
        records.append(
            {
                "frame": index,
                "coverage_fraction": round(coverage, 6),
                "mean_abs_color_difference_0_255": round(
                    difference, 4
                ),
            }
        )
    minimum_coverage = float(audit["minimum_coverage_fraction"])
    maximum_difference = float(
        audit["maximum_mean_abs_color_difference_0_255"]
    )
    return {
        "passed": min(coverage_values) >= minimum_coverage
        and max(difference_values) <= maximum_difference,
        "reference_pixel_count": int(mask.sum()),
        "minimum_coverage_fraction": round(min(coverage_values), 6),
        "maximum_mean_abs_color_difference_0_255": round(
            max(difference_values), 4
        ),
        "thresholds": {
            "minimum_coverage_fraction": minimum_coverage,
            "maximum_mean_abs_color_difference_0_255": maximum_difference,
        },
        "records": records,
    }


def _color_mass_trend_audit(
    frames: list[np.ndarray],
    audit: dict[str, Any],
) -> dict[str, Any]:
    height, width = frames[0].shape[:2]
    roi = audit["roi_normalized_xyxy"]
    left = int(round(float(roi[0]) * width))
    top = int(round(float(roi[1]) * height))
    right = int(round(float(roi[2]) * width))
    bottom = int(round(float(roi[3]) * height))
    counts = []
    scores = []
    for frame in frames:
        crop = frame[top:bottom, left:right].astype(np.int16)
        mask = (
            (crop[:, :, 0] >= int(audit["r_min"]))
            & (
                crop[:, :, 0] - crop[:, :, 1]
                >= int(audit["minimum_r_minus_g"])
            )
            & (
                crop[:, :, 2] - crop[:, :, 1]
                >= int(audit["minimum_b_minus_g"])
            )
        )
        counts.append(int(mask.sum()))
        excess = np.maximum(
            crop[:, :, 0]
            - crop[:, :, 1]
            - int(audit["minimum_r_minus_g"]),
            0,
        )
        scores.append(float(excess[mask].sum()))
    initial = max(counts[0], 1)
    final_fraction = counts[-1] / initial
    initial_score = max(scores[0], 1.0)
    final_score_fraction = scores[-1] / initial_score
    maximum_score_fraction = max(scores) / initial_score
    maximum_increase = max(
        [right_value - left_value for left_value, right_value in zip(
            counts, counts[1:]
        )]
        or [0]
    )
    allowed_increase = float(
        audit["maximum_step_increase_fraction_of_initial"]
    ) * initial
    maximum_final_fraction = float(
        audit["maximum_final_fraction_of_initial"]
    )
    maximum_allowed_score_fraction = float(
        audit["maximum_score_fraction_of_initial"]
    )
    return {
        "passed": counts[0] >= int(audit["minimum_initial_pixels"])
        and final_fraction <= maximum_final_fraction
        and final_score_fraction <= maximum_final_fraction
        and maximum_score_fraction <= maximum_allowed_score_fraction
        and maximum_increase <= allowed_increase,
        "colored_pixel_counts": counts,
        "integrated_color_scores": [
            round(value, 3) for value in scores
        ],
        "initial_count": counts[0],
        "final_count": counts[-1],
        "final_fraction_of_initial": round(final_fraction, 6),
        "initial_integrated_color_score": round(scores[0], 3),
        "final_integrated_color_score": round(scores[-1], 3),
        "final_score_fraction_of_initial": round(
            final_score_fraction, 6
        ),
        "maximum_score_fraction_of_initial": round(
            maximum_score_fraction, 6
        ),
        "maximum_step_increase_pixels": maximum_increase,
        "allowed_step_increase_pixels": round(allowed_increase, 4),
        "thresholds": {
            "maximum_final_fraction_of_initial": (
                maximum_final_fraction
            ),
            "maximum_score_fraction_of_initial": (
                maximum_allowed_score_fraction
            ),
            "minimum_initial_pixels": audit["minimum_initial_pixels"],
        },
        "measurement_zh": (
            "只在规格给出的液体 ROI 内按颜色差计算局部显色像素量，"
            "检查它是否总体消散而不是重新增长。"
        ),
    }


def _save_preview(
    frames: list[np.ndarray],
    output_root: Path,
    fps: float,
) -> tuple[Path, list[int]]:
    indices = np.linspace(0, len(frames) - 1, 9, dtype=int).tolist()
    width, height = frames[0].shape[1], frames[0].shape[0]
    label_height = 34
    panels = []
    samples = output_root / "samples"
    samples.mkdir(exist_ok=True)
    for index in indices:
        image = Image.fromarray(frames[index])
        image.save(samples / f"frame_{index:03d}.png")
        panel = Image.new("RGB", (width, height + label_height), "white")
        panel.paste(image, (0, label_height))
        ImageDraw.Draw(panel).text(
            (10, 10),
            f"Frame {index} · {index / fps:.2f} s",
            fill=(20, 35, 42),
        )
        panels.append(panel)
    sheet = Image.new(
        "RGB",
        (width * 3, (height + label_height) * 3),
        (236, 232, 222),
    )
    for index, panel in enumerate(panels):
        sheet.paste(
            panel,
            (
                (index % 3) * width,
                (index // 3) * (height + label_height),
            ),
        )
    path = output_root / "generated-frames.jpg"
    sheet.save(path, quality=92, subsampling=0)
    return path, indices


def run_video_experiment(
    spec: dict[str, Any],
    output_root: Path,
    *,
    server: str = "http://127.0.0.1:8188",
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    prepared_path = output_root / "_work" / "prepare.json"
    if not prepared_path.is_file():
        prepare_video_experiment(spec, output_root)
    prepared = load_json(prepared_path)
    workflow = load_json(output_root / prepared["workflow"]["path"])
    _request_json(f"{server}/system_stats")
    queued = _request_json(f"{server}/prompt", {"prompt": workflow})
    prompt_id = queued["prompt_id"]
    print(f"Queued ComfyUI prompt: {prompt_id}", flush=True)
    history, elapsed = _wait_for_result(
        server, prompt_id, timeout_seconds
    )
    record = _find_video_record(history.get("outputs", {}))
    if not record:
        raise RuntimeError("SaveVideo completed without an MP4 record")
    comfy_video = (
        COMFY_OUTPUT
        / record.get("subfolder", "")
        / record["filename"]
    )
    if not comfy_video.is_file():
        raise FileNotFoundError(comfy_video)
    video_path = output_root / "transition.mp4"
    shutil.copy2(comfy_video, video_path)

    video_info, frames = _decode_video(video_path)
    expected_count = int(spec["settings"]["frame_count"])
    if len(frames) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} frames, decoded {len(frames)}"
        )
    width, height = int(video_info["width"]), int(video_info["height"])
    first_reference = _fitted_reference(
        Path(spec["source"]["first_frame"]), (width, height)
    )
    last_reference = _fitted_reference(
        Path(spec["source"]["last_frame"]), (width, height)
    )
    consecutive = [
        float(
            np.mean(
                np.abs(
                    frames[index].astype(np.float32)
                    - frames[index - 1].astype(np.float32)
                )
            )
        )
        for index in range(1, len(frames))
    ]
    preview_path, sample_indices = _save_preview(
        frames, output_root, float(video_info["fps"])
    )
    anchors = None
    if "fixed_anchors" in spec["audit"]:
        anchors = _anchor_audit(
            frames, spec["audit"]["fixed_anchors"]
        )
    radial = None
    if "radial_propagation" in spec["audit"]:
        radial = _radial_propagation_audit(
            frames, spec["audit"]["radial_propagation"]
        )
    color_identity = None
    if "color_identity_reference_sequence" in spec["audit"]:
        color_identity = _color_identity_reference_audit(
            frames,
            spec["audit"]["color_identity_reference_sequence"],
        )
    component_division = None
    if "color_component_division" in spec["audit"]:
        component_division = _color_component_division_audit(
            frames,
            spec["audit"]["color_component_division"],
        )
    region_topology = None
    if "color_region_topology" in spec["audit"]:
        region_topology = _color_region_topology_audit(
            frames,
            spec["audit"]["color_region_topology"],
        )
    reference_stability = None
    if "reference_color_mask_stability" in spec["audit"]:
        reference_stability = _reference_color_mask_stability(
            frames,
            Path(spec["source"]["first_frame"]),
            spec["audit"]["reference_color_mask_stability"],
        )
    color_mass = None
    if "color_mass_trend" in spec["audit"]:
        color_mass = _color_mass_trend_audit(
            frames, spec["audit"]["color_mass_trend"]
        )
    first_metrics = _comparison_metrics(frames[0], first_reference)
    last_metrics = _comparison_metrics(frames[-1], last_reference)
    thresholds = spec["audit"]["thresholds"]
    hard_checks = [
        {
            "name": "video_contract_matches_requested_dimensions",
            "passed": (
                width == int(spec["settings"]["width"])
                and height == int(spec["settings"]["height"])
                and len(frames) == expected_count
            ),
            "evidence": video_info,
        },
        {
            "name": "first_and_last_frames_follow_inputs",
            "passed": (
                first_metrics["mean_absolute_pixel_error_0_255"]
                <= float(thresholds["maximum_endpoint_mae_0_255"])
                and last_metrics["mean_absolute_pixel_error_0_255"]
                <= float(thresholds["maximum_endpoint_mae_0_255"])
            ),
            "evidence": {
                "first": first_metrics,
                "last": last_metrics,
                "threshold": thresholds["maximum_endpoint_mae_0_255"],
            },
        },
        {
            "name": "no_single_frame_jump_exceeds_smoke_threshold",
            "passed": max(consecutive)
            <= float(thresholds["maximum_consecutive_frame_mae_0_255"]),
            "evidence": {
                "maximum": round(max(consecutive), 4),
                "p95": round(float(np.percentile(consecutive, 95)), 4),
                "mean": round(float(np.mean(consecutive)), 4),
                "threshold": thresholds[
                    "maximum_consecutive_frame_mae_0_255"
                ],
            },
        },
    ]
    if anchors is not None:
        hard_checks.append(
            {
                "name": (
                    "fixed_semantic_anchors_remain_visible_and_stable"
                ),
                "passed": anchors["passed"],
                "evidence": {
                    key: value
                    for key, value in anchors.items()
                    if key != "frames"
                },
            }
        )
    if radial is not None:
        hard_checks.append(
            {
                "name": "declared_radial_motion_moves_outward",
                "passed": radial["passed"],
                "evidence": {
                    key: value
                    for key, value in radial.items()
                    if key not in {"records", "measurement_zh"}
                },
            }
        )
    if color_identity is not None:
        hard_checks.append(
            {
                "name": (
                    "colored_object_identities_follow_program_rigid_motion"
                ),
                "passed": color_identity["passed"],
                "evidence": {
                    key: value
                    for key, value in color_identity.items()
                    if key not in {"records", "measurement_zh"}
                },
            }
        )
    if component_division is not None:
        hard_checks.append(
            {
                "name": (
                    "colored_parent_population_divides_into_balanced_groups"
                ),
                "passed": component_division["passed"],
                "evidence": {
                    key: value
                    for key, value in component_division.items()
                    if key not in {"records", "measurement_zh"}
                },
            }
        )
    if region_topology is not None:
        hard_checks.append(
            {
                "name": (
                    "colored_region_changes_topology_with_connected_trunk"
                ),
                "passed": region_topology["passed"],
                "evidence": {
                    key: value
                    for key, value in region_topology.items()
                    if key not in {"records", "measurement_zh"}
                },
            }
        )
    if reference_stability is not None:
        hard_checks.append(
            {
                "name": "declared_static_color_mask_remains_stable",
                "passed": reference_stability["passed"],
                "evidence": {
                    key: value
                    for key, value in reference_stability.items()
                    if key != "records"
                },
            }
        )
    if color_mass is not None:
        hard_checks.append(
            {
                "name": "declared_color_mass_follows_expected_trend",
                "passed": color_mass["passed"],
                "evidence": {
                    key: value
                    for key, value in color_mass.items()
                    if key not in {"colored_pixel_counts", "measurement_zh"}
                },
            }
        )
    result = {
        "schema_version": "1.0",
        "experiment_id": spec["experiment_id"],
        "case_id": spec["case_id"],
        "motion_class": spec["motion_class"],
        "status": "generated_pending_visual_review",
        "classification": "LTX-2.3 first-last-frame conditioned video",
        "experiment_spec_sha256": sha256_path(
            Path(spec["_spec_path"])
        ),
        "prompt_id": prompt_id,
        "generation_seconds": round(elapsed, 3),
        "model_runs": {"image": 0, "video": 1},
        "video": {
            **video_info,
            **artifact_record(video_path, output_root),
        },
        "preview": artifact_record(preview_path, output_root),
        "sample_indices": sample_indices,
        "endpoint_comparison": {
            "first": first_metrics,
            "last": last_metrics,
        },
        "temporal_metrics": {
            "mean_consecutive_frame_mae_0_255": round(
                float(np.mean(consecutive)), 4
            ),
            "p95_consecutive_frame_mae_0_255": round(
                float(np.percentile(consecutive, 95)), 4
            ),
            "maximum_consecutive_frame_mae_0_255": round(
                max(consecutive), 4
            ),
        },
        "anchor_audit": anchors,
        "radial_propagation_audit": radial,
        "color_identity_reference_audit": color_identity,
        "color_component_division_audit": component_division,
        "color_region_topology_audit": region_topology,
        "reference_color_mask_stability_audit": reference_stability,
        "color_mass_trend_audit": color_mass,
        "hard_checks": hard_checks,
    }
    write_json(output_root / "_work" / "run.json", result)
    return result


def reaudit_video_experiment(
    spec: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Add deterministic audits to an existing video without a model call."""

    run_path = output_root / "_work" / "run.json"
    result = load_json(run_path)
    result["experiment_spec_sha256"] = sha256_path(
        Path(spec["_spec_path"])
    )
    video_path = output_root / result["video"]["path"]
    _, frames = _decode_video(video_path)
    audit_count = 0

    def replace_check(check: dict[str, Any]) -> None:
        result["hard_checks"] = [
            item
            for item in result["hard_checks"]
            if item["name"] != check["name"]
        ]
        result["hard_checks"].append(check)

    radial_spec = spec["audit"].get("radial_propagation")
    if radial_spec is not None:
        radial = _radial_propagation_audit(frames, radial_spec)
        result["radial_propagation_audit"] = radial
        replace_check(
            {
                "name": "declared_radial_motion_moves_outward",
                "passed": radial["passed"],
                "evidence": {
                    key: value
                    for key, value in radial.items()
                    if key not in {"records", "measurement_zh"}
                },
            }
        )
        audit_count += 1
    stability_spec = spec["audit"].get(
        "reference_color_mask_stability"
    )
    if stability_spec is not None:
        stability = _reference_color_mask_stability(
            frames,
            Path(spec["source"]["first_frame"]),
            stability_spec,
        )
        result["reference_color_mask_stability_audit"] = stability
        replace_check(
            {
                "name": "declared_static_color_mask_remains_stable",
                "passed": stability["passed"],
                "evidence": {
                    key: value
                    for key, value in stability.items()
                    if key != "records"
                },
            }
        )
        audit_count += 1
    mass_spec = spec["audit"].get("color_mass_trend")
    if mass_spec is not None:
        mass = _color_mass_trend_audit(frames, mass_spec)
        result["color_mass_trend_audit"] = mass
        replace_check(
            {
                "name": "declared_color_mass_follows_expected_trend",
                "passed": mass["passed"],
                "evidence": {
                    key: value
                    for key, value in mass.items()
                    if key
                    not in {
                        "colored_pixel_counts",
                        "integrated_color_scores",
                        "measurement_zh",
                    }
                },
            }
        )
        audit_count += 1
    division_spec = spec["audit"].get("color_component_division")
    if division_spec is not None:
        division = _color_component_division_audit(
            frames, division_spec
        )
        result["color_component_division_audit"] = division
        replace_check(
            {
                "name": (
                    "colored_parent_population_divides_into_balanced_groups"
                ),
                "passed": division["passed"],
                "evidence": {
                    key: value
                    for key, value in division.items()
                    if key not in {"records", "measurement_zh"}
                },
            }
        )
        audit_count += 1
    topology_spec = spec["audit"].get("color_region_topology")
    if topology_spec is not None:
        topology = _color_region_topology_audit(
            frames, topology_spec
        )
        result["color_region_topology_audit"] = topology
        replace_check(
            {
                "name": (
                    "colored_region_changes_topology_with_connected_trunk"
                ),
                "passed": topology["passed"],
                "evidence": {
                    key: value
                    for key, value in topology.items()
                    if key not in {"records", "measurement_zh"}
                },
            }
        )
        audit_count += 1
    if audit_count == 0:
        raise ValueError("spec does not declare a reusable re-audit")
    write_json(run_path, result)
    return result
