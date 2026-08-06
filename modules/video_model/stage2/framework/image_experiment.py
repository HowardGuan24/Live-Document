"""Auditable SDXL + ControlNet experiments for Stage 2.

The experiment runner is case-agnostic: it reads a clean keyframe and declared
semantic layers, compiles prompt parts, derives alternative control images,
and freezes every model input before loading the GPU pipeline.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .contracts import artifact_record, load_json, sha256_path, write_json


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _resize(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.convert("RGB").resize(
        (width, height), Image.Resampling.LANCZOS
    )


def _draw_geometry(
    draw: ImageDraw.ImageDraw,
    geometry: dict[str, Any],
    *,
    scale_x: float,
    scale_y: float,
    width: int,
) -> None:
    kind = geometry["kind"]
    if kind == "ellipse":
        box = geometry["bbox_xyxy"]
        draw.ellipse(
            (
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y,
            ),
            outline=255,
            width=width,
        )
    elif kind in {"polygon", "polyline"}:
        points = [
            (point[0] * scale_x, point[1] * scale_y)
            for point in geometry["points"]
        ]
        if kind == "polygon":
            draw.line((*points, points[0]), fill=255, width=width)
        else:
            draw.line(points, fill=255, width=width)


def _identity_boundary(
    layer_manifest: dict[str, Any],
    program_root: Path,
    *,
    width: int,
    height: int,
) -> Image.Image:
    canvas = layer_manifest["canvas"]
    scale_x = width / canvas["width"]
    scale_y = height / canvas["height"]
    result = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(result)
    line_width = max(2, round(width / 340))
    # The locked camera/tank frame is a hard scene boundary, while semantic
    # object identities contribute only their declared geometry.
    margin_x = round(7 * scale_x)
    margin_y = round(7 * scale_y)
    draw.rectangle(
        (margin_x, margin_y, width - margin_x, height - margin_y),
        outline=255,
        width=line_width,
    )
    object_layers = [
        layer
        for layer in layer_manifest["layers"]
        if layer["layer_type"] == "object_identity"
    ]
    for layer in object_layers:
        payload = load_json(program_root / layer["data"]["path"])
        for item in payload["items"]:
            _draw_geometry(
                draw,
                item["geometry"],
                scale_x=scale_x,
                scale_y=scale_y,
                width=line_width,
            )
    return result.convert("RGB")


def _hard_boundary(
    layer_manifest: dict[str, Any],
    program_root: Path,
    *,
    width: int,
    height: int,
) -> Image.Image:
    """Rasterize only semantic layers explicitly typed as hard boundaries."""

    layers = [
        layer
        for layer in layer_manifest["layers"]
        if layer["layer_type"] == "hard_boundary"
    ]
    if not layers:
        return Image.new("RGB", (width, height), (0, 0, 0))
    combined = None
    for layer in layers:
        if layer["data"]["encoding"] != "npy":
            raise ValueError("hard-boundary data must use NPY encoding")
        array = np.load(
            program_root / layer["data"]["path"], allow_pickle=False
        )
        binary = np.asarray(array > 0, dtype=np.uint8) * 255
        combined = (
            binary
            if combined is None
            else np.maximum(combined, binary)
        )
    assert combined is not None
    image = Image.fromarray(combined, mode="L").resize(
        (width, height), Image.Resampling.NEAREST
    )
    return image.convert("RGB")


def _dense_canny(source: Image.Image) -> Image.Image:
    import cv2

    rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # The program renders small-amplitude water waves with deliberately soft
    # contrast.  Low thresholds are required for the negative control to
    # actually encode those continuous ridges instead of silently becoming
    # another sparse-boundary control.
    edge = cv2.Canny(gray, 5, 15, L2gradient=True)
    edge = cv2.dilate(edge, np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(np.repeat(edge[:, :, None], 3, axis=2), mode="RGB")


def compile_prompt(parts: dict[str, str]) -> str:
    required = (
        "scene_identity",
        "material_goal",
        "state_delta",
        "must_preserve",
    )
    missing = [name for name in required if not parts.get(name, "").strip()]
    if missing:
        raise ValueError(f"prompt parts are empty: {missing}")
    return ", ".join(parts[name].strip(" ,.") for name in required)


def _token_preflight(
    base_path: Path,
    prompt: str,
    negative_prompt: str,
) -> dict[str, Any]:
    from transformers import CLIPTokenizer

    result = {}
    violations = []
    for label, text in (
        ("positive", prompt),
        ("negative", negative_prompt),
    ):
        counts = {}
        limits = {}
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
                    f"{label}/{subfolder}: {count} > {limit}"
                )
        result[label] = {
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


def prepare_experiment(
    spec: dict[str, Any],
    experiment_root: Path,
) -> dict[str, Any]:
    experiment_root.mkdir(parents=True, exist_ok=True)
    input_root = experiment_root / "inputs"
    control_root = experiment_root / "controls"
    work_root = experiment_root / "_work"
    for path in (input_root, control_root, work_root):
        path.mkdir(parents=True, exist_ok=True)

    source_path = Path(spec["source"]["clean_frame"])
    semantic_path = Path(spec["source"]["semantic_layers"])
    program_root = semantic_path.parents[2]
    layer_manifest = load_json(semantic_path)
    width = int(spec["render"]["width"])
    height = int(spec["render"]["height"])
    source = _resize(Image.open(source_path), width, height)
    source_output = input_root / "clean_keyframe.png"
    source.save(source_output, optimize=False)
    prompt = compile_prompt(spec["prompt_parts"])
    negative_prompt = spec["negative_artifacts"].strip()
    prompt_path = input_root / "positive_prompt.txt"
    negative_path = input_root / "negative_prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    negative_path.write_text(negative_prompt + "\n", encoding="utf-8")

    controls = {
        "control_off": Image.new("RGB", (width, height), (0, 0, 0)),
        "dense_canny": _dense_canny(source),
        "sparse_hard_boundary": _hard_boundary(
            layer_manifest,
            program_root,
            width=width,
            height=height,
        ),
        "sparse_identity_boundary": _identity_boundary(
            layer_manifest,
            program_root,
            width=width,
            height=height,
        ),
    }
    for route, override_path_value in spec.get(
        "control_overrides", {}
    ).items():
        override_path = Path(override_path_value)
        if not override_path.is_file():
            raise FileNotFoundError(
                f"control override is missing: {override_path}"
            )
        controls[route] = _resize(
            Image.open(override_path), width, height
        )
    control_records = {}
    for route, image in controls.items():
        path = control_root / f"{route}.png"
        image.save(path, optimize=False)
        edge_fraction = float(
            (np.asarray(image.convert("L")) > 0).mean()
        )
        control_records[route] = {
            **artifact_record(path, experiment_root),
            "edge_fraction": round(edge_fraction, 8),
        }
    derivation = {
        "schema_version": "1.0",
        "control_off": (
            "全黑控制图且 ControlNet 强度为 0，表示完全关闭边缘约束。"
        ),
        "dense_canny": (
            "从无标注程序底图灰度化后，用 Canny 阈值 5/15 提取，并膨胀 1 次；"
            "它会保留底图中的全部明显亮暗边缘，包括主体轮廓、内部结构和辅助线。"
            "这是高约束控制；若底图有粗糙像素、文字或无关纹理，也会一并约束模型。"
        ),
        "sparse_identity_boundary": (
            "只从锁定画框和 object_identity 语义层的几何生成白线；"
            "不包含波峰、波谷、箭头或文字。"
        ),
        "sparse_hard_boundary": (
            "只合并程序明确声明为 hard_boundary 的数值层；不从成图猜边，"
            "也不加入区域、连续场、箭头或文字。"
        ),
        "source_semantic_layers": str(semantic_path.resolve()),
    }
    for route, explanation in spec.get(
        "control_override_explanations", {}
    ).items():
        derivation[route] = explanation
    derivation_path = control_root / "derivation.json"
    write_json(derivation_path, derivation)
    prepared = {
        "schema_version": "1.0",
        "experiment_id": spec["experiment_id"],
        "case_id": spec["case_id"],
        "source": {
            "clean_keyframe": artifact_record(
                source_output, experiment_root
            ),
            "original_clean_keyframe": {
                "path": str(source_path.resolve()),
                "sha256": sha256_path(source_path),
            },
            "semantic_layers": {
                "path": str(semantic_path.resolve()),
                "sha256": sha256_path(semantic_path),
            },
        },
        "prompt_parts": spec["prompt_parts"],
        "positive_prompt": {
            **artifact_record(prompt_path, experiment_root),
            "text": prompt,
        },
        "negative_prompt": {
            **artifact_record(negative_path, experiment_root),
            "text": negative_prompt,
        },
        "controls": control_records,
        "control_derivation": artifact_record(
            derivation_path, experiment_root
        ),
        "render": spec["render"],
        "configurations": spec["configurations"],
    }
    prepared_path = work_root / "prepare.json"
    write_json(prepared_path, prepared)
    return prepared


def _package_versions() -> dict[str, str | None]:
    result = {}
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
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _candidate_metrics(
    candidate: Image.Image,
    control: Image.Image,
) -> dict[str, float]:
    import cv2

    rgb = np.asarray(candidate.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 110, L2gradient=True) > 0
    control_binary = np.asarray(control.convert("L")) > 0
    if control_binary.any():
        dilated = cv2.dilate(
            np.uint8(edges) * 255,
            np.ones((9, 9), np.uint8),
            iterations=1,
        ) > 0
        coverage = float(dilated[control_binary].mean())
    else:
        coverage = 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return {
        "candidate_edge_fraction": round(float(edges.mean()), 8),
        "control_edge_coverage_within_4px": round(coverage, 8),
        "mean_saturation_0_255": round(float(hsv[:, :, 1].mean()), 5),
        "luminance_std_0_255": round(float(gray.std()), 5),
    }


def _sheet(
    records: list[dict[str, Any]],
    experiment_root: Path,
    path: Path,
    *,
    blind: bool,
) -> None:
    columns = 3
    thumb_width, thumb_height = 512, 288
    label_height = 34
    gutter = 10
    rows = (len(records) + columns - 1) // columns
    width = columns * thumb_width + (columns + 1) * gutter
    height = rows * (thumb_height + label_height) + (rows + 1) * gutter
    sheet = Image.new("RGB", (width, height), (13, 31, 37))
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        left = gutter + column * (thumb_width + gutter)
        top = gutter + row * (thumb_height + label_height + gutter)
        image = Image.open(experiment_root / record["path"]).convert("RGB")
        image.thumbnail((thumb_width, thumb_height))
        sheet.paste(image, (left, top))
        draw.rectangle(
            (
                left,
                top + thumb_height,
                left + thumb_width,
                top + thumb_height + label_height,
            ),
            fill=(5, 23, 28),
        )
        label = (
            record["blind_id"]
            if blind
            else f"{record['configuration_id']} · seed {record['seed']}"
        )
        draw.text(
            (left + 8, top + thumb_height + 10),
            label,
            fill=(232, 244, 239),
        )
    sheet.save(path, quality=92, subsampling=0)


def build_candidate_sheets(
    metadata: dict[str, Any],
    experiment_root: Path,
) -> dict[str, dict[str, Any]]:
    records = list(metadata["candidates"])
    labeled_path = experiment_root / "candidates-labeled.jpg"
    _sheet(records, experiment_root, labeled_path, blind=False)
    blind = list(records)
    random.Random(int(metadata["blind_shuffle_seed"])).shuffle(blind)
    blind_map = {}
    for index, record in enumerate(blind):
        blind_id = chr(ord("A") + index)
        record["blind_id"] = blind_id
        blind_map[blind_id] = {
            "configuration_id": record["configuration_id"],
            "seed": record["seed"],
            "sha256": record["sha256"],
        }
        for original in metadata["candidates"]:
            if original["sha256"] == record["sha256"]:
                original["blind_id"] = blind_id
                break
    blind_path = experiment_root / "candidates-blind.jpg"
    _sheet(blind, experiment_root, blind_path, blind=True)
    blind_map_path = experiment_root / "_work" / "blind_map.json"
    write_json(blind_map_path, blind_map)
    return {
        "labeled": artifact_record(labeled_path, experiment_root),
        "blind": artifact_record(blind_path, experiment_root),
        "blind_map": artifact_record(blind_map_path, experiment_root),
    }


def _materialize_external_reuse(
    *,
    configuration: dict[str, Any],
    seed: int,
    prepared: dict[str, Any],
    spec: dict[str, Any],
    fingerprints: dict[str, Any],
    target: Path,
) -> dict[str, Any] | None:
    reuse = configuration.get("reuse_from")
    if not reuse:
        return None
    reference_root = Path(reuse["experiment_root"])
    reference_prepared = load_json(
        reference_root / "_work" / "prepare.json"
    )
    reference_metadata = load_json(
        reference_root / "_work" / "generate.json"
    )
    comparisons = {
        "clean input": (
            prepared["source"]["clean_keyframe"]["sha256"],
            reference_prepared["source"]["clean_keyframe"]["sha256"],
        ),
        "positive prompt": (
            prepared["positive_prompt"]["sha256"],
            reference_prepared["positive_prompt"]["sha256"],
        ),
        "negative prompt": (
            prepared["negative_prompt"]["sha256"],
            reference_prepared["negative_prompt"]["sha256"],
        ),
        "control image": (
            prepared["controls"][configuration["control_route"]]["sha256"],
            reference_prepared["controls"][
                reuse["control_route"]
            ]["sha256"],
        ),
        "render settings": (
            spec["render"],
            reference_metadata["render"],
        ),
        "model fingerprints": (
            fingerprints["models"],
            reference_metadata["model_fingerprints"]["models"],
        ),
    }
    mismatches = [
        name for name, (current, previous) in comparisons.items()
        if current != previous
    ]
    if mismatches:
        raise ValueError(
            "external candidate reuse input mismatch: "
            + ", ".join(mismatches)
        )
    records = [
        item
        for item in reference_metadata["candidates"]
        if item["configuration_id"] == reuse["configuration_id"]
        and int(item["seed"]) == seed
    ]
    if len(records) != 1:
        raise ValueError(
            "external reuse must resolve exactly one candidate: "
            f"{reuse['configuration_id']}/seed {seed}"
        )
    record = records[0]
    if float(record["controlnet_conditioning_scale"]) != float(
        configuration["controlnet_conditioning_scale"]
    ):
        raise ValueError("external reuse ControlNet scale mismatch")
    if configuration.get("pipeline_mode", "controlnet_img2img") == (
        "controlnet_img2img"
    ):
        reference_strength = float(
            record.get(
                "img2img_strength",
                reference_metadata["render"]["strength"],
            )
        )
        current_strength = float(
            configuration.get(
                "img2img_strength", spec["render"]["strength"]
            )
        )
        if reference_strength != current_strength:
            raise ValueError("external reuse Img2Img strength mismatch")
    source = reference_root / record["path"]
    if sha256_path(source) != record["sha256"]:
        raise ValueError(f"external reuse source hash mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256_path(target) != record["sha256"]:
        raise ValueError(f"external reuse copy hash mismatch: {target}")
    return {
        "experiment_root": str(reference_root.resolve()),
        "experiment_id": reference_metadata["experiment_id"],
        "configuration_id": record["configuration_id"],
        "seed": seed,
        "sha256": record["sha256"],
        "reason": (
            "clean input, prompts, control image, render settings, model "
            "fingerprints, ControlNet scale and seed are byte-identical"
        ),
    }


def generate_experiment(
    spec: dict[str, Any],
    experiment_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Generate or reuse all fixed-seed candidates in one GPU load."""

    prepared = prepare_experiment(spec, experiment_root)
    from modules.video_model.stage1.keyframe_render.enhance import (
        MODEL_IDS,
        _diffusers_runtime,
        fingerprint_models,
        model_paths,
    )

    paths = model_paths()
    missing = [name for name, path in paths.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing local model directories: {missing}")
    fingerprints = fingerprint_models(experiment_root, paths)
    prompt = prepared["positive_prompt"]["text"]
    negative = prepared["negative_prompt"]["text"]
    token_preflight = _token_preflight(
        paths["sdxl_base"], prompt, negative
    )
    metadata_path = experiment_root / "_work" / "generate.json"
    previous = load_json(metadata_path) if metadata_path.is_file() else {}
    previous_by_key = {
        (record["configuration_id"], int(record["seed"])): record
        for record in previous.get("candidates", [])
    }
    jobs = []
    for configuration in spec["configurations"]:
        control = prepared["controls"][configuration["control_route"]]
        for seed in spec["render"]["seeds"]:
            target = (
                experiment_root
                / "raw"
                / configuration["configuration_id"]
                / f"seed_{seed}.png"
            )
            signature = _stable_hash(
                {
                    "experiment_id": spec["experiment_id"],
                    "source_sha256": prepared["source"][
                        "clean_keyframe"
                    ]["sha256"],
                    "prompt_sha256": prepared["positive_prompt"]["sha256"],
                    "negative_sha256": prepared["negative_prompt"]["sha256"],
                    "control_sha256": control["sha256"],
                    "configuration": configuration,
                    "render": spec["render"],
                    "seed": seed,
                    "models": fingerprints["models"],
                }
            )
            old = previous_by_key.get(
                (configuration["configuration_id"], int(seed))
            )
            cached_reuse = bool(
                target.is_file()
                and not force
                and old
                and old.get("input_signature") == signature
                and old.get("sha256") == sha256_path(target)
            )
            external_reuse = None
            if not cached_reuse and configuration.get("reuse_from"):
                external_reuse = _materialize_external_reuse(
                    configuration=configuration,
                    seed=int(seed),
                    prepared=prepared,
                    spec=spec,
                    fingerprints=fingerprints,
                    target=target,
                )
            reused = cached_reuse or external_reuse is not None
            jobs.append(
                {
                    "configuration": configuration,
                    "seed": int(seed),
                    "target": target,
                    "control": control,
                    "signature": signature,
                    "reused": reused,
                    "external_reuse": external_reuse,
                    "old": old,
                }
            )
    to_generate = [job for job in jobs if not job["reused"]]
    pipeline_modes = {
        job["configuration"].get(
            "pipeline_mode", "controlnet_img2img"
        )
        for job in to_generate
    }
    if len(pipeline_modes) > 1:
        raise ValueError(
            "one experiment invocation cannot mix pipeline modes"
        )
    pipeline_mode = (
        next(iter(pipeline_modes))
        if pipeline_modes
        else previous.get("pipeline_mode", "controlnet_img2img")
    )
    if pipeline_mode not in {
        "controlnet_img2img",
        "controlnet_t2i",
    }:
        raise ValueError(f"unsupported pipeline mode: {pipeline_mode}")
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "generating",
        "experiment_id": spec["experiment_id"],
        "case_id": spec["case_id"],
        "hypothesis_zh": spec["hypothesis_zh"],
        "single_variable_zh": spec["single_variable_zh"],
        "render": spec["render"],
        "pipeline_mode": pipeline_mode,
        "models": {
            name: {
                "model_id": MODEL_IDS[name],
                "path": str(path.resolve()),
                "variant": "fp16",
            }
            for name, path in paths.items()
        },
        "model_fingerprints": fingerprints,
        "package_versions": _package_versions(),
        "prompt_token_preflight": token_preflight,
        "blind_shuffle_seed": int(spec["blind_shuffle_seed"]),
        "candidates": [],
    }
    pipeline = None
    controlnet = None
    torch = None
    if to_generate:
        torch, ControlNetModel, Img2ImgPipeline = _diffusers_runtime()
        Pipeline = (
            Img2ImgPipeline
            if pipeline_mode == "controlnet_img2img"
            else getattr(
                importlib.import_module("diffusers"),
                "StableDiffusionXLControlNetPipeline",
            )
        )
        if not torch.cuda.is_available():
            raise RuntimeError("registered GPU runtime is unavailable")
        metadata["runtime"] = {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
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
    write_json(metadata_path, metadata)

    source = Image.open(
        experiment_root / prepared["source"]["clean_keyframe"]["path"]
    ).convert("RGB")
    for job in jobs:
        configuration = job["configuration"]
        target = job["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        peak_memory = None
        if not job["reused"]:
            assert torch is not None and pipeline is not None
            control = Image.open(
                experiment_root / job["control"]["path"]
            ).convert("RGB")
            torch.cuda.reset_peak_memory_stats()
            generator = torch.Generator(device="cuda").manual_seed(
                job["seed"]
            )
            common_arguments = {
                "prompt": prompt,
                "negative_prompt": negative,
                "num_inference_steps": int(spec["render"]["steps"]),
                "guidance_scale": float(
                    spec["render"]["guidance_scale"]
                ),
                "controlnet_conditioning_scale": float(
                    configuration["controlnet_conditioning_scale"]
                ),
                "generator": generator,
            }
            if pipeline_mode == "controlnet_img2img":
                candidate = pipeline(
                    **common_arguments,
                    image=source,
                    control_image=control,
                    strength=float(
                        configuration.get(
                            "img2img_strength",
                            spec["render"]["strength"],
                        )
                    ),
                ).images[0]
            else:
                candidate = pipeline(
                    **common_arguments,
                    image=control,
                    width=int(spec["render"]["width"]),
                    height=int(spec["render"]["height"]),
                ).images[0]
            candidate.save(target)
            torch.cuda.synchronize()
            peak_memory = int(torch.cuda.max_memory_allocated())
        control_image = Image.open(
            experiment_root / job["control"]["path"]
        ).convert("RGB")
        candidate_image = Image.open(target).convert("RGB")
        record_pipeline_mode = configuration.get(
            "pipeline_mode", "controlnet_img2img"
        )
        record = {
            "configuration_id": configuration["configuration_id"],
            "control_route": configuration["control_route"],
            "pipeline_mode": record_pipeline_mode,
            "controlnet_conditioning_scale": configuration[
                "controlnet_conditioning_scale"
            ],
            "img2img_strength": (
                float(
                    configuration.get(
                        "img2img_strength", spec["render"]["strength"]
                    )
                )
                if record_pipeline_mode == "controlnet_img2img"
                else None
            ),
            "seed": job["seed"],
            **artifact_record(target, experiment_root),
            "classification": (
                "raw SDXL ControlNet Img2Img output"
                if record_pipeline_mode == "controlnet_img2img"
                else "raw SDXL ControlNet text-to-image output"
            ),
            "input_signature": job["signature"],
            "reused": job["reused"],
            "reused_from": job["external_reuse"],
            "inference_seconds": round(
                time.perf_counter() - started, 3
            ),
            "peak_gpu_memory_bytes": peak_memory,
            "metrics": _candidate_metrics(
                candidate_image, control_image
            ),
        }
        metadata["candidates"].append(record)
        write_json(metadata_path, metadata)
    sheets = build_candidate_sheets(metadata, experiment_root)
    metadata["sheets"] = sheets
    metadata["cache"] = {
        "generated": len(to_generate),
        "reused": len(jobs) - len(to_generate),
        "policy": (
            "reuse only if clean input, prompts, control, configuration, "
            "seed, render settings and model fingerprints all match"
        ),
    }
    metadata["status"] = "generated"
    metadata["model_runs"] = {
        "image_candidates": len(to_generate),
        "video_candidates": 0,
    }
    write_json(metadata_path, metadata)
    for directory, text in (
        (
            experiment_root / "composite",
            "本实验只比较模型原始图，没有生成程序合成图。\n",
        ),
        (
            experiment_root / "final",
            "本实验尚未选择生产用最终图；候选不能冒充 final。\n",
        ),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "README.md").write_text(text, encoding="utf-8")
    if pipeline is not None:
        del pipeline, controlnet
        gc.collect()
        assert torch is not None
        torch.cuda.empty_cache()
    return metadata
