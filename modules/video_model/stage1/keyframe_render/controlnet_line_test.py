"""Compare sparse Canny and detailed line art with pure SDXL ControlNet.

Unlike the earlier Stage 1.1 Route B, this experiment does not provide an
img2img initialization image. The only changed variable is the control image.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import importlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .enhance import _diffusers_runtime, fingerprint_models, model_paths
from .first_frame_test import (
    PROMPT_ROOT,
    SETTINGS as FIRST_FRAME_SETTINGS,
    _contact_sheet,
    _load_prompt,
    _package_versions,
    _validate_prompt_lengths,
)


STAGE_ROOT = Path(__file__).resolve().parents[1]
FIRST_FRAME_ROOT = STAGE_ROOT / "output" / "keyframe_render" / "first_frame_test"
OUTPUT_ROOT = FIRST_FRAME_ROOT / "controlnet_line_test"

SETTINGS = {
    "width": FIRST_FRAME_SETTINGS["width"],
    "height": FIRST_FRAME_SETTINGS["height"],
    "steps": FIRST_FRAME_SETTINGS["steps"],
    "guidance_scale": FIRST_FRAME_SETTINGS["guidance_scale"],
    "controlnet_conditioning_scale": 0.50,
    "seeds": [3101, 3102, 3103, 3104],
    "pipeline": "StableDiffusionXLControlNetPipeline",
    "img2img_initial_image": None,
    "strength": None,
    "dtype": "float16",
}

CONDITIONS = ("sparse_canny", "detailed_lineart", "reference_canny")


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


def _curve_points(
    xs: np.ndarray,
    ys: np.ndarray,
    scale: int,
) -> list[tuple[int, int]]:
    return [
        (int(round(x * scale)), int(round(y * scale)))
        for x, y in zip(xs, ys)
    ]


def detailed_lineart(
    size: tuple[int, int] = (SETTINGS["width"], SETTINGS["height"]),
) -> Image.Image:
    """Create a reproducible high-resolution semantic line drawing.

    White geometry represents:
    * a naturally varying coast split by one river mouth;
    * two gently meandering river banks;
    * restrained land contours and near-shore depth contours;
    * a widening suspended-sediment plume with internal streamlines.
    """

    width, height = size
    scale = 2
    canvas = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(canvas)

    center_y = height / 2
    mouth_x = width * 0.395
    samples_x = np.linspace(0, mouth_x, 220)
    progress = samples_x / mouth_x
    upper_bank = (
        center_y
        - 58
        - 24 * progress**1.8
        + 5 * np.sin(samples_x / 82)
        + 2 * np.sin(samples_x / 27)
    )
    lower_bank = (
        center_y
        + 58
        + 24 * progress**1.8
        + 5 * np.sin(samples_x / 91 + 0.7)
        + 2 * np.sin(samples_x / 31)
    )
    line_width = 3 * scale
    draw.line(
        _curve_points(samples_x, upper_bank, scale),
        fill=255,
        width=line_width,
        joint="curve",
    )
    draw.line(
        _curve_points(samples_x, lower_bank, scale),
        fill=255,
        width=line_width,
        joint="curve",
    )

    upper_end = float(upper_bank[-1])
    lower_end = float(lower_bank[-1])
    upper_y = np.linspace(0, upper_end, 180)
    lower_y = np.linspace(lower_end, height - 1, 180)

    def coast_x(ys: np.ndarray) -> np.ndarray:
        mouth_bulge = 20 * np.exp(-((ys - center_y) / 150) ** 2)
        return (
            mouth_x
            + mouth_bulge
            + 13 * np.sin(ys / 73)
            + 5 * np.sin(ys / 24 + 0.4)
        )

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

    # Interior land contours: enough detail for texture placement, but no
    # hatching dense enough to overwhelm the main silhouette.
    land_x = np.linspace(25, mouth_x - 24, 160)
    for base_y, phase in (
        (72, 0.0),
        (142, 0.8),
        (222, 1.7),
        (height - 220, 0.4),
        (height - 145, 1.2),
        (height - 72, 2.0),
    ):
        contour_y = base_y + 8 * np.sin(land_x / 88 + phase)
        draw.line(
            _curve_points(land_x, contour_y, scale),
            fill=150,
            width=2 * scale,
        )

    # Near-shore bathymetry follows the coast instead of repeating the
    # rectangular source geometry.
    all_y = np.linspace(8, height - 8, 240)
    for offset, tone in ((74, 155), (158, 125), (270, 95)):
        depth_x = (
            coast_x(all_y)
            + offset
            + 28 * np.exp(-((all_y - center_y) / 145) ** 2)
        )
        draw.line(
            _curve_points(depth_x, all_y, scale),
            fill=tone,
            width=2 * scale,
        )

    # A continuous plume opens into the sea. Boundary and internal streamlines
    # are weaker than the coast so the model may treat them as suspended
    # material rather than as an island outline.
    plume_x = np.linspace(mouth_x + 8, width * 0.87, 220)
    plume_progress = (plume_x - plume_x[0]) / (plume_x[-1] - plume_x[0])
    plume_center = center_y + 11 * np.sin(plume_x / 105)
    plume_spread = 24 + 92 * plume_progress**0.85
    for factor, tone, plume_width in (
        (-1.0, 205, 2),
        (-0.55, 135, 2),
        (0.0, 165, 2),
        (0.55, 135, 2),
        (1.0, 205, 2),
    ):
        plume_y = (
            plume_center
            + factor * plume_spread
            + 4 * np.sin(plume_x / 39 + factor)
        )
        draw.line(
            _curve_points(plume_x, plume_y, scale),
            fill=tone,
            width=plume_width * scale,
        )

    downsampled = canvas.resize(size, Image.Resampling.LANCZOS)
    # Keep the Canny convention: sparse white edges on a black background.
    array = np.asarray(downsampled)
    return Image.fromarray(np.uint8(array >= 42) * 255, mode="L")


def _edge_stats(image: Image.Image) -> dict[str, Any]:
    array = np.asarray(image.convert("L"))
    nonzero = array > 0
    return {
        "size": list(image.size),
        "edge_pixels": int(nonzero.sum()),
        "edge_fraction": round(float(nonzero.mean()), 6),
        "unique_values": [int(value) for value in np.unique(array)],
    }


def prepare_conditions(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    source_root = output_root / "_work" / "source"
    prompt_root = output_root / "_work" / "prompts"
    source_root.mkdir(parents=True, exist_ok=True)
    prompt_root.mkdir(parents=True, exist_ok=True)

    sparse_source = FIRST_FRAME_ROOT / "_work" / "source" / "coastline_canny.png"
    if not sparse_source.is_file():
        raise FileNotFoundError(
            f"run first_frame_test --prepare before this test: {sparse_source}"
        )
    sparse_target = source_root / "sparse_canny.png"
    detailed_target = source_root / "detailed_lineart.png"
    reference_source = (
        FIRST_FRAME_ROOT
        / "review"
        / "free_topdown"
        / "free_topdown_s3103.png"
    )
    if not reference_source.is_file():
        raise FileNotFoundError(
            f"run first_frame_test --generate before this test: {reference_source}"
        )
    reference_target = source_root / "reference_image.png"
    reference_canny_target = source_root / "reference_canny.png"
    shutil.copyfile(sparse_source, sparse_target)
    detailed_lineart().save(detailed_target)
    shutil.copyfile(reference_source, reference_target)
    import cv2

    reference_gray = np.asarray(Image.open(reference_target).convert("L"))
    Image.fromarray(
        cv2.Canny(reference_gray, 100, 200),
        mode="L",
    ).save(reference_canny_target)

    prompt_target = prompt_root / "topdown.txt"
    negative_target = prompt_root / "topdown_negative.txt"
    shutil.copyfile(PROMPT_ROOT / "topdown.txt", prompt_target)
    shutil.copyfile(PROMPT_ROOT / "topdown_negative.txt", negative_target)

    conditions = {}
    for name, path in (
        ("sparse_canny", sparse_target),
        ("detailed_lineart", detailed_target),
        ("reference_canny", reference_canny_target),
    ):
        image = Image.open(path).convert("L")
        conditions[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            **_edge_stats(image),
        }

    _contact_sheet(
        [
            ("A: current sparse Canny", sparse_target),
            ("B: synthetic semantic line map", detailed_target),
            ("C source: natural SDXL aerial image", reference_target),
            ("C control: Canny extracted from source", reference_canny_target),
        ],
        output_root / "source-comparison.jpg",
        columns=2,
    )
    result = {
        "status": "prepared",
        "experiment_design": {
            "changed_variable": "control image only",
            "fixed": [
                "SDXL Base model and weights",
                "SDXL Canny ControlNet and weights",
                "pure text-to-image ControlNet pipeline",
                "prompt and negative prompt",
                "four seeds",
                "resolution, steps, guidance, and control scale",
            ],
            "img2img_initial_image": None,
            "mask_projection": False,
        },
        "conditions": conditions,
        "detailed_lineart_contents": [
            "naturalized continuous coastline",
            "one gently meandering river with widening mouth",
            "six restrained land contours",
            "three near-shore bathymetric contours",
            "five widening plume streamlines",
        ],
        "reference_image": {
            "purpose": (
                "demo-like reconstruction control only; its geography is not "
                "a valid target composition"
            ),
            "path": str(reference_target.resolve()),
            "sha256": _sha256(reference_target),
            "size": list(Image.open(reference_target).size),
            "canny_method": "OpenCV Canny thresholds 100/200",
        },
        "prompt": {
            "path": str(prompt_target.resolve()),
            "sha256": _sha256(prompt_target),
            "text": prompt_target.read_text(encoding="utf-8").strip(),
        },
        "negative_prompt": {
            "path": str(negative_target.resolve()),
            "sha256": _sha256(negative_target),
            "text": negative_target.read_text(encoding="utf-8").strip(),
        },
        "settings": SETTINGS,
    }
    _write_json(output_root / "_work" / "prepare_manifest.json", result)
    return result


def generate(
    output_root: Path = OUTPUT_ROOT,
    *,
    force: bool = False,
) -> dict[str, Any]:
    prepare = prepare_conditions(output_root)
    metadata_path = output_root / "_work" / "metadata.json"
    previous_by_name: dict[str, dict[str, Any]] = {}
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_by_name = {
            record["name"]: record for record in previous.get("candidates", [])
        }
    paths = model_paths()
    missing = [name for name, path in paths.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing local model directories: {missing}")

    prompt_preflight = _validate_prompt_lengths(paths["sdxl_base"])
    torch, ControlNetModel, _ = _diffusers_runtime()
    diffusers_module = importlib.import_module("diffusers")
    Pipeline = getattr(diffusers_module, "StableDiffusionXLControlNetPipeline")

    fingerprints = fingerprint_models(output_root, paths)
    metadata: dict[str, Any] = {
        "status": "generating",
        "settings": SETTINGS,
        "package_versions": _package_versions(),
        "prompt_token_preflight": {
            "topdown": prompt_preflight["topdown"],
            "topdown_negative": prompt_preflight["topdown_negative"],
        },
        "model_fingerprints": fingerprints,
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "conditions": prepare["conditions"],
        "candidates": [],
    }
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
    metadata["model_load_seconds"] = round(time.perf_counter() - load_started, 3)
    metadata["scheduler"] = {
        "class": type(pipeline.scheduler).__name__,
        "config": dict(pipeline.scheduler.config),
    }
    metadata["started_at_unix"] = time.time()

    prompt = _load_prompt("topdown")
    negative = _load_prompt("topdown_negative")
    for condition_name in CONDITIONS:
        control_image = Image.open(
            prepare["conditions"][condition_name]["path"]
        ).convert("RGB")
        candidate_root = output_root / "review" / condition_name
        candidate_root.mkdir(parents=True, exist_ok=True)
        for seed in SETTINGS["seeds"]:
            path = candidate_root / f"{condition_name}_s{seed}.png"
            started = time.perf_counter()
            reused = path.is_file() and not force
            previous_record = previous_by_name.get(path.stem, {})
            peak_memory: int | None = previous_record.get(
                "peak_gpu_memory_bytes"
            )
            if not reused:
                torch.cuda.reset_peak_memory_stats()
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=control_image,
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
                    "condition": condition_name,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "size": list(Image.open(path).size),
                    "inference_seconds": (
                        previous_record.get("inference_seconds")
                        if reused and previous_record
                        else round(time.perf_counter() - started, 3)
                    ),
                    "peak_gpu_memory_bytes": peak_memory,
                    "reused": reused,
                }
            )
            _write_json(metadata_path, metadata)

    metadata["completed_at_unix"] = time.time()
    metadata["total_generation_seconds"] = round(
        metadata["completed_at_unix"] - metadata["started_at_unix"], 3
    )
    metadata["status"] = "generated"
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
    for condition in CONDITIONS:
        records = [
            record for record in candidates if record["condition"] == condition
        ]
        _contact_sheet(
            [
                (
                    f"{condition} | seed {record['seed']}",
                    Path(record["path"]),
                )
                for record in records
            ],
            output_root / "review" / condition / "contact-sheet.jpg",
            columns=2,
        )
    reference_records = [
        record
        for record in candidates
        if record["condition"] == "reference_canny"
    ]
    _contact_sheet(
        [
            (
                "source image | free SDXL seed 3103",
                output_root / "_work" / "source" / "reference_image.png",
            ),
            *[
                (
                    f"ControlNet from its Canny | seed {record['seed']}",
                    Path(record["path"]),
                )
                for record in reference_records
            ],
        ],
        output_root / "reference-reconstruction.jpg",
        columns=2,
    )
    paired = []
    by_key = {
        (record["condition"], record["seed"]): record for record in candidates
    }
    for seed in SETTINGS["seeds"]:
        for condition in CONDITIONS:
            record = by_key[(condition, seed)]
            paired.append(
                (
                    f"{condition} | seed {seed}",
                    Path(record["path"]),
                )
            )
    _contact_sheet(
        paired,
        output_root / "comparison-labeled.jpg",
        columns=3,
    )
    rng = np.random.default_rng(1910)
    blind_order = list(rng.permutation(len(candidates)))
    blind_records = [candidates[index] for index in blind_order]
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
            "condition": record["condition"],
            "seed": record["seed"],
        }
        for index, record in enumerate(blind_records)
    }
    _write_json(output_root / "_work" / "blind_order.json", mapping)
    return mapping


def _review_text(output_root: Path) -> str:
    review_path = output_root / "_work" / "review.json"
    if not review_path.is_file():
        return "尚未完成审图。"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    paired_findings = "\n".join(
        f"- {finding}" for finding in review["paired_findings"]
    )
    return f"""### 总结

{review["conclusion"]}

- 稀疏 Canny：{review["sparse_canny"]}
- 详细线图：{review["detailed_lineart"]}
- 参考图实边 Canny：{review["reference_canny"]}
- 根因判断：{review["root_cause"]}

### 盲评与逐组复核

{review["blind_review"]}

{paired_findings}

- 稀疏组相对最好：`{review["best_sparse"]}`，但不合格。
- 详细组相对最好：`{review["best_detailed"]}`，但不合格。
- 参考边缘组代表图：`{review["best_reference"]}`，用于验证还原能力，不是目标构图。

### 这次实验能证明什么、不能证明什么

{review["scope"]}

### 下一步

{review["next_step"]}"""


def write_report(output_root: Path = OUTPUT_ROOT) -> Path:
    prepare = json.loads(
        (output_root / "_work" / "prepare_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = json.loads(
        (output_root / "_work" / "metadata.json").read_text(encoding="utf-8")
    )
    rows = [
        "| 文件 | 控制图 | seed | 秒 | 本次调用 | SHA-256 |",
        "|---|---|---:|---:|---|---|",
    ]
    for record in metadata["candidates"]:
        run_status = "复用原图" if record["reused"] else "新生成"
        rows.append(
            f"| `{Path(record['path']).name}` | {record['condition']} | "
            f"{record['seed']} | {record['inference_seconds']:.3f} | "
            f"{run_status} | "
            f"`{record['sha256']}` |"
        )
    weights = []
    for model_id, files in metadata["model_fingerprints"]["models"].items():
        weights.append(f"- `{model_id}`")
        for filename, digest in files.items():
            weights.append(f"  - `{filename}`: `{digest}`")
    tokens = metadata["prompt_token_preflight"]
    edge_ratio = (
        prepare["conditions"]["detailed_lineart"]["edge_pixels"]
        / prepare["conditions"]["sparse_canny"]["edge_pixels"]
    )
    reused_count = sum(
        bool(record["reused"]) for record in metadata["candidates"]
    )
    generated_count = len(metadata["candidates"]) - reused_count
    report = f"""# ControlNet 线稿对照实验

## 结论

{_review_text(output_root)}

## 1. 这个实验回答什么

旧 Route B 同时使用了 `smooth_base` img2img 和 Canny，因此无法区分结果差是因为平滑
底图，还是因为控制线条太少。本实验移除 img2img，只运行
`StableDiffusionXLControlNetPipeline`，并且只改变控制图：

- A：原来的 `sparse_canny.png`；
- B：新生成的 `detailed_lineart.png`；
- C：从已有自然航拍图逐像素执行 Canny 得到的 `reference_canny.png`。

模型、prompt、negative prompt、四个 seed、分辨率、步数、CFG 和 ControlNet scale
全部相同。没有 strength，因为本实验没有 img2img 初始图；没有 mask projection。
其中 C 最接近网上“从已有图像提取线条再重绘”的 demo，但它的源图不是合格目标构图，
所以只检验 ControlNet 能否利用真实图像边缘，不参与最终地理构图优选。

## 2. 三张控制图

先看 `source-comparison.jpg`。

### A：稀疏 Canny

路径：`{prepare["conditions"]["sparse_canny"]["path"]}`

- edge pixels：{prepare["conditions"]["sparse_canny"]["edge_pixels"]}
- edge fraction：{prepare["conditions"]["sparse_canny"]["edge_fraction"]}
- SHA-256：`{prepare["conditions"]["sparse_canny"]["sha256"]}`
- 内容：原 96×64 机制陆地掩码放大后得到的笔直海岸和矩形河道边缘。

### B：详细合成语义线图

路径：`{prepare["conditions"]["detailed_lineart"]["path"]}`

- edge pixels：{prepare["conditions"]["detailed_lineart"]["edge_pixels"]}
- edge fraction：{prepare["conditions"]["detailed_lineart"]["edge_fraction"]}
- SHA-256：`{prepare["conditions"]["detailed_lineart"]["sha256"]}`
- 内容：自然曲线海岸、单一缓弯河道、陆地等高线、近岸水深线，以及向海域扩张的
  五条泥沙羽流流线。

它的边缘像素是 A 的 {edge_ratio:.1f} 倍。这里故意称它为“合成语义线图”，而不是
“高质量线稿”：它由 `detailed_lineart()` 确定性生成，没有调用另一个图像模型，也
不是人工挑选后隐藏来源的图片；但其中的等高线、水深线和羽流流线只是我们希望表达的
概念，并不都是最终照片中真实可见的物体边缘。这个区别直接影响实验结果。

### C：自然参考图的真实 Canny 边缘

源图路径：`{prepare["reference_image"]["path"]}`

- 源图 SHA-256：`{prepare["reference_image"]["sha256"]}`
- Canny 路径：`{prepare["conditions"]["reference_canny"]["path"]}`
- Canny edge pixels：{prepare["conditions"]["reference_canny"]["edge_pixels"]}
- Canny edge fraction：{prepare["conditions"]["reference_canny"]["edge_fraction"]}
- Canny SHA-256：`{prepare["conditions"]["reference_canny"]["sha256"]}`
- 提取方法：{prepare["reference_image"]["canny_method"]}

源图是前一轮纯 SDXL 生成的 `free_topdown_s3103.png`。它有连续、自然的真实可见岸线
和地表纹理，但实际是斜视狭长水道，不符合本阶段的“左侧河流进入开阔海域”。选择它
不是为了偷换目标，而是为了复刻网上 demo 的前提：控制边缘来自一幅视觉上已经成立的
图像。

## 3. 为什么网上的 Canny demo 看起来更好

典型 Canny demo 的控制图通常来自一张已经有合理构图、透视、轮廓和物体比例的参考图。
Canny 只删除颜色和纹理，却保留了这些真实可见边缘；扩散模型的工作主要是把纹理和
风格重新填回去。

本实验的 A 来自 96×64 机制图，只有笔直海岸和矩形河道，缺少足够几何信息。B 虽然
线更多，却加入了地图式等高线、水深线和羽流内部流线。Canny ControlNet 不知道哪条线
是“仅供理解的说明”，会把它们都当成应该出现在成图里的强边界，于是生成沟槽、额外
水道和地质分界。悬浮泥沙本来是软密度场，也没有适合用 Canny 表达的硬轮廓。

所以真正的差别不是简单的“有没有线稿”，而是控制图是否来自一幅视觉上已经成立的
目标构图，以及其中每条边是否应该成为成图的可见边缘。C 组就是对此判断的直接检查。

## 4. 实际提示词

正向 prompt（两个 tokenizer 均为
{tokens["topdown"]["counts_including_special_tokens"]["tokenizer"]}/77 tokens）：

```text
{prepare["prompt"]["text"]}
```

负向 prompt（两个 tokenizer 均为
{tokens["topdown_negative"]["counts_including_special_tokens"]["tokenizer"]}/77 tokens）：

```text
{prepare["negative_prompt"]["text"]}
```

## 5. 固定推理参数

- pipeline：`{SETTINGS["pipeline"]}`
- size：{SETTINGS["width"]}×{SETTINGS["height"]}
- steps：{SETTINGS["steps"]}
- guidance scale：{SETTINGS["guidance_scale"]}
- ControlNet scale：{SETTINGS["controlnet_conditioning_scale"]}
- seeds：{SETTINGS["seeds"]}
- scheduler：`{metadata["scheduler"]["class"]}`
- img2img initial image：无
- strength：无
- model load：{metadata["model_load_seconds"]:.3f} 秒
- 本次调用 generation：{metadata["total_generation_seconds"]:.3f} 秒
- 本次调用新生成 / 复用：{generated_count} / {reused_count} 张
- GPU：`{metadata["runtime"]["gpu"]}`
- HIP：`{metadata["runtime"]["hip"]}`

这里的 total 只统计最后一次增量调用：A/B 两组已在上一批实际生成并通过哈希复用，
本次新增 C 组四张。表中每张的秒数保留其实际生成时记录；使用复现命令中的 `--force`
会在一次调用中重新生成全部 12 张。

## 6. 全部候选

{chr(10).join(rows)}

查看方式：

- `comparison-blind.jpg`：12 张随机排序，不显示控制图名称；
- `comparison-labeled.jpg`：每行一个 seed，依次为稀疏、详细、参考边缘；
- `reference-reconstruction.jpg`：自然源图与四张参考边缘重建结果；
- `review/sparse_canny/contact-sheet.jpg`：稀疏控制内部对比；
- `review/detailed_lineart/contact-sheet.jpg`：详细线图内部对比；
- `review/reference_canny/contact-sheet.jpg`：真实参考边缘内部对比。

## 7. 模型权重

{chr(10).join(weights)}

## 8. 复现

从仓库根目录运行：

```bash
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.controlnet_line_test \\
  --prepare
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.controlnet_line_test \\
  --generate --force
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.controlnet_line_test \\
  --report
```

`_work/prepare_manifest.json` 保存控制图生成内容和哈希，`_work/metadata.json` 保存每张
候选的参数、耗时和哈希，`_work/review.json` 保存人工审图结论。
"""
    report_path = output_root / "report.md"
    report_path.write_text(report, encoding="utf-8")
    (output_root / "report.html").write_text(
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>ControlNet 线稿对照实验</title>
<style>body{max-width:1100px;margin:32px auto;padding:0 24px;background:#f7fafb;
color:#172127;font:16px/1.65 system-ui,sans-serif}pre{white-space:pre-wrap;
background:white;padding:28px;border-radius:12px;box-shadow:0 2px 16px #0001}
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
        result = prepare_conditions(args.output)
    if args.generate:
        result = generate(args.output, force=args.force)
    if args.report:
        result = {"report": str(write_report(args.output).resolve())}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
