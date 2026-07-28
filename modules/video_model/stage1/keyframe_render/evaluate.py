"""Evaluate prepared/projected keyframes and report model availability honestly."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .enhance import probe_environment
from .prepare import KEYFRAME_OUTPUT_ROOT, WORK_DIR_NAME


def _visual_metrics(path: Path) -> dict[str, float]:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    gray = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    gradient_x = np.abs(np.diff(gray, axis=1)).mean()
    gradient_y = np.abs(np.diff(gray, axis=0)).mean()
    return {
        "mean_luminance": round(float(gray.mean()), 6),
        "contrast_std": round(float(gray.std()), 6),
        "mean_gradient": round(float((gradient_x + gradient_y) / 2.0), 6),
    }


def _candidate_sheet(
    records: list[dict[str, Any]],
    path_key: str,
    output: Path,
) -> None:
    tile_width, tile_height = 384, 256
    columns = min(3, len(records))
    rows = int(np.ceil(len(records) / columns))
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), "#10242c")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    for index, record in enumerate(records):
        image_path = record[path_key]
        image = Image.open(image_path).convert("RGB").resize(
            (tile_width, tile_height), Image.Resampling.LANCZOS
        )
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((7, 7, 278, 31), fill=(5, 22, 29, 215))
        label = f"{record['candidate']} / {record['keyframe']}"
        draw.text((13, 10), label, font=font, fill="white")
        sheet.paste(
            image,
            ((index % columns) * tile_width, (index // columns) * tile_height),
        )
    sheet.save(output, quality=94, subsampling=0)


def _pair_stability(
    first_record: dict[str, Any],
    last_record: dict[str, Any],
    prepare_manifest: dict[str, Any],
) -> dict[str, Any]:
    first = np.asarray(Image.open(first_record["output"]).convert("RGB"), dtype=np.float64)
    last = np.asarray(Image.open(last_record["output"]).convert("RGB"), dtype=np.float64)
    categories = list(prepare_manifest["keyframes"]["first"]["masks"])
    first_labels = np.full(first.shape[:2], -1, dtype=np.int16)
    last_labels = np.full(last.shape[:2], -2, dtype=np.int16)
    for label, category in enumerate(categories):
        first_path = prepare_manifest["keyframes"]["first"]["masks"][category]["path"]
        last_path = prepare_manifest["keyframes"]["last"]["masks"][category]["path"]
        first_labels[np.asarray(Image.open(first_path).convert("L")) > 127] = label
        last_labels[np.asarray(Image.open(last_path).convert("L")) > 127] = label
    stable = first_labels == last_labels
    absolute_difference = np.abs(first - last) / 255.0
    return {
        "stable_pixels": int(stable.sum()),
        "stable_fraction": round(float(stable.mean()), 6),
        "mean_absolute_rgb_difference": round(
            float(absolute_difference[stable].mean()), 6
        ),
        "interpretation": "lower is more style-consistent on unchanged semantic regions",
    }


def evaluate(output_root: Path = KEYFRAME_OUTPUT_ROOT) -> dict[str, Any]:
    work_root = output_root / WORK_DIR_NAME
    final_root = output_root / "final"
    review_root = output_root / "review"
    final_root.mkdir(parents=True, exist_ok=True)
    review_root.mkdir(parents=True, exist_ok=True)
    prepare_manifest = json.loads(
        (work_root / "prepare_manifest.json").read_text(encoding="utf-8")
    )
    model_status = probe_environment(output_root)
    projected_dir = work_root / "constrained_candidates"
    projection_manifests = (
        sorted(projected_dir.glob("*.json")) if projected_dir.is_dir() else []
    )
    candidates = []
    projection_records: list[dict[str, Any]] = []
    for manifest_path in projection_manifests:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        projection_records.append(record)
        output_path = Path(record["output"])
        categories = record["projection"]["categories"]
        candidates.append(
            {
                "candidate": record["candidate"],
                "keyframe": record["keyframe"],
                "path": str(output_path.resolve()),
                "geometry": {
                    "mask_source": "mechanism state",
                    "exclusive_exhaustive": record["projection"][
                        "exclusive_exhaustive_masks"
                    ],
                    "proposal_color_layout_used": record["projection"][
                        "proposal_color_layout_used"
                    ],
                    "boundary_weight_max": max(
                        value["boundary_model_weight"] for value in categories.values()
                    ),
                    "mask_sha256": record["mask_sha256"],
                },
                "visual_metrics": _visual_metrics(output_path),
            }
        )
    by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for manifest_path in projection_manifests:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_candidate.setdefault(record["candidate"], {})[record["keyframe"]] = record
    pair_stability = {
        candidate: _pair_stability(records["first"], records["last"], prepare_manifest)
        for candidate, records in by_candidate.items()
        if "first" in records and "last" in records
    }
    tail_records = [
        record for record in projection_records if record["keyframe"] == "last"
    ]
    tail_candidates = [
        candidate for candidate in candidates if candidate["keyframe"] == "last"
    ]
    status = (
        "selected_pair_ready"
        if pair_stability
        else (
            "candidates_ready_for_style_review"
            if tail_candidates
            else "blocked_pending_local_sdxl_weights"
        )
    )
    contact_sheets: dict[str, str] = {}
    if tail_records:
        proposal_sheet = review_root / "raw-style-proposals.jpg"
        projected_sheet = review_root / "constrained-style-candidates.jpg"
        _candidate_sheet(tail_records, "proposal", proposal_sheet)
        _candidate_sheet(tail_records, "output", projected_sheet)
        contact_sheets = {
            "raw_style_proposals": str(proposal_sheet.resolve()),
            "constrained_style_candidates": str(projected_sheet.resolve()),
        }
    selected_candidate: str | None = None
    if pair_stability:
        selected_candidate = sorted(pair_stability)[0]
        pair_records = by_candidate[selected_candidate]
        proposal_pair_sheet = review_root / "selected-pair-proposals.jpg"
        projected_pair_sheet = final_root / "comparison.jpg"
        ordered_pair = [pair_records["first"], pair_records["last"]]
        _candidate_sheet(ordered_pair, "proposal", proposal_pair_sheet)
        _candidate_sheet(ordered_pair, "output", projected_pair_sheet)
        shutil.copy2(pair_records["first"]["output"], final_root / "first.png")
        shutil.copy2(pair_records["last"]["output"], final_root / "last.png")
        contact_sheets.update(
            {
                "selected_pair_proposals": str(proposal_pair_sheet.resolve()),
                "final_comparison": str(projected_pair_sheet.resolve()),
            }
        )
    result = {
        "status": status,
        "prepared_keyframes": {
            name: {
                "display_frame": record["display_frame"],
                "state_frame": record["state_frame"],
                "beat_id": record["beat_id"],
                "clean_base_metrics": _visual_metrics(Path(record["clean_base"])),
            }
            for name, record in prepare_manifest["keyframes"].items()
        },
        "model_status": model_status,
        "projected_candidates": candidates,
        "pair_stability": pair_stability,
        "contact_sheets": contact_sheets,
        "selected_candidate": selected_candidate,
        "geometry_gate": {
            "passed": all(
                candidate["geometry"]["exclusive_exhaustive"]
                and not candidate["geometry"]["proposal_color_layout_used"]
                and candidate["geometry"]["boundary_weight_max"] <= 0.14
                for candidate in candidates
            )
            if candidates
            else None,
            "reason": (
                "Every output pixel is assigned by one mechanism-derived semantic mask; "
                "model contribution is reduced to 0.14 at boundaries."
            ),
        },
        "human_style_selection_required": bool(tail_candidates) and not bool(pair_stability),
    }
    (work_root / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    missing = (
        ", ".join(model_status["missing_models"] + model_status["missing_packages"])
        or "none"
    )
    candidate_rows = "\n".join(
        f"| `{candidate['candidate']}` | {candidate['visual_metrics']['mean_gradient']:.6f} | "
        f"{candidate['visual_metrics']['contrast_std']:.6f} |"
        for candidate in tail_candidates
    )
    generation_manifest_path = work_root / "raw_proposals" / "candidate_manifest.json"
    generation_manifest = (
        json.loads(generation_manifest_path.read_text(encoding="utf-8"))
        if generation_manifest_path.is_file()
        else {}
    )
    generation_runtime = generation_manifest.get("runtime", {})
    model_rows = "\n".join(
        f"| `{record['model_id']}` | `{record['path']}` | "
        f"{'可用' if record['available'] else '缺失'} |"
        for record in model_status["required_models"].values()
    )
    package_rows = "\n".join(
        f"| `{name}` | `{version or 'missing'}` |"
        for name, version in model_status.get("package_versions", {}).items()
    )
    fingerprint_path = work_root / "model_fingerprints.json"
    fingerprints = (
        json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if fingerprint_path.is_file()
        else {}
    )
    fingerprint_rows = "\n".join(
        f"| `{model_id}` | `{relative_path}` | `{digest}` |"
        for model_id, files in fingerprints.get("models", {}).items()
        for relative_path, digest in files.items()
    )
    prompt_root = Path(__file__).with_name("prompts")
    negative_prompt = (prompt_root / "negative.txt").read_text(encoding="utf-8").strip()
    selected_style = (
        selected_candidate.rsplit("_s", 1)[0] if selected_candidate else "尚未选择"
    )
    selected_seed = (
        selected_candidate.rsplit("_s", 1)[1] if selected_candidate else "尚未选择"
    )
    selected_prompt_path = prompt_root / f"{selected_style}.txt"
    selected_prompt = (
        selected_prompt_path.read_text(encoding="utf-8").strip()
        if selected_prompt_path.is_file()
        else "尚未选择"
    )
    if pair_stability and selected_candidate:
        selected_stability = pair_stability[selected_candidate]
        status_section = f"""## 9. 本次选择和检查结果

- 选定候选：`{selected_candidate}`，即风格 `{selected_style}`、seed `{selected_seed}`。
- 生成模型：SDXL Base 1.0 FP16 + SDXL Canny ControlNet FP16。
- 尾帧比较了 {len(tail_candidates)} 个候选（三种提示风格 × 两个 seed）。
- 几何门禁：`{"通过" if result["geometry_gate"]["passed"] else "未通过"}`。
- 首尾帧未变化语义区域占比：`{selected_stability["stable_fraction"]}`。
- 这些未变化区域的平均 RGB 差异：`{selected_stability["mean_absolute_rgb_difference"]}`；
  数值越低，表示首尾帧风格越一致。
- 最终首帧 SHA-256：`{pair_records["first"]["output_sha256"]}`。
- 最终尾帧 SHA-256：`{pair_records["last"]["output_sha256"]}`。

| 候选 | 平均梯度 | 对比度 |
|---|---:|---:|
{candidate_rows}

`平均梯度` 和 `对比度` 只用于发现过度模糊或异常平坦的图片，并不自动决定哪张最好。
最终风格是根据约束后联系表人工选择的，判断重点是边界清晰、纹理克制、沉积与新陆地
仍能区分。选中风格后，才用相同 prompt、seed 和推理参数生成首帧。
"""
    elif tail_candidates:
        status_section = f"""## 9. 当前候选

- 已用指定的 SDXL Base 1.0 FP16 + SDXL Canny ControlNet FP16 生成 {len(tail_candidates)} 张尾帧候选。
- 三种风格各 2 个 seed；原始模型图和受机制约束后的图都保存在内部工作目录。
- 几何门禁：`{"通过" if result["geometry_gate"]["passed"] else "未通过"}`。
- 尚未生成选定风格的首帧，因此目前只有尾帧候选。

| 候选 | 平均梯度 | 对比度 |
|---|---:|---:|
{candidate_rows}
"""
    else:
        status_section = f"""## 9. 当前阻塞

缺少：`{missing}`。

按 Stage 1 约束，未静默替换 SDXL Base 1.0 FP16 或 SDXL Canny ControlNet FP16，
因此当前没有伪造“模型增强成功”的候选图。将本地权重放到 `_work/model_status.json` 中列出的
路径（或设置 `SDXL_BASE_PATH` / `SDXL_CANNY_CONTROLNET_PATH`）并安装对应运行时后，
运行 `python -m modules.video_model.stage1.keyframe_render.enhance --generate-candidates`。
"""
    final_section = (
        """- [最终首帧](final/first.png)
- [最终尾帧](final/last.png)
- [首尾对比图](final/comparison.jpg)"""
        if pair_stability
        else "首尾帧对尚未生成，因此当前没有 `final/` 成品。"
    )
    review_section = (
        """- [模型原始尾帧候选](review/raw-style-proposals.jpg)：查看 SDXL 原本生成了什么。
- [受机制约束的尾帧候选](review/constrained-style-candidates.jpg)：比较可用风格。
- [选定风格的原始首尾提案](review/selected-pair-proposals.jpg)：检查约束前后的差异。"""
        if pair_stability
        else (
            """- [模型原始尾帧候选](review/raw-style-proposals.jpg)
- [受机制约束的尾帧候选](review/constrained-style-candidates.jpg)"""
            if tail_candidates
            else "模型候选尚未生成。"
        )
    )
    report = f"""# Stage 1 Track B 报告：高质量关键帧生成

状态：`{status}`

## 1. 这份报告说明什么

Track A 先用机制模型算出了三角洲变化，但程序渲染的画面纹理较简单。Track B 的任务是：

1. 从已经验证的机制状态中取出首尾两个关键帧；
2. 用 SDXL + Canny ControlNet 提议更自然的视觉纹理；
3. 用机制生成的语义区域图把模型限制回原来的海岸线、河道和沉积范围；
4. 比较六个尾帧候选，选定风格后再生成同风格首帧；
5. 输出可供下一阶段动画合成使用的首尾图片。

这意味着模型只负责视觉增强，不能决定“哪里是陆地”或“河道如何分流”。

最终结果：

{final_section}

完整的机制动画生成过程见
[Track A 报告](../causal_delta/report.md)。

## 2. 全流程概览

```text
Track A 的 states.jsonl + timeline.json
  ↓ prepare.py：按教学阶段名称寻找首尾状态
base_images：去掉箭头和文字的机制底图
  ├─ control_edges：底图的 Canny 边缘，作为 ControlNet 输入
  └─ semantic_masks：五类机制区域，作为生成后的硬约束
  ↓ enhance.py：SDXL img2img + Canny ControlNet
raw_proposals：模型原始提案，只用于提供纹理，不是成品
  ↓ project_texture：按 semantic_masks 投回底图
constrained_candidates：几何仍由机制决定的候选
  ↓ 人工比较六个尾帧候选，选定 physical_geography_s3102
用同一 prompt、seed、参数生成首帧
  ↓ evaluate.py：几何门禁、首尾稳定性、联系表
final/first.png + final/last.png
```

对应代码：

| 文件 | 负责的步骤 |
|---|---|
| [`prepare.py`](../../keyframe_render/prepare.py) | 选状态、画底图、生成边缘图和语义区域图 |
| [`enhance.py`](../../keyframe_render/enhance.py) | 模型推理、纹理投影、候选与首尾帧生成 |
| [`evaluate.py`](../../keyframe_render/evaluate.py) | 指标、几何门禁、联系表、最终文件和报告 |
| [`prompts/`](../../keyframe_render/prompts/) | 三种正向风格提示词和公共负向提示词 |

## 3. 首尾关键帧从哪里来

输入不是动画 PNG 的截图，而是
[`states.jsonl`](../causal_delta/mechanism/states.jsonl) 中的机制状态。
`prepare.py` 读取 [`timeline.json`](../causal_delta/timeline.json)，按阶段名称取每个阶段
最后一个状态，避免以后时间线长度改变时选错帧：

| 名称 | 查找的教学阶段 | 展示帧 | 机制状态 | 选择原因 |
|---|---|---:|---:|---|
| 首帧 | `accumulate` | {prepare_manifest["keyframes"]["first"]["display_frame"]} | {prepare_manifest["keyframes"]["first"]["state_frame"]} | 水下沉积已形成，但新陆地尚未出水 |
| 尾帧 | `threshold_change` | {prepare_manifest["keyframes"]["last"]["display_frame"]} | {prepare_manifest["keyframes"]["last"]["state_frame"]} | 沙洲已经出水，适合表现阶段差异 |

每个状态重新绘制为 768×512 的 `base_images`。这一版底图关闭新陆地黄色网格边，
并去除流向箭头、悬沙粒子、标题、图例、字幕、进度点和底栏，只保留地貌语义。

## 4. control edges 和 semantic masks 到底是什么

### control edges

`control_edges` 是把底图转为灰度后，用 OpenCV Canny 阈值 100/200 得到的黑白边缘图。
它作为 `control_image` 送入 Canny ControlNet，告诉扩散模型主要边缘应当在哪里。
它不是最终效果图，也不会叠到最终图片上。

### semantic masks

`semantic_masks` 是从机制状态直接计算的黑白位置图，不是 AI 猜出来的标注：

| 区域 | 计算方式 | 为什么需要 |
|---|---|---|
| `original_land` | 模拟开始前的固定陆地 | 保持原始海岸和陆地范围 |
| `river` | 海岸线以内且不是陆地的水格 | 防止模型把河道填成陆地 |
| `ocean` | 不属于其他四类的水格 | 保持开阔水域 |
| `underwater_deposit` | 厚度 > 0.001、尚未出水且位于海岸外 | 保留水下沉积范围 |
| `new_land` | 机制状态中已经出水的格子 | 锁定新生陆地形状 |

五张 mask 必须互不重叠，并且合起来覆盖画面的每一个像素。程序会验证覆盖计数恰好为
1；有重叠或空洞就停止，而不是继续输出。

ControlNet 的边缘约束只能减少漂移，不能保证完全不漂移；semantic masks 是生成之后
执行的第二道硬约束。两者用途不同，所以都需要保留在复现材料中。

## 5. 模型、环境和精确参数

本次没有使用替代模型：

| 模型 ID | 本地路径 | 状态 |
|---|---|---|
{model_rows}

本次推理环境：

| 软件 | 版本 |
|---|---|
{package_rows}
| `GPU` | `{generation_runtime.get("gpu", "未记录")}` |
| `HIP/CUDA runtime` | `{generation_runtime.get("hip", "未记录")}` |

模型参数：

| 参数 | 值 | 作用 |
|---|---:|---|
| 输出尺寸 | 768×512 | 与机制底图一致 |
| dtype | FP16 | 降低显存占用 |
| 推理步数 | 36 | 扩散去噪步数 |
| img2img strength | 0.50 | 保留底图结构同时允许生成纹理 |
| guidance scale | 6.5 | 提示词引导强度 |
| ControlNet scale | 1.35 | Canny 边缘约束强度 |
| 尾帧 seeds | 3101、3102 | 每种风格生成两个候选 |
| 选定 seed | {selected_seed} | 首尾帧使用同一 seed |

选定的正向提示词来自
[`physical_geography.txt`](../../keyframe_render/prompts/physical_geography.txt)：

```text
{selected_prompt}
```

公共负向提示词来自
[`negative.txt`](../../keyframe_render/prompts/negative.txt)：

```text
{negative_prompt}
```

为识别本次实际使用的权重，五个主要 FP16 权重文件已记录 SHA-256：

| 模型 | 权重文件 | SHA-256 |
|---|---|---|
{fingerprint_rows or "| 未记录 | 未记录 | 未记录 |"}

`--generate-candidates` 和 `--generate-pair` 会在实际推理前重新计算这些哈希；
若必需权重文件缺失，会直接停止。

机器可读版本见
[`_work/model_status.json`](_work/model_status.json) 和
[`_work/model_fingerprints.json`](_work/model_fingerprints.json)。

## 6. 六个模型候选是怎样生成的

先只生成尾帧，因为尾帧同时包含原始陆地、河道、海洋、水下沉积和新生陆地，
最适合暴露模型是否改变结构。三种风格分别是：

1. `physical_geography`：物理地理教材插图；
2. `museum_infographic`：博物馆科学信息图；
3. `remote_sensing`：遥感图风格。

每种风格使用 seed 3101、3102，共六张。模型调用实际收到四类主要输入：

```text
prompt          = 对应风格提示词
negative_prompt = 公共负向提示词
image           = _work/base_images/last.png
control_image   = _work/control_edges/last_canny.png
```

模型原始输出保存在 `_work/raw_proposals/`。这些图片可能改变颜色布局或地形，
所以明确标为 proposal，不直接进入 `final/`。

## 7. 模型纹理如何被限制回机制地形

`project_texture` 对每张原始提案执行以下处理：

1. 把模型提案缩放到机制底图尺寸。
2. 对提案做半径 7 的高斯模糊。
3. 用“原提案 − 模糊提案”取得中高频纹理残差。
4. 在每个 semantic mask 内计算提案的中位亮度，只允许 0.82–1.18 范围的局部明暗变化。
5. 每张 mask 向内腐蚀 4 像素，区分区域内部和边界。
6. 区域内部使用 0.62 的纹理权重；边界降到 0.14，减少跨岸线污染。
7. 逐 mask 合成；机制底图继续提供颜色类别和几何，模型的整体颜色布局被丢弃。

简化后的计算是：

```text
texture_residual = proposal - gaussian_blur(proposal, radius=7)
shading = clip(local_luminance / region_median_luminance, 0.82, 1.18)
textured = base × shading + 0.72 × texture_residual
final_pixel = base × (1 - weight) + textured × weight

weight = 0.62  # 区域内部
weight = 0.14  # 四像素边界带
```

所以最终图可以借用模型的细节和明暗，但岸线、新陆地和沉积区域仍来自机制 mask。
约束后候选保存在 `_work/constrained_candidates/`。

## 8. 为什么选 `{selected_candidate}`

六张尾帧先排成：

- [模型原始候选联系表](review/raw-style-proposals.jpg)
- [约束后候选联系表](review/constrained-style-candidates.jpg)

选择不是由单一数值指标自动完成。本次人工比较的标准是：

- 海岸线和河道清楚；
- 不出现模型虚构的岛屿、文字或箭头；
- 陆地、水下沉积和新生陆地仍可区分；
- 纹理比程序底图自然，但不过分写实或抢夺教学信息；
- 适合后续重新叠加机制箭头、粒子和字幕。

按这些标准选定 `{selected_candidate}`。之后用完全相同的模型、正负提示词、
seed `{selected_seed}` 和推理参数分别生成尾帧与首帧，再执行同一套纹理投影。
原始首尾提案可在
[selected-pair-proposals.jpg](review/selected-pair-proposals.jpg) 对照。

{status_section}

几何门禁要求所有 semantic masks 互斥且完备、模型颜色布局未进入输出、边界模型权重
不超过 0.14。稳定性指标只比较首尾帧中语义类别没有变化的像素；它衡量风格一致性，
不代替人工检查或科学验证。

## 10. 从零复现

所有命令都从仓库根目录 `Live-Document/` 执行。

### 10.1 先复现 CPU 机制状态

```bash
python -m venv .venv
.venv/bin/python -m pip install -r modules/video_model/stage1/requirements.txt
.venv/bin/python -m modules.video_model.stage1.causal_delta.simulate
.venv/bin/python -m modules.video_model.stage1.causal_delta.validate
.venv/bin/python -m modules.video_model.stage1.causal_delta.export
```

必须先看到 `validation.json` 中的 `passed: true`，否则不要生成关键帧。

### 10.2 准备 GPU 环境和模型

先安装与你的 NVIDIA CUDA 或 AMD ROCm 环境匹配的 PyTorch；PyTorch 的安装命令取决于
显卡和驱动，不能跨机器统一。下面的 `/opt/venv/bin/python` 是本次服务器实际使用的
ROCm 环境；其他机器应替换成自己的 GPU Python。然后在同一个环境安装图像库版本：

```bash
MODEL_PYTHON=/opt/venv/bin/python
$MODEL_PYTHON -m pip install \\
  diffusers==0.35.2 transformers==4.57.6 accelerate==1.13.0 \\
  safetensors==0.7.0 opencv-python-headless==4.13.0.92
```

准备以下两个 Hugging Face 模型的 FP16 Diffusers 目录：

```text
stabilityai/stable-diffusion-xl-base-1.0
diffusers/controlnet-canny-sdxl-1.0
```

如果本机还没有权重，可用 Hugging Face CLI 下载完整仓库；代码会明确选择其中的
`variant="fp16"` 文件。完整快照占用空间较大，应先确认磁盘容量：

```bash
$MODEL_PYTHON -m pip install huggingface_hub
$MODEL_PYTHON -m huggingface_hub.cli.hf download \\
  stabilityai/stable-diffusion-xl-base-1.0 \\
  --local-dir /absolute/path/to/sdxl-base-1.0
$MODEL_PYTHON -m huggingface_hub.cli.hf download \\
  diffusers/controlnet-canny-sdxl-1.0 \\
  --local-dir /absolute/path/to/controlnet-canny-sdxl-1.0
```

默认路径是本报告第 5 节列出的路径。也可以放在其他位置并设置：

```bash
export SDXL_BASE_PATH=/absolute/path/to/sdxl-base-1.0
export SDXL_CANNY_CONTROLNET_PATH=/absolute/path/to/controlnet-canny-sdxl-1.0
```

检查环境；输出必须是 `status: ready`，并且 `substitution_used: false`：

```bash
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.prepare
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.enhance --status-only
```

### 10.3 重新生成六个尾帧候选

```bash
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.enhance \\
  --generate-candidates --force
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.evaluate
```

此时查看 `review/raw-style-proposals.jpg` 和
`review/constrained-style-candidates.jpg`。如果目标是复现本次选择，继续使用
`physical_geography` 和 seed `{selected_seed}`；如果要重新选风格，应先记录选择理由。

### 10.4 生成选定风格的首尾帧并评估

```bash
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.enhance \\
  --generate-pair --selected-style physical_geography --seed {selected_seed} --force
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.evaluate
$MODEL_PYTHON -m pytest -q modules/video_model/stage1
```

最后一条 `evaluate` 会把选中的约束后图片复制到 `final/first.png` 和
`final/last.png`，并重建联系表、指标和本报告。

注意：`python -m modules.video_model.stage1.run` 会重跑 Track A、准备关键帧并评估
已有模型结果，但为了避免意外占用 GPU，它不会自动执行
`--generate-candidates` 或 `--generate-pair`。

## 11. 输出文件和删除建议

| 目录或文件 | 实际用途 | 最终成品需要 | 能否删除 |
|---|---|---:|---|
| `final/first.png` | 动画合成使用的首帧，768×512 | 是 | 否 |
| `final/last.png` | 动画合成使用的尾帧，768×512 | 是 | 否 |
| `final/comparison.jpg` | 带标签的首尾快速对比，不是合成输入 | 复核时 | 可以 |
| `review/` | 原始提案和候选联系表 | 否 | 可以；不影响 final |
| `_work/base_images/` | SDXL img2img 的机制底图 | 重跑时 | 可以；`prepare` 会重建 |
| `_work/control_edges/` | ControlNet 的 Canny 输入 | 重跑时 | 可以；`prepare` 会重建 |
| `_work/semantic_masks/` | 几何约束和审计依据 | 重跑/审计时 | 可以；但会失去直接证据 |
| `_work/raw_proposals/` | 六张模型原始尾帧和生成记录 | 重新选风格时 | 可以；但需重新做模型推理 |
| `_work/constrained_candidates/` | 约束后候选及逐图哈希 | 重新选风格/审计时 | 可以；但需重新计算 |
| `_work/selected_pair/` | 选定风格的原始首尾模型图 | 审计时 | 可以；不影响现有 final |
| `_work/*.json` | 参数、模型状态、哈希和指标 | 复现/审计时 | 建议保留 |

复核图片：

{review_section}

## 12. 可复现性的边界和已知限制

- seed 能固定本机推理，但更换 GPU 架构、PyTorch、Diffusers 或底层算子后，扩散结果
  可能不是逐字节一致；模型权重 SHA-256 用于确认最重要的输入一致。
- semantic masks 保证类别位置和边界来源于机制，但模型增强权重较克制，因此视觉提升
  主要是细纹理和局部明暗，不会变成完全写实的卫星照片。
- 候选选择包含人工视觉判断；平均梯度、对比度和 RGB 稳定性不是完整的美学评分。
- Track B 只生成两张静态关键帧，不生成二者之间的视频过渡。
- 本流程继承 Track A 的教学模型假设；它保证不篡改机制输出，但不把简化模型升级为
  真实水动力工程模拟。
    """
    (output_root / "report.md").write_text(report, encoding="utf-8")
    html_final = (
        """<p><a href="final/first.png">最终首帧</a> ·
<a href="final/last.png">最终尾帧</a> ·
<a href="final/comparison.jpg">首尾对比图</a></p>
<img src="final/comparison.jpg" alt="selected first and last keyframes">"""
        if pair_stability
        else "<p>首尾帧对尚未生成。</p>"
    )
    html_model_rows = "".join(
        f"<tr><td><code>{html.escape(record['model_id'])}</code></td>"
        f"<td><code>{html.escape(record['path'])}</code></td>"
        f"<td>{'可用' if record['available'] else '缺失'}</td></tr>"
        for record in model_status["required_models"].values()
    )
    html_package_rows = "".join(
        f"<tr><td><code>{html.escape(name)}</code></td>"
        f"<td><code>{html.escape(version or 'missing')}</code></td></tr>"
        for name, version in model_status.get("package_versions", {}).items()
    )
    html_candidate_rows = "".join(
        f"<tr><td><code>{html.escape(candidate['candidate'])}</code></td>"
        f"<td>{candidate['visual_metrics']['mean_gradient']:.6f}</td>"
        f"<td>{candidate['visual_metrics']['contrast_std']:.6f}</td></tr>"
        for candidate in tail_candidates
    )
    html_status = (
        f"""<h2>复核图片</h2>
<p>下面两张图仅用于比较模型原始提案和受机制约束后的候选，不是额外交付物。</p>
<img src="review/raw-style-proposals.jpg">
<img src="review/constrained-style-candidates.jpg">
<p>几何门禁：<code>{'通过' if result['geometry_gate']['passed'] else '未通过'}</code></p>"""
        if tail_candidates
        else f"<h2>当前阻塞</h2><p>缺少：<code>{html.escape(missing)}</code>。未使用替代模型。</p>"
    )
    report_html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Stage 1 Track B Report</title>
<style>body{{font:16px/1.6 system-ui;max-width:980px;margin:40px auto;padding:0 24px;color:#19313a}}
img{{max-width:100%;margin:8px 1% 20px 0}}code{{background:#eef3f3;padding:2px 5px}}
table{{border-collapse:collapse;width:100%;margin:10px 0 24px}}th,td{{border:1px solid #ccd8da;padding:7px 10px;text-align:left}}
pre{{background:#eef3f3;padding:14px;overflow:auto;white-space:pre-wrap}}</style>
<h1>Stage 1 Track B 报告：高质量关键帧生成</h1>
<p>状态：<code>{html.escape(status)}</code></p>
<h2>1. 目标与最终结果</h2>
<p>Track A 先计算三角洲机制；Track B 从已验证状态取两个关键帧，让 SDXL 提供纹理，
再用机制区域图把海岸线、河道和沉积范围锁回原位置。模型不决定地形。</p>
{html_final}
<p><a href="../causal_delta/report.html">查看 Track A 机制动画报告</a></p>
<h2>2. 生成流程</h2>
<pre>states.jsonl + timeline.json
→ 选 accumulate 末尾和 threshold_change 末尾
→ 机制底图 + Canny 边缘图 + 五类语义 mask
→ SDXL img2img + Canny ControlNet 原始提案
→ mask 约束纹理、降低边界权重
→ 比较六个尾帧候选
→ 用选定 prompt 和 seed 生成首尾帧
→ 几何门禁、稳定性检查、final/</pre>
<h2>3. 输入关键帧</h2>
<table><tr><th>名称</th><th>阶段</th><th>展示帧</th><th>机制状态</th></tr>
<tr><td>首帧</td><td><code>accumulate</code></td>
<td>{prepare_manifest["keyframes"]["first"]["display_frame"]}</td>
<td>{prepare_manifest["keyframes"]["first"]["state_frame"]}</td></tr>
<tr><td>尾帧</td><td><code>threshold_change</code></td>
<td>{prepare_manifest["keyframes"]["last"]["display_frame"]}</td>
<td>{prepare_manifest["keyframes"]["last"]["state_frame"]}</td></tr></table>
<h2>4. control edges 与 semantic masks</h2>
<p><code>control_edges</code> 是底图经 Canny 100/200 得到的 ControlNet 输入，只用于
减少边缘漂移，不是成品。<code>semantic_masks</code> 直接从机制状态计算，不是 AI 标注；
五张图互斥并完整覆盖所有像素。</p>
<table><tr><th>mask</th><th>保护内容</th></tr>
<tr><td>original_land</td><td>原有陆地和海岸</td></tr>
<tr><td>river</td><td>海岸线以内的河道水格</td></tr>
<tr><td>ocean</td><td>其余开阔水域</td></tr>
<tr><td>underwater_deposit</td><td>尚未出水的水下沉积</td></tr>
<tr><td>new_land</td><td>机制判定已经出水的新陆地</td></tr></table>
<h2>5. 模型与参数</h2>
<table><tr><th>模型 ID</th><th>本地路径</th><th>状态</th></tr>{html_model_rows}</table>
<table><tr><th>软件</th><th>版本</th></tr>{html_package_rows}
<tr><td>GPU</td><td><code>{html.escape(str(generation_runtime.get("gpu", "未记录")))}</code></td></tr></table>
<p>768×512、FP16、36 steps、strength 0.50、guidance 6.5、
ControlNet scale 1.35；每种尾帧风格使用 seed 3101 和 3102。</p>
<h2>6. 纹理约束</h2>
<p>模型提案先提取中高频纹理和受限明暗，再逐 mask 混回机制底图。
mask 向内腐蚀 4 像素后，内部权重为 0.62，边界权重降为 0.14；
模型的整体颜色布局不会进入成品。</p>
<pre>residual = proposal - gaussian_blur(proposal, radius=7)
shading = clip(local_luminance / region_median, 0.82, 1.18)
textured = base × shading + 0.72 × residual
final = base × (1 - weight) + textured × weight</pre>
<h2>7. 候选选择与验证</h2>
<p>本次人工选择 <code>{html.escape(str(selected_candidate))}</code>。数值指标只辅助发现
异常模糊图片，不自动决定风格。</p>
<table><tr><th>候选</th><th>平均梯度</th><th>对比度</th></tr>{html_candidate_rows}</table>
{html_status}
<h2>8. 从零复现</h2>
<pre>python -m venv .venv
.venv/bin/python -m pip install -r modules/video_model/stage1/requirements.txt
.venv/bin/python -m modules.video_model.stage1.causal_delta.simulate
.venv/bin/python -m modules.video_model.stage1.causal_delta.validate
.venv/bin/python -m modules.video_model.stage1.causal_delta.export

MODEL_PYTHON=/opt/venv/bin/python
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.prepare
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.enhance --status-only
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.enhance --generate-candidates --force
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.evaluate
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.enhance \\
  --generate-pair --selected-style physical_geography --seed {html.escape(str(selected_seed))} --force
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.evaluate
$MODEL_PYTHON -m pytest -q modules/video_model/stage1</pre>
<p>需要先安装与显卡匹配的 PyTorch，并准备
<code>stabilityai/stable-diffusion-xl-base-1.0</code> 和
<code>diffusers/controlnet-canny-sdxl-1.0</code> 的 FP16 本地目录。
环境变量、权重 SHA-256、逐文件用途、删除建议和可复现性边界见
<a href="report.md">完整 Markdown 报告</a>。</p>
</html>"""
    (output_root / "report.html").write_text(report_html, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=KEYFRAME_OUTPUT_ROOT)
    args = parser.parse_args()
    result = evaluate(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
