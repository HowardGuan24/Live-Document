"""Build the 105-frame timeline and export PNG, MP4, GIF and reports."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import FRAMES_ROOT, MECHANISM_ROOT, OUTPUT_ROOT, SimulationConfig
from .render import find_font, render_state
from .validate import load_states


STORYBOARD_PATH = Path(__file__).with_name("storyboard.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_timeline(
    storyboard: dict[str, Any],
    states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for beat_index, beat in enumerate(storyboard["beats"]):
        selected = np.linspace(
            beat["state_start"],
            beat["state_end"],
            beat["dynamic_frames"],
        ).round().astype(int)
        state_indices = selected.tolist() + [beat["state_end"]] * beat["hold_frames"]
        for local_index, state_index in enumerate(state_indices):
            display_index = len(timeline)
            state = states[state_index]
            timeline.append(
                {
                    "display_frame": display_index,
                    "time_seconds": round(display_index / storyboard["fps"], 4),
                    "state_frame": state_index,
                    "beat_id": beat["id"],
                    "caption": beat["caption"],
                    "is_hold": local_index >= beat["dynamic_frames"],
                    "source": {
                        "states_file": "mechanism/states.jsonl",
                        "state_line": state_index + 1,
                    },
                    "state_stats": state["stats"],
                    "beat_index": beat_index,
                }
            )
    if len(timeline) != 105:
        raise ValueError(f"storyboard must produce exactly 105 frames, got {len(timeline)}")
    return timeline


def _encode_mp4(frames_root: Path, output: Path, fps: int) -> list[str]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_root / "%04d.png"),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "FFmpeg MP4 encoding failed")
    return command


def _global_palette(frames: list[Image.Image], output: Path) -> Image.Image:
    samples = frames[:: max(1, len(frames) // 12)]
    thumbnails = [frame.resize((192, 128), Image.Resampling.LANCZOS) for frame in samples]
    atlas = Image.new("RGB", (192 * len(thumbnails), 128))
    for index, thumbnail in enumerate(thumbnails):
        atlas.paste(thumbnail, (index * 192, 0))
    palette_source = atlas.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    raw_palette = palette_source.getpalette()[: 128 * 3]
    swatch = Image.new("P", (16, 8))
    swatch.putpalette(raw_palette + [0] * (768 - len(raw_palette)))
    swatch.putdata(list(range(128)))
    swatch.save(output)
    return palette_source


def _encode_gif(
    frames: list[Image.Image],
    output: Path,
    palette_output: Path,
) -> None:
    palette = _global_palette(frames, palette_output)
    quantized = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]
    durations = [80, 80, 90] * (len(quantized) // 3)
    durations += [80, 80, 90][: len(quantized) - len(durations)]
    quantized[0].save(
        output,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )


def _contact_sheet(
    frames: list[Image.Image],
    timeline: list[dict[str, Any]],
    output: Path,
) -> None:
    indices = [0, 20, 21, 39, 64, 71, 85, 104]
    tile_w, tile_h = 384, 256
    sheet = Image.new("RGB", (tile_w * 4, tile_h * 2), "#0b2029")
    font_path, _ = find_font()
    font = ImageFont.truetype(str(font_path), 14)
    for tile_index, frame_index in enumerate(indices):
        image = frames[frame_index].resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image, "RGBA")
        label = f"display {frame_index}  /  state {timeline[frame_index]['state_frame']}"
        draw.rectangle((7, 7, 205, 29), fill=(5, 22, 29, 210))
        draw.text((13, 9), label, font=font, fill="white")
        sheet.paste(image, ((tile_index % 4) * tile_w, (tile_index // 4) * tile_h))
    sheet.save(output, quality=92, subsampling=0)


def _report(
    output_root: Path,
    summary: dict[str, Any],
    validation: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    config = json.loads(
        (output_root / "mechanism" / "simulation_config.json").read_text(
            encoding="utf-8"
        )
    )
    storyboard = json.loads(STORYBOARD_PATH.read_text(encoding="utf-8"))
    check_explanations = {
        "state_count": "状态数量与配置一致，避免少算或多算帧。",
        "thickness_monotonic": "每个网格的沉积厚度只能增加，不能凭空消失。",
        "new_land_monotonic": "已经出水的陆地必须持续存在。",
        "land_threshold_exact": "只有“沉积厚度大于当地水深”才能产生新陆地。",
        "arrival_before_settling": "泥沙必须先到达河口，之后才能发生沉降。",
        "visible_underwater_stage": "出水前必须有足够长的水下沉积阶段。",
        "emergence_in_threshold_stage": "首次出水必须发生在教学时间线的出水阶段。",
        "mouth_deceleration": "河口平均流速必须明显低于上游。",
        "final_channel_count": "最终必须形成 2–3 条可辨认通道。",
        "stable_channels": "分流结果必须稳定至少配置要求的帧数。",
        "new_land_connected": "新生陆地必须是一个连通体，不是零散噪点。",
        "land_and_deposit_extent": "陆地与水下沉积必须向海侧推进到最低距离。",
        "state_traceability": "每个状态都必须带帧号、阶段、流场采样和统计量。",
    }
    checks = "\n".join(
        f"| `{item['name']}` | {check_explanations[item['name']]} | "
        f"{'PASS' if item['passed'] else 'FAIL'} | "
        f"`{json.dumps(item['evidence'], ensure_ascii=False)}` |"
        for item in validation["checks"]
    )
    beat_rows = "\n".join(
        f"| `{beat['id']}` | {beat['caption']} | "
        f"{beat['state_start']}–{beat['state_end']} | "
        f"{beat['dynamic_frames']} | {beat['hold_frames']} |"
        for beat in storyboard["beats"]
    )
    report = f"""# Stage 1 Track A 报告：三角洲机制动画

状态：`{"PASSED" if validation["passed"] else "FAILED"}`

## 1. 这份报告说明什么

这条流水线不是让视频模型猜测三角洲如何形成，而是先用一套可检查的网格规则计算
“输沙 → 减速 → 沉积 → 出水 → 分流”，再把计算状态画成动画。机制状态是事实来源；
渲染器只能读取状态，不能修改岸线、沉积厚度或水流。

本报告记录输入、算法、执行顺序、输出文件和验证结果。第一次接触项目的读者可从
第 8 节开始按命令复现。

最终媒体：

- [MP4 动画](delta_causal.mp4)
- [GIF 动画](delta_causal.gif)
- [八个代表帧](contact-sheet.jpg)
- [Track B 高质量关键帧报告](../keyframe_render/report.md)

## 2. 输入和代码入口

| 输入 | 用途 |
|---|---|
| [`config.py`](../../causal_delta/config.py) | 默认网格、随机种子、流速、沉降和水深参数 |
| [`primitives.py`](../../causal_delta/primitives.py) | 流场、粒子搬运、沉积、出水和绕流规则 |
| [`simulate.py`](../../causal_delta/simulate.py) | 运行 120 个机制状态并写入 JSONL |
| [`validate.py`](../../causal_delta/validate.py) | 在渲染前独立检查机制是否成立 |
| [`storyboard.json`](../../causal_delta/storyboard.json) | 把 120 个机制状态编排成 105 个展示帧 |
| [`render.py`](../../causal_delta/render.py) | 把状态画成 768×512 图片并叠加教学信息 |
| [`export.py`](../../causal_delta/export.py) | 编码 MP4/GIF、生成联系表、元数据和本报告 |

实际使用的完整参数保存在
[`mechanism/simulation_config.json`](mechanism/simulation_config.json)。关键参数如下：

| 参数 | 本次值 | 含义 |
|---|---:|---|
| 网格 | {config["grid_width"]}×{config["grid_height"]} | 机制计算分辨率 |
| 画布 | {config["canvas_width"]}×{config["canvas_height"]} | 输出图片分辨率 |
| 状态数 | {config["state_count"]} | 机制更新次数 |
| 随机种子 | {config["random_seed"]} | 控制粒子初始位置和横向扰动 |
| 每帧新粒子 | {config["sediment_per_frame"]} | 泥沙输入量 |
| 上游流速 | {config["river_speed"]} | 河道内的基准速度 |
| 海域最低流速 | {config["sea_min_speed"]} | 避免海域流场降为零 |
| 基础沉降率 | {config["base_settling_rate"]} | 减速后粒子的沉降概率基数 |
| 单粒沉积质量 | {config["deposit_mass"]} | 每个沉降粒子增加的总厚度 |
| 沉积分散半径 | {config["dispersion_radius"]} | 沉积核半径，单位为网格 |

本次运行环境：Python `{metadata["runtime"]["python"]}`、NumPy
`{metadata["runtime"]["numpy"]}`、Pillow `{metadata["runtime"]["Pillow"]}`、
imageio-ffmpeg `{metadata["runtime"]["imageio-ffmpeg"]}`。

## 3. 一个机制状态是怎样算出来的

每个状态按固定顺序执行：

1. 根据当前陆地计算流场。河道内使用基准流速；入海后速度按离岸距离下降。
2. 注入 {config["sediment_per_frame"]} 个悬浮泥沙粒子。
3. 粒子沿流场移动，并加入由 seed `{config["random_seed"]}` 控制的小幅横向扰动。
4. 粒子撞到新生陆地时，在附近下游水格中寻找替代路线。
5. 粒子越过河口保护区后，按局部减速程度计算沉降概率：

   ```text
   slowdown = 1 - local_speed / river_speed
   settling_probability = base_settling_rate × slowdown²
   ```

6. 沉降粒子通过归一化高斯核增加周围网格的沉积厚度；旧厚度不会减少。
7. 对每个水格应用唯一的出水规则：

   ```text
   new_land = not original_land and sediment_thickness > water_depth
   land = original_land or new_land
   ```

8. 新生陆地作为障碍反馈到流场，下一状态的水和粒子从其上下两侧绕行。
9. 把粒子、厚度、陆地、流场采样和统计量写成
   [`states.jsonl`](mechanism/states.jsonl) 中的一行。

这里的水深不是目标三角洲轮廓，而是连续的河口浅滩标量场。最终陆地仍只能由
“沉积厚度大于水深”这一条规则产生。

## 4. 为什么有 120 个状态，却只有 105 张展示帧

机制状态用于计算，展示帧用于讲解。`storyboard.json` 从每个阶段抽取部分状态，
再在阶段末尾停留 7 帧，给观众阅读时间。`timeline.json` 逐张记录展示帧对应的
机制状态，因此不会丢失追溯关系。

| 阶段 | 画面含义 | 状态范围 | 动态展示帧 | 末尾停留帧 |
|---|---|---:|---:|---:|
{beat_rows}

合计 {metadata["display_frames"]} 张展示帧，{metadata["fps"]} fps，
时长 {metadata["duration_seconds"]} 秒。

## 5. 渲染和视频编码

每张展示帧先从状态绘制海水、原有陆地、水下沉积和新生陆地，再叠加：

- 蓝色箭头：保存于状态中的流向和速度；
- 赭色圆点：悬浮粒子的固定抽样，避免粒子太密；
- 标题、图例、阶段字幕和进度点：仅用于讲解，不进入机制计算。

PNG 位于 [`frames/`](frames/)。MP4 使用 H.264、CRF 18、`yuv420p` 编码。
GIF 从整段动画抽样建立同一套 128 色全局调色板，并关闭抖动，以减少逐帧色闪。
实际 FFmpeg 命令保存在 [`metadata.json`](metadata.json) 的 `mp4_command` 字段。

## 6. 本次结果

- {metadata["mechanism_states"]} 个机制状态，经五段教学时间线编排为
  {metadata["display_frames"]} 张展示帧。
- 正式画布 {metadata["canvas"][0]}×{metadata["canvas"][1]}，
  {metadata["fps"]} fps，时长 {metadata["duration_seconds"]} 秒。
- 泥沙第 {summary["first_coast_arrival_frame"]} 帧到岸，第 {summary["first_settling_frame"]} 帧首次沉降。
- 第 {summary["first_emergence_frame"]} 帧首次出水；最终新生陆地 {summary["final_new_land_cells"]} 格。
- 最终 {summary["final_channel_count"]} 条通道，稳定尾段 {summary["stable_channel_tail_frames"]} 帧。
- 模拟与渲染均未使用 GPU 或生成模型。

## 7. 自动验证

验证在渲染之前执行。任一门禁失败，`export.py` 会拒绝生成正式媒体。

| 检查 | 检查的含义 | 结果 | 本次证据 |
|---|---|---|---|
{checks}

完整机器可读结果见
[`mechanism/validation.json`](mechanism/validation.json)。

## 8. 从零复现

以下命令从仓库根目录 `Live-Document/` 执行。Track A 只需要 CPU：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r modules/video_model/stage1/requirements.txt

.venv/bin/python -m modules.video_model.stage1.causal_delta.simulate
.venv/bin/python -m modules.video_model.stage1.causal_delta.validate
.venv/bin/python -m modules.video_model.stage1.causal_delta.export

.venv/bin/python -m pytest -q modules/video_model/stage1/causal_delta/tests
```

若系统没有中文字体，画面会自动使用英文字幕。要复现中文排版，请准备支持简体中文的
TrueType/OpenType 字体并在运行前设置：

```bash
export DELTA_FONT=/absolute/path/to/NotoSansCJKsc-Regular.otf
```

复现成功后应看到：

- `simulate` 输出 `state_count: 120`；
- `validate` 输出 `passed: true`；
- `export` 生成 105 张 PNG、MP4、GIF、联系表和报告；
- 测试全部通过。

也可运行：

```bash
.venv/bin/python -m modules.video_model.stage1.run
```

这个入口会重跑 Track A，并准备/评估 Track B 已存在的文件；它**不会自动执行耗时的
SDXL 候选生成**。Track B 的完整 GPU 命令见其
[报告](../keyframe_render/report.md)。

## 9. 文件如何追溯

```text
config.py
  ↓ simulate
mechanism/simulation_config.json + mechanism/states.jsonl
  ↓ validate
mechanism/validation.json
  ↓ storyboard + render
timeline.json + frames/*.png
  ↓ encode
delta_causal.mp4 + delta_causal.gif
```

渲染器只读取状态；岸线、沉积厚度、新生陆地、颗粒位置与流向均不能在渲染层反向修改。

| 输出 | 是否最终交付 | 用途 |
|---|---:|---|
| `delta_causal.mp4` | 是 | 正式 H.264 动画 |
| `delta_causal.gif` | 是 | 循环预览 |
| `contact-sheet.jpg` | 是 | 快速检查八个代表帧 |
| `frames/` | 否 | 视频编码源帧，可由状态重建 |
| `timeline.json` | 否 | 展示帧到机制状态的映射 |
| `mechanism/states.jsonl` | 否，但应保留审计 | 每个机制状态的事实记录 |
| `mechanism/validation.json` | 否，但应保留审计 | 自动门禁和证据 |
| `metadata.json` | 否，但应保留复现 | 编码参数、哈希、字体和耗时 |

## 10. 可复现性的边界

- 同一代码、参数、NumPy 版本和 seed 应产生相同机制状态。
- 字体文件、Pillow 或 FFmpeg 版本不同，可能使 PNG、MP4、GIF 的字节哈希不同，
  但不应改变机制状态和验证结果。
- 这是用于解释因果链的简化教学模型，不是经过观测数据校准的水动力或泥沙工程模型，
  不能用于真实工程预测。

本次 MP4：{metadata["artifacts"]["mp4"]["size_bytes"]} bytes，
SHA-256 `{metadata["artifacts"]["mp4"]["sha256"]}`。

本次 GIF：{metadata["artifacts"]["gif"]["size_bytes"]} bytes，
SHA-256 `{metadata["artifacts"]["gif"]["sha256"]}`。

本次机制状态 `states.jsonl` 的 SHA-256：
`{metadata["traceability"]["states_sha256"]}`。
"""
    (output_root / "report.md").write_text(report, encoding="utf-8")
    rows = "".join(
        f"<tr><td><code>{html.escape(item['name'])}</code></td>"
        f"<td>{html.escape(check_explanations[item['name']])}</td>"
        f"<td>{'PASS' if item['passed'] else 'FAIL'}</td>"
        f"<td><code>{html.escape(json.dumps(item['evidence'], ensure_ascii=False))}</code></td></tr>"
        for item in validation["checks"]
    )
    html_beats = "".join(
        f"<tr><td><code>{html.escape(beat['id'])}</code></td>"
        f"<td>{html.escape(beat['caption'])}</td>"
        f"<td>{beat['state_start']}–{beat['state_end']}</td>"
        f"<td>{beat['dynamic_frames']}</td><td>{beat['hold_frames']}</td></tr>"
        for beat in storyboard["beats"]
    )
    report_html = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>Stage 1 Track A Report</title>
<style>body{{font:16px/1.6 system-ui;max-width:980px;margin:40px auto;padding:0 24px;color:#19313a}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd7da;padding:8px;text-align:left}}
code{{white-space:pre-wrap}}video,img{{max-width:100%}}pre{{background:#eef3f3;padding:14px;overflow:auto}}</style>
<h1>Stage 1 Track A 报告：三角洲机制动画</h1>
<p><strong>状态：{'PASSED' if validation['passed'] else 'FAILED'}</strong></p>
<h2>过程概览</h2>
<p>先在 96×64 网格上计算输沙、减速、沉积、出水和分流，再把状态渲染成动画。
机制状态是事实来源，渲染器不能修改机制。</p>
<pre>配置 → 120 个机制状态 → 自动验证 → 105 个展示帧 → MP4 / GIF</pre>
<video controls loop muted src="delta_causal.mp4"></video>
<h2>代表帧</h2><img src="contact-sheet.jpg" alt="contact sheet">
<h2>教学时间线</h2><table><tr><th>阶段</th><th>含义</th><th>状态</th><th>动态帧</th><th>停留帧</th></tr>{html_beats}</table>
<h2>自动验证</h2><table><tr><th>检查</th><th>含义</th><th>结果</th><th>证据</th></tr>{rows}</table>
<h2>从零复现</h2><pre>python -m venv .venv
.venv/bin/python -m pip install -r modules/video_model/stage1/requirements.txt
.venv/bin/python -m modules.video_model.stage1.causal_delta.simulate
.venv/bin/python -m modules.video_model.stage1.causal_delta.validate
.venv/bin/python -m modules.video_model.stage1.causal_delta.export
.venv/bin/python -m pytest -q modules/video_model/stage1/causal_delta/tests</pre>
<p>算法、公式、输入文件、输出目录、字体设置和可复现性边界请查看同目录的
<a href="report.md">完整 Markdown 报告</a>。</p></html>"""
    (output_root / "report.html").write_text(report_html, encoding="utf-8")


def export(
    mechanism_root: Path = MECHANISM_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    started = time.perf_counter()
    states = load_states(mechanism_root / "states.jsonl")
    storyboard = json.loads(STORYBOARD_PATH.read_text(encoding="utf-8"))
    validation = json.loads((mechanism_root / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("passed"):
        raise RuntimeError("mechanism validation must pass before rendering")
    summary = json.loads((mechanism_root / "simulation_summary.json").read_text(encoding="utf-8"))
    config_data = json.loads((mechanism_root / "simulation_config.json").read_text(encoding="utf-8"))
    config = SimulationConfig(**config_data)
    frames_root = output_root / "frames"
    if frames_root.exists():
        shutil.rmtree(frames_root)
    frames_root.mkdir(parents=True)
    timeline = build_timeline(storyboard, states)
    frames: list[Image.Image] = []
    render_info: dict[str, Any] = {}
    for entry in timeline:
        frame, render_info = render_state(
            states[entry["state_frame"]],
            beat_index=entry["beat_index"],
            config=config,
        )
        frame_path = frames_root / f"{entry['display_frame']:04d}.png"
        frame.save(frame_path, compress_level=6)
        frames.append(frame)
        entry["rendered_file"] = f"frames/{frame_path.name}"
    (output_root / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    mp4_path = output_root / "delta_causal.mp4"
    gif_path = output_root / "delta_causal.gif"
    mp4_command = _encode_mp4(frames_root, mp4_path, storyboard["fps"])
    _encode_gif(frames, gif_path, output_root / "palette.png")
    _contact_sheet(frames, timeline, output_root / "contact-sheet.jpg")
    metadata = {
        "status": "success",
        "canvas": storyboard["canvas"],
        "fps": storyboard["fps"],
        "display_frames": len(timeline),
        "mechanism_states": len(states),
        "duration_seconds": len(timeline) / storyboard["fps"],
        "render_seconds": round(time.perf_counter() - started, 3),
        "runtime": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "Pillow": importlib.metadata.version("Pillow"),
            "imageio-ffmpeg": importlib.metadata.version("imageio-ffmpeg"),
        },
        "model_usage": [],
        "gpu_usage": False,
        "font": render_info,
        "gif": {
            "global_palette": True,
            "max_colors": 128,
            "dither": "none",
            "frame_durations_ms_cycle": [80, 80, 90],
        },
        "mp4_command": mp4_command,
        "artifacts": {
            "mp4": {
                "path": str(mp4_path.resolve()),
                "size_bytes": mp4_path.stat().st_size,
                "sha256": sha256(mp4_path),
            },
            "gif": {
                "path": str(gif_path.resolve()),
                "size_bytes": gif_path.stat().st_size,
                "sha256": sha256(gif_path),
            },
            "contact_sheet": {
                "path": str((output_root / "contact-sheet.jpg").resolve()),
                "sha256": sha256(output_root / "contact-sheet.jpg"),
            },
        },
        "traceability": {
            "states": str((mechanism_root / "states.jsonl").resolve()),
            "states_sha256": sha256(mechanism_root / "states.jsonl"),
            "timeline": str((output_root / "timeline.json").resolve()),
            "all_frames_exist": all(
                (output_root / entry["rendered_file"]).is_file() for entry in timeline
            ),
        },
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _report(output_root, summary, validation, metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-root", type=Path, default=MECHANISM_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    metadata = export(args.mechanism_root, args.output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
