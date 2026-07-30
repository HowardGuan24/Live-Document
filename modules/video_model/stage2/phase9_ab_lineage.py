"""Build a visual, file-level explanation of the actual A→B lineage."""

from __future__ import annotations

import argparse
import html
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .framework.contracts import (
    artifact_record,
    load_json,
    sha256_path,
    write_json,
)


STAGE2_ROOT = Path(__file__).resolve().parent
PHASE2_ROOT = STAGE2_ROOT / "output/phase-2"
PHASE7_ROOT = STAGE2_ROOT / "output/phase-7"
PHASE8_ROOT = STAGE2_ROOT / "output/phase-8"
OUTPUT_ROOT = STAGE2_ROOT / "output/phase-9"
ASSET_ROOT = OUTPUT_ROOT / "report-assets"
REPORT_PATH = OUTPUT_ROOT / "ab-lineage-report.html"
MANIFEST_PATH = OUTPUT_ROOT / "phase9-manifest.json"
KEYFRAMES = (
    "00_start",
    "01_mechanism",
    "02_result",
    "03_end",
)
LABELS = {
    "00_start": "START",
    "01_mechanism": "MECHANISM",
    "02_result": "RESULT",
    "03_end": "END",
}
CHEM_A_ROOT = (
    PHASE7_ROOT
    / "route-a/experiments/EXP-P7-A-chem-01-00_start"
)
CHEM_A_BASE = (
    CHEM_A_ROOT
    / "raw/semantic_control_065/seed_7101.png"
)
CHEM_B_ROOT = PHASE7_ROOT / "route-b/CHEM-01"
CHEM_B_VARIANT = "optical_tint_plus_drop"
CHEM_CONTROL_SOURCE_ROOT = (
    STAGE2_ROOT / "experiments/EXP-20260729-009"
)
CHEM_CONTROL_BUILDER = (
    CHEM_CONTROL_SOURCE_ROOT / "build_control.py"
)
CHEM_CONTROL_TEMPLATE = (
    CHEM_CONTROL_SOURCE_ROOT
    / "semantic_apparatus_line_art.png"
)
CHEM_A_CONTROL = (
    CHEM_A_ROOT / "controls/phase7_semantic_control.png"
)


def _href(path: Path) -> str:
    return os.path.relpath(path, REPORT_PATH.parent).replace(
        os.sep, "/"
    )


def _layer_path(
    case_id: str, keyframe_id: str, layer_id: str
) -> Path:
    root = PHASE2_ROOT / case_id / "keyframes" / keyframe_id
    manifest = load_json(root / "semantic_layers.json")
    layer = next(
        item
        for item in manifest["layers"]
        if item["layer_id"] == layer_id
    )
    return PHASE2_ROOT / case_id / layer["data"]["path"]


def _sheet(
    cells: list[tuple[str, Path]],
    output: Path,
    *,
    columns: int = 4,
    thumb: tuple[int, int] = (400, 225),
) -> None:
    label_height = 38
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * thumb[0],
            rows * (thumb[1] + label_height),
        ),
        (12, 31, 38),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(cells):
        column = index % columns
        row = index // columns
        x = column * thumb[0]
        y = row * (thumb[1] + label_height)
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb, Image.Resampling.LANCZOS)
        tile = Image.new("RGB", thumb, (232, 229, 219))
        tile.paste(
            image,
            (
                (thumb[0] - image.width) // 2,
                (thumb[1] - image.height) // 2,
            ),
        )
        sheet.paste(tile, (x, y))
        draw.text(
            (x + 9, y + thumb[1] + 9),
            label,
            fill=(235, 246, 241),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, subsampling=0)


def _save_mask(mask: np.ndarray, output: Path) -> None:
    preview = np.zeros((*mask.shape, 3), dtype=np.uint8)
    preview[:] = (16, 34, 41)
    preview[mask] = (242, 242, 235)
    Image.fromarray(preview, mode="RGB").save(
        output, optimize=False
    )


def _save_ph_field(
    ph: np.ndarray, mask: np.ndarray, output: Path
) -> None:
    normalized = np.clip((ph - 2.0) / 10.0, 0, 1)
    preview = np.zeros((*ph.shape, 3), dtype=np.uint8)
    preview[:] = (16, 34, 41)
    red = np.clip(normalized * 300.0, 0, 255)
    green = np.clip(
        (1.0 - np.abs(normalized - 0.5) * 1.7)
        * 210.0,
        0,
        255,
    )
    blue = np.clip((1.0 - normalized) * 255.0, 0, 255)
    colors = np.stack((red, green, blue), axis=-1)
    preview[mask] = np.uint8(np.rint(colors[mask]))
    Image.fromarray(preview, mode="RGB").save(
        output, optimize=False
    )


def _build_state_assets() -> tuple[list[dict[str, Any]], dict[str, Path]]:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    program_cells = []
    mask_cells = []
    field_cells = []
    records = []
    for keyframe_id in KEYFRAMES:
        root = (
            PHASE2_ROOT
            / "CHEM-01/keyframes"
            / keyframe_id
        )
        program_cells.append(
            (LABELS[keyframe_id], root / "clean.png")
        )
        mask = (
            np.load(
                _layer_path(
                    "CHEM-01",
                    keyframe_id,
                    "chem01_liquid_region",
                ),
                allow_pickle=False,
            )
            > 0
        )
        ph = np.load(
            _layer_path(
                "CHEM-01", keyframe_id, "chem01_ph_field"
            ),
            allow_pickle=False,
        ).astype(np.float32)
        mask_path = (
            ASSET_ROOT / f"{keyframe_id}-liquid-mask.png"
        )
        field_path = (
            ASSET_ROOT / f"{keyframe_id}-ph-field.png"
        )
        _save_mask(mask, mask_path)
        _save_ph_field(ph, mask, field_path)
        mask_cells.append((LABELS[keyframe_id], mask_path))
        field_cells.append((LABELS[keyframe_id], field_path))
        state = load_json(root / "state.json")
        records.append(
            {
                "keyframe_id": keyframe_id,
                "bulk_ph": state["bulk_ph"],
                "liquid_level_y": state["liquid_level_y"],
                "liquid_mask_area_px": int(mask.sum()),
                "ph_min_in_liquid": round(
                    float(ph[mask].min()), 6
                ),
                "ph_mean_in_liquid": round(
                    float(ph[mask].mean()), 6
                ),
                "ph_max_in_liquid": round(
                    float(ph[mask].max()), 6
                ),
                "program": artifact_record(
                    root / "clean.png", STAGE2_ROOT
                ),
                "semantic_layers": artifact_record(
                    root / "semantic_layers.json", STAGE2_ROOT
                ),
            }
        )
    assets = {
        "program_sequence": ASSET_ROOT
        / "chem-program-sequence.jpg",
        "mask_sequence": ASSET_ROOT
        / "chem-liquid-masks.jpg",
        "ph_sequence": ASSET_ROOT / "chem-ph-fields.jpg",
    }
    _sheet(program_cells, assets["program_sequence"])
    _sheet(mask_cells, assets["mask_sequence"])
    _sheet(field_cells, assets["ph_sequence"])
    return records, assets


def _build_a_assets(assets: dict[str, Path]) -> None:
    assets["a_inputs"] = ASSET_ROOT / "chem-a-inputs.jpg"
    _sheet(
        [
            (
                "PROGRAM START",
                PHASE2_ROOT
                / "CHEM-01/keyframes/00_start/clean.png",
            ),
            (
                "A STRUCTURE CONTROL",
                CHEM_A_CONTROL,
            ),
            ("A SELECTED FROZEN BASE", CHEM_A_BASE),
        ],
        assets["a_inputs"],
        columns=3,
        thumb=(480, 270),
    )
    assets["a_control_comparison"] = (
        ASSET_ROOT / "chem-a-control-comparison.jpg"
    )
    _sheet(
        [
            (
                "PROGRAM CLEAN FRAME",
                PHASE2_ROOT
                / "CHEM-01/keyframes/00_start/clean.png",
            ),
            (
                "AUTOMATIC DENSE CANNY",
                CHEM_A_ROOT / "controls/dense_canny.png",
            ),
            (
                "ACTUAL SEMANTIC CONTROL",
                CHEM_A_CONTROL,
            ),
            ("RAW SDXL + CONTROLNET OUTPUT", CHEM_A_BASE),
        ],
        assets["a_control_comparison"],
    )
    assets["a_rejected_direct_sequence"] = (
        ASSET_ROOT / "chem-a-rejected-direct-sequence.jpg"
    )
    _sheet(
        [
            (
                LABELS[keyframe_id],
                PHASE7_ROOT
                / "route-a/experiments"
                / f"EXP-P7-A-chem-01-{keyframe_id}"
                / "raw/semantic_control_065/seed_7101.png",
            )
            for keyframe_id in KEYFRAMES
        ],
        assets["a_rejected_direct_sequence"],
    )


def _build_calibration_asset(
    assets: dict[str, Path]
) -> None:
    base = Image.open(CHEM_A_BASE).convert("RGB")
    annotated = base.copy()
    draw = ImageDraw.Draw(annotated)
    polygon = (
        (350, 404),
        (675, 404),
        (652, 493),
        (370, 493),
    )
    draw.line(
        (*polygon, polygon[0]),
        fill=(255, 43, 153),
        width=6,
        joint="curve",
    )
    calibration = ASSET_ROOT / "chem-real-surface.png"
    annotated.save(calibration, optimize=False)
    assets["calibration"] = calibration
    assets["b_mapping"] = ASSET_ROOT / "chem-b-mapping.jpg"
    _sheet(
        [
            (
                "PROGRAM LIQUID MASK",
                ASSET_ROOT / "03_end-liquid-mask.png",
            ),
            ("REAL LIQUID SURFACE", calibration),
            (
                "MAPPED END STATE",
                CHEM_B_ROOT
                / "variants"
                / CHEM_B_VARIANT
                / "03_end.png",
            ),
        ],
        assets["b_mapping"],
        columns=3,
        thumb=(480, 270),
    )
    assets["final_sequence"] = (
        ASSET_ROOT / "chem-final-b-sequence.jpg"
    )
    _sheet(
        [
            (
                LABELS[keyframe_id],
                CHEM_B_ROOT
                / "variants"
                / CHEM_B_VARIANT
                / f"{keyframe_id}.png",
            )
            for keyframe_id in KEYFRAMES
        ],
        assets["final_sequence"],
    )


def _build_other_case_assets(
    assets: dict[str, Path]
) -> None:
    assets["other_lineages"] = (
        ASSET_ROOT / "other-actual-lineages.jpg"
    )
    _sheet(
        [
            (
                "BIO: PHASE-3 DONOR -> B",
                PHASE7_ROOT / "report-assets/bio-process.jpg",
            ),
            (
                "GEO: A CANDIDATES, NO B",
                PHASE7_ROOT
                / "route-a/experiments/"
                "EXP-P7-A-geo-02-02_result-terrain_only/"
                "candidates-labeled.jpg",
            ),
            (
                "MATH: PHASE-3 DONOR -> B2 TEST",
                PHASE8_ROOT / "report-assets/math-b-process.jpg",
            ),
            (
                "PHYS: PHASE-3 DONOR -> B3 TEST",
                PHASE8_ROOT / "report-assets/phys-b-process.jpg",
            ),
        ],
        assets["other_lineages"],
        columns=2,
        thumb=(640, 360),
    )


def _links_resolve(report: str) -> bool:
    links = re.findall(r'(?:href|src)="([^"]+)"', report)
    local = [
        link
        for link in links
        if not link.startswith(("http:", "https:", "#"))
    ]
    return bool(local) and all(
        (REPORT_PATH.parent / link).resolve().exists()
        or (REPORT_PATH.parent / link).resolve()
        == MANIFEST_PATH.resolve()
        for link in local
    )


def _state_rows(states: list[dict[str, Any]]) -> str:
    names = {
        "00_start": "起始",
        "01_mechanism": "局部反应",
        "02_result": "接近终点前",
        "03_end": "滴定终点",
    }
    rows = []
    for item in states:
        rows.append(
            "<tr>"
            f"<td>{names[item['keyframe_id']]}</td>"
            f"<td>{item['bulk_ph']:.3f}</td>"
            f"<td>{item['ph_min_in_liquid']:.3f} / "
            f"{item['ph_mean_in_liquid']:.3f} / "
            f"{item['ph_max_in_liquid']:.3f}</td>"
            f"<td>{item['liquid_level_y']}</td>"
            f"<td>{item['liquid_mask_area_px']}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_report(
    states: list[dict[str, Any]],
    assets: dict[str, Path],
    manifest: dict[str, Any],
) -> str:
    positive = (
        CHEM_A_ROOT / "inputs/positive_prompt.txt"
    ).read_text(encoding="utf-8")
    negative = (
        CHEM_A_ROOT / "inputs/negative_prompt.txt"
    ).read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A→B 实际生成流程与文件血缘</title>
<style>
:root{{--ink:#17343b;--muted:#5a7178;--paper:#f2efe6;--card:#fffdf7;
--line:#c8d4d0;--green:#14745a;--blue:#236f92;--amber:#9a6616;--red:#a14242}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.72 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
main{{max-width:1180px;margin:auto;padding:34px 24px 80px}}h1{{font-size:clamp(34px,5vw,58px);
line-height:1.08;margin-bottom:18px}}h2{{margin-top:58px;font-size:30px;
border-top:1px solid var(--line);padding-top:30px}}h3{{margin-top:34px}}
.lead{{font-size:20px;max-width:980px}}.card,details,.step{{background:var(--card);
border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0}}
.truth{{border-left:7px solid var(--green)}}.warning{{border-left:7px solid var(--amber)}}
.steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:24px 0}}
.step strong{{display:block;font-size:19px}}.a{{border-top:5px solid var(--blue)}}
.b{{border-top:5px solid var(--green)}}img{{display:block;width:100%;height:auto;
border:1px solid var(--line);border-radius:9px;background:#dfe3df}}figure{{margin:22px 0}}
figcaption,.small{{font-size:14px;color:var(--muted)}}table{{width:100%;
border-collapse:collapse;background:var(--card)}}th,td{{padding:12px;text-align:left;
border-bottom:1px solid var(--line);vertical-align:top}}pre{{white-space:pre-wrap;
overflow-wrap:anywhere;background:#132c32;color:#eef8f4;padding:15px;border-radius:8px;
font:13px/1.55 ui-monospace,monospace}}code{{background:#e1e7e4;padding:.15em .35em;
border-radius:4px}}a{{color:#116850}}.pass{{color:var(--green);font-weight:750}}
.no{{color:var(--red);font-weight:750}}summary{{font-weight:750;cursor:pointer}}
.term{{font-weight:760;color:#174f66}}.io td:first-child{{font-weight:720;white-space:nowrap}}
@media(max-width:800px){{.steps{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="small">Live-Document / Stage 2 / Phase 9 documentation audit</p>
<h1>一张 A 底图，怎样变成<br>四张 B 关键帧</h1>
<p class="lead">这份报告不再分别介绍抽象的“路线 A”和“路线 B”，而是沿着一个实际文件
从生成到消费的路径解释。Phase 7 中只有化学滴定完整执行了本轮 A→B，因此它是本文
主案例；其他案例单独标明真实来源，不再假装都经过了 A。</p>

<div class="card truth"><strong>先说最重要的关系：</strong>
A 只负责生成一次可信外观；B 读取这张冻结图和四份程序状态，生产四张关键帧。
B 不会把 A 的四张候选各拿一张，也不会每帧重新调用扩散模型。</div>

<h2>先直接回答这三个问题</h2>
<ol>
 <li><strong>“Canny → Canny ControlNet → 残差 → SDXL”对吗？</strong>
 对，这是标准 Canny ControlNet 链路。但“残差”不是两张图片相减得到的像素差，
 而是 ControlNet 计算出的多层特征修正张量。本次被选中的烧杯图有一个重要例外：
 它没有把自动 Canny 结果送进 ControlNet，而是送入了单独绘制的器材线稿。</li>
 <li><strong>烧杯的 <code>phase7_semantic_control.png</code> 和程序图有关系吗？</strong>
 有“表达同一个滴定装置”的人工设计关系，但<strong>没有自动文件派生关系</strong>。
 它不是从本帧 <code>clean.png</code> 检边得到，也没有读取
 <code>semantic_layers.json</code>。早期实验的
 <a href="{_href(CHEM_CONTROL_BUILDER)}">build_control.py</a>
 用固定像素坐标画了烧杯、滴定管、活塞和液面；Phase 7 把那张冻结模板原样复制为
 控制输入。因此当前实现存在模板与程序图以后可能不一致的风险。</li>
 <li><strong>哪些参数真正控制了结果？</strong>
 本次真正生效的是提示词、1024×576 尺寸、30 个去噪步、guidance 6.0、
 ControlNet scale 0.65、seed 7101、Euler scheduler 和 FP16 权重。
 配置里虽然还写着 <code>strength: 0.5</code>，但 T2I 分支没有把它传给模型，
 所以它没有参与这张图。后文逐项解释每个参数增大或减小时会发生什么。</li>
</ol>

<div class="steps">
 <div class="step"><strong>0 程序事实</strong>四个时刻的 pH、液面和局部羽流</div>
 <div class="step a"><strong>1 A 生成外观</strong>结构控制 + 外观提示词 → 六张候选</div>
 <div class="step a"><strong>2 冻结一张</strong>选中同一个烧杯、相机和光照</div>
 <div class="step b"><strong>3 B 生成状态</strong>同一底图 + 四个程序状态 → 四帧</div>
</div>

<h2>先把 Canny、ControlNet、SDXL 和 FP16 分开</h2>
<p>“SDXL Base 1.0 FP16 + SDXL Canny ControlNet FP16”看起来像一个很长的模型名，
实际是两个神经网络权重一起运行，再加上一个普通图像算法。它们不是同一个东西：</p>
<table><thead><tr><th>名称</th><th>它是什么</th><th>输入</th><th>输出</th>
<th>本项目里负责什么</th></tr></thead><tbody class="io">
<tr><td>Canny</td><td>经典边缘检测算法，不是生成模型</td>
<td>一张普通 RGB/灰度图片和高低阈值</td><td>黑底白线的二值边缘图</td>
<td>可以把程序截图自动转成 ControlNet 能读的线图；但本次选中结果没有使用自动
dense Canny</td></tr>
<tr><td>SDXL Canny ControlNet FP16</td><td>专门学习“边缘图应怎样约束 SDXL”的辅助神经网络</td>
<td>控制图、当前带噪 latent、当前去噪时间步、文字语义</td>
<td><strong>多层控制残差，不是 RGB 图片</strong></td>
<td>告诉 SDXL 的不同网络层：烧杯、滴定管和液面结构应出现在哪里</td></tr>
<tr><td>SDXL Base 1.0 FP16</td><td>真正负责生成图像内容的主扩散模型</td>
<td>随机 latent、正负提示词语义、ControlNet 残差和当前时间步</td>
<td>逐步去噪后的 latent；随后由 VAE 解码为图片</td>
<td>生成玻璃、折射、阴影、实验室背景和整体真实感</td></tr>
<tr><td>FP16</td><td>模型权重和计算使用 16-bit floating point</td>
<td>同一个模型的半精度权重文件</td><td>更低显存占用和较快计算</td>
<td>它不是画风，也不表示最终 PNG 是“16 位图”；最终保存的是普通 RGB PNG</td></tr>
</tbody></table>

<h3>Canny 到底做了什么</h3>
<p>Canny 会先计算图像亮度梯度，保留局部最强边缘，再通过高低阈值和连通关系决定哪些
像素变成白线。Phase 7 保存的自动 dense Canny 使用阈值 5/15，并膨胀一次。
它只知道“这里亮度突然变化”，不知道哪条线是烧杯、文字、液面还是无关装饰。</p>
<p>本次接受的 <code>phase7_semantic_control.png</code> 则不是对程序截图直接运行
Canny 得到的，也不是由 Phase 7 的程序语义层自动生成的。准确血缘是：
早期实验的 <code>build_control.py</code> 在 1024×576 画布上用硬编码坐标绘制器材
→ 保存 <code>semantic_apparatus_line_art.png</code>
→ Phase 7 原样复制为 <code>phase7_semantic_control.png</code>。
这张模板只留下一个烧杯、一个滴定管、活塞、滴嘴和液面。因为它仍是黑底白线、
形态接近 Canny 边缘，所以可以输入训练在 Canny 图上的 ControlNet；ControlNet
并不要求推理时一定再调用一次 Canny 算法。</p>
<pre>标准链路：
普通图片 / 程序图 → Canny 算法 → 边缘图 → Canny ControlNet → 特征残差 → SDXL

本次接受链路：
固定坐标器材模板 ─────────────→ 器材线图 → Canny ControlNet → 特征残差 → SDXL

本次仅作对照：
程序 clean.png → dense Canny（阈值 5/15 + 膨胀一次）→ 保存，但没有送入选中管线</pre>
<figure><img src="{_href(assets['a_control_comparison'])}">
<figcaption>从左到右：程序图；OpenCV 自动 dense Canny；本次真正送入 ControlNet 的
语义线图；模型 raw 输出。第三张才是选中管线的实际控制输入。</figcaption></figure>

<h3>ControlNet 与 SDXL 的连接关系</h3>
<pre>正向/负向文字
    │
    └→ SDXL tokenizer + 两个 text encoder → 文字语义向量 ───────────┐
                                                                   │
固定器材模板 → 黑白结构控制图 → Canny ControlNet ─→ 多层控制残差 ───┤
                              ↑            × 0.65                  │
随机 latent + 当前时间步 + 文字语义 ────────┘                       │
复现编号 7101 → 随机初始 latent ─→ Euler scheduler，循环 30 步 ──────┤
                                                                   ▼
                                                        SDXL Base UNet
                                                                   │
                                                        去噪后的 latent
                                                                   │
                                                              SDXL VAE
                                                                   │
                                                        1024×576 RGB PNG</pre>
<p><span class="term">ControlNet 不替代 SDXL。</span>它在每个去噪时间步观察控制图，
同时接收当前带噪 latent、时间步和文字语义，输出一组与 SDXL 不同层尺寸对应的
残差特征。Diffusers 把这些输出乘 0.65，再分别作为
<code>down_block_additional_residuals</code> 和
<code>mid_block_additional_residual</code> 加进 SDXL UNet。
这里的“残差”可理解为网络内部的结构修正信号，不是 RGB 图片，也不是最终图减原图。
SDXL 仍然负责决定白线之间填什么材质、是什么光照，以及背景长什么样。如果只有
ControlNet 而没有 SDXL Base，不会得到最终图片。</p>

<h3>一个去噪步内部实际发生什么</h3>
<ol>
 <li>Euler scheduler 按当前时间步缩放带噪 latent。</li>
 <li>ControlNet 读取该 latent、时间步、文字语义和器材线图，产生 down-block 与
 mid-block 的多层特征残差，并乘 ControlNet scale。</li>
 <li>SDXL UNet 在相同 latent 上运行，同时接收文字语义和上述残差，分别预测
 “无条件/负向”与“正向”两份噪声。</li>
 <li>guidance 6.0 合并两份预测；Euler scheduler 据此把 latent 更新到下一步。</li>
</ol>
<pre>noise = noise_uncond + 6.0 × (noise_text - noise_uncond)
latent_next = Euler.step(noise, current_timestep, latent_current)</pre>
<p>以上循环 30 次，最后 VAE 才把 latent 解码为人能看到的 RGB 图片。</p>

<h3>本次实际管线的输入和输出</h3>
<table><thead><tr><th>组件</th><th>本次实际输入</th><th>本次实际输出/用途</th></tr></thead>
<tbody class="io">
<tr><td>文字 tokenizer</td><td>完整正向提示词 65 token、负向提示词 59 token</td>
<td>两个 SDXL text encoder 使用的语义向量；两者都低于 77 token 上限，没有截断</td></tr>
<tr><td>随机输入</td><td><code>torch.Generator(device="cuda").manual_seed(7101)</code></td>
<td>生成初始随机 latent；项目没有另存一张“噪声图片”，复现依靠编号和完整运行配置</td></tr>
<tr><td>ControlNet 图像输入</td><td>1024×576 的
<code>phase7_semantic_control.png</code></td>
<td>ControlNet 多层残差，缩放系数 0.65</td></tr>
<tr><td>SDXL Base</td><td>随机 latent + 文字语义 + ControlNet 残差</td>
<td><code>raw/semantic_control_065/seed_7101.png</code></td></tr>
<tr><td>B</td><td>上面的 PNG + 四帧液体遮罩和 pH 数值场</td>
<td>四张确定性关键帧；不再调用 SDXL 或 ControlNet</td></tr>
</tbody></table>

<div class="card warning"><strong>一个很容易误解的实现细节：</strong>
本次 A 使用的是 <code>controlnet_t2i</code>，即从随机噪声开始的 text-to-image。
程序的 <code>clean.png</code> 和 <code>semantic_layers.json</code> 被登记为实验来源，
供人审计和后续 B 使用，但在本次 A 中既没有作为 img2img 初始图片送进 SDXL，
也没有用来生成实际器材控制图。真正进入图片端口的只有那张冻结黑白模板。
配置中的 <code>strength: 0.5</code> 来自共享实验 schema；在这条 T2I 调用中没有传给
pipeline，因此不影响选中结果。</div>

<h2>参数字典：每个数到底控制什么</h2>
<p>下面区分三件常被混淆的事：<strong>guidance</strong> 控制文字牵引，
<strong>ControlNet scale</strong> 控制结构线牵引，<strong>img2img strength</strong>
控制对初始图片的改写幅度。三者不是同一个旋钮。</p>
<table><thead><tr><th>参数</th><th>本次值/是否生效</th><th>它控制什么</th>
<th>增大或减小的通常影响</th></tr></thead><tbody class="io">
<tr><td><code>pipeline_mode</code></td><td><code>controlnet_t2i</code>，生效</td>
<td>决定从随机 latent 开始，还是从一张初始图加噪后开始</td>
<td>本次是 T2I；没有可调的“保留原图百分比”</td></tr>
<tr><td>正向提示词</td><td>65 token，生效且未截断</td>
<td>希望出现的场景、器材、玻璃材质、视角和光照</td>
<td>改词会改变语义和外观；不是固定空间位置的可靠手段</td></tr>
<tr><td>负向提示词</td><td>59 token，生效且未截断</td>
<td>在 CFG 的“无条件/负向”分支中压制多余器材、文字、塑料感等</td>
<td>过长或互相冲突可能削弱效果；本次两套 tokenizer 上限均为 77</td></tr>
<tr><td><code>guidance_scale</code><br>也叫 CFG</td><td>6.0，生效</td>
<td>放大正向预测与负向预测之间的差，决定模型多用力服从文字</td>
<td>更高通常更贴文字，但可能过饱和、生硬或出现伪影；更低更自由，也更可能漏掉要求。
它不控制线稿强度</td></tr>
<tr><td><code>controlnet_conditioning_scale</code></td>
<td>试 0.65 / 0.80；选中 0.65，生效</td>
<td>乘在 ControlNet 多层残差上的幅度系数；这里“scale”是特征信号强度，
不是把图片尺寸缩放到 65%</td>
<td>更高通常更严格贴线，但会压缩 SDXL 的材质自由度，可能留下硬线、塑料感；
更低材质更自由，但器材形状更可能漂移</td></tr>
<tr><td><code>strength</code><br>img2img strength</td><td>配置写 0.5，
<strong>本次未生效</strong></td><td>仅在 img2img 中决定给初始图加入多少噪声、
允许离开初始图多远</td><td>若在 img2img 使用：高值改写更大，低值更保留原图。
本次 T2I 调用根本没有这个参数；不要把 0.5 与 ControlNet 的 0.65 混淆</td></tr>
<tr><td><code>num_inference_steps</code></td><td>30，生效</td>
<td>UNet + ControlNet + scheduler 的去噪循环次数</td>
<td>更多步更慢，通常给模型更多细化机会，但不保证单调变好；减少则更快</td></tr>
<tr><td><code>seed</code></td><td>7101，生效；另试 7102、7103</td>
<td>初始化 GPU 随机 latent；它是复现编号，不是场景语义或质量分数</td>
<td>换 seed 相当于换噪声起点，构图、反射和细节都会变化。精确复现还需相同权重、
软件、硬件与参数</td></tr>
<tr><td><code>width × height</code></td><td>1024×576，生效</td>
<td>输出与控制的空间网格、16:9 画幅</td>
<td>提高尺寸增加显存和计算量，也可能改变构图；它与 ControlNet scale 无关</td></tr>
<tr><td><code>scheduler</code></td><td><code>EulerDiscreteScheduler</code>，生效</td>
<td>安排 30 个噪声时间步，并把 UNet 的噪声预测更新成下一份 latent</td>
<td>换 scheduler 即使 seed 不变也可能改变结果；它不是另一个模型</td></tr>
<tr><td><code>control_guidance_start/end</code></td><td>未显式传入，
使用 Diffusers 默认 0.0 / 1.0</td><td>ControlNet 在整个去噪进度的哪一段介入</td>
<td>本次从第一步持续到最后一步；缩短区间可让后段更自由，但本轮未做此扫描</td></tr>
<tr><td><code>guess_mode</code></td><td>未显式传入，默认 false</td>
<td>是否让 ControlNet 尝试脱离提示词自行识别控制图内容</td>
<td>本次关闭；没有把它作为优化变量</td></tr>
<tr><td><code>dtype / variant</code></td><td>FP16，生效</td>
<td>权重加载与张量计算精度，主要影响显存、速度和极小数值差异</td>
<td>它不控制“真实感强度”，也不表示输出 PNG 是 16-bit 图</td></tr>
</tbody></table>

<details open><summary>把本次选中结果还原成一次模型调用</summary>
<p>项目代码先加载 SDXL Base、Canny ControlNet 与它们各自的 FP16 本地权重，
然后实际调用等价于：</p>
<pre>generator = torch.Generator(device="cuda").manual_seed(7101)
result = pipeline(
    prompt=positive_prompt,
    negative_prompt=negative_prompt,
    image=phase7_semantic_control_png,  # ControlNet 图，不是 img2img 初始图
    width=1024,
    height=576,
    num_inference_steps=30,
    guidance_scale=6.0,
    controlnet_conditioning_scale=0.65,
    generator=generator,
).images[0]

# 注意：这里没有 clean.png，也没有 strength=0.5</pre>
<p>Phase 7 实际分支可在
<a href="{_href(STAGE2_ROOT / 'framework/image_experiment.py')}">image_experiment.py</a>
中查看。历史 <code>generate.json</code> 对每条候选都统一写了
<code>img2img_strength: 0.5</code>，classification 也沿用了
“Img2Img output”；这两个字段是共享记录代码留下的误导性元数据，不能覆盖
<code>pipeline_mode: controlnet_t2i</code> 和实际函数分支。报告以实际调用代码为准。</p>
</details>

<details><summary>“两个 FP16 模型”对应哪些本地权重</summary>
<p>主模型 ID：<code>stabilityai/stable-diffusion-xl-base-1.0</code>；
辅助模型 ID：<code>diffusers/controlnet-canny-sdxl-1.0</code>。
主模型包含两个 text encoder、UNet 和 VAE；ControlNet 有自己的
<code>diffusion_pytorch_model.fp16.safetensors</code>。具体本地路径和每个权重文件
SHA-256 位于
<a href="{_href(CHEM_A_ROOT / '_work/model_fingerprints.json')}">model_fingerprints.json</a>。
运行时使用 Diffusers 0.35.2、Torch 2.9.1、EulerDiscreteScheduler、30 步。</p></details>

<h2>步骤 0：程序先决定“发生什么”</h2>
<p>下面四张不是模型图，而是 Phase 2 程序的无标注关键帧。程序在这一阶段已经确定：
液体区域在哪里、液面多高、整体 pH 是多少、机制帧是否有局部高 pH 羽流。A/B 都不得
修改这些因果关系。</p>
<figure><img src="{_href(assets['program_sequence'])}">
<figcaption>四张程序关键帧。它们是事实源，不是最终美术稿。</figcaption></figure>
<table><thead><tr><th>状态</th><th>整体 pH</th><th>液体内 min / mean / max</th>
<th>程序液面 y</th><th>液体像素</th></tr></thead><tbody>
{_state_rows(states)}</tbody></table>

<h3>mask（遮罩）实际是什么</h3>
<p>白色区域是 <code>chem01_liquid_region</code>：程序明确声明“这是液体”。
它不是给人看的装饰，也不是模型自动猜出来的。B 用它裁出允许承载 pH 状态的区域；
黑色区域不能被液体颜色修改。</p>
<figure><img src="{_href(assets['mask_sequence'])}">
<figcaption>四个实际液体遮罩。滴定液增加后，液体面积随液面变化。</figcaption></figure>

<h3>pH scalar field（标量场）实际是什么</h3>
<p>每个液体像素保存一个浮点 pH。下图只是把数值着色方便人查看：蓝色偏酸，
红色偏碱。机制帧中间的暖色区域就是局部羽流；起始和结果帧接近均匀酸性；
终点整杯进入碱性指示区。B 读取的是原始 <code>.npy</code> 数值，不是这张彩色预览。</p>
<figure><img src="{_href(assets['ph_sequence'])}">
<figcaption>四份程序 pH 场的可视化。只有机制帧存在明显的局部高值。</figcaption></figure>

<h2>步骤 1：A 只生成“真实烧杯长什么样”</h2>
<p>A 的模型输入有两部分。第一部分是黑底白线的结构控制图，固定一个烧杯、一个滴定管、
正视相机和液面大位置；这就是 ControlNet 的 conditioning。第二部分是文字，只描述
玻璃、折射、光照和场景外观。需要再次强调：这里的线图是固定器材模板，不是当前
程序关键帧自动产生的。</p>
<figure><img src="{_href(assets['a_inputs'])}">
<figcaption>从左到右：程序起始图；模型实际看到的结构控制；最终选中的 A 底图。
A 不读取四帧 pH 场来生产最终序列。</figcaption></figure>

<details open><summary>A 选中底图时使用的完整提示词和参数</summary>
<h4>正向提示词</h4><pre>{html.escape(positive)}</pre>
<h4>负向提示词</h4><pre>{html.escape(negative)}</pre>
<p>模型：SDXL Base 1.0 FP16 + SDXL Canny ControlNet FP16；尺寸 1024×576；
30 步；CFG guidance 6.0；ControlNet scale 试 0.65 和 0.80；每档三个固定复现编号。
所以起始场景共有 2×3=6 张 raw 候选。</p></details>

<figure><img src="{_href(CHEM_A_ROOT / 'candidates-labeled.jpg')}">
<figcaption>A 的六张起始场景候选。标签中的数字是固定随机噪声起点，不是画面含义。
最终选择控制强度 0.65、复现编号 7101。</figcaption></figure>

<h2>步骤 2：为什么只冻结一张，而不是让 A 直接画四张</h2>
<p>Phase 7 确实做过“同一复现编号 + 四条状态提示词，各生成一帧”的实验。
下面就是实际结果。单张玻璃很好，但不同状态改变了背景光、液体外观和构图；文字中的
“局部粉色”也经常被扩大。它们被保留为失败证据，不进入最终序列。</p>
<figure><img src="{_href(assets['a_rejected_direct_sequence'])}">
<figcaption>被拒绝的 A 直接四帧。看似顺序相关，实际是四次相互独立的扩散生成。</figcaption></figure>
<div class="card warning"><strong>生产规则因此改变：</strong>未来不会再为每一个关键帧
分别运行 A。A 只在没有合适底图时运行一次；候选选中后记录文件路径和 SHA-256，
所有 B 帧共同读取同一个文件。</div>

<h3>被选中的冻结文件</h3>
<pre>{html.escape(str(CHEM_A_BASE.relative_to(STAGE2_ROOT)))}
SHA-256: {sha256_path(CHEM_A_BASE)}</pre>
<p>化学 B manifest 中的 donor 路径和哈希与上面完全相同。这是本轮真正的 A→B
文件级顺承，而不是报告文字上的推测。</p>

<h2>步骤 3：B 怎样把程序状态放进真实底图</h2>
<p>B 同时读取两类输入：</p>
<ol>
 <li><strong>不变输入：</strong>A 选中的同一张真实烧杯图。</li>
 <li><strong>逐帧输入：</strong>每一帧的液体遮罩、pH 场、液面高度和机制状态。</li>
</ol>
<p>程序图是 640×360，A 底图是 1024×576，不能把程序矩形直接贴上去。B 先做
surface calibration（表面标定）：把程序液体裁到自身最小矩形，再映射到真实烧杯中的
梯形表面。下图粉色边线就是实际使用的真实液体目标区。</p>
<figure><img src="{_href(assets['b_mapping'])}">
<figcaption>程序遮罩 → 真实图片中的液体梯形 → 映射后的终点状态。
这一步解决了早期“粉色矩形跑到杯外”的错误。</figcaption></figure>

<details open><summary>B 的实际数值变换</summary>
<p>真实表面的左右范围是 x=350–675，底边 y=493；左右下角收窄到 x=370 和 652。
顶边 y 根据每帧 <code>liquid_level_y</code> 线性换算，边缘做 5 px 羽化。</p>
<pre>indicator = 1 / (1 + exp(-(pH - 8.15) * 2.5))
alpha = indicator × 0.52 × feathered_liquid_mask</pre>
<p>indicator 把程序 pH 转成酚酞粉色强度。粉色目标还会按 A 底图原有亮度缩放，
所以杯中的高光和反射不会被纯色矩形盖住。机制帧额外在滴定管尖端加入一枚羽化液滴；
光学边缘强度为 0.38。B 在此阶段没有调用 SDXL，全部是确定性数组计算。</p></details>

<h2>步骤 4：最终得到的四张 B 关键帧</h2>
<figure><img src="{_href(assets['final_sequence'])}">
<figcaption>同一张 A 底图产生的最终四帧：无色 → 局部羽流和液滴 → 褪色 →
稳定浅粉。玻璃、相机、背景、桌面反射完全共享。</figcaption></figure>
<p>自动检查确认：</p>
<ul>
 <li>四帧指示剂液体均值顺序为 0、0.143768、0.000015、0.879632；</li>
 <li>器材之外的冻结背景最大像素差为 0；</li>
 <li>B 输出 3 个变体 × 4 帧，但生产选择固定为
 <code>{CHEM_B_VARIANT}</code>，重跑不重新挑图；</li>
 <li>B 的图片模型调用数为 0。</li>
</ul>

<h2>其他案例当时到底有没有 A→B</h2>
<figure><img src="{_href(assets['other_lineages'])}">
<figcaption>四条真实血缘。只有化学使用了 Phase 7 的新 A 底图；其他案例各有不同来源。</figcaption></figure>
<table><thead><tr><th>案例</th><th>真实外观来源</th><th>后续</th>
<th>能否称为 Phase 7 A→B</th></tr></thead><tbody>
<tr><td>CHEM-01</td><td>Phase 7 A 的烧杯候选 7101</td>
<td>Phase 7 B 生成四帧</td><td class="pass">可以，完整直接血缘</td></tr>
<tr><td>BIO-01</td><td>Phase 3 已保存的细胞供体 3104</td>
<td>Phase 7 B 抽取纹理统计</td><td class="no">不可以，是历史供体→B</td></tr>
<tr><td>GEO-02</td><td>Phase 7 A 的地形候选</td>
<td>没有继续生成 B 关键帧</td><td class="no">不可以，只有 A</td></tr>
<tr><td>MATH-02</td><td>Phase 3 木纹供体 3101</td>
<td>Phase 7 C；Phase 8 B2 对照</td><td class="no">不可以，是历史供体→后端</td></tr>
<tr><td>PHYS-01</td><td>Phase 3 水体供体 3101/3102</td>
<td>Phase 7 C；Phase 8 B3 对照</td><td class="no">不可以，是历史供体→后端</td></tr>
</tbody></table>

<h2>放回通用项目后的真实运行规则</h2>
<pre>程序输出：四个状态 + 遮罩/对象/高度场
                    │
                    ├─ 已经有合适照片或冻结供体 ─────────────┐
                    │                                        │
                    └─ 没有合适外观 → A 生成候选 → 冻结一张 ─┤
                                                             ▼
                                                           B
                                          ┌──────────────────┼──────────────────┐
                                          B1                 B2                 B3
                                      区域/标量状态       对象附着纹理       高度/法线光学
                                          └──────────────────┼──────────────────┘
                                                             ▼
                                                     多张正确关键帧
                                                             ▼
                                                    视频模型生成过渡</pre>
<p>A 是可选的一次性准备步骤；B 是必需的多帧生产步骤。即使跳过 A，B 也必须记录冻结
外观来自哪里。即使使用 A，B 也只能读取一张已冻结结果，不能每帧重新生成。</p>

<h2>怎样复现这条实际链</h2>
<p>以下命令不会重新抽候选：Phase 7 B/C 使用现有冻结文件重建；Phase 9 只重建预览、
文件血缘和本报告。</p>
<pre>.venv/bin/python -m modules.video_model.stage2.phase7_hybrid_pbr
.venv/bin/python -m modules.video_model.stage2.phase9_ab_lineage
.venv/bin/python -m pytest modules/video_model/stage2/tests/test_phase9.py -q</pre>
<p>代码与证据：</p>
<ul>
 <li><a href="{_href(STAGE2_ROOT / 'phase7_semifree.py')}">A 候选生成代码</a></li>
 <li><a href="{_href(STAGE2_ROOT / 'phase7_hybrid_pbr.py')}">B 状态投影代码</a></li>
 <li><a href="{_href(CHEM_A_ROOT / 'spec.json')}">A 实验完整配置</a></li>
 <li><a href="{_href(CHEM_B_ROOT / 'manifest.json')}">B 输入、输出和指标</a></li>
 <li><a href="{_href(MANIFEST_PATH)}">本报告血缘清单</a></li>
</ul>
<p class="small">状态：{manifest['status']}。本报告新增模型调用：
{manifest['model_runs']['image']} 张图片、{manifest['model_runs']['video']} 段视频。</p>
</main></body></html>"""


def build(*, check_only: bool = False) -> dict[str, Any]:
    if not check_only:
        states, assets = _build_state_assets()
        _build_a_assets(assets)
        _build_calibration_asset(assets)
        _build_other_case_assets(assets)
    else:
        states = load_json(MANIFEST_PATH)["chem_states"]
        assets = {
            "program_sequence": ASSET_ROOT
            / "chem-program-sequence.jpg",
            "mask_sequence": ASSET_ROOT
            / "chem-liquid-masks.jpg",
            "ph_sequence": ASSET_ROOT / "chem-ph-fields.jpg",
            "a_inputs": ASSET_ROOT / "chem-a-inputs.jpg",
            "a_control_comparison": ASSET_ROOT
            / "chem-a-control-comparison.jpg",
            "a_rejected_direct_sequence": ASSET_ROOT
            / "chem-a-rejected-direct-sequence.jpg",
            "calibration": ASSET_ROOT
            / "chem-real-surface.png",
            "b_mapping": ASSET_ROOT / "chem-b-mapping.jpg",
            "final_sequence": ASSET_ROOT
            / "chem-final-b-sequence.jpg",
            "other_lineages": ASSET_ROOT
            / "other-actual-lineages.jpg",
        }
    chem_b = load_json(CHEM_B_ROOT / "manifest.json")
    generate = load_json(CHEM_A_ROOT / "_work/generate.json")
    selected_candidate = next(
        item
        for item in generate["candidates"]
        if item["configuration_id"] == "semantic_control_065"
        and item["seed"] == 7101
    )
    donor_matches = (
        chem_b["donor"]["sha256"] == sha256_path(CHEM_A_BASE)
        and (
            STAGE2_ROOT / chem_b["donor"]["path"]
        ).resolve()
        == CHEM_A_BASE.resolve()
    )
    selected_records = [
        item
        for item in chem_b["records"]
        if item["variant"] == CHEM_B_VARIANT
    ]
    manifest = {
        "schema_version": "1.0",
        "phase": 9,
        "status": "passed",
        "classification": "documentation_and_lineage_audit",
        "model_runs": {"image": 0, "video": 0},
        "actual_relation_zh": (
            "CHEM-01 是 Phase 7 唯一直接执行本轮 A→B 的"
            "案例；A 的一张冻结输出被 B 的四个程序状态共同读取。"
        ),
        "chem_a_selected_base": artifact_record(
            CHEM_A_BASE, STAGE2_ROOT
        ),
        "chem_a_pipeline": {
            "pipeline_mode": generate["pipeline_mode"],
            "program_clean_used_as_img2img_initial_image": False,
            "program_clean_used_to_derive_actual_control": False,
            "semantic_layers_used_to_derive_actual_control": False,
            "shared_schema_strength_applied": False,
            "actual_control_input": artifact_record(
                CHEM_A_CONTROL,
                STAGE2_ROOT,
            ),
            "actual_control_source": {
                "classification": (
                    "fixed-coordinate deterministic apparatus "
                    "template, not derived from current program frame"
                ),
                "builder": artifact_record(
                    CHEM_CONTROL_BUILDER, STAGE2_ROOT
                ),
                "frozen_template": artifact_record(
                    CHEM_CONTROL_TEMPLATE, STAGE2_ROOT
                ),
                "copied_without_pixel_change": (
                    sha256_path(CHEM_CONTROL_TEMPLATE)
                    == sha256_path(CHEM_A_CONTROL)
                ),
            },
            "automatic_dense_canny_not_selected": artifact_record(
                CHEM_A_ROOT / "controls/dense_canny.png",
                STAGE2_ROOT,
            ),
            "positive_tokens": generate[
                "prompt_token_preflight"
            ]["positive"],
            "negative_tokens": generate[
                "prompt_token_preflight"
            ]["negative"],
            "models": generate["models"],
            "scheduler": generate["scheduler"],
            "selected_configuration_id": selected_candidate[
                "configuration_id"
            ],
            "selected_controlnet_scale": selected_candidate[
                "controlnet_conditioning_scale"
            ],
            "selected_seed": selected_candidate["seed"],
            "raw_output": artifact_record(
                CHEM_A_BASE, STAGE2_ROOT
            ),
        },
        "chem_b_manifest": artifact_record(
            CHEM_B_ROOT / "manifest.json", STAGE2_ROOT
        ),
        "chem_states": states,
        "other_actual_lineages": [
            {
                "case_id": "BIO-01",
                "source": "Phase 3 model donor",
                "consumer": "Phase 7 B",
                "direct_phase7_A_to_B": False,
            },
            {
                "case_id": "GEO-02",
                "source": "Phase 7 A",
                "consumer": None,
                "direct_phase7_A_to_B": False,
            },
            {
                "case_id": "MATH-02",
                "source": "Phase 3 model donor",
                "consumer": "Phase 7 C / Phase 8 B2 test",
                "direct_phase7_A_to_B": False,
            },
            {
                "case_id": "PHYS-01",
                "source": "Phase 3 model donors",
                "consumer": "Phase 7 C / Phase 8 B3 test",
                "direct_phase7_A_to_B": False,
            },
        ],
        "artifacts": {
            name: artifact_record(path, STAGE2_ROOT)
            for name, path in assets.items()
        },
        "checks": [
            {
                "name": "chem_B_donor_is_exactly_selected_A_output",
                "passed": donor_matches,
                "evidence": {
                    "A_sha256": sha256_path(CHEM_A_BASE),
                    "B_donor_sha256": chem_b["donor"][
                        "sha256"
                    ],
                },
            },
            {
                "name": "four_program_states_documented",
                "passed": len(states) == 4,
            },
            {
                "name": "four_selected_B_frames_exist",
                "passed": len(selected_records) == 4
                and all(
                    (
                        CHEM_B_ROOT
                        / item["output"]["path"]
                    ).is_file()
                    for item in selected_records
                ),
            },
            {
                "name": "B_used_no_new_image_or_video_model",
                "passed": chem_b["model_runs"]
                == {"image": 0, "video": 0},
            },
            {
                "name": "A_actual_pipeline_is_controlnet_text_to_image",
                "passed": (
                    generate["pipeline_mode"] == "controlnet_t2i"
                    and selected_candidate["pipeline_mode"]
                    == "controlnet_t2i"
                ),
            },
            {
                "name": "A_actual_models_are_sdxl_base_and_canny_controlnet",
                "passed": (
                    generate["models"]["sdxl_base"]["model_id"]
                    == "stabilityai/stable-diffusion-xl-base-1.0"
                    and generate["models"]["controlnet_canny"][
                        "model_id"
                    ]
                    == "diffusers/controlnet-canny-sdxl-1.0"
                ),
            },
            {
                "name": "A_actual_control_is_frozen_template_copy",
                "passed": (
                    sha256_path(CHEM_CONTROL_TEMPLATE)
                    == sha256_path(CHEM_A_CONTROL)
                ),
                "evidence": {
                    "template_sha256": sha256_path(
                        CHEM_CONTROL_TEMPLATE
                    ),
                    "phase7_control_sha256": sha256_path(
                        CHEM_A_CONTROL
                    ),
                },
            },
        ],
    }
    report = _render_report(states, assets, manifest)
    manifest["checks"].append(
        {
            "name": "report_links_resolve",
            "passed": _links_resolve(report),
        }
    )
    if not all(item["passed"] for item in manifest["checks"]):
        manifest["status"] = "failed"
    if not check_only:
        REPORT_PATH.write_text(report, encoding="utf-8")
        write_json(MANIFEST_PATH, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = build(check_only=args.check_only)
    print(
        f"Phase 9 A→B report: {result['status']} · "
        f"{len(result['checks'])} checks"
    )


if __name__ == "__main__":
    main()
