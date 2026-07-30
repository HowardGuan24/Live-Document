"""Review and report the three Phase 7 image-rendering routes.

This module does not run a diffusion model.  It inventories Route A model
outputs, verifies Routes B/C, builds explanatory contact sheets, records the
visual review, and writes a beginner-readable HTML report.
"""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .framework.contracts import (
    artifact_record,
    load_json,
    write_json,
)


STAGE2_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = STAGE2_ROOT / "output/phase-7"
REPORT_PATH = OUTPUT_ROOT / "report.html"
MANIFEST_PATH = OUTPUT_ROOT / "phase7-manifest.json"
REVIEW_PATH = OUTPUT_ROOT / "phase7-review.json"
ASSET_ROOT = OUTPUT_ROOT / "report-assets"
KEYFRAMES = (
    "00_start",
    "01_mechanism",
    "02_result",
    "03_end",
)
KEYFRAME_LABELS = {
    "00_start": "起始",
    "01_mechanism": "机制出现",
    "02_result": "阶段结果",
    "03_end": "结束",
}


def select_appearance_route(
    *,
    has_exact_geometry_or_field: bool,
    has_frozen_real_base: bool,
    state_is_local_or_semantic: bool,
) -> str:
    """Select an appearance route from data properties, not a case name."""

    if has_exact_geometry_or_field:
        return "C_exact_renderer"
    if has_frozen_real_base and state_is_local_or_semantic:
        return "B_frozen_base_projection"
    return "A_once_then_B"


def _href(path: Path) -> str:
    return os.path.relpath(path, REPORT_PATH.parent).replace(
        os.sep, "/"
    )


def _sheet(
    cells: list[tuple[str, Path]],
    output: Path,
    *,
    columns: int,
    thumb: tuple[int, int] = (384, 216),
) -> None:
    label_height = 42
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * thumb[0],
            rows * (thumb[1] + label_height),
        ),
        (12, 29, 37),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(cells):
        column = index % columns
        row = index // columns
        x = column * thumb[0]
        y = row * (thumb[1] + label_height)
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb, Image.Resampling.LANCZOS)
        tile = Image.new("RGB", thumb, (228, 226, 215))
        tile.paste(
            image,
            (
                (thumb[0] - image.width) // 2,
                (thumb[1] - image.height) // 2,
            ),
        )
        sheet.paste(tile, (x, y))
        draw.text(
            (x + 10, y + thumb[1] + 10),
            label,
            fill=(237, 245, 242),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, subsampling=0)


def _semantic_layer(
    case_id: str, keyframe_id: str, layer_id: str
) -> Path:
    keyframe_root = (
        STAGE2_ROOT
        / "output/phase-2"
        / case_id
        / "keyframes"
        / keyframe_id
    )
    manifest = load_json(keyframe_root / "semantic_layers.json")
    layer = next(
        item
        for item in manifest["layers"]
        if item["layer_id"] == layer_id
    )
    return (
        STAGE2_ROOT
        / "output/phase-2"
        / case_id
        / layer["data"]["path"]
    )


def _save_scalar_preview(source: Path, output: Path) -> None:
    field = np.load(source, allow_pickle=False).astype(np.float32)
    minimum = float(field.min())
    maximum = float(field.max())
    normalized = (field - minimum) / max(maximum - minimum, 1e-8)
    red = np.clip(255.0 * normalized, 0, 255)
    green = np.clip(
        255.0 * (1.0 - np.abs(normalized - 0.5) * 2.0),
        0,
        255,
    )
    blue = np.clip(255.0 * (1.0 - normalized), 0, 255)
    preview = np.stack((red, green, blue), axis=-1)
    Image.fromarray(
        np.uint8(np.rint(preview)), mode="RGB"
    ).save(output, optimize=False)


def _save_mask_preview(source: Path, output: Path) -> None:
    mask = np.load(source, allow_pickle=False)
    preview = np.zeros((*mask.shape, 3), dtype=np.uint8)
    preview[:] = (18, 38, 46)
    preview[mask > 0] = (119, 224, 171)
    Image.fromarray(preview, mode="RGB").save(
        output, optimize=False
    )


def _selected_paths(
    route: str, case_id: str, variant: str
) -> list[Path]:
    return [
        OUTPUT_ROOT
        / route
        / case_id
        / "variants"
        / variant
        / f"{keyframe_id}.png"
        for keyframe_id in KEYFRAMES
    ]


def _build_assets() -> dict[str, Path]:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    route_a_chem = []
    for keyframe_id in KEYFRAMES:
        route_a_chem.append(
            (
                KEYFRAME_LABELS[keyframe_id],
                OUTPUT_ROOT
                / "route-a/experiments"
                / f"EXP-P7-A-chem-01-{keyframe_id}"
                / "raw/semantic_control_065/seed_7101.png",
            )
        )
    paths["route_a_chem_direct"] = (
        ASSET_ROOT / "route-a-chem-direct-sequence.jpg"
    )
    _sheet(route_a_chem, paths["route_a_chem_direct"], columns=4)

    selected = {
        "chem": (
            "route-b",
            "CHEM-01",
            "optical_tint_plus_drop",
        ),
        "bio": (
            "route-b",
            "BIO-01",
            "stable_material_plus_depth",
        ),
        "math": ("route-c", "MATH-02", "studio_pbr"),
        "phys": ("route-c", "PHYS-01", "specular_water"),
    }
    for name, (route, case_id, variant) in selected.items():
        paths[f"{name}_selected"] = (
            ASSET_ROOT / f"{name}-selected-sequence.jpg"
        )
        _sheet(
            list(
                zip(
                    (
                        KEYFRAME_LABELS[item]
                        for item in KEYFRAMES
                    ),
                    _selected_paths(route, case_id, variant),
                    strict=True,
                )
            ),
            paths[f"{name}_selected"],
            columns=4,
        )

    chemistry_base = (
        OUTPUT_ROOT
        / "route-a/experiments/"
        "EXP-P7-A-chem-01-00_start/raw/"
        "semantic_control_065/seed_7101.png"
    )
    paths["chem_process"] = ASSET_ROOT / "chem-process.jpg"
    _sheet(
        [
            (
                "1 PROGRAM STATE",
                STAGE2_ROOT
                / "output/phase-2/CHEM-01/keyframes/"
                "03_end/clean.png",
            ),
            (
                "2 APPARATUS CONTROL",
                OUTPUT_ROOT
                / "route-a/experiments/"
                "EXP-P7-A-chem-01-00_start/controls/"
                "phase7_semantic_control.png",
            ),
            ("3 FROZEN REAL BASE", chemistry_base),
            (
                "4 BASE + PROGRAM STATE",
                _selected_paths(
                    "route-b",
                    "CHEM-01",
                    "optical_tint_plus_drop",
                )[-1],
            ),
        ],
        paths["chem_process"],
        columns=4,
    )

    bio_mask = ASSET_ROOT / "bio-cell-mask.png"
    _save_mask_preview(
        _semantic_layer(
            "BIO-01", "01_mechanism", "bio01_cell_region"
        ),
        bio_mask,
    )
    paths["bio_mask"] = bio_mask
    paths["bio_process"] = ASSET_ROOT / "bio-process.jpg"
    _sheet(
        [
            (
                "1 PROGRAM FACTS",
                STAGE2_ROOT
                / "output/phase-2/BIO-01/keyframes/"
                "01_mechanism/clean.png",
            ),
            ("2 MUTABLE CELL MASK", bio_mask),
            (
                "3 MATERIAL DONOR",
                STAGE2_ROOT
                / "output/phase-3/EXP-20260729-013/"
                "raw/t2i_dense_cell_and_chromatids/"
                "seed_3104.png",
            ),
            (
                "4 MATERIAL + FACTS",
                _selected_paths(
                    "route-b",
                    "BIO-01",
                    "stable_material_plus_depth",
                )[1],
            ),
        ],
        paths["bio_process"],
        columns=4,
    )

    paths["math_process"] = ASSET_ROOT / "math-process.jpg"
    _sheet(
        [
            (
                "1 PROGRAM GEOMETRY",
                STAGE2_ROOT
                / "output/phase-2/MATH-02/keyframes/"
                "01_mechanism/clean.png",
            ),
            (
                "2 WOOD MATERIAL DONOR",
                STAGE2_ROOT
                / "output/phase-3/EXP-20260729-012/"
                "raw/t2i_wood_material_donor/seed_3101.png",
            ),
            (
                "3 OBJECT-LOCAL TEXTURE",
                _selected_paths(
                    "route-c", "MATH-02", "studio_pbr"
                )[1],
            ),
            (
                "4 EXACT END LAYOUT",
                _selected_paths(
                    "route-c", "MATH-02", "studio_pbr"
                )[-1],
            ),
        ],
        paths["math_process"],
        columns=4,
    )

    height_preview = ASSET_ROOT / "phys-height-field.png"
    _save_scalar_preview(
        _semantic_layer(
            "PHYS-01", "03_end", "phys01_surface_height"
        ),
        height_preview,
    )
    paths["phys_height"] = height_preview
    paths["phys_process"] = ASSET_ROOT / "phys-process.jpg"
    _sheet(
        [
            (
                "1 PROGRAM DIAGRAM",
                STAGE2_ROOT
                / "output/phase-2/PHYS-01/keyframes/"
                "03_end/clean.png",
            ),
            ("2 HEIGHT FIELD", height_preview),
            (
                "3 SAFE NORMAL LIGHTING",
                _selected_paths(
                    "route-c", "PHYS-01", "normal_diffuse"
                )[-1],
            ),
            (
                "4 SELECTED SPECULAR WATER",
                _selected_paths(
                    "route-c", "PHYS-01", "specular_water"
                )[-1],
            ),
        ],
        paths["phys_process"],
        columns=4,
    )

    paths["chem_before_after"] = (
        ASSET_ROOT / "chem-phase6-vs-phase7.jpg"
    )
    _sheet(
        [
            (
                "PHASE 6: +1.88 / 255 DETAIL",
                STAGE2_ROOT
                / "output/phase-6/image-regressions/"
                "CHEM-01/final/03_end.png",
            ),
            (
                "PHASE 7: REAL GLASS + STATE",
                _selected_paths(
                    "route-b",
                    "CHEM-01",
                    "optical_tint_plus_drop",
                )[-1],
            ),
        ],
        paths["chem_before_after"],
        columns=2,
    )
    paths["bio_before_after"] = (
        ASSET_ROOT / "bio-phase6-vs-phase7.jpg"
    )
    _sheet(
        [
            (
                "PHASE 6: +1.28 / 255 DETAIL",
                STAGE2_ROOT
                / "output/phase-6/image-regressions/"
                "BIO-01/final/03_end.png",
            ),
            (
                "PHASE 7: MICROSCOPE MATERIAL + FACTS",
                _selected_paths(
                    "route-b",
                    "BIO-01",
                    "stable_material_plus_depth",
                )[-1],
            ),
        ],
        paths["bio_before_after"],
        columns=2,
    )
    return paths


def _route_a_records() -> list[dict[str, Any]]:
    records = []
    experiment_root = OUTPUT_ROOT / "route-a/experiments"
    for root in sorted(
        path
        for path in experiment_root.iterdir()
        if path.is_dir()
    ):
        spec = load_json(root / "spec.json")
        generated = load_json(root / "_work/generate.json")
        records.append(
            {
                "experiment_id": spec["experiment_id"],
                "case_id": spec["case_id"],
                "keyframe_id": spec["source"]["keyframe_id"],
                "configuration_count": len(
                    spec["configurations"]
                ),
                "candidate_count": len(
                    generated["candidates"]
                ),
                "positive_prompt": (
                    root / "inputs/positive_prompt.txt"
                ).read_text(encoding="utf-8"),
                "negative_prompt": (
                    root / "inputs/negative_prompt.txt"
                ).read_text(encoding="utf-8"),
                "render": spec["render"],
                "control": artifact_record(
                    root
                    / "controls/phase7_semantic_control.png",
                    STAGE2_ROOT,
                ),
                "candidate_sheet": artifact_record(
                    root / "candidates-labeled.jpg",
                    STAGE2_ROOT,
                ),
            }
        )
    return records


def _review() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "reviewed",
        "review_method_zh": (
            "先检查程序事实和遮罩硬指标，再并排盲看所有候选，"
            "最后检查跨帧相机、对象、背景和局部状态是否稳定。"
        ),
        "score_scale_zh": "1=不可用，3=勉强可用，5=可进入后续视频",
        "route_a": {
            "status": "partial",
            "accepted_scope": (
                "生成一张真实的标准场景或材质底图；生成后冻结，"
                "后续状态不再分别重抽。"
            ),
            "rejected_scope": (
                "分别用四段自然语言生成四张教学关键帧。"
            ),
            "experiments": [
                {
                    "case": "CHEM-01 direct four-frame generation",
                    "scores": {
                        "single_frame_material": 4.6,
                        "structure": 4.1,
                        "state_accuracy": 2.2,
                        "cross_frame_consistency": 2.0,
                    },
                    "verdict": "donor_only",
                    "reason_zh": (
                        "玻璃、折射和光照很真实；但同一复现编号在"
                        "不同提示词下仍会换背景，粉色局部也常扩大成"
                        "整杯或环境灯光。"
                    ),
                },
                {
                    "case": "GEO-02 semantic landscape control",
                    "scores": {
                        "single_frame_material": 2.4,
                        "structure": 1.7,
                        "state_accuracy": 1.6,
                        "cross_frame_consistency": 1.5,
                    },
                    "verdict": "rejected",
                    "reason_zh": (
                        "把山体、云边和雨线同时交给 Canny 后，"
                        "模型把雨线误读成山脊、幕布或建筑。"
                    ),
                },
                {
                    "case": "GEO-02 terrain-only control",
                    "scores": {
                        "single_frame_material": 3.8,
                        "structure": 3.7,
                        "state_accuracy": 2.1,
                        "cross_frame_consistency": 2.0,
                    },
                    "verdict": "donor_only",
                    "reason_zh": (
                        "只控制山体轮廓后材质明显改善，但文字提示仍"
                        "不能可靠地把雨限制在迎风坡。"
                    ),
                },
            ],
            "preflight_failure": {
                "status": "rejected_before_generation",
                "reason_zh": (
                    "第一版正向提示词有 83 个 CLIP token，超过"
                    "本管线 77 token 上限；缩短到 65–75 token"
                    "后才生成。没有截断后冒充完整提示词。"
                ),
                "evidence_limit_zh": (
                    "失败发生在候选生成之前，因此没有图片；当时的"
                    "83-token 草稿也未单独落盘。这是本轮确认的日志"
                    "缺口，当前 review 只保存计数和修正说明。"
                ),
            },
        },
        "route_b": {
            "status": "accepted_for_material_cases",
            "cases": [
                {
                    "case_id": "CHEM-01",
                    "selected": "optical_tint_plus_drop",
                    "scores": {
                        "material": 4.6,
                        "fact_accuracy": 4.6,
                        "cross_frame_consistency": 5.0,
                    },
                    "rejected": [],
                    "reason_zh": (
                        "真实玻璃底图完全冻结；程序 pH 场只改变液体"
                        "内部颜色，机制帧额外显示滴液。"
                    ),
                },
                {
                    "case_id": "BIO-01",
                    "selected": "stable_material_plus_depth",
                    "scores": {
                        "material": 4.1,
                        "fact_accuracy": 4.7,
                        "cross_frame_consistency": 4.7,
                    },
                    "rejected": ["raw_underlay"],
                    "reason_zh": (
                        "选中版本只抽取供体的纹理统计，染色体和细胞"
                        "轮廓仍由程序绘制；直接贴原图会带入不存在的"
                        "细胞器，因此拒绝。"
                    ),
                },
            ],
        },
        "route_c": {
            "status": "accepted_for_exact_geometry_and_fields",
            "cases": [
                {
                    "case_id": "MATH-02",
                    "selected": "studio_pbr",
                    "scores": {
                        "material": 4.2,
                        "fact_accuracy": 5.0,
                        "cross_frame_consistency": 5.0,
                    },
                    "rejected": [],
                    "reason_zh": (
                        "四块三角形的身份、面积和变换来自程序 JSON；"
                        "木纹在对象局部坐标中采样，并增加倒角和阴影。"
                    ),
                },
                {
                    "case_id": "PHYS-01",
                    "selected": "specular_water",
                    "scores": {
                        "material": 4.3,
                        "fact_accuracy": 4.9,
                        "cross_frame_consistency": 5.0,
                    },
                    "rejected": ["refractive_water"],
                    "reason_zh": (
                        "波峰波谷完全来自程序高度场；适度镜面光最"
                        "清楚。强折射版本在干涉区出现棋盘伪影，拒绝。"
                    ),
                },
            ],
        },
        "generic_route_rule": [
            {
                "when_zh": "必须精确保留几何、对象身份或连续物理场",
                "route": "C",
            },
            {
                "when_zh": "场景可冻结，变化是局部状态或语义对象",
                "route": "B",
            },
            {
                "when_zh": "还没有可信的真实场景或材质底图",
                "route": "A once, then B",
            },
        ],
    }


def _render_prompt_experiments(
    records: list[dict[str, Any]],
) -> str:
    sections = []
    for record in records:
        title = (
            f"{record['case_id']} · "
            f"{KEYFRAME_LABELS[record['keyframe_id']]}"
        )
        if record["experiment_id"].endswith("terrain_only"):
            title += " · 第二轮：只控制山体"
        sections.append(
            f"""
            <details class="experiment">
              <summary>{html.escape(title)}：控制图、完整提示词与六张候选</summary>
              <div class="two">
                <figure><img src="{_href(STAGE2_ROOT / record['control']['path'])}">
                  <figcaption>模型实际看到的黑白结构控制图。</figcaption></figure>
                <div>
                  <h4>正向提示词</h4>
                  <pre>{html.escape(record['positive_prompt'])}</pre>
                  <h4>负向提示词</h4>
                  <pre>{html.escape(record['negative_prompt'])}</pre>
                  <p>两档控制强度 × 三个复现编号 = 六张候选。
                  “复现编号”只是随机噪声的固定起点；相同模型、参数和编号可重现同一张图，
                  它不是画面含义或质量分数。</p>
                </div>
              </div>
              <figure><img src="{_href(STAGE2_ROOT / record['candidate_sheet']['path'])}">
                <figcaption>所有候选均保留；标签给出控制强度和复现编号。</figcaption></figure>
            </details>
            """
        )
    return "\n".join(sections)


def _report(
    route_a: list[dict[str, Any]],
    assets: dict[str, Path],
    review: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    route_rows = """
      <tr><td>A 半自由扩散</td><td>单张标准场景 / 材质供体</td>
      <td class="partial">部分接受</td><td>单帧真实，但不能独立生成四帧</td></tr>
      <tr><td>B 冻结真实底图 + 程序状态</td><td>化学、生物等材质型案例</td>
      <td class="pass">接受</td><td>真实感与事实控制分工清楚</td></tr>
      <tr><td>C 对象 / 物理场渲染</td><td>数学几何、波场等精确案例</td>
      <td class="pass">接受</td><td>对象身份和数值场不交给扩散模型猜</td></tr>
    """
    score_rows = []
    for route_key in ("route_a", "route_b", "route_c"):
        section = review[route_key]
        items = section.get("experiments", section.get("cases", []))
        for item in items:
            case = item.get("case", item.get("case_id"))
            scores = item["scores"]
            score_rows.append(
                "<tr>"
                f"<td>{html.escape(route_key[-1].upper())}</td>"
                f"<td>{html.escape(case)}</td>"
                f"<td>{scores.get('single_frame_material', scores.get('material'))}</td>"
                f"<td>{scores.get('state_accuracy', scores.get('fact_accuracy'))}</td>"
                f"<td>{scores['cross_frame_consistency']}</td>"
                f"<td>{html.escape(item.get('verdict', item.get('selected', '')))}</td>"
                "</tr>"
            )
    prompts = _render_prompt_experiments(route_a)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 2 · Phase 7 三路线材质增强实验报告</title>
<style>
:root{{--ink:#183138;--muted:#587078;--paper:#f3f0e7;--card:#fffdf6;
--line:#cad5d2;--green:#19745b;--amber:#9b6618;--red:#a43f3f;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.72 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
main{{max-width:1180px;margin:auto;padding:34px 24px 80px}}
h1{{font-size:clamp(30px,5vw,58px);line-height:1.08;margin:0 0 18px}}
h2{{margin-top:58px;font-size:30px;border-top:1px solid var(--line);padding-top:30px}}
h3{{margin-top:34px}} h4{{margin-bottom:6px}} p{{max-width:900px}}
.lead{{font-size:20px;color:#304e55;max-width:980px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}}
.card,details,.callout{{background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:18px}} .number{{font-size:32px;font-weight:750}}
.small,.caption,figcaption{{color:var(--muted);font-size:14px}}
table{{width:100%;border-collapse:collapse;background:var(--card)}}
th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--line);vertical-align:top}}
.pass{{color:var(--green);font-weight:700}} .partial{{color:var(--amber);font-weight:700}}
.fail{{color:var(--red);font-weight:700}} figure{{margin:22px 0}}
img{{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:9px;
background:#dfe3dd}} figcaption{{margin-top:8px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}}
.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:24px 0}}
.flow div{{background:#dfece7;border-radius:10px;padding:16px;position:relative}}
.flow div:not(:last-child)::after{{content:"→";position:absolute;right:-14px;top:34%;
font-size:24px;z-index:1}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#13292f;
color:#edf8f4;padding:14px;border-radius:8px;font:13px/1.55 ui-monospace,monospace}}
summary{{font-weight:700;cursor:pointer}} details{{margin:14px 0}}
code{{background:#e2e8e5;padding:.15em .35em;border-radius:4px}}
a{{color:#126750}} ol li,ul li{{margin:.45em 0}}
@media(max-width:800px){{.cards,.flow,.two{{grid-template-columns:1fr}}.flow div::after{{display:none}}}}
</style>
</head>
<body><main>
<p class="small">Live-Document / video_model / Stage 2 / Phase 7</p>
<h1>让模型负责“像真的”，<br>让程序负责“讲得对”</h1>
<p class="lead">本阶段不是继续把微弱噪声叫作材质增强，而是把三条路线各自跑完、
保留失败，并决定它们在通用系统中分别负责什么。结果：A 适合生成一次真实底图；
B 适合在底图上精确改变教学状态；C 适合精确几何和物理场。</p>

<div class="cards">
 <div class="card"><div class="number">{manifest['counts']['route_a_model_images']}</div>
 <div>张新 SDXL 候选</div><div class="small">6 组实验，每组 6 张</div></div>
 <div class="card"><div class="number">{manifest['counts']['route_b_c_renders']}</div>
 <div>张确定性渲染</div><div class="small">4 案例 × 3 变体 × 4 帧</div></div>
 <div class="card"><div class="number">4</div><div>条选中关键帧序列</div>
 <div class="small">化学、生物、数学、物理</div></div>
 <div class="card"><div class="number">4</div><div>类明确失败</div>
 <div class="small">未删除、未伪装成成功</div></div>
</div>

<h2>先看结论</h2>
<table><thead><tr><th>路线</th><th>现在负责什么</th><th>结论</th><th>原因</th></tr></thead>
<tbody>{route_rows}</tbody></table>
<figure><img src="{_href(assets['chem_before_after'])}">
<figcaption>上阶段化学图只改变了约 1.88/255 的局部细节；本阶段是真实玻璃底图，
再由程序控制杯内状态。两张不是同尺寸，因此这里只做视觉比较，不伪造像素指标。</figcaption></figure>
<figure><img src="{_href(assets['bio_before_after'])}">
<figcaption>生物案例同理：左图接近原程序图，右图有明显的显微材质和深度，
但染色体仍由程序决定。</figcaption></figure>

<h2>为什么 Phase 6 看不出“材质增强”</h2>
<p>旧方法从一张模型图中只取高频残差：先做模糊，用“原图减模糊图”得到细小纹理，
把残差截断在 ±10，再乘 0.30–0.35，只贴进允许修改的区域。可以写成：</p>
<pre>纹理残差 = clip(模型图 − 模糊(模型图), -10, +10)
旧输出 = 程序图 + 0.30~0.35 × median4(纹理残差)</pre>
<p>它的优点是几乎不可能破坏事实；缺点也正是几乎什么都没改变。化学、生物和地理
的平均变化分别只有 1.90、1.30、0.87（0–255 标度）。Phase 6 的“通过”只表示
安全合成器没有越界，不表示真实感达标。本报告把这两个判断拆开。</p>

<h2>统一流程：四类信息各司其职</h2>
<div class="flow">
 <div><strong>程序帧</strong><br>定义对象、位置和状态</div>
 <div><strong>语义层</strong><br>区域遮罩、对象身份或数值场</div>
 <div><strong>材质来源</strong><br>SDXL 标准场景或材质样本</div>
 <div><strong>路由 B / C</strong><br>把材质放回事实坐标</div>
</div>
<p>这里的 <strong>mask（遮罩）</strong>不是多余中间产物。它是一张只有区域含义的图：
白色表示允许改变，黑色表示不得改变。<strong>conditioning（控制条件）</strong>则是模型
实际看到的黑白结构图；它告诉 ControlNet 哪里应该有器材或山脊。两者都会在下面显示，
并说明怎样得到、用在哪里。</p>

<h2>路线 A：SDXL + Canny ControlNet 半自由生成</h2>
<p>输入不是像素程序图本身，而是程序语义层绘出的简洁结构控制图。SDXL 从噪声开始生成
光照、玻璃、岩石和空气感；Canny ControlNet 用黑白线约束大结构。模型为
<code>stabilityai/stable-diffusion-xl-base-1.0</code> 和
<code>diffusers/controlnet-canny-sdxl-1.0</code>，本地 FP16 权重哈希记录在每组
<code>model_fingerprints.json</code> 中。统一参数为 1024×576、30 步、
guidance 6.0；控制强度每组试两档。</p>
<figure><img src="{_href(assets['route_a_chem_direct'])}">
<figcaption>同一复现编号、四段状态提示词直接生成的化学序列。单张很好，
但背景和光线仍会漂移；局部粉色也不够可靠。因此这组不作为最终四帧。</figcaption></figure>
<div class="callout"><strong>第一次自迭代：</strong>地理控制图同时画山、云和雨，模型把雨线
解释成山脊或幕布，拒绝。<strong>第二次自迭代：</strong>只保留山体轮廓后自然度提高，
但“只在迎风坡下雨”仍不能仅靠文字稳定实现。因此 A 的通用职责被收缩为
“生成一次标准场景”，不让它逐帧猜教学状态。</div>
<details><summary>控制图到底怎样从程序数据画出来</summary>
<p>化学控制图使用确定性的器材模板：黑底白线，只画一个烧杯、一个滴定管和液面，
不含粉色、文字或箭头。地理第一轮读取 <code>geo02_terrain_region</code>：
逐列取地形区域最上方像素并连成山脊；再从
<code>geo02_humidity_cloud_rain</code> 取阈值 0.22 的外包络，每隔 3 像素画云边，
并在 x=210–340 范围每隔 15 像素画雨线。失败后第二轮删除云雨，只保留地形最上沿。
最后用最近邻放大到 1024×576，避免灰边。精确实现见
<a href="{_href(STAGE2_ROOT / 'phase7_semifree.py')}">phase7_semifree.py</a>，
而不是靠报告外的手工修图。</p></details>
{prompts}

<h2>路线 B：冻结真实底图，再放回程序状态</h2>
<p>路线 B 解决 A 的跨帧漂移：先选一张真实底图并永久冻结，再把程序的语义状态映射到
底图中的目标表面。通用抽象是 <code>surface calibration → state field projection →
material-aware compositing</code>：先标定“程序区域对应真实图哪里”，再投影状态场，
最后保留原图亮度、反射和纹理地合成。新帧不会重新抽噪声。</p>

<h3>化学：pH 场 → 指示剂光学颜色</h3>
<figure><img src="{_href(assets['chem_process'])}">
<figcaption>从左到右：程序终态、模型实际看到的器材控制图、一次生成后冻结的真实底图、
把程序 pH 场映射进真实液体后的结果。杯外不允许被状态色修改。</figcaption></figure>
<figure><img src="{_href(assets['chem_selected'])}">
<figcaption>选中四帧：起始无色；机制帧有滴液和局部粉色羽流；中间结果褪色；
结束为整杯稳定浅粉。相机、玻璃和背景完全来自同一张冻结底图。</figcaption></figure>
<details><summary>化学表面映射与光学合成的精确参数</summary>
<p>程序液体区域先裁成自己的最小矩形，再双线性缩放到真实底图的 x=350–675。
底边固定 y=493；顶边由程序的 <code>liquid_level_y</code> 线性换算。真实可修改区域
是四点梯形 (350,y0)、(675,y0)、(652,493)、(370,493)，边缘做 5 px 羽化。
pH 经逻辑函数 <code>1/(1+exp(-(pH-8.15)*2.5))</code> 变成指示剂强度。
三轮着色权重为 0.34、0.48、0.52；选中版本再用 0.38 的状态梯度边缘高光，
机制帧在滴定管尖端增加一枚羽化液滴。合成时按原底图亮度缩放粉色目标，因此玻璃反射
没有被一块纯色盖掉。器材外背景最大像素差实测为 0。</p></details>
<details><summary>三轮变体全部展开：为什么选最后一轮</summary>
<figure><img src="{_href(OUTPUT_ROOT / 'route-b/CHEM-01/all-variants-sequence.jpg')}">
<figcaption>第一轮只有平面着色；第二轮增加液体边缘光；第三轮再增加滴液。
第一版曾把程序矩形直接贴到真实图，造成杯外粉色方块；该错误已修正为烧杯梯形表面映射。</figcaption></figure>
</details>

<h3>生物：只借材质统计，不借模型画出的细胞器</h3>
<figure><img src="{_href(assets['bio_process'])}">
<figcaption>绿色遮罩来自 <code>bio01_cell_region</code> 语义层；只允许细胞内部换材质。
紫色染色体和蓝色纺锤极仍从程序图抽取，因此不会被供体图臆造。</figcaption></figure>
<figure><img src="{_href(assets['bio_selected'])}">
<figcaption>选中序列。终态分成两个细胞，染色体位置随程序状态变化；背景和材质基调稳定。</figcaption></figure>
<details><summary>生物材质怎样避免把供体内容照搬进来</summary>
<p>先把供体转为局部纹理统计，不保留它的细胞器位置。选中版本的细胞质基色为
RGB(66,136,111)，标准化纹理分别乘 (9,17,13)，再加入从细胞中心向边缘衰减、
峰值 24 的深度项。细胞膜强度为 0.85。紫色染色体和蓝色纺锤极从每一帧程序图重新
抽取后覆盖，所以供体只贡献“粗糙度”，不贡献教学事实。四帧渲染细胞面积与程序遮罩
逐像素一致。</p></details>
<details><summary>三轮变体与失败样本</summary>
<figure><img src="{_href(OUTPUT_ROOT / 'route-b/BIO-01/all-variants-sequence.jpg')}">
<figcaption>第一排“直接底图”带入了与本课无关的假细胞器，拒绝；第二排归一化细胞质；
第三排再增加稳定的径向深度和膜边缘，选中。</figcaption></figure>
</details>

<h2>路线 C：对象坐标 / 物理场的确定性材质渲染</h2>
<p>当知识点本身就是几何关系或连续场，像素合成仍不够强。路线 C 直接读取程序的结构数据：
几何案例读取对象多边形和身份；物理案例读取浮点高度场。模型只提供木纹或水色参考，
不决定面积、相位、波峰或对象数量。</p>

<h3>数学：木纹固定在每个三角形自己的坐标里</h3>
<figure><img src="{_href(assets['math_process'])}">
<figcaption>木纹通过每个三角形的局部重心坐标采样，所以三角形移动和旋转时，
纹理跟着物体走，不会像贴在屏幕上的噪声。四个对象身份和像素面积逐帧核对。</figcaption></figure>
<figure><img src="{_href(assets['math_selected'])}">
<figcaption>选中 studio PBR 变体：木纹、倒角、接触阴影和清楚的浅色嵌板；
最后四块准确重排到目标正方形。</figcaption></figure>
<details><summary>数学对象局部纹理怎样计算</summary>
<p>每个三角形从 <code>math02_piece_identity</code> 读取三个顶点和稳定对象 ID。
以三个顶点建立 2×2 仿射逆矩阵，把每个屏幕像素变成该三角形的局部 u/v 坐标，
再用 u/v 去木材供体采样；因此移动、旋转时纹理跟随对象。9 px 内缩遮罩产生倒角，
阴影偏移 (6,8) px、模糊 5 px、强度 0.24。硬检查确认每帧始终四个对象，
材质覆盖面积始终等于程序多边形面积 22,204 px。</p></details>
<details><summary>三轮数学材质</summary>
<figure><img src="{_href(OUTPUT_ROOT / 'route-c/MATH-02/all-variants-sequence.jpg')}">
<figcaption>从单纯木纹映射，到倒角，再到工作室阴影。早期浅色证明区域曾融入桌面，
本轮改为有边界的凹入嵌板。</figcaption></figure></details>

<h3>物理：波高求梯度，再计算水面光照</h3>
<figure><img src="{_href(assets['phys_process'])}">
<figcaption>彩色图是程序输出的浮点波高，不是模型图片。对高度场求 x/y 梯度得到法线，
再计算漫反射或镜面反射；所以干涉条纹的位置仍是物理计算结果。</figcaption></figure>
<figure><img src="{_href(assets['phys_selected'])}">
<figcaption>选中的适度镜面水面。亮暗变化增强水的质感，但不移动程序定义的波峰波谷。</figcaption></figure>
<details><summary>水面法线和镜面光怎样计算</summary>
<p>从 <code>phys01_surface_height</code> 读取浮点高度 h，对 h 求 x/y 梯度，
组成法线 <code>normalize(-16·dh/dx, -16·dh/dy, 1)</code>。
光向量固定为 normalize(-0.38,-0.48,0.79)，蓝色基底是 RGB(38,139,177)，
漫反射系数为 <code>0.62 + 0.52·max(N·L,0)</code>。选中版本再用 Blinn 半程向量、
42 次幂和峰值 150 加镜面光。四帧中高度梯度与成图亮度的相关系数绝对值均大于 0.63，
说明看到的波纹确实受程序高度场驱动。</p></details>
<details><summary>三轮水面与失败样本</summary>
<figure><img src="{_href(OUTPUT_ROOT / 'route-c/PHYS-01/all-variants-sequence.jpg')}">
<figcaption>第一排安全但平；第二排适度镜面，选中；第三排强折射在干涉中心产生棋盘伪影，
明确拒绝，不删除。</figcaption></figure></details>

<h2>统一评分与自动路由规则</h2>
<p>评分顺序是先硬事实、再真实感：如果对象数量、状态区域或数值场不对，即使好看也不能通过。
5 分表示可进入后续视频，3 分表示只能做供体，1 分表示不可用。</p>
<table><thead><tr><th>路线</th><th>实验</th><th>材质</th><th>事实/状态</th>
<th>跨帧</th><th>选择</th></tr></thead><tbody>{''.join(score_rows)}</tbody></table>
<ol>
 <li><strong>有精确对象或物理场：</strong>走 C。纹理在对象坐标中，光照从数值场计算。</li>
 <li><strong>场景可冻结、变化是局部状态：</strong>走 B。真实底图不变，程序只改允许区域。</li>
 <li><strong>没有可信底图：</strong>先用 A 生成一次标准场景，人工或自动评分后冻结，再转 B。</li>
 <li><strong>禁止的默认路径：</strong>不再让 A 独立生成每个关键帧，也不再把 ±10 的微弱残差
当作“真实感已增强”。</li>
</ol>

<h2>失败记录：我们具体学到了什么</h2>
<ul>
 <li><strong>提示词超长：</strong>首版 83 token，在生成前拒绝；缩短至 65–75 token。
没有静默截断。由于失败发生在候选生成前，没有失败图片；当时的草稿文本也没有单独
落盘，这是本轮仍存在的日志缺口，报告不假装能够复原那段草稿。</li>
 <li><strong>控制图语义过载：</strong>山、云、雨同时画给 Canny 会互相误读；结构控制只保留
高置信硬边界，天气改由后续语义层处理。</li>
 <li><strong>直接贴真实图：</strong>生物供体会带入不存在的细胞器；只能提取材质统计。</li>
 <li><strong>过强物理特效：</strong>折射能增加“水感”，也会在高频干涉区形成棋盘伪影；
选用适度镜面，并保留被拒版本作为回归样本。</li>
</ul>

<h2>怎样完整复现</h2>
<p>从仓库根目录 <code>Live-Document</code> 执行。第一条真正调用 SDXL，若文件和参数未变会
命中缓存；第二条重建 48 张确定性变体；第三条重新评分、生成本报告并检查链接。</p>
<pre>/opt/venv/bin/python -m modules.video_model.stage2.phase7_semifree
.venv/bin/python -m modules.video_model.stage2.phase7_hybrid_pbr
.venv/bin/python -m modules.video_model.stage2.phase7
.venv/bin/python -m pytest modules/video_model/stage2/tests/test_phase7.py -q</pre>
<p>本次已经把路线 B/C 的 48 张图完整重建一次，并逐文件比较重跑前后的 SHA-256：
48/48 未变化。该结果记录在 <code>hybrid-pbr-manifest.json</code> 的
<code>determinism_replay</code> 字段，并由总清单再次检查。</p>
<p>模型、提示词、控制强度、尺寸、步数和每个复现编号均在
<a href="{_href(STAGE2_ROOT / 'phase7_semifree_matrix.json')}">phase7_semifree_matrix.json</a>
与每个实验的 <code>spec.json</code> 中；选择和拒绝理由在
<a href="{_href(REVIEW_PATH)}">phase7-review.json</a>；所有文件哈希与计数在
<a href="{_href(MANIFEST_PATH)}">phase7-manifest.json</a>。路线 B/C 的完整公式和参数见
<a href="{_href(STAGE2_ROOT / 'phase7_hybrid_pbr.py')}">phase7_hybrid_pbr.py</a>。</p>

<h2>本阶段没有宣称完成的部分</h2>
<p>路线 B 当前的“表面标定”仍由案例适配器提供：化学要知道真实烧杯的液体表面，
生物要知道细胞区域。通用框架已经明确输入合同，但自动从任意生成图估计目标表面、
自动检查视频模型是否保持遮罩边界，仍是下一阶段工作。因此本阶段结论是
<strong>路线分工与可复现实现成立</strong>，不是“所有新概念已经零适配”。</p>

<p class="small">报告状态：{manifest['status']} ·
输出清单：<a href="{_href(MANIFEST_PATH)}">phase7-manifest.json</a></p>
</main></body></html>"""


def _links_resolve(report_text: str) -> bool:
    import re

    links = re.findall(r'(?:href|src)="([^"]+)"', report_text)
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


def build_phase7(*, check_only: bool = False) -> dict[str, Any]:
    route_a = _route_a_records()
    route_a_count = sum(
        item["candidate_count"] for item in route_a
    )
    hybrid = load_json(
        OUTPUT_ROOT / "hybrid-pbr-manifest.json"
    )
    review = _review()
    if not check_only:
        write_json(REVIEW_PATH, review)
        assets = _build_assets()
    else:
        assets = {
            "route_a_chem_direct": (
                ASSET_ROOT / "route-a-chem-direct-sequence.jpg"
            ),
            "chem_selected": (
                ASSET_ROOT / "chem-selected-sequence.jpg"
            ),
            "bio_selected": (
                ASSET_ROOT / "bio-selected-sequence.jpg"
            ),
            "math_selected": (
                ASSET_ROOT / "math-selected-sequence.jpg"
            ),
            "phys_selected": (
                ASSET_ROOT / "phys-selected-sequence.jpg"
            ),
            "chem_process": ASSET_ROOT / "chem-process.jpg",
            "bio_process": ASSET_ROOT / "bio-process.jpg",
            "math_process": ASSET_ROOT / "math-process.jpg",
            "phys_process": ASSET_ROOT / "phys-process.jpg",
            "chem_before_after": (
                ASSET_ROOT / "chem-phase6-vs-phase7.jpg"
            ),
            "bio_before_after": (
                ASSET_ROOT / "bio-phase6-vs-phase7.jpg"
            ),
        }
    selected = {
        "CHEM-01": {
            "route": "B",
            "variant": "optical_tint_plus_drop",
            "sequence": artifact_record(
                assets["chem_selected"], STAGE2_ROOT
            ),
        },
        "BIO-01": {
            "route": "B",
            "variant": "stable_material_plus_depth",
            "sequence": artifact_record(
                assets["bio_selected"], STAGE2_ROOT
            ),
        },
        "MATH-02": {
            "route": "C",
            "variant": "studio_pbr",
            "sequence": artifact_record(
                assets["math_selected"], STAGE2_ROOT
            ),
        },
        "PHYS-01": {
            "route": "C",
            "variant": "specular_water",
            "sequence": artifact_record(
                assets["phys_selected"], STAGE2_ROOT
            ),
        },
    }
    route_a_token_checks = []
    for item in route_a:
        candidate_sheet = (
            STAGE2_ROOT / item["candidate_sheet"]["path"]
        )
        generated = load_json(
            candidate_sheet.parent / "_work/generate.json"
        )
        preflight = generated["prompt_token_preflight"]
        route_a_token_checks.append(
            not preflight["positive"]["would_truncate"]
            and not preflight["negative"]["would_truncate"]
        )
    chemistry_manifest = load_json(
        OUTPUT_ROOT / "route-b/CHEM-01/manifest.json"
    )
    chemistry_records = [
        item
        for item in chemistry_manifest["records"]
        if item["variant"] == "optical_tint_plus_drop"
    ]
    chemistry_state = [
        item["metrics"]["indicator_mean_in_liquid"]
        for item in chemistry_records
    ]
    chemistry_base = np.asarray(
        Image.open(
            OUTPUT_ROOT
            / "route-a/experiments/"
            "EXP-P7-A-chem-01-00_start/raw/"
            "semantic_control_065/seed_7101.png"
        ).convert("RGB"),
        dtype=np.int16,
    )
    yy, xx = np.mgrid[
        0 : chemistry_base.shape[0],
        0 : chemistry_base.shape[1],
    ]
    outside_apparatus = (
        (xx < 320)
        | (xx > 720)
        | (yy < 200)
        | (yy > 520)
    )
    chemistry_background_max = max(
        int(
            np.abs(
                np.asarray(
                    Image.open(path).convert("RGB"),
                    dtype=np.int16,
                )
                - chemistry_base
            )[outside_apparatus].max(initial=0)
        )
        for path in _selected_paths(
            "route-b", "CHEM-01", "optical_tint_plus_drop"
        )
    )
    biology_manifest = load_json(
        OUTPUT_ROOT / "route-b/BIO-01/manifest.json"
    )
    biology_records = [
        item
        for item in biology_manifest["records"]
        if item["variant"] == "stable_material_plus_depth"
    ]
    math_manifest = load_json(
        OUTPUT_ROOT / "route-c/MATH-02/manifest.json"
    )
    math_records = [
        item
        for item in math_manifest["records"]
        if item["variant"] == "studio_pbr"
    ]
    physics_manifest = load_json(
        OUTPUT_ROOT / "route-c/PHYS-01/manifest.json"
    )
    physics_records = [
        item
        for item in physics_manifest["records"]
        if item["variant"] == "specular_water"
    ]
    manifest = {
        "schema_version": "1.0",
        "phase": 7,
        "status": "passed_with_documented_boundaries",
        "classification": "self_iterated_three_route_prototype",
        "counts": {
            "route_a_experiments": len(route_a),
            "route_a_model_images": route_a_count,
            "route_b_c_renders": hybrid["output_count"],
            "total_visual_outputs": (
                route_a_count + hybrid["output_count"]
            ),
        },
        "model_runs_during_phase7": {
            "image": route_a_count,
            "video": 0,
        },
        "route_status": {
            "A": "partial_donor_only",
            "B": "accepted_for_stable_scene_material_cases",
            "C": "accepted_for_exact_geometry_and_fields",
        },
        "executable_route_examples": {
            "exact_field": select_appearance_route(
                has_exact_geometry_or_field=True,
                has_frozen_real_base=False,
                state_is_local_or_semantic=False,
            ),
            "frozen_scene_local_state": select_appearance_route(
                has_exact_geometry_or_field=False,
                has_frozen_real_base=True,
                state_is_local_or_semantic=True,
            ),
            "no_real_base_yet": select_appearance_route(
                has_exact_geometry_or_field=False,
                has_frozen_real_base=False,
                state_is_local_or_semantic=True,
            ),
        },
        "selected": selected,
        "route_a_experiments": route_a,
        "review": artifact_record(REVIEW_PATH, STAGE2_ROOT),
        "report": {
            "path": str(REPORT_PATH.relative_to(STAGE2_ROOT)),
        },
        "checks": [
            {
                "name": "route_a_has_36_model_candidates",
                "passed": route_a_count == 36,
            },
            {
                "name": "routes_b_c_have_48_renders",
                "passed": hybrid["output_count"] == 48,
            },
            {
                "name": "routes_b_c_repeat_byte_for_byte",
                "passed": hybrid["determinism_replay"][
                    "passed"
                ],
                "evidence": hybrid["determinism_replay"],
            },
            {
                "name": "four_selected_sequences_exist",
                "passed": all(
                    all(path.exists() for path in _selected_paths(
                        (
                            "route-b"
                            if item["route"] == "B"
                            else "route-c"
                        ),
                        case_id,
                        item["variant"],
                    ))
                    for case_id, item in selected.items()
                ),
            },
            {
                "name": "route_a_prompts_do_not_truncate",
                "passed": all(route_a_token_checks),
                "evidence": {
                    "experiment_count": len(
                        route_a_token_checks
                    ),
                    "preflight": "actual CLIP tokenizers",
                },
            },
            {
                "name": "chemistry_state_order_and_static_background",
                "passed": (
                    chemistry_state[0] < 0.001
                    and chemistry_state[1] > 0.1
                    and chemistry_state[2] < 0.001
                    and chemistry_state[3] > 0.8
                    and chemistry_background_max == 0
                ),
                "evidence": {
                    "indicator_mean_in_liquid": chemistry_state,
                    "outside_apparatus_max_difference_0_255": (
                        chemistry_background_max
                    ),
                },
            },
            {
                "name": "biology_program_cell_area_is_exact",
                "passed": all(
                    item["metrics"]["program_cell_area_px"]
                    == item["metrics"][
                        "rendered_cell_area_px"
                    ]
                    for item in biology_records
                ),
                "evidence": {
                    "keyframe_count": len(biology_records)
                },
            },
            {
                "name": "math_object_identity_and_area_are_exact",
                "passed": all(
                    item["metrics"]["object_count"] == 4
                    and item["metrics"]["program_piece_area_px"]
                    == item["metrics"][
                        "rendered_material_mask_area_px"
                    ]
                    for item in math_records
                ),
                "evidence": {
                    "keyframe_count": len(math_records),
                    "object_count": 4,
                },
            },
            {
                "name": "physics_uses_visible_program_height_field",
                "passed": (
                    len(physics_records) == 4
                    and all(
                        abs(
                            item["metrics"][
                                "height_gradient_to_luminance_correlation"
                            ]
                        )
                        > 0.6
                        for item in physics_records
                    )
                ),
                "evidence": {
                    "gradient_luminance_correlations": [
                        item["metrics"][
                            "height_gradient_to_luminance_correlation"
                        ]
                        for item in physics_records
                    ]
                },
            },
        ],
    }
    report = _report(route_a, assets, review, manifest)
    links_pass = _links_resolve(report)
    manifest["checks"].append(
        {
            "name": "report_links_resolve",
            "passed": links_pass,
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
    result = build_phase7(check_only=args.check_only)
    print(
        "Phase 7:",
        result["status"],
        "·",
        result["counts"]["total_visual_outputs"],
        "route outputs",
    )


if __name__ == "__main__":
    main()
