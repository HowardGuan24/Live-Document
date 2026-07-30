"""Build Phase 1 generic contracts and ten model-free case fixtures."""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .framework.contracts import (
    artifact_record,
    load_json,
    sha256_path,
    validate_concept_spec,
    validate_fixture_manifest,
    validate_layer_manifest,
    validate_schema_documents,
    validate_sequence_spec,
    validate_states,
    write_json,
)
from .framework.fixture_builder import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    build_fixture,
)


STAGE2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STAGE2_ROOT.parents[2]
OUTPUT_ROOT = STAGE2_ROOT / "output" / "phase-1"
FIXTURES_ROOT = OUTPUT_ROOT / "fixtures"

REGISTRY_PATH = STAGE2_ROOT / "case_registry.json"
TEMPLATE_PATH = STAGE2_ROOT / "framework" / "fixture_templates.json"
PHASE0_MANIFEST_PATH = (
    STAGE2_ROOT / "output" / "phase-0" / "phase0_manifest.json"
)

LAYER_LABELS = {
    "hard_boundary": "硬边界",
    "region": "对象或允许修改区域",
    "scalar_field": "连续标量场",
    "vector_field": "方向与速度场",
    "height_or_normal": "高度或表面法线",
    "object_identity": "带稳定身份的对象",
    "annotation": "教学叠加层",
}

LAYER_EXPLANATIONS = {
    "hard_boundary": "适合表示岸线、容器或拼图轮廓；可以成为稀疏控制候选。",
    "region": "表示对象范围或允许改变区域，用来保护区域外像素。",
    "scalar_field": "每个像素是连续数值，例如浓度、温度或振幅；不应强转密集线稿。",
    "vector_field": "保存方向和大小，预览箭头只用于审计，最终图不默认显示箭头。",
    "height_or_normal": "保存表面起伏与受光方向，让程序守住大形状。",
    "object_identity": "保存跨帧稳定 ID，避免模型复制、吞掉或交换对象。",
    "annotation": "箭头、文字和重点提示在写实底图之后由程序叠加。",
}


def _fixture_root(case_id: str) -> Path:
    return FIXTURES_ROOT / case_id


def _validate_fixture_tree(
    case: dict[str, Any], fixture_root: Path
) -> dict[str, Any]:
    manifest_path = fixture_root / "fixture_manifest.json"
    manifest = load_json(manifest_path)
    layer_path = fixture_root / manifest["semantic_layers"]["path"]
    layers = load_json(layer_path)
    concept_path = fixture_root / manifest["concept_spec"]["path"]
    sequence_path = fixture_root / manifest["sequence_spec"]["path"]
    states_path = fixture_root / manifest["states"]["path"]
    concept = load_json(concept_path)
    sequence = load_json(sequence_path)

    validate_concept_spec(concept, expected_case_id=case["case_id"])
    validate_sequence_spec(
        sequence,
        expected_case_id=case["case_id"],
        expected_layer_ids={
            layer["layer_id"] for layer in layers["layers"]
        },
    )
    states = validate_states(states_path)
    validate_layer_manifest(layers, fixture_root)
    validate_fixture_manifest(manifest, fixture_root)

    expected_paths = {
        Path("fixture_manifest.json"),
        *(
            Path(manifest[field]["path"])
            for field in (
                "concept_spec",
                "sequence_spec",
                "states",
                "clean_frame",
                "program_frame",
                "semantic_layers",
            )
        ),
    }
    for layer in layers["layers"]:
        expected_paths.add(Path(layer["data"]["path"]))
        expected_paths.add(Path(layer["preview"]["path"]))
    preview = manifest["control"].get("control_preview")
    if preview:
        expected_paths.add(Path(preview["path"]))
    actual_paths = {
        path.relative_to(fixture_root)
        for path in fixture_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError(
            f"{case['case_id']} fixture file set mismatch: "
            f"extra={sorted(str(p) for p in actual_paths - expected_paths)}, "
            f"missing={sorted(str(p) for p in expected_paths - actual_paths)}"
        )

    return {
        "case_id": case["case_id"],
        "title_zh": case["title_zh"],
        "discipline": case["discipline"],
        "discipline_zh": case["discipline_zh"],
        "sentinel": case["sentinel"],
        "fixture_root": str(fixture_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_path(manifest_path),
        "layer_types": [layer["layer_type"] for layer in layers["layers"]],
        "layer_count": len(layers["layers"]),
        "control": manifest["control"],
        "program_frame": manifest["program_frame"],
        "clean_frame": manifest["clean_frame"],
        "layers": layers["layers"],
        "states": states,
        "model_runs": manifest["model_runs"],
    }


def _build_contact_sheet(fixtures: list[dict[str, Any]]) -> Path:
    columns = 2
    rows = (len(fixtures) + columns - 1) // columns
    label_height = 28
    gutter = 12
    tile_width = CANVAS_WIDTH
    tile_height = CANVAS_HEIGHT + label_height
    width = columns * tile_width + (columns + 1) * gutter
    height = rows * tile_height + (rows + 1) * gutter
    sheet = Image.new("RGB", (width, height), (18, 35, 40))
    draw = ImageDraw.Draw(sheet)
    for index, fixture in enumerate(fixtures):
        row, column = divmod(index, columns)
        x = gutter + column * (tile_width + gutter)
        y = gutter + row * (tile_height + gutter)
        root = Path(fixture["fixture_root"])
        frame = Image.open(
            root / fixture["program_frame"]["path"]
        ).convert("RGB")
        sheet.paste(frame, (x, y))
        draw.rectangle(
            (x, y + CANVAS_HEIGHT, x + tile_width, y + tile_height),
            fill=(8, 24, 30),
        )
        label = (
            f"{fixture['case_id']}  |  {fixture['discipline'].upper()}  |  "
            f"{fixture['layer_count']} LAYERS"
        )
        draw.text(
            (x + 9, y + CANVAS_HEIGHT + 7),
            label,
            fill=(228, 244, 239),
        )
    path = OUTPUT_ROOT / "fixture-contact-sheet.jpg"
    sheet.save(path, quality=92, subsampling=0)
    return path


def _fixture_checks(
    fixtures: list[dict[str, Any]],
    schema_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_ids = [fixture["case_id"] for fixture in fixtures]
    if len(case_ids) != 10 or len(case_ids) != len(set(case_ids)):
        raise ValueError("Phase 1 requires ten unique fixtures")
    if any(
        fixture["model_runs"] != {"image": 0, "video": 0}
        for fixture in fixtures
    ):
        raise ValueError("a Phase 1 fixture claims a model run")
    if any(len(fixture["states"]) != 4 for fixture in fixtures):
        raise ValueError("every fixture must contain four ordered states")
    layer_counts = Counter(
        layer_type
        for fixture in fixtures
        for layer_type in fixture["layer_types"]
    )
    if set(layer_counts) != set(LAYER_LABELS):
        raise ValueError("the fixtures do not cover all seven layer types")
    control_counts = Counter(
        fixture["control"]["route"] for fixture in fixtures
    )
    if control_counts != {
        "sparse_hard_boundary_candidate": 5,
        "off": 5,
    }:
        raise ValueError(f"unexpected control-route coverage: {control_counts}")
    annotations = [
        layer
        for fixture in fixtures
        for layer in fixture["layers"]
        if layer["layer_type"] == "annotation"
    ]
    if len(annotations) != 10 or any(
        layer["model_input_policy"] != "never"
        or layer["used_as_model_input"]
        for layer in annotations
    ):
        raise ValueError("annotation layers are not safely separated")
    if any(
        layer["used_as_model_input"]
        for fixture in fixtures
        for layer in fixture["layers"]
    ):
        raise ValueError("Phase 1 cannot use a semantic layer as model input")
    if len(schema_records) != 5:
        raise ValueError("five JSON Schema documents are required")
    return [
        {
            "name": "five_contract_schemas",
            "passed": True,
            "evidence": [item["name"] for item in schema_records],
        },
        {
            "name": "ten_model_free_fixtures",
            "passed": True,
            "evidence": case_ids,
        },
        {
            "name": "four_ordered_states_per_fixture",
            "passed": True,
            "evidence": {case_id: 4 for case_id in case_ids},
        },
        {
            "name": "seven_semantic_layer_types_covered",
            "passed": True,
            "evidence": dict(sorted(layer_counts.items())),
        },
        {
            "name": "control_route_is_data_driven",
            "passed": True,
            "evidence": dict(sorted(control_counts.items())),
        },
        {
            "name": "annotations_are_post_generation_only",
            "passed": True,
            "evidence": {
                "annotation_layer_count": len(annotations),
                "used_as_model_input": False,
            },
        },
        {
            "name": "zero_model_runs",
            "passed": True,
            "evidence": {"image": 0, "video": 0},
        },
        {
            "name": "fixture_hashes_and_file_sets",
            "passed": True,
            "evidence": {
                fixture["case_id"]: fixture["manifest_sha256"]
                for fixture in fixtures
            },
        },
    ]


def _href(path: Path) -> str:
    return os.path.relpath(path, OUTPUT_ROOT).replace(os.sep, "/")


def _render_report(
    fixtures: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    schema_records: list[dict[str, Any]],
    contact_sheet_path: Path,
) -> str:
    total_layers = sum(item["layer_count"] for item in fixtures)
    control_counts = Counter(
        item["control"]["route"] for item in fixtures
    )
    layer_examples: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for fixture in fixtures:
        for layer in fixture["layers"]:
            layer_examples.setdefault(
                layer["layer_type"], (fixture, layer)
            )

    case_cards = []
    for fixture in fixtures:
        root = Path(fixture["fixture_root"])
        program_href = _href(root / fixture["program_frame"]["path"])
        clean_href = _href(root / fixture["clean_frame"]["path"])
        layer_figures = "".join(
            "<figure>"
            f'<a href="{html.escape(_href(root / layer["preview"]["path"]))}">'
            f'<img src="{html.escape(_href(root / layer["preview"]["path"]))}" '
            f'alt="{html.escape(layer["title_zh"])}"></a>'
            f"<figcaption><strong>{html.escape(layer['title_zh'])}</strong>"
            f"{html.escape(layer['meaning_zh'])}<br>"
            f"<span>本次输入模型：否；策略："
            f"{html.escape(layer['model_input_policy'])}</span></figcaption>"
            "</figure>"
            for layer in fixture["layers"]
        )
        control = fixture["control"]
        control_label = (
            "保存稀疏硬边界候选"
            if control["route"] == "sparse_hard_boundary_candidate"
            else "明确关闭控制"
        )
        case_cards.append(
            f"""<article class="case" id="{fixture['case_id']}">
<div class="case-head"><div><p class="eyebrow">{html.escape(fixture['discipline_zh'])}
· {'哨兵案例' if fixture['sentinel'] else '扩展案例'}</p>
<h3><code>{fixture['case_id']}</code> {html.escape(fixture['title_zh'])}</h3></div>
<span class="route">{control_label}</span></div>
<p class="fixture-warning">这是用于验证文件协议的抽象 fixture，不是完成的科学程序图，
也没有经过图像模型。</p>
<div class="pair">
<figure><a href="{clean_href}"><img src="{clean_href}" alt="干净底图"></a>
<figcaption><strong>干净底图</strong>不含教学箭头，未来可作为模型路线的视觉底图。</figcaption>
</figure>
<figure><a href="{program_href}"><img src="{program_href}" alt="程序图"></a>
<figcaption><strong>程序图</strong>在同一底图上叠加程序生成的教学箭头。</figcaption>
</figure></div>
<details><summary>展开 {fixture['layer_count']} 个语义层和普通话说明</summary>
<div class="layer-grid">{layer_figures}</div></details>
<p><strong>控制选择：</strong>{html.escape(control['reason_zh'])}</p>
<p class="links"><a href="{html.escape(_href(root / 'concept_spec.json'))}">
概念规格</a> · <a href="{html.escape(_href(root / 'sequence_spec.json'))}">
关键帧规格</a> · <a href="{html.escape(_href(root / 'semantic_layers.json'))}">
图层清单</a> · <a href="{html.escape(_href(root / 'fixture_manifest.json'))}">
完整 manifest</a></p>
</article>"""
        )

    layer_cards = "".join(
        f"""<article><h3>{html.escape(LAYER_LABELS[layer_type])}</h3>
<p>{html.escape(LAYER_EXPLANATIONS[layer_type])}</p>
<figure><a href="{html.escape(_href(Path(fixture['fixture_root']) / layer['preview']['path']))}">
<img src="{html.escape(_href(Path(fixture['fixture_root']) / layer['preview']['path']))}"
alt="{html.escape(LAYER_LABELS[layer_type])}"></a>
<figcaption>示例来自 <code>{fixture['case_id']}</code>，数据文件与预览分开保存。</figcaption>
</figure></article>"""
        for layer_type, (fixture, layer) in layer_examples.items()
    )
    check_rows = "".join(
        "<tr>"
        f"<td>{html.escape(check['name'])}</td><td class=\"pass\">通过</td>"
        f"<td><code>{html.escape(json.dumps(check['evidence'], ensure_ascii=False))}</code></td>"
        "</tr>"
        for check in checks
    )
    schema_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td><code>{html.escape(str(item['id']))}</code></td>"
        f"<td><code>{item['sha256'][:18]}…</code></td></tr>"
        for item in schema_records
    )
    contact_href = _href(contact_sheet_path)
    phase0_href = _href(
        STAGE2_ROOT / "output" / "phase-0" / "report.html"
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 2 Phase 1｜通用契约与十案例 Fixtures</title>
<style>
:root{{--ink:#173038;--muted:#63777b;--paper:#f4f2e9;--card:#fff;--line:#d8dfda;
--accent:#126c65;--accent2:#d17a3d;--ok:#18734a}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--paper);
font:16px/1.7 system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{padding:64px 24px 48px;color:#fff;background:linear-gradient(125deg,#102e35,
#176b65 62%,#cd713a)}}header>div,main{{max-width:1200px;margin:auto}}
h1{{font-size:clamp(34px,6vw,64px);line-height:1.08;margin:8px 0 18px}}
h2{{font-size:30px;line-height:1.25;margin:0 0 12px}}h3{{margin:0 0 8px;line-height:1.35}}
p{{margin:8px 0 14px}}.lede{{max-width:900px;font-size:19px;opacity:.95}}
.eyebrow{{font-size:13px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;
color:var(--accent)}}header .eyebrow{{color:#c9fff3}}
nav{{position:sticky;top:0;z-index:5;display:flex;gap:9px;overflow:auto;padding:12px
max(24px,calc((100vw - 1200px)/2));background:rgba(244,242,233,.97);
border-bottom:1px solid var(--line)}}nav a{{white-space:nowrap;text-decoration:none;
color:var(--ink);background:#fff;padding:5px 10px;border-radius:18px}}
main{{padding:28px 24px 80px}}section,.case{{margin:28px 0;padding:28px;border:1px
solid var(--line);border-radius:20px;background:rgba(255,255,255,.78)}}.grid{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:15px}}
.stat,.grid>article{{padding:17px;border:1px solid var(--line);border-radius:14px;
background:#fff}}.stat strong{{display:block;font-size:34px;color:var(--accent)}}
.status{{display:inline-block;padding:7px 12px;border-radius:20px;background:#daf4e7;
color:#0f5b38;font-weight:800}}.warning,.fixture-warning{{border-left:5px solid
var(--accent2);padding:12px 16px;background:#fff0e4;border-radius:8px}}
.plain{{border-left:5px solid var(--accent);padding:12px 16px;background:#e8f5f2;
border-radius:8px}}figure{{margin:0;padding:9px;border:1px solid var(--line);
border-radius:13px;background:#fff}}figure img{{display:block;width:100%;height:auto;
border-radius:8px}}figcaption{{margin-top:8px;color:var(--muted);font-size:14px}}
.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
.layer-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
gap:12px;margin-top:14px}}details{{margin:18px 0}}summary{{cursor:pointer;font-weight:800;
color:var(--accent)}}.case-head{{display:flex;justify-content:space-between;gap:15px;
align-items:flex-start}}.route{{padding:5px 9px;border-radius:13px;background:#e8f3f1;
white-space:nowrap;font-size:13px;font-weight:700}}table{{width:100%;border-collapse:
collapse;background:#fff}}th,td{{padding:10px;border:1px solid var(--line);text-align:left;
vertical-align:top}}th{{background:#e7f0ed}}.table-wrap{{overflow:auto}}code{{padding:
2px 5px;border-radius:4px;background:#edf1ef;font-size:.9em}}pre{{overflow:auto;
padding:15px;border-radius:11px;background:#102d33;color:#e8fff9}}a{{color:#086860}}
.pass{{color:var(--ok);font-weight:800}}.links{{font-size:14px}}.small{{font-size:14px;
color:var(--muted)}}@media(max-width:720px){{section,.case{{padding:18px 14px}}
.pair{{grid-template-columns:1fr}}.case-head{{display:block}}.route{{display:inline-block;
margin-bottom:8px}}}}
</style></head><body>
<header><div><p class="eyebrow">Stage 2 · Phase 1 · Model-free contracts</p>
<h1>十个案例先说同一种“机器语言”</h1>
<p class="lede">这份报告展示的不是模型成品，而是所有后续案例都必须遵守的数据接口：
概念怎样拆段、状态怎样选关键帧、七类程序数据怎样保存、控制路线为什么开启或关闭，
以及每个文件如何被哈希审计。</p><span class="status">Phase 1 契约冒烟全部通过</span>
</div></header>
<nav><a href="#result">结果</a><a href="#contract">一个 fixture 有什么</a>
<a href="#layers">七类图层</a><a href="#cases">十个案例</a>
<a href="#control">控制路线</a><a href="#audit">验证</a><a href="#next">下一步</a></nav>
<main>
<section id="result"><p class="eyebrow">01 · 本阶段结果</p><h2>契约已实现，科学动画尚未实现</h2>
<div class="grid"><div class="stat"><strong>10</strong>个确定性 fixture</div>
<div class="stat"><strong>40</strong>个有序状态</div>
<div class="stat"><strong>7</strong>种统一语义层</div>
<div class="stat"><strong>{total_layers}</strong>个图层实例</div>
<div class="stat"><strong>5 / 5</strong>开启候选 / 明确关闭控制</div>
<div class="stat"><strong>0</strong>次图片或视频模型调用</div></div>
<p class="warning"><strong>不要把这些图当成案例成果。</strong>它们是抽象、低成本的
序列化样本，用于证明十个案例能走同一接口。下一阶段才会为五个哨兵案例实现真实机制
程序。每张程序图都在页面上保留这个分类。</p>
<figure style="margin-top:18px"><a href="{contact_href}"><img src="{contact_href}"
alt="十个 fixture 程序图总览"></a><figcaption>十个 fixture 的程序图总览。相同画布、
文件结构和验证器，不代表它们使用相同的案例机制。</figcaption></figure></section>

<section id="contract"><p class="eyebrow">02 · 文件契约</p><h2>一个 fixture 到底包含什么</h2>
<div class="grid"><article><h3>concept_spec.json</h3><p>用普通话记录教学目标、假设、
四个因果片段以及禁止的偷懒方式。</p></article>
<article><h3>states.jsonl</h3><p>四个有序状态。Phase 1 只有进度值，并明确标为
<code>fixture_only</code>；Phase 2 必须换成真实机制变量。</p></article>
<article><h3>sequence_spec.json</h3><p>把状态映射到关键帧，并记录每段唯一主要变化、
固定内容和图像/视频模型职责。</p></article>
<article><h3>semantic_layers.json</h3><p>每层都有普通话含义、数据文件、预览、来源、
模型输入策略和本次是否真正输入模型。</p></article>
<article><h3>clean_frame.png</h3><p>不含教学箭头的底图。以后模型路线只能从被允许的
底图和程序控制出发。</p></article>
<article><h3>program_frame.png</h3><p>在干净底图上加入程序教学箭头，用于解释状态，
不直接交给生成模型。</p></article></div>
<p class="plain">数据文件与预览图分开：例如 <code>scalar_field.npy</code> 保存真实
浮点数，<code>scalar_field_preview.png</code> 只是把数值染色给人看。后续程序读取
前者，报告显示后者。</p></section>

<section id="layers"><p class="eyebrow">03 · 七类语义层</p><h2>不再把所有中间图都叫 mask</h2>
<div class="grid">{layer_cards}</div></section>

<section id="cases"><p class="eyebrow">04 · 十案例逐项证据</p>
<h2>干净底图、程序图和所有图层都能打开</h2>
{''.join(case_cards)}</section>

<section id="control"><p class="eyebrow">05 · 控制路线</p><h2>有硬边界才保存线稿候选</h2>
<div class="grid"><div class="stat"><strong>{control_counts['sparse_hard_boundary_candidate']}</strong>
个 fixture 声明硬边界</div><div class="stat"><strong>{control_counts['off']}</strong>
个 fixture 明确 <code>control=off</code></div></div>
<p>数学精确几何、器材边界或河岸可以保存稀疏硬边界候选。水波高度、结晶状态、染色体
身份、气孔形变和大气连续场不能为了使用当前 Canny ControlNet 被压成密集白线。
Phase 1 只记录选择，所有 <code>used_as_model_input</code> 都是 false。</p></section>

<section id="audit"><p class="eyebrow">06 · 自动验证</p><h2>Schema、文件内容和哈希一起检查</h2>
<div class="table-wrap"><table><thead><tr><th>检查</th><th>结果</th><th>证据</th></tr>
</thead><tbody>{check_rows}</tbody></table></div>
<h3 style="margin-top:24px">五份 JSON Schema</h3>
<div class="table-wrap"><table><thead><tr><th>文件</th><th>Schema ID</th><th>SHA-256</th>
</tr></thead><tbody>{schema_rows}</tbody></table></div>
<h3 style="margin-top:24px">复现命令</h3>
<pre>.venv/bin/python -m modules.video_model.stage2.phase1
.venv/bin/python -m modules.video_model.stage2.phase1 --check
.venv/bin/python -m pytest -q modules/video_model/stage2/tests</pre>
<p><a href="phase1_manifest.json">打开 Phase 1 构建 manifest</a> ·
<a href="{phase0_href}">回看 Phase 0 的基线与评分协议</a></p></section>

<section id="next"><p class="eyebrow">07 · 下一检查点</p>
<h2>Phase 2：五个哨兵案例的真实确定性程序</h2>
<p>下一步按学科实现勾股拼图、水波干涉、酸碱滴定、有丝分裂和地形雨。届时状态选择
必须来自面积、相位、pH、对象分配、温湿度等真实机制变量，替换当前
<code>fixture_progress</code>。仍先完成程序动画和报告，再加载图像模型。</p></section>
</main></body></html>"""


def _source_records() -> list[dict[str, Any]]:
    paths = [
        STAGE2_ROOT / "case.txt",
        STAGE2_ROOT / "loop.md",
        REGISTRY_PATH,
        TEMPLATE_PATH,
        PHASE0_MANIFEST_PATH,
        Path(__file__).resolve(),
        STAGE2_ROOT / "framework" / "contracts.py",
        STAGE2_ROOT / "framework" / "fixture_builder.py",
        *sorted(
            (STAGE2_ROOT / "framework" / "schemas").glob("*.json")
        ),
    ]
    return [artifact_record(path, REPO_ROOT) for path in paths]


def _manifest(
    fixtures: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    schema_records: list[dict[str, Any]],
    contact_sheet_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "phase": "phase-1",
        "status": "passed",
        "classification": (
            "generic contracts and model-free fixtures; "
            "not finished scientific animations"
        ),
        "model_runs": {"image": 0, "video": 0},
        "fixture_count": len(fixtures),
        "state_count": sum(len(item["states"]) for item in fixtures),
        "semantic_layer_count": sum(
            item["layer_count"] for item in fixtures
        ),
        "checks": checks,
        "schemas": schema_records,
        "fixtures": [
            {
                "case_id": fixture["case_id"],
                "manifest": artifact_record(
                    Path(fixture["manifest_path"]), OUTPUT_ROOT
                ),
                "layer_types": fixture["layer_types"],
                "control_route": fixture["control"]["route"],
            }
            for fixture in fixtures
        ],
        "source_files": _source_records(),
        "contact_sheet": artifact_record(contact_sheet_path, OUTPUT_ROOT),
        "report": artifact_record(report_path, OUTPUT_ROOT),
        "next_phase": "phase-2 five deterministic sentinel programs",
    }


def _load_existing_fixtures(
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    fixtures = []
    for case in registry["cases"]:
        fixtures.append(_validate_fixture_tree(case, _fixture_root(case["case_id"])))
    return fixtures


def build_phase1(*, check_only: bool = False) -> dict[str, Any]:
    registry = load_json(REGISTRY_PATH)
    template_set = load_json(TEMPLATE_PATH)
    if set(template_set["cases"]) != {
        case["case_id"] for case in registry["cases"]
    }:
        raise ValueError("fixture templates do not match the case registry")
    schema_records = validate_schema_documents()

    if check_only:
        fixtures = _load_existing_fixtures(registry)
    else:
        FIXTURES_ROOT.mkdir(parents=True, exist_ok=True)
        for case in registry["cases"]:
            build_fixture(
                case,
                template_set["cases"][case["case_id"]],
                _fixture_root(case["case_id"]),
            )
        fixtures = _load_existing_fixtures(registry)

    checks = _fixture_checks(fixtures, schema_records)
    contact_sheet_path = OUTPUT_ROOT / "fixture-contact-sheet.jpg"
    if not check_only:
        _build_contact_sheet(fixtures)
    if not contact_sheet_path.is_file():
        raise FileNotFoundError(contact_sheet_path)

    report_text = _render_report(
        fixtures, checks, schema_records, contact_sheet_path
    )
    report_path = OUTPUT_ROOT / "report.html"
    manifest_path = OUTPUT_ROOT / "phase1_manifest.json"
    if check_only:
        if not report_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("Phase 1 report or manifest is missing")
        if report_path.read_text(encoding="utf-8") != report_text:
            raise ValueError("Phase 1 report is stale")
        expected_manifest = _manifest(
            fixtures,
            checks,
            schema_records,
            contact_sheet_path,
            report_path,
        )
        actual_manifest = load_json(manifest_path)
        if actual_manifest != expected_manifest:
            raise ValueError("Phase 1 manifest is stale")
        return actual_manifest

    report_path.write_text(report_text, encoding="utf-8")
    manifest = _manifest(
        fixtures,
        checks,
        schema_records,
        contact_sheet_path,
        report_path,
    )
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or validate Stage 2 Phase 1 model-free fixtures."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate existing fixtures, report, and manifest without writes",
    )
    args = parser.parse_args()
    manifest = build_phase1(check_only=args.check)
    mode = "checked" if args.check else "built"
    print(
        f"Phase 1 {mode}: {manifest['fixture_count']} fixtures, "
        f"{manifest['semantic_layer_count']} semantic layers, "
        f"{len(manifest['checks'])} checks passed; model runs=0"
    )


if __name__ == "__main__":
    main()
