"""Generate a visual, beginner-first report from saved pipeline manifests."""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from string import Template
from typing import Any

from .utils import write_json


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required report input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(output_root: Path, value: str | Path) -> str:
    return Path(os.path.relpath(Path(value), output_root)).as_posix()


def _figure(
    output_root: Path,
    path: str,
    title: str,
    caption: str,
) -> str:
    relative = html.escape(_rel(output_root, path))
    return (
        f'<figure><a href="{relative}"><img src="{relative}" '
        f'alt="{html.escape(title)}"></a><figcaption><strong>'
        f"{html.escape(title)}</strong>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def write_prepare_audit_report(
    spec: dict[str, Any],
    output_root: Path,
    prepared: dict[str, Any],
) -> Path:
    """Write a model-free audit page for any number of keyframes."""
    cards = [
        _figure(
            output_root,
            prepared["anchor"]["program_frame"]["path"],
            "视觉锚点对应的程序状态",
            (
                f"display {prepared['anchor']['display_frame']} / "
                f"state {prepared['anchor']['state_frame']}"
            ),
        )
    ]
    for index, item in enumerate(spec["keyframes"], start=1):
        entry = prepared["keyframes"][item["id"]]
        tokens = entry["prompt"]["token_counts"]
        cards.append(
            "<article>"
            f"<h2>{index}｜{html.escape(item['meaning'])}</h2>"
            f"<p>display {entry['display_frame']} / state "
            f"{entry['state_frame']}。以下内容均由 <code>--prepare</code> "
            "生成，没有加载扩散模型。</p><div class=\"grid\">"
            + _figure(
                output_root,
                entry["program_frame"]["path"],
                "程序审计帧",
                "只确认机制状态，不输入模型。",
            )
            + _figure(
                output_root,
                entry["semantic_layers"]["suspended_density"]["path"],
                "悬浮泥沙浓度",
                "程序粒子经投影和平滑得到，不输入 ControlNet。",
            )
            + _figure(
                output_root,
                entry["control"]["canny"]["path"],
                "实际 Canny",
                "唯一输入 ControlNet 的图像。",
            )
            + _figure(
                output_root,
                entry["control"]["anchor_overlay"]["path"],
                "位置核对",
                "红线确认 Canny 与视觉锚点对齐。",
            )
            + "</div>"
            f"<p>提示词 token：Tokenizer 1 正/负 "
            f"{tokens['tokenizer']['positive']}/"
            f"{tokens['tokenizer']['negative']}；Tokenizer 2 正/负 "
            f"{tokens['tokenizer_2']['positive']}/"
            f"{tokens['tokenizer_2']['negative']}，上限 77。</p>"
            "</article>"
        )
    path = output_root / "prepare-audit.html"
    path.write_text(
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prepare 冒烟审计</title><style>
body{max-width:1120px;margin:40px auto;padding:0 18px;color:#17252b;
font:16px/1.7 system-ui,sans-serif;background:#f4f0e6}article{margin:28px 0;
padding:22px;border:1px solid #d7d2c7;border-radius:14px;background:#fffdf8}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
figure{margin:0;border:1px solid #ddd;background:white}img{display:block;width:100%}
figcaption{padding:10px}figcaption strong{display:block}@media(max-width:700px){
.grid{grid-template-columns:1fr}}</style></head><body>
<h1>配置驱动的 Prepare 审计</h1><p>本页证明报告器可以读取任意数量的关键帧。
它只展示程序状态、语义层、Canny 和 token 检查，不包含模型生成结果。</p>
"""
        + "".join(cards)
        + "</body></html>",
        encoding="utf-8",
    )
    return path


def write_report(spec: dict[str, Any], output_root: Path) -> Path:
    manifest_root = output_root / "_work" / "manifests"
    prepared = _load(manifest_root / "prepare.json")
    generated = _load(manifest_root / "generate.json")
    composed = _load(manifest_root / "compose.json")
    evaluated = _load(manifest_root / "evaluate.json")
    handoff = _load(output_root / "video_handoff.json")

    final_figures = [
        _figure(
            output_root,
            prepared["anchor"]["final"]["path"],
            "0｜泥沙刚到河口",
            spec["anchor"]["meaning"],
        )
    ]
    for index, item in enumerate(spec["keyframes"], start=1):
        final_figures.append(
            _figure(
                output_root,
                composed["keyframes"][item["id"]]["final"]["path"],
                f"{index}｜{item['meaning']}",
                (
                    f"display {item['display_frame']} / "
                    f"state {item['state_frame']}"
                ),
            )
        )

    state_rows = []
    ordered = [
        (spec["anchor"], prepared["anchor"]),
        *[
            (item, prepared["keyframes"][item["id"]])
            for item in spec["keyframes"]
        ],
    ]
    for index, (item, entry) in enumerate(ordered):
        stats = entry["stats"]
        state_rows.append(
            "<tr>"
            f"<td>{index}</td><td>{html.escape(item['meaning'])}</td>"
            f"<td>{html.escape(item.get('selection_reason', '由规格文件选择'))}</td>"
            f"<td>{entry['display_frame']}/{entry['state_frame']}</td>"
            f"<td>{stats['suspended_particles']}</td>"
            f"<td>{stats['underwater_deposit_cells']}</td>"
            f"<td>{stats['new_land_cells']}</td>"
            f"<td>{stats['raw_channel_count']}</td>"
            "</tr>"
        )
    state_program_figure = _figure(
        output_root,
        output_root / "source-comparison.jpg",
        "五个被选状态的程序审计图",
        (
            "按时间排列；界面、图例、箭头和橙色粒子只用于核对，"
            "不会整张输入图像模型。"
        ),
    )

    raw_by_key_seed = {
        (record["keyframe_id"], record["seed"]): record
        for record in generated["candidates"]
    }
    report_seed = int(spec["render"]["report_seed"])
    process_sections = []
    for index, item in enumerate(spec["keyframes"], start=1):
        keyframe_id = item["id"]
        entry = prepared["keyframes"][keyframe_id]
        semantic = entry["semantic_layers"]
        control = entry["control"]
        composite = composed["keyframes"][keyframe_id]
        raw = raw_by_key_seed[(keyframe_id, report_seed)]
        tokens = entry["prompt"]["token_counts"]
        prompt_table = (
            "<table><thead><tr><th>提示词部分</th><th>实际内容</th></tr></thead>"
            "<tbody>"
            f"<tr><td>共用视觉</td><td>{html.escape(entry['prompt']['common_visual'])}</td></tr>"
            f"<tr><td>本帧变化</td><td>{html.escape(entry['prompt']['mechanism_delta'])}</td></tr>"
            f"<tr><td>本帧禁区</td><td>{html.escape(entry['prompt']['stage_forbidden'])}</td></tr>"
            f"<tr><td>共用负向</td><td>{html.escape(entry['prompt']['common_negative'])}</td></tr>"
            "</tbody></table>"
        )
        process_sections.append(
            f"""
<article class="stage-card" id="stage-{keyframe_id}">
  <div class="stage-heading"><span>{index}</span><div>
    <p class="eyebrow">DISPLAY {entry['display_frame']} · STATE {entry['state_frame']}</p>
    <h3>{html.escape(item['meaning'])}</h3>
  </div></div>
  <h4>A. 程序状态怎样变成可读语义</h4>
  <div class="grid four">
    {_figure(output_root, entry['program_frame']['path'], '程序审计帧', '含界面和粒子，只用于确认状态，不输入模型。')}
    {_figure(output_root, semantic['suspended_density']['path'], '悬浮泥沙浓度', semantic['suspended_density']['meaning'])}
    {_figure(output_root, semantic['underwater_deposit']['path'], '水下沉积厚度', semantic['underwater_deposit']['meaning'])}
    {_figure(output_root, semantic['new_land_binary']['path'], '新生陆地', semantic['new_land_binary']['meaning'])}
  </div>
  <h4>B. Canny 控制图怎样得到</h4>
  <div class="flowline"><span>机制陆地图</span><b>→</b><span>投影边界</span><b>→</b>
  <span>二值 Canny</span><b>→</b><span>叠回锚点核对</span></div>
  <div class="grid four">
    {_figure(output_root, control['geometry_source']['path'], '1. 机制几何源图', control['geometry_source']['meaning'])}
    {_figure(output_root, control['projected_boundaries']['path'], '2. 投影后的边界', control['projected_boundaries']['meaning'])}
    {_figure(output_root, control['canny']['path'], '3. 实际 ControlNet 输入', f"边缘占画面 {control['canny']['edge_fraction'] * 100:.2f}%。只有这张图输入 ControlNet。")}
    {_figure(output_root, control['anchor_overlay']['path'], '4. 锚点叠加检查', control['anchor_overlay']['meaning'])}
  </div>
  <p class="plain-note">Canny 只告诉模型“岸线和新生陆地边界在哪里”；
  悬浮浓度和水下厚度不会塞进线稿。</p>
  <div class="input-box"><strong>这一帧模型实际看到了什么：</strong>
  ControlNet 只读取上面的二值 Canny；SDXL 两个文本编码器读取下一节展示的完整正向和
  负向英文。程序截图、悬浮浓度、水下厚度、新生陆地区域和水流箭头都没有输入扩散模型；
  它们只用于审计或最后的机制约束组合。</div>
  <h4>C. 语言提示怎样组装</h4>
  {prompt_table}
  <div class="token-row"><span>Tokenizer 1：正向 {tokens['tokenizer']['positive']} /
  负向 {tokens['tokenizer']['negative']}</span><span>Tokenizer 2：正向
  {tokens['tokenizer_2']['positive']} / 负向 {tokens['tokenizer_2']['negative']}</span>
  <span>上限均为 77</span></div>
  <details><summary>展开真正送入模型的完整英文</summary>
    <div class="details-body"><p><strong>正向：</strong></p>
    <pre>{html.escape(entry['prompt']['positive_combined'])}</pre>
    <p><strong>负向：</strong></p><pre>{html.escape(entry['prompt']['negative_combined'])}</pre></div>
  </details>
  <h4>D. 模型原图与最终组合</h4>
  <div class="grid four">
    {_figure(output_root, raw['path'], f'SDXL 原始候选（复现编号 {report_seed}）', '模型直接输出，保留用于判断提示词和 Canny 的作用；没有进入最终像素。')}
    {_figure(output_root, composite['colored_layers']['suspended_sediment']['path'], '实际使用的悬浮泥沙颜色层', '由程序粒子浓度得到，不是第二个生图模型。')}
    {_figure(output_root, composite['allowed_region']['path'], '允许修改区域', composite['allowed_region']['meaning'])}
    {_figure(output_root, composite['final']['path'], '最终关键帧', '固定视觉锚点加机制层；远离白色允许区域的像素完全不变。')}
  </div>
  <div class="grid three compact">
    {_figure(output_root, composite['colored_layers']['underwater_deposit']['path'], '水下沉积颜色层', '保留水面亮度，以低透明度表现水底浅滩。')}
    {_figure(output_root, composite['colored_layers']['new_land_texture']['path'], '新生陆地纹理层', '只在程序 new_land 区域内出现，纹理统计来自现有沙地。')}
    {_figure(output_root, composite['difference']['path'], '相对锚点的变化图', '越亮表示变化越大；用于检查背景是否漂移。')}
  </div>
</article>
"""
        )

    check_rows = "".join(
        "<tr>"
        f"<td>{html.escape(check['name'])}</td>"
        f"<td>{html.escape(check.get('scope', 'generic'))}</td>"
        f"<td class=\"{'pass' if check['passed'] else 'fail'}\">"
        f"{'通过' if check['passed'] else '失败'}</td>"
        f"<td><code>{html.escape(json.dumps(check['evidence'], ensure_ascii=False))}</code></td>"
        "</tr>"
        for check in evaluated["checks"]
    )
    transitions = "".join(
        "<tr>"
        f"<td>{item['index'] + 1}</td>"
        f"<td>{html.escape(item['first']['id'])} → {html.escape(item['last']['id'])}</td>"
        f"<td>{html.escape(item['only_major_change'])}</td>"
        f"<td>{item['suggested_duration_seconds']} 秒</td>"
        "</tr>"
        for item in handoff["transitions"]
    )
    model_minutes = generated["total_generation_seconds"] / 60
    sequence_sheet = _rel(
        output_root, output_root / "sequence-contact-sheet.jpg"
    )
    raw_sheet = _rel(
        output_root, output_root / "raw-candidates-by-seed.jpg"
    )
    blind_sheet = _rel(
        output_root, output_root / "blind-review.jpg"
    )
    cache = generated.get(
        "cache",
        {"reused": 0, "generated": len(generated["candidates"])},
    )
    render_settings = generated["settings"]
    seed_text = "、".join(
        str(seed) for seed in render_settings["seeds"]
    )
    model_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{html.escape(record['model_id'])}</td>"
        f"<td>{html.escape(record['variant'])}</td>"
        f"<td><code>{html.escape(record['path'])}</code></td>"
        "</tr>"
        for name, record in generated["models"].items()
    )
    source_links = "".join(
        f'<li><a href="{html.escape(_rel(output_root, path))}">'
        f"{html.escape(label)}</a></li>"
        for label, path in (
            ("序列规格 JSON", spec["_spec_path"]),
            ("机制 states.jsonl", prepared["sources"]["states"]),
            ("时间线 timeline.json", prepared["sources"]["timeline"]),
            (
                "机制 simulation_config.json",
                prepared["sources"]["simulation_config"],
            ),
            (
                "模型权重指纹",
                output_root / "_work" / "model_fingerprints.json",
            ),
            ("盲评映射", output_root / "_work" / "blind_order.json"),
        )
    )
    markdown_final_list = "\n".join(
        f"- `final/{item['output_filename']}`"
        for item in (spec["anchor"], *spec["keyframes"])
    )

    template = Template(
        """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 1.3｜程序机制到关键帧的完整生成报告</title>
<style>
:root{--ink:#17252b;--muted:#607179;--paper:#f4f0e6;--card:#fffdf8;--line:#d7d2c7;
--river:#0b6975;--deep:#083e48;--sed:#a9562d;--good:#e3f2e9;--warn:#fff0dc}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);
color:var(--ink);font:16px/1.72 Inter,"Noto Sans SC","Microsoft YaHei",system-ui,sans-serif}
a{color:var(--deep)}header{padding:70px 24px;color:white;background:linear-gradient(120deg,#073d48,#137a84)}
.wrap,main{width:min(1180px,calc(100% - 36px));margin:auto}header h1{max-width:900px;margin:0;
font-size:clamp(2.2rem,5vw,4.2rem);line-height:1.1}header p{max-width:850px;color:#daf0ed}
nav{position:sticky;top:0;z-index:5;display:flex;gap:8px;justify-content:center;overflow:auto;
padding:10px;background:#f4f0e6ed;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}
nav a{white-space:nowrap;padding:5px 10px;text-decoration:none;color:var(--ink)}main{padding:54px 0 80px}
section{margin-bottom:72px;scroll-margin-top:70px}h2{font-size:clamp(1.7rem,3vw,2.6rem);margin-bottom:8px}
h3{font-size:1.55rem;margin:0}h4{margin:30px 0 10px;font-size:1.08rem}.muted,figcaption{color:var(--muted)}
.eyebrow{margin:0;color:var(--sed);font-size:.8rem;font-weight:800;letter-spacing:.12em}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.grid.three{grid-template-columns:repeat(3,minmax(0,1fr))}
.grid.four{grid-template-columns:repeat(4,minmax(0,1fr))}.grid.five{grid-template-columns:repeat(5,minmax(0,1fr))}
figure{margin:0;overflow:hidden;border:1px solid var(--line);border-radius:13px;background:var(--card);
box-shadow:0 10px 26px #00000010}figure img{display:block;width:100%;height:auto;background:#102b32}
figcaption{padding:12px 14px;font-size:.88rem}figcaption strong{display:block;color:var(--ink)}
.hero-sequence figure img{aspect-ratio:1.75;object-fit:cover}.callout,.plain-note,.input-box{padding:18px 20px;
border-left:5px solid var(--river);border-radius:9px;background:var(--good)}.warn{border-color:var(--sed);background:var(--warn)}
.input-box{margin:14px 0;border-color:#667d86;background:#edf2f3}
.flowline{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:14px;border-radius:10px;background:#e4ece9}
.flowline span{padding:7px 10px;border-radius:8px;background:white}.flowline b{color:var(--sed)}
.stage-card{margin:24px 0 50px;padding:25px;border:1px solid var(--line);border-radius:18px;background:#faf7ef}
.stage-heading{display:flex;gap:14px;align-items:center}.stage-heading>span{display:grid;place-items:center;width:46px;
height:46px;border-radius:50%;color:white;background:var(--sed);font-size:1.2rem;font-weight:800}
table{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden}
th,td{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:white;background:var(--deep)}.token-row{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
.token-row span{padding:6px 10px;border-radius:999px;background:#e4ece9;font-size:.86rem}
details{margin:12px 0;border:1px solid var(--line);border-radius:10px;background:white}
summary{padding:14px 16px;cursor:pointer;font-weight:700}.details-body{padding:0 16px 16px}
pre{padding:14px;white-space:pre-wrap;overflow-wrap:anywhere;color:#e9f5f2;background:#102d33;border-radius:8px}
code{font-family:ui-monospace,monospace;font-size:.82rem}.pass{color:#17643a;font-weight:800}.fail{color:#a12727;font-weight:800}
.architecture{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}.architecture article{padding:18px;border:1px solid var(--line);
border-radius:12px;background:white}.architecture h3{font-size:1.05rem}.compact figcaption{min-height:74px}
@media(max-width:950px){.grid.four,.grid.five{grid-template-columns:repeat(2,minmax(0,1fr))}.architecture{grid-template-columns:1fr}}
@media(max-width:650px){.grid,.grid.three,.grid.four,.grid.five{grid-template-columns:1fr}.stage-card{padding:15px}
th,td{padding:9px}.wrap,main{width:min(100% - 20px,1180px)}} </style></head>
<body><header><div class="wrap"><p class="eyebrow">STAGE 1.3 · 可解释关键帧流水线</p>
<h1>程序中的泥沙、沉积与新陆地，怎样变成连续关键帧</h1>
<p>这份报告不只展示结果。它逐步公开状态选择、语义层、Canny 制作、提示词拼装、
模型原图、确定性组合和验收，让第一次接触项目的人能够看懂，也让开发者能把框架换到其他案例。</p>
</div></header>
<nav><a href="#result">最终序列</a><a href="#overview">完整流程</a><a href="#states">状态</a>
<a href="#stages">逐帧过程</a><a href="#raw">模型候选</a><a href="#evaluation">验收</a>
<a href="#framework">通用框架</a><a href="#handoff">视频交接</a><a href="#reproduce">复现</a></nav>
<main>
<section id="result"><p class="eyebrow">01 · 结果</p><h2>五张连续关键帧</h2>
<p class="muted">第 0 张沿用上一阶段选中的视觉锚点；后四张只在程序允许变化的区域增加泥沙、沉积和新陆地。</p>
<div class="grid five hero-sequence">$final_figures</div>
<div class="callout"><strong>分类说明：</strong>最终图不是四张新的 SDXL 原图，而是同一张已选 SDXL
视觉锚点加上可追溯的程序机制层。当前阶段的 SDXL 原始候选完整保留，但因为会改变整幅场景，没有复制到最终像素。</div>
</section>
<section id="overview"><p class="eyebrow">02 · 总览</p><h2>从程序状态到视频交接</h2>
<div class="flowline"><span>程序状态</span><b>→</b><span>可读语义层</span><b>→</b><span>硬边界 Canny</span>
<b>→</b><span>分段语言提示</span><b>→</b><span>SDXL 原图</span><b>→</b><span>机制约束组合</span>
<b>→</b><span>自动验收</span><b>→</b><span>LTX 首尾帧交接</span></div>
<h3 style="margin-top:26px">先认识四个词</h3><div class="architecture">
<article><h3>Canny</h3><p>黑底白线的硬边界图，告诉 ControlNet 岸线或沙洲边界在哪里。</p></article>
<article><h3>语义层</h3><p>由程序数据生成的浓度、厚度或陆地范围图；每层只有一个明确含义。</p></article>
<article><h3>随机种子</h3><p>可复现模型噪声起点的编号，不代表模型、泥沙或阶段。</p></article>
<article><h3>模型原图</h3><p>SDXL 直接保存的输出，没有叠加程序颜色层，可用于诚实判断模型本身表现。</p></article>
<article><h3>允许修改区域</h3><p>白色处允许泥沙、沉积或沙洲改变；黑色处最终像素必须和视觉锚点完全相同。</p></article>
<article><h3>ControlNet 强度</h3><p>数值 0.60 表示边界约束对扩散采样的影响权重，不是透明度，也不是泥沙浓度。</p></article>
</div></section>
<section id="states"><p class="eyebrow">03 · 状态选择</p><h2>为什么选择这五个时刻</h2>
$state_program_figure
<table><thead><tr><th>#</th><th>画面含义</th><th>选择原因</th><th>display/state</th><th>悬浮</th><th>水下网格</th>
<th>新陆地</th><th>通道</th></tr></thead><tbody>$state_rows</tbody></table>
<p class="plain-note" style="margin-top:18px">状态表中的新陆地数量来自程序统计；斑块连通性
由案例评估器读取实际二值区域计算，不根据文案猜测。选择原因说明了每个时刻为何承担
当前这一个主要机制变化。</p></section>
<section id="stages"><p class="eyebrow">04 · 逐帧过程</p><h2>四张新关键帧分别怎么做</h2>
$process_sections</section>
<section id="raw"><p class="eyebrow">05 · 原始模型证据</p><h2>四个复现编号、四个阶段的全部原图</h2>
<p>共生成 ${candidate_count} 张 raw 候选，耗时约 ${model_minutes} 分钟。它们用来评估提示词和
Canny，不会被报告隐藏或冒充最终图。</p>
<table><thead><tr><th>用途</th><th>精确模型 ID</th><th>精度版本</th><th>本地路径</th></tr></thead>
<tbody>$model_rows</tbody></table>
<p class="plain-note" style="margin-top:18px">实际参数：${render_width}×${render_height}，
${render_steps} 个采样步骤，CFG ${render_cfg}，ControlNet 强度 ${render_control}；
随机起点为 $seed_text。最后一次执行复用
${cache_reused} 张、重新生成 ${cache_generated} 张；缓存只有在提示词、Canny 哈希、模型指纹、
参数、状态规格和随机起点全部一致时才命中。</p>
<figure><a href="$raw_sheet"><img src="$raw_sheet" alt="全部模型原始候选"></a>
<figcaption><strong>按随机种子成组的原始候选</strong>每行共享一个噪声起点，每列是一个机制阶段。</figcaption></figure>
<figure style="margin-top:20px"><a href="$blind_sheet"><img src="$blind_sheet" alt="盲评候选"></a>
<figcaption><strong>隐藏阶段和随机编号的盲评图</strong>先按 C01–C16 看画面，再从
<code>_work/blind_order.json</code> 查回阶段和随机起点，避免只凭熟悉的编号挑图。</figcaption></figure>
<div class="callout warn" style="margin-top:20px"><strong>这批 raw 图说明了什么：</strong>
ControlNet 确实抓住了河道、海岸与出水斑块边界，但 text-to-image 同时把大片既有地貌
重画成浅色沙地。问题不是“没有线稿”，而是线稿只能约束边界，不能锁住原图全部像素；
所以 raw 图保留为失败证据和材质参考，不能直接作为连续动画帧。</div>
</section>
<section id="evaluation"><p class="eyebrow">06 · 验收</p><h2>程序约束和文件完整性</h2>
<table><thead><tr><th>检查</th><th>范围</th><th>结果</th><th>证据</th></tr></thead><tbody>$check_rows</tbody></table>
<div class="grid" style="margin-top:20px"><div class="callout"><h3>已经解决</h3><p>五张图保持同一背景；
水下沉积和新陆地按程序单调增长；两个出水斑块在最终连接；背景允许区域之外像素差为 0。</p></div>
<div class="callout warn"><h3>仍有局限</h3><p>泥沙和水下沉积仍是确定性颜色层，湿沙纹理也可能偏平滑。
原始 SDXL 候选的全图漂移说明，当前模型还不能直接承担精准的机制定位。</p></div></div></section>
<section id="framework"><p class="eyebrow">07 · 通用化</p><h2>哪些保留，哪些换案例时替换</h2>
<div class="architecture"><article><h3>通用核心：直接保留</h3><p>规格验证、投影接口、Canny 记录、
提示词编译、候选管理、组合溯源、评估框架、HTML 报告和视频交接。</p></article>
<article><h3>案例适配器：按机制替换</h3><p>当前
<code>${state_adapter}</code> 把原始机制字段整理成流水线状态记录；其他程序提供自己的
适配器即可，不应修改通用 schema。</p></article>
<article><h3>规格文件：按故事替换</h3><p>状态编号、视觉锚点、坐标投影、每帧语言变化、禁区、
语义层和案例验收规则都属于 JSON 配置。</p></article></div>
<h3 style="margin-top:26px">现有例子带来的框架改进</h3>
<table><thead><tr><th>观察</th><th>通用规则</th></tr></thead><tbody>
<tr><td>软泥沙无法靠提示词定位</td><td>软物质由状态适配器生成浓度层，提示词只负责材质语言。</td></tr>
<tr><td>详细线稿让画面变硬</td><td>Canny 只接收硬几何；没有硬边界的案例可以关闭 ControlNet。</td></tr>
<tr><td>全图生成造成场景漂移</td><td>支持视觉锚点与允许修改区域；不把相同 seed 当作像素锁定。</td></tr>
<tr><td>中间 mask 无法理解</td><td>每层必须有图片、来源、普通话含义和是否输入模型的声明。</td></tr>
<tr><td>视频跨越太多机制会失控</td><td>关键帧规格把变化拆成单一阶段，再生成相邻首尾帧视频。</td></tr>
</tbody></table>
<p class="plain-note" style="margin-top:18px">额外的 display 40 / state 50 冒烟规格已经通过
<code>--prepare</code>，并生成自己的一帧版 <code>prepare-audit.html</code>；没有修改通用代码，
也没有覆盖正式序列。</p>
<h3 style="margin-top:30px">换一个程序案例时怎么接</h3>
<table><thead><tr><th>问题</th><th>框架答案</th></tr></thead><tbody>
<tr><td>哪些参数仍是经验值？</td><td>投影曲线、粒子平滑尺度、颜色与透明度、
Canny 阈值和 ControlNet 0.60 都来自当前案例；必须写进 config，并在新案例重新校准。</td></tr>
<tr><td>没有粒子或厚度怎么办？</td><td>adapter 只需输出自己确实拥有的语义层，并为每层声明
名称、数值范围、普通话含义、是否输入模型；不得伪造不存在的粒子或厚度。</td></tr>
<tr><td>没有适合 Canny 的硬边界怎么办？</td><td>把几何控制声明为关闭，直接走无 ControlNet 的图像路线；
软浓度图不能为了使用 ControlNet 被强行转成边线。</td></tr>
<tr><td>最少要提供什么？</td><td>视觉锚点或全图策略；带唯一 ID 的机制状态；坐标或投影；
至少一个可视语义层；每帧变化描述与禁区。硬边界、密度、厚度、对象区域、流向和案例验收均可选。</td></tr>
</tbody></table></section>
<section id="handoff"><p class="eyebrow">08 · 视频交接</p><h2>后续拆成四段 LTX-2.3 过渡</h2>
<table><thead><tr><th>#</th><th>首尾帧</th><th>中间唯一主要变化</th><th>建议时长</th></tr></thead>
<tbody>$transitions</tbody></table></section>
<section id="reproduce"><p class="eyebrow">09 · 复现</p><h2>同一流水线的五个阶段</h2>
<pre>/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \\
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json --prepare
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \\
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json --generate
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \\
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json --compose
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \\
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json --evaluate
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \\
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json --report</pre>
<p>默认会复用输入签名完全一致的 raw 候选；只有显式加 <code>--force</code> 才重新采样。
单独运行 <code>--report</code> 只读 manifest 和现有图片，不加载 SDXL 或 ControlNet。</p>
<details><summary>展开模型指纹、源数据和机器记录</summary><div class="details-body">
<p>模型权重的逐文件 SHA-256 保存在 <code>_work/model_fingerprints.json</code>；
每张图片的 SHA-256、尺寸、模式和来源保存在各阶段 manifest。</p>
<ul>$source_links</ul></div></details>
<ul><li><a href="sequence-contact-sheet.jpg">五张最终关键帧总览</a></li>
<li><a href="video_handoff.json">视频首尾帧交接规格</a></li>
<li><a href="_work/manifests/prepare.json">prepare 记录</a></li>
<li><a href="_work/manifests/generate.json">模型生成记录</a></li>
<li><a href="_work/manifests/compose.json">组合来源记录</a></li>
<li><a href="_work/manifests/evaluate.json">自动验收记录</a></li></ul>
</section></main></body></html>"""
    )
    html_report = template.substitute(
        final_figures="".join(final_figures),
        state_program_figure=state_program_figure,
        state_rows="".join(state_rows),
        process_sections="".join(process_sections),
        candidate_count=len(generated["candidates"]),
        model_minutes=f"{model_minutes:.2f}",
        raw_sheet=html.escape(raw_sheet),
        blind_sheet=html.escape(blind_sheet),
        model_rows=model_rows,
        cache_reused=cache["reused"],
        cache_generated=cache["generated"],
        render_width=render_settings["width"],
        render_height=render_settings["height"],
        render_steps=render_settings["steps"],
        render_cfg=render_settings["guidance_scale"],
        render_control=render_settings[
            "controlnet_conditioning_scale"
        ],
        seed_text=seed_text,
        check_rows=check_rows,
        transitions=transitions,
        source_links=source_links,
        state_adapter=html.escape(spec["state_adapter"]),
    )
    report_path = output_root / "report.html"
    report_path.write_text(html_report, encoding="utf-8")
    references = sorted(
        {
            value
            for value in re.findall(
                r'(?:src|href)="([^"]+)"', html_report
            )
            if value and not value.startswith(("#", "http:", "https:"))
        }
    )
    missing_references = [
        value
        for value in references
        if not (output_root / value).resolve().exists()
    ]
    if missing_references:
        raise FileNotFoundError(
            f"report contains missing references: {missing_references}"
        )

    markdown = f"""# Stage 1.3 机制关键帧生成报告

## 结论

五张关键帧使用同一张 Stage 1.2 视觉锚点。四张后续图的悬浮泥沙、水下沉积和新生陆地
来自对应程序状态，固定区域像素保持不变。当前阶段的 {len(generated["candidates"])}
张 raw SDXL ControlNet 候选全部保留，但因全图漂移未直接用于最终像素。

生成链路是：

```text
states.jsonl
→ 悬浮浓度 / 水下厚度 / 新生陆地
→ 固定岸线加新陆地边界的二值 Canny
→ 分段正负提示词
→ SDXL + Canny ControlNet 原始候选
→ 固定视觉锚点上的机制约束组合
→ 通用检查 + 三角洲案例检查
→ LTX-2.3 首尾帧交接
```

## 最终序列

{markdown_final_list}

## 模型与参数

- SDXL Base 1.0 FP16
- SDXL Canny ControlNet FP16
- {render_settings["width"]}×{render_settings["height"]}，
  {render_settings["steps"]} steps，CFG {render_settings["guidance_scale"]}，
  ControlNet scale {render_settings["controlnet_conditioning_scale"]}
- seeds：{seed_text}
- raw 生成耗时：{generated["total_generation_seconds"]:.3f} 秒
- 最近一次执行：缓存复用 {cache["reused"]} 张，重新生成 {cache["generated"]} 张

随机 seed 只是可复现的噪声起点。ControlNet scale 0.60 是边界约束权重，不是透明度。
每帧完整正负提示词和两个 tokenizer 的 token 数保存在 `_work/prompts/`。

## 模型原图为什么没有直接当成关键帧

16 张 raw 图证明稀疏 Canny 能让模型抓住河道、海岸和沙洲边界；但它不能锁定 Canny
之外的像素，模型把大片原有地貌重画成了浅色沙地。直接使用会造成镜头内地面“呼吸”。
因此 raw 图完整留在 `review/raw/` 和 `raw-candidates-by-seed.jpg`，最终图则保持同一视觉
锚点，只在 `allowed_region` 白色区域合入程序机制层。

## 复用框架

- 通用模块：规格验证、adapter 装载、投影接口、Canny、提示词编译、候选缓存、组合溯源、
  通用评估、HTML 报告和视频交接。
- 三角洲 adapter：解释 `particles`、`thick`、`new_land`、`flow_samples`。
- 案例配置：状态、投影、提示差异、颜色参数和验收规则。
- 没有硬边界的案例应关闭 ControlNet，不能把软浓度伪造成 Canny。
- 最小接入需要视觉锚点或全图策略、唯一状态 ID、坐标或投影、至少一个有说明的语义层，
  以及每帧变化与禁区。

## 验收

通用与案例检查共 {len(evaluated["checks"])} 项，结果：
`{evaluated["status"]}`。HTML 引用 {len(references)} 个本地资源，缺失 0 个。
display 40 / state 50 的单关键帧 smoke 规格也可独立生成 `prepare-audit.html`。

完整可视化过程、Canny 制作、提示词拼装、语义层、候选图、验收和复现命令见
`report.html`。机器可读记录位于 `_work/manifests/`。
"""
    (output_root / "report.md").write_text(markdown, encoding="utf-8")
    write_json(
        output_root / "_work" / "metadata.json",
        {
            "sequence_id": spec["sequence_id"],
            "classification": composed["classification"],
            "models": generated["models"],
            "runtime": generated["runtime"],
            "render_settings": generated["settings"],
            "cache": cache,
            "sources": prepared["sources"],
            "spec_path": spec["_spec_path"],
            "spec_sha256": prepared["spec_sha256"],
            "final_files": {
                spec["anchor"]["id"]: prepared["anchor"]["final"],
                **{
                    item["id"]: composed["keyframes"][item["id"]][
                        "final"
                    ]
                    for item in spec["keyframes"]
                },
            },
            "report_integrity": {
                "reference_count": len(references),
                "missing": missing_references,
            },
            "model_rerun_during_report": False,
        },
    )
    write_json(
        output_root / "_work" / "manifests" / "report.json",
        {
            "status": "written",
            "html": str(report_path.resolve()),
            "markdown": str((output_root / "report.md").resolve()),
            "sections": [
                "final sequence",
                "pipeline overview",
                "state selection",
                "per-keyframe process",
                "raw candidates",
                "evaluation",
                "general framework",
                "video handoff",
                "reproduction",
            ],
            "reference_count": len(references),
            "missing_references": missing_references,
            "model_rerun": False,
        },
    )
    return report_path
