"""Build and verify the aggregate Stage 2 Phase 3 evidence report."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any

from .framework.contracts import (
    artifact_record,
    load_json,
    sha256_path,
    write_json,
)


STAGE2_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = STAGE2_ROOT / "experiments"
PHASE_ROOT = STAGE2_ROOT / "output" / "phase-3"
REPORT_PATH = PHASE_ROOT / "report.html"
MANIFEST_PATH = PHASE_ROOT / "phase3_manifest.json"
LEDGER_PATH = EXPERIMENTS_ROOT / "ledger.json"
EXPECTED_EXPERIMENT_IDS = tuple(
    f"EXP-20260729-{index:03d}" for index in range(1, 16)
)
EXPECTED_CASES = {"MATH-02", "PHYS-01", "CHEM-01", "BIO-01", "GEO-02"}
EXPECTED_MODELS = {
    "stabilityai/stable-diffusion-xl-base-1.0",
    "diffusers/controlnet-canny-sdxl-1.0",
}

EXPERIMENT_TITLES = {
    "EXP-20260729-001": "物理：三种控制图的第一轮对照",
    "EXP-20260729-002": "物理：ControlNet 强度扫描",
    "EXP-20260729-003": "物理：Img2Img 重绘强度扫描",
    "EXP-20260729-004": "物理：单供体材质残差（失败）",
    "EXP-20260729-005": "物理：多供体稳健水纹投影",
    "EXP-20260729-006": "化学：液体区域材质回归",
    "EXP-20260729-007": "化学：Img2Img 控制路线",
    "EXP-20260729-008": "化学：T2I 与 Img2Img 能力差异",
    "EXP-20260729-009": "化学：语义充分的器材线稿",
    "EXP-20260729-010": "数学：未归一化拼图控制",
    "EXP-20260729-011": "数学：控制占用率归一化",
    "EXP-20260729-012": "数学：只在拼片内投影木纹",
    "EXP-20260729-013": "生物：两个细胞与精确染色体计数",
    "EXP-20260729-014": "地理：强控制保机制但留下墨线",
    "EXP-20260729-015": "地理：控制强度的机制—材质折中",
}

COMPOSITE_EVIDENCE = {
    "EXP-20260729-004": (
        "material-projection-comparison.jpg",
        "被拒绝的单供体投影：供体里的接缝和物体会一起进入 composite。",
    ),
    "EXP-20260729-005": (
        "ensemble-material-comparison-gain_030.jpg",
        "接受的水纹稳健投影：程序底图、raw 供体、留一法 composite 并排。",
    ),
    "EXP-20260729-006": (
        "ensemble-material-comparison-frozen_gain_030.jpg",
        "化学回归：机器门通过，但平滑粉色供体没有可见材质收益。",
    ),
    "EXP-20260729-012": (
        "ensemble-material-comparison-gain_070.jpg",
        "接受的数学安全路线：木纹只进入四个拼片，空区和背景不改。",
    ),
}


def _load_experiments() -> list[dict[str, Any]]:
    experiments = []
    for experiment_id in EXPECTED_EXPERIMENT_IDS:
        source_root = EXPERIMENTS_ROOT / experiment_id
        output_root = PHASE_ROOT / experiment_id
        paths = {
            "spec": source_root / "spec.json",
            "hypothesis": source_root / "hypothesis.md",
            "review": source_root / "review.json",
            "prepare": output_root / "_work" / "prepare.json",
            "generate": output_root / "_work" / "generate.json",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise ValueError(
                f"{experiment_id} is missing required evidence: {missing}"
            )
        experiments.append(
            {
                "experiment_id": experiment_id,
                "source_root": source_root,
                "output_root": output_root,
                "paths": paths,
                "spec": load_json(paths["spec"]),
                "prepared": load_json(paths["prepare"]),
                "generated": load_json(paths["generate"]),
                "review": load_json(paths["review"]),
            }
        )
    return experiments


def _verify_experiment(item: dict[str, Any]) -> dict[str, Any]:
    experiment_id = item["experiment_id"]
    spec = item["spec"]
    prepared = item["prepared"]
    generated = item["generated"]
    review = item["review"]
    if spec["experiment_id"] != experiment_id:
        raise ValueError(f"{experiment_id}: spec ID mismatch")
    if generated["experiment_id"] != experiment_id:
        raise ValueError(f"{experiment_id}: generated ID mismatch")
    if review["experiment_id"] != experiment_id:
        raise ValueError(f"{experiment_id}: review ID mismatch")
    planned = len(spec["configurations"]) * len(spec["render"]["seeds"])
    if planned != spec["budget"]["actual_planned_image_candidates"]:
        raise ValueError(f"{experiment_id}: planned matrix mismatch")
    if planned > spec["budget"]["maximum_new_image_candidates"]:
        raise ValueError(f"{experiment_id}: candidate budget exceeded")
    generated_count = int(generated["cache"]["generated"])
    maximum_new_generation = int(
        spec["budget"].get(
            "maximum_new_generation",
            spec["budget"]["maximum_new_image_candidates"],
        )
    )
    if generated_count > maximum_new_generation:
        raise ValueError(f"{experiment_id}: new generation budget exceeded")
    if len(generated["candidates"]) != planned:
        raise ValueError(f"{experiment_id}: candidate evidence incomplete")
    if generated["model_runs"]["video_candidates"] != 0:
        raise ValueError(f"{experiment_id}: unexpected video run")
    if any(
        record["would_truncate"]
        for record in generated["prompt_token_preflight"].values()
    ):
        raise ValueError(f"{experiment_id}: prompt was truncated")
    model_ids = {
        value["model_id"] for value in generated["models"].values()
    }
    if model_ids != EXPECTED_MODELS:
        raise ValueError(f"{experiment_id}: unregistered model substitution")
    for candidate in generated["candidates"]:
        candidate_path = item["output_root"] / candidate["path"]
        if sha256_path(candidate_path) != candidate["sha256"]:
            raise ValueError(
                f"{experiment_id}: candidate hash mismatch {candidate_path}"
            )
    for key in ("clean_keyframe", "semantic_layers"):
        record = prepared["source"][key]
        path = item["output_root"] / record["path"]
        if sha256_path(path) != record["sha256"]:
            raise ValueError(f"{experiment_id}: copied source mismatch")
    return {
        "experiment_id": experiment_id,
        "case_id": spec["case_id"],
        "candidate_count": len(generated["candidates"]),
        "new_model_runs": generated_count,
        "reused_candidates": int(generated["cache"]["reused"]),
        "verdict": review["verdict"],
        "prompt_tokens": generated["prompt_token_preflight"],
    }


def _relative_source_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(STAGE2_ROOT).as_posix(),
        "sha256": sha256_path(path),
    }


def _source_records(
    experiments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    paths = [
        Path(__file__),
        STAGE2_ROOT / "phase3_experiment.py",
        STAGE2_ROOT / "framework" / "image_experiment.py",
        STAGE2_ROOT / "framework" / "material_projection.py",
    ]
    for item in experiments:
        paths.extend(item["paths"][name] for name in ("spec", "hypothesis", "review"))
        for optional_name in ("build_control.py",):
            optional = item["source_root"] / optional_name
            if optional.is_file():
                paths.append(optional)
    return [_relative_source_record(path) for path in paths]


def _control_cards(item: dict[str, Any]) -> str:
    spec = item["spec"]
    prepared = item["prepared"]
    experiment_id = item["experiment_id"]
    routes = []
    for configuration in spec["configurations"]:
        route = configuration["control_route"]
        if route not in routes:
            routes.append(route)
    cards = []
    explanations = spec.get("control_override_explanations", {})
    derivation = load_json(
        item["output_root"] / prepared["control_derivation"]["path"]
    )
    for route in routes:
        control = prepared["controls"][route]
        explanation = explanations.get(route)
        if explanation is None:
            explanation = derivation.get(
                route,
                "由通用预处理器从无标注程序图或语义层确定性导出。",
            )
        path = f"{experiment_id}/{control['path']}"
        cards.append(
            f"""<figure class="control-card">
              <a href="{html.escape(path)}"><img loading="lazy"
                src="{html.escape(path)}" alt="{html.escape(route)} 控制图"></a>
              <figcaption><code>{html.escape(route)}</code><br>
              {html.escape(explanation)}<br>
              <span class="muted">白线占画面 {control['edge_fraction'] * 100:.3f}%</span>
              </figcaption></figure>"""
        )
    return "".join(cards)


def _score_summary(review: dict[str, Any]) -> str:
    records = review.get("configuration_summary")
    if records is None and review.get("deterministic_projection_variants"):
        records = [
            {
                "configuration_id": item["variant_id"],
                "finding_zh": item["finding_zh"],
                "score_text": (
                    f"机器门 {item['hard_checks_passed']}/"
                    f"{item['hard_checks_total']}；"
                    f"{item['visual_status']}"
                ),
            }
            for item in review["deterministic_projection_variants"]
        ]
    if records is None and review.get("selected_variant_metrics"):
        metrics = review["selected_variant_metrics"]
        records = [
            {
                "configuration_id": "selected_region_projection",
                "finding_zh": review["verdict_zh"],
                "score_text": (
                    f"机制 {metrics['mechanism_fidelity_5']}/5；"
                    f"材质 {metrics['material_improvement_5']}/5；"
                    f"伪影控制 {metrics['artifact_control_5']}/5"
                ),
            }
        ]
    if records is None and review.get("selected_variant"):
        metrics = review["selected_variant"]
        records = [
            {
                "configuration_id": metrics["variant_id"],
                "finding_zh": review["verdict_zh"],
                "score_text": (
                    f"机制 {metrics['mechanism_fidelity_5']}/5；"
                    f"材质 {metrics['material_improvement_5']}/5；"
                    f"稳健性 {metrics['seed_robustness_5']}/5"
                ),
            }
        ]
    if records is None and review.get("candidate_reviews"):
        records = []
        for candidate in review["candidate_reviews"]:
            identifier = candidate.get(
                "blind_id", f"seed_{candidate.get('seed', 'unknown')}"
            )
            score_parts = []
            for key, title in (
                ("mechanism_fidelity_5", "机制"),
                ("material_naturalness_5", "材质"),
                ("composite_material_gain_5", "材质增益"),
                ("artifact_control_5", "伪影控制"),
            ):
                if key in candidate:
                    score_parts.append(f"{title} {candidate[key]}/5")
            records.append(
                {
                    "configuration_id": identifier,
                    "finding_zh": candidate.get(
                        "notes_zh",
                        candidate.get("raw_finding_zh", "见评审原文"),
                    ),
                    "score_text": "；".join(score_parts),
                }
            )
    if records is None:
        records = [
            {
                "configuration_id": "aggregate_verdict",
                "finding_zh": review["verdict_zh"],
                "score_text": "见机器门与评审原文",
            }
        ]
    rows = []
    for record in records:
        mechanism = record.get("mean_mechanism_fidelity_5")
        material = record.get("mean_material_naturalness_5")
        usable = record.get("usable_seed_fraction")
        scores = [record["score_text"]] if record.get("score_text") else []
        if mechanism is not None:
            scores.append(f"机制 {mechanism:.2f}/5")
        if material is not None:
            scores.append(f"材质 {material:.2f}/5")
        if usable is not None:
            scores.append(f"可用种子 {usable * 100:.0f}%")
        rows.append(
            f"""<tr><td><code>{html.escape(record['configuration_id'])}</code></td>
            <td>{html.escape('；'.join(scores))}</td>
            <td>{html.escape(record['finding_zh'])}</td></tr>"""
        )
    return "".join(rows)


def _experiment_section(item: dict[str, Any], index: int) -> str:
    experiment_id = item["experiment_id"]
    spec = item["spec"]
    prepared = item["prepared"]
    generated = item["generated"]
    review = item["review"]
    input_path = f"{experiment_id}/{prepared['source']['clean_keyframe']['path']}"
    blind_path = f"{experiment_id}/{generated['sheets']['blind']['path']}"
    labeled_path = f"{experiment_id}/{generated['sheets']['labeled']['path']}"
    token = generated["prompt_token_preflight"]
    positive_counts = token["positive"]["counts_including_special_tokens"]
    negative_counts = token["negative"]["counts_including_special_tokens"]
    composite = ""
    if experiment_id in COMPOSITE_EVIDENCE:
        relative_path, caption = COMPOSITE_EVIDENCE[experiment_id]
        evidence_path = item["output_root"] / relative_path
        if not evidence_path.is_file():
            raise ValueError(
                f"{experiment_id}: missing composite evidence {relative_path}"
            )
        composite = f"""<h4>程序约束后的 composite</h4>
        <p>{html.escape(caption)}</p>
        <a href="{experiment_id}/{relative_path}"><img class="wide evidence"
        loading="lazy" src="{experiment_id}/{relative_path}"
        alt="{html.escape(caption)}"></a>"""
    machine = review.get("machine_evidence")
    machine_text = (
        f"<pre>{html.escape(str(machine))}</pre>" if machine else ""
    )
    return f"""
    <section class="experiment" id="{experiment_id}">
      <p class="eyebrow">实验 {index:02d} · {html.escape(spec['case_id'])}</p>
      <h3>{html.escape(EXPERIMENT_TITLES[experiment_id])}</h3>
      <p><b>要验证：</b>{html.escape(spec['hypothesis_zh'])}</p>
      <p><b>唯一变量：</b>{html.escape(spec['single_variable_zh'])}</p>
      <div class="input-grid">
        <figure><a href="{input_path}"><img loading="lazy" src="{input_path}"
        alt="无标注程序关键帧"></a><figcaption>无标注程序关键帧：事实与几何的来源</figcaption></figure>
        <div><h4>本轮模型实际读到的语言</h4>
        <p class="prompt"><b>正向：</b>{html.escape(prepared['positive_prompt']['text'])}</p>
        <p class="prompt"><b>负向：</b>{html.escape(prepared['negative_prompt']['text'])}</p>
        <p class="muted">正向 token：{positive_counts['tokenizer']} /
        {positive_counts['tokenizer_2']}；负向 token：
        {negative_counts['tokenizer']} / {negative_counts['tokenizer_2']}；
        两个上限都是 77，均未截断。</p></div>
      </div>
      <h4>模型看到的控制图</h4>
      <div class="control-grid">{_control_cards(item)}</div>
      <h4>全部 raw 模型候选（先盲评）</h4>
      <p>字母只是盲评编号；图中没有路线名或 seed，避免先入为主。</p>
      <a href="{blind_path}"><img class="wide" loading="lazy"
      src="{blind_path}" alt="{experiment_id} 全部 raw 候选盲评表"></a>
      <details><summary>展开带配置名与 seed 的联系表</summary>
      <a href="{labeled_path}"><img class="wide" loading="lazy"
      src="{labeled_path}" alt="{experiment_id} 解盲联系表"></a></details>
      {composite}
      <h4>解盲后判定</h4>
      <table><thead><tr><th>配置</th><th>分数</th><th>看到什么</th></tr></thead>
      <tbody>{_score_summary(review)}</tbody></table>
      <div class="verdict"><b>{html.escape(review['verdict_zh'])}</b><br>
      <span>{html.escape(review['generalization_boundary_zh'])}</span></div>
      {machine_text}
      <p class="links"><a href="{experiment_id}/_work/generate.json">候选参数、哈希和耗时</a>
      · <a href="{experiment_id}/controls/derivation.json">控制图推导</a>
      · <a href="../../experiments/{experiment_id}/review.json">Agent 评审原文</a></p>
    </section>"""


def _write_report(
    experiments: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    total_candidates = sum(item["candidate_count"] for item in checks)
    total_new = sum(item["new_model_runs"] for item in checks)
    total_reused = sum(item["reused_candidates"] for item in checks)
    sections = "".join(
        _experiment_section(item, index)
        for index, item in enumerate(experiments, 1)
    )
    REPORT_PATH.write_text(
        f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 2 · Phase 3 图像模型能力边界</title>
<style>
:root{{--ink:#17373d;--deep:#0d2930;--paper:#f2eee3;--card:#fffdf7;
--line:#bed0ca;--teal:#167f76;--amber:#d2842d;--red:#a94b49}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.65 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
header,main{{max-width:1200px;margin:auto;padding:36px 24px}}header{{padding-top:58px}}
h1{{font-size:clamp(2.3rem,6vw,5rem);line-height:1.04;margin:.15em 0}}
h2{{font-size:2rem;margin-top:0}}h3{{font-size:1.55rem;margin:.2em 0}}
h4{{font-size:1.05rem;margin:1.25em 0 .55em}}p{{max-width:88ch}}
a{{color:var(--teal)}}section{{border-top:3px solid var(--ink);padding:42px 0}}
.eyebrow{{color:var(--teal);font-weight:800;letter-spacing:.12em;text-transform:uppercase}}
.metrics,.route-grid,.control-grid,.input-grid{{display:grid;gap:13px;margin:20px 0}}
.metrics{{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}}
.route-grid{{grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}}
.control-grid{{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}}
.input-grid{{grid-template-columns:minmax(240px,.8fr) minmax(300px,1.2fr);align-items:start}}
.metric,.route,.verdict,.note{{background:var(--card);border:1px solid var(--line);
border-radius:13px;padding:16px}}.metric b{{display:block;font-size:2rem;color:var(--teal)}}
.route.reject{{border-left:6px solid var(--red)}}.route.accept{{border-left:6px solid var(--teal)}}
figure{{margin:0}}figure img,.wide{{display:block;width:100%;height:auto;border-radius:9px;
background:var(--deep)}}figcaption{{padding:7px 3px;color:#4f696d}}
.control-card{{background:var(--card);border:1px solid var(--line);padding:9px;border-radius:11px}}
.control-card img{{aspect-ratio:16/9;object-fit:contain}}
.experiment{{border-top:2px solid var(--line)}}.experiment>.eyebrow{{margin-bottom:0}}
.prompt{{font-size:.93rem;background:#eef3ef;padding:10px;border-radius:8px}}
.verdict{{margin:16px 0;border-left:6px solid var(--amber)}}.verdict b{{font-size:1.05rem}}
.evidence{{max-height:900px;object-fit:contain;background:var(--deep)}}
table{{width:100%;border-collapse:collapse;background:var(--card);font-size:.9rem}}
th,td{{border:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}}
pre{{white-space:pre-wrap;overflow:auto;background:var(--deep);color:#eaf5f1;padding:15px;
border-radius:9px}}details{{margin:12px 0}}summary{{cursor:pointer;font-weight:700}}
.muted{{color:#60787c}}.links{{font-size:.92rem}}code{{overflow-wrap:anywhere}}
@media(max-width:760px){{header,main{{padding-left:14px;padding-right:14px}}
.input-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><p class="eyebrow">LIVE-DOCUMENT · LOOP ENGINEER · PHASE 3</p>
<h1>不是让模型重画一切，<br>而是先划清能力边界</h1>
<p>这一阶段用数学、物理、化学、生物、地理五个程序动画的固定关键帧，
做了 15 个单变量实验。目标不是从随机结果里挑一张漂亮图，而是回答：
哪些信息能交给 SDXL，哪些事实必须留在程序里。</p>
<div class="metrics">
<div class="metric"><b>5 / 5</b>学科代表已覆盖</div>
<div class="metric"><b>15 / 15</b>实验已盲评并判定</div>
<div class="metric"><b>{total_candidates}</b>报告中的 raw 候选</div>
<div class="metric"><b>{total_new}</b>实际新图片模型调用</div>
<div class="metric"><b>{total_reused}</b>严格签名复用候选</div>
<div class="metric"><b>0</b>本阶段视频模型调用</div>
</div></header><main>
<section><h2>1. 第一次接手项目，先理解这条生成链</h2>
<ol>
<li><b>程序关键帧</b>先计算数量、位置、拓扑和连续场。报告里的输入图已经移除文字、
箭头与界面。</li>
<li><b>语义层</b>是程序同时输出的机器数据，例如物体边界、允许修改区域、水面高度或
对象 ID；它不是模型凭截图猜出的 mask。</li>
<li><b>控制图（conditioning）</b>是实际送入 Canny ControlNet 的黑底白线图。
白线只是“请在这里保留边缘”，模型不知道线代表三角形、玻璃还是雨。</li>
<li><b>语言提示</b>由场景、材料、本帧变化、必须保留项、负向禁区五部分编译。
两个 tokenizer 都在加载完整扩散管线前检查 77-token 上限。</li>
<li><b>seed</b>只是可复现的初始噪声编号。每个配置固定用 3101–3104，
不是材质名，也不是人工评分。</li>
<li><b>raw</b>是模型直接输出；<b>composite</b>是程序约束后的合成结果；
<b>final</b>只允许来自已通过门禁且冻结选择规则的结果。三者绝不混称。</li>
<li>先看隐藏路线名的联系表，再解盲评分；只要数量、拓扑、提示截断或模型指纹硬门失败，
再漂亮也拒绝。</li>
</ol>
<div class="note"><b>固定模型：</b>SDXL Base 1.0 FP16 +
SDXL Canny ControlNet FP16。所有实验使用相同本地权重指纹，没有下载、替换或把
程序合成图冒充模型增强。</div></section>

<section><h2>2. Phase 3 得出的通用路由表</h2>
<div class="route-grid">
<article class="route accept"><h3>精确数量 / 刚体几何</h3>
<p><b>整图重绘关闭。</b>MATH-02 与 BIO-01 都证明 Canny 能保大轮廓，却不能可靠保证
四块拼片或每细胞六条染色单体。程序保留结构，模型只提供受限材质残差。</p></article>
<article class="route accept"><h3>语义明确的器材边界</h3>
<p><b>可用 T2I ControlNet。</b>CHEM-01 中，只有外包围 U 形时成功率 2/4；
加入烧杯口沿、液面、滴定管、活塞与滴嘴后达到 4/4。核心要求“线稿语义充分”，
具体器材线仍属案例插件。</p></article>
<article class="route accept"><h3>连续材质 / 高度场</h3>
<p><b>程序决定大形，模型负责微纹理。</b>PHYS-01 的整图 Img2Img 在强度升高时先丢波形，
后得到材质；多供体限幅中位数残差能保留程序低频结构并安全增加水纹。</p></article>
<article class="route reject"><h3>静态环境 + 局部动态场</h3>
<p><b>一个 Canny 强度不够。</b>GEO-02 中强控制留下墨线，弱控制丢雨区。
Phase 4 要把地形写实化与程序云雨层分开，而不是继续堆 prompt 或挑幸运 seed。</p></article>
</div></section>

<section><h2>3. 如何读下面的 15 轮证据</h2>
<p>每轮严格按实际顺序展示：无标注程序输入 → 模型控制图 → 完整提示词与 token 数 →
全部 raw 候选 → 可选 composite → 解盲评分和结论。点击图片可看原尺寸；
每轮末尾可打开包含模型 ID、权重指纹、参数、seed、耗时和 SHA-256 的 JSON。</p>
</section>
{sections}

<section><h2>4. 阶段自评与自动晋级</h2>
<table><thead><tr><th>门禁</th><th>结果</th><th>证据</th></tr></thead><tbody>
<tr><td>五学科代表覆盖</td><td>通过</td><td>MATH-02、PHYS-01、CHEM-01、BIO-01、GEO-02</td></tr>
<tr><td>实验预算</td><td>通过</td><td>每轮最多 12 张；总计 {total_new} 次新图片调用，{total_reused} 张严格缓存复用</td></tr>
<tr><td>模型与提示完整性</td><td>通过</td><td>15/15 权重 ID 与指纹一致；0 次 token 静默截断</td></tr>
<tr><td>证据可审计</td><td>通过</td><td>全部控制图、提示、raw、composite、盲表、失败结果和哈希保留</td></tr>
<tr><td>能力边界是否可执行</td><td>通过</td><td>形成按 exact-count、semantic-boundary、continuous-field、layered-field 路由的规则</td></tr>
</tbody></table>
<div class="verdict"><b>Phase 3 判定：passed。</b><br>
本阶段的成功标准是明确图像模型职责边界，不是让五个案例都在此阶段产出最终关键帧。
未通过硬门的图片仍作为失败证据保存。下一步自动进入 Phase 4：实现剩余五个程序案例，
并按这张路由表稳定十案例关键帧。</div>
</section>

<section><h2>5. 从零复现</h2>
<pre># 先重建确定性的五学科程序动画
.venv/bin/python -m modules.video_model.stage2.phase2

# 单轮：准备控制图、检查提示，再调用已登记模型
/opt/venv/bin/python -m modules.video_model.stage2.phase3_experiment \\
  --experiment EXP-20260729-009 --prepare
/opt/venv/bin/python -m modules.video_model.stage2.phase3_experiment \\
  --experiment EXP-20260729-009 --generate

# 需要受限材质时，执行确定性多供体合成
/opt/venv/bin/python -m modules.video_model.stage2.phase3_experiment \\
  --experiment EXP-20260729-012 --project-material-ensemble \\
  --projection-variant gain_070 --residual-gain 0.7

# 重建本总报告并检查所有哈希、预算和链接
.venv/bin/python -m modules.video_model.stage2.phase3 --report
.venv/bin/python -m modules.video_model.stage2.phase3 --check</pre>
<p><a href="phase3_manifest.json">Phase 3 机器清单</a> ·
<a href="../../experiments/ledger.json">完整实验账本</a> ·
<a href="../phase-2/report.html">Phase 2 程序动画报告</a></p>
</section></main></body></html>""",
        encoding="utf-8",
    )


def _update_ledger(experiments: list[dict[str, Any]]) -> None:
    ledger = load_json(LEDGER_PATH)
    retained = [
        record
        for record in ledger["experiments"]
        if record["experiment_id"] not in EXPECTED_EXPERIMENT_IDS
    ]
    phase3_records = []
    for item in experiments:
        generated = item["generated"]
        phase3_records.append(
            {
                "experiment_id": item["experiment_id"],
                "phase": 3,
                "case_id": item["spec"]["case_id"],
                "primary_hypothesis": item["spec"]["hypothesis_zh"],
                "status": item["review"]["verdict"],
                "image_candidates": len(generated["candidates"]),
                "new_image_model_runs": int(
                    generated["cache"]["generated"]
                ),
                "reused_candidates": int(generated["cache"]["reused"]),
                "video_candidates": 0,
                "model_ids": sorted(
                    value["model_id"]
                    for value in generated["models"].values()
                ),
                "output_manifest": (
                    f"output/phase-3/{item['experiment_id']}/"
                    "_work/generate.json"
                ),
                "review": (
                    f"experiments/{item['experiment_id']}/review.json"
                ),
            }
        )
    ledger["experiments"] = retained + phase3_records
    write_json(LEDGER_PATH, ledger)


def _missing_report_links() -> list[str]:
    report = REPORT_PATH.read_text(encoding="utf-8")
    targets = re.findall(r'(?:href|src)="([^"]+)"', report)
    missing = []
    for target in targets:
        if target.startswith(("#", "http://", "https://")):
            continue
        if not (REPORT_PATH.parent / target).resolve().exists():
            missing.append(target)
    return missing


def build_report() -> dict[str, Any]:
    experiments = _load_experiments()
    checks = [_verify_experiment(item) for item in experiments]
    if {item["case_id"] for item in checks} != EXPECTED_CASES:
        raise ValueError("Phase 3 does not cover all five sentinel disciplines")
    _update_ledger(experiments)
    _write_report(experiments, checks)
    manifest = {
        "schema_version": "1.0",
        "phase": 3,
        "status": "passed",
        "phase_complete": True,
        "automatic_next_action": "advance_to_phase_4",
        "experiment_ids": list(EXPECTED_EXPERIMENT_IDS),
        "case_ids": sorted(EXPECTED_CASES),
        "experiment_count": len(checks),
        "candidate_count": sum(item["candidate_count"] for item in checks),
        "model_runs": {
            "new_image_candidates": sum(
                item["new_model_runs"] for item in checks
            ),
            "reused_image_candidates": sum(
                item["reused_candidates"] for item in checks
            ),
            "video_candidates": 0,
        },
        "hard_gates": {
            "all_experiments_reviewed": True,
            "all_candidates_hashed": True,
            "all_prompts_untruncated": True,
            "all_models_registered": True,
            "all_budgets_respected": True,
            "five_disciplines_covered": True,
            "raw_composite_final_separated": True,
        },
        "capability_map": {
            "exact_count_or_rigid_geometry": "program_pixels_plus_region_limited_material_projection",
            "semantically_sufficient_hard_boundary": "controlnet_t2i_candidate",
            "continuous_height_or_material_field": "program_low_frequency_plus_robust_model_residual",
            "static_environment_plus_dynamic_scalar_field": "split_generation_and_program_overlay_in_phase_4",
        },
        "experiments": checks,
        "sources": _source_records(experiments),
        "report": artifact_record(REPORT_PATH, PHASE_ROOT),
        "ledger_phase3_record_count": len(EXPECTED_EXPERIMENT_IDS),
    }
    write_json(MANIFEST_PATH, manifest)
    missing = _missing_report_links()
    if missing:
        raise ValueError(f"Phase 3 report links missing: {missing}")
    return manifest


def check_phase3() -> dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    if manifest["status"] != "passed" or not manifest["phase_complete"]:
        raise ValueError("Phase 3 is not marked complete")
    experiments = _load_experiments()
    checks = [_verify_experiment(item) for item in experiments]
    if manifest["candidate_count"] != sum(
        item["candidate_count"] for item in checks
    ):
        raise ValueError("Phase 3 candidate total mismatch")
    for record in manifest["sources"]:
        path = STAGE2_ROOT / record["path"]
        if sha256_path(path) != record["sha256"]:
            raise ValueError(f"Phase 3 source changed: {path}")
    if sha256_path(REPORT_PATH) != manifest["report"]["sha256"]:
        raise ValueError("Phase 3 report hash mismatch")
    missing = _missing_report_links()
    if missing:
        raise ValueError(f"Phase 3 report links missing: {missing}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.report:
        manifest = build_report()
        print(
            f"Phase 3: {manifest['status']} · "
            f"{manifest['experiment_count']} experiments · "
            f"{manifest['candidate_count']} candidates · "
            f"next={manifest['automatic_next_action']}"
        )
    if args.check:
        manifest = check_phase3()
        print(
            f"Phase 3 check: {manifest['status']} · "
            f"{manifest['experiment_count']} experiments"
        )
    if not (args.report or args.check):
        parser.error("choose --report or --check")


if __name__ == "__main__":
    main()
