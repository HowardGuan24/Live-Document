"""Build the five remaining deterministic programs for Phase 4."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .cases.remaining_programs import PROGRAMS
from .framework.contracts import (
    artifact_record,
    load_json,
    sha256_path,
    write_json,
)
from .framework.program_runner import (
    HEIGHT,
    WIDTH,
    build_program,
    validate_program_tree,
)
from .phase4_routes import OUTPUT_PATH as ROUTE_PLAN_PATH
from .phase4_routes import build_route_plan


STAGE2_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = STAGE2_ROOT / "output" / "phase-4"
PROGRAM_ROOT = OUTPUT_ROOT / "programs"
REPORT_PATH = OUTPUT_ROOT / "program-report.html"
MANIFEST_PATH = OUTPUT_ROOT / "phase4_program_manifest.json"
LEDGER_PATH = STAGE2_ROOT / "experiments" / "ledger.json"
CASE_IDS = ("MATH-01", "PHYS-02", "CHEM-02", "BIO-02", "GEO-01")


def _overview(programs: list[dict[str, Any]]) -> Path:
    thumb = (320, 180)
    gutter, label_height = 10, 28
    width = thumb[0] * 4 + gutter * 5
    height = len(programs) * (thumb[1] + label_height) + gutter * (
        len(programs) + 1
    )
    sheet = Image.new("RGB", (width, height), (13, 31, 37))
    draw = ImageDraw.Draw(sheet)
    for row, manifest in enumerate(programs):
        root = PROGRAM_ROOT / manifest["case_id"]
        for column, keyframe in enumerate(manifest["keyframes"]):
            x = gutter + column * (thumb[0] + gutter)
            y = gutter + row * (thumb[1] + label_height + gutter)
            image = Image.open(
                root / keyframe["program_frame"]["path"]
            ).convert("RGB")
            sheet.paste(image.resize(thumb), (x, y))
            draw.rectangle(
                (x, y + thumb[1], x + thumb[0], y + thumb[1] + label_height),
                fill=(5, 23, 28),
            )
            draw.text(
                (x + 7, y + thumb[1] + 7),
                f"{manifest['case_id']} · K{column}",
                fill=(232, 244, 239),
            )
    path = OUTPUT_ROOT / "remaining-keyframes.jpg"
    sheet.save(path, quality=92, subsampling=0)
    return path


def _program_sections(programs: list[dict[str, Any]]) -> str:
    sections = []
    for manifest in programs:
        case_id = manifest["case_id"]
        validation = load_json(
            PROGRAM_ROOT / case_id / manifest["validation"]["path"]
        )
        frames = "".join(
            f"""<figure><a href="programs/{case_id}/{keyframe['program_frame']['path']}">
            <img loading="lazy" src="programs/{case_id}/{keyframe['program_frame']['path']}"
            alt="{case_id} {keyframe['keyframe_id']}"></a>
            <figcaption>{html.escape(keyframe['keyframe_id'])} ·
            t={keyframe['progress']:.3f}</figcaption></figure>"""
            for keyframe in manifest["keyframes"]
        )
        checks = "".join(
            f"<li><code>{html.escape(check['name'])}</code>："
            f"{'通过' if check['passed'] else '失败'}</li>"
            for check in validation["mechanism_checks"]
        )
        layers = manifest["keyframes"][2]["layers"]
        layer_rows = "".join(
            f"<tr><td><code>{html.escape(layer['layer_id'])}</code></td>"
            f"<td>{html.escape(layer['layer_type'])}</td>"
            f"<td>{html.escape(layer['meaning_zh'])}</td>"
            f"<td>{'禁止' if layer['model_input_policy'] == 'never' else '由路由决定'}</td></tr>"
            for layer in layers
        )
        sections.append(
            f"""<section id="{case_id}"><p class="eyebrow">{case_id}</p>
            <h2>{html.escape(manifest['title_zh'])}</h2>
            <p><b>机制目标：</b>{html.escape(manifest['primary_mechanism_zh'])}</p>
            <video controls muted loop preload="metadata"
            poster="programs/{case_id}/{manifest['keyframes'][0]['program_frame']['path']}">
            <source src="programs/{case_id}/{manifest['animation']['path']}"
            type="video/mp4"></video>
            <div class="frames">{frames}</div>
            <h3>程序状态断言</h3><ul>{checks}</ul>
            <h3>K2 的语义层</h3><table><thead><tr><th>ID</th><th>类型</th>
            <th>含义</th><th>模型策略</th></tr></thead><tbody>{layer_rows}</tbody></table>
            <p><a href="programs/{case_id}/program_manifest.json">完整程序清单</a> ·
            <a href="programs/{case_id}/keyframe-contact-sheet.jpg">四帧原尺寸总览</a></p>
            </section>"""
        )
    return "".join(sections)


def _write_report(
    programs: list[dict[str, Any]], route_plan: dict[str, Any]
) -> None:
    route_rows = "".join(
        f"<tr><td><code>{html.escape(route['case_id'])}</code></td>"
        f"<td><code>{html.escape(route['route_id'])}</code></td>"
        f"<td>{html.escape(route['model_role_zh'])}</td></tr>"
        for route in route_plan["routes"]
    )
    accepted_review_path = (
        STAGE2_ROOT
        / "phase4_experiments/EXP-P4-20260729-002/review.json"
    )
    biology_review_path = (
        STAGE2_ROOT
        / "phase4_experiments/EXP-P4-20260729-003/review.json"
    )
    chemistry_review_path = (
        STAGE2_ROOT
        / "phase4_experiments/EXP-P4-20260729-004/review.json"
    )
    mathematics_review_path = (
        STAGE2_ROOT
        / "phase4_experiments/EXP-P4-20260729-005/review.json"
    )
    physics_review_path = (
        STAGE2_ROOT
        / "phase4_experiments/EXP-P4-20260729-006/review.json"
    )
    smoke_section = ""
    if accepted_review_path.is_file():
        review = load_json(accepted_review_path)
        selected = review["selected_variant"]
        smoke_section = f"""<section><h2>第一项路线冒烟：GEO-01 水体材质</h2>
        <p>第一次输入审图发现程序宽折线有规则针孔，因此 EXP-P4-001 即使机器门
        3/3 通过也被拒绝；修正程序后新建 EXP-P4-002，没有覆盖旧证据。Phase 5
        又用真实水色连通域发现“状态称已隔离、像素仍连通”的第二层错误：现已把沉积塞
        放到旧河道支路，并从 raster 反算出主河 1 个贯通组件 + 牛轭湖 1 个独立组件。
        既有四张水纹供体直接重新投影，新增图片模型调用为 0。</p>
        <img class="overview" loading="lazy"
        src="experiments/EXP-P4-20260729-002/ensemble-material-comparison-gain_040.jpg"
        alt="牛轭湖程序底图、模型水纹供体和受限 composite 对比">
        <p><b>{html.escape(review['verdict_zh'])}</b></p>
        <table><tbody>
        <tr><th>非允许区最大像素差</th><td>{selected['non_allowed_max_abs_difference_0_255']}</td></tr>
        <tr><th>低频结构 MSE</th><td>{selected['low_frequency_mse_vs_program']}</td></tr>
        <tr><th>留一法最大 MAE</th><td>{selected['leave_one_out_maximum_mae_0_255']}</td></tr>
        <tr><th>选择</th><td><code>{html.escape(selected['variant_id'])}</code></td></tr>
        </tbody></table>
        <p><a href="experiments/EXP-P4-20260729-002/candidates-blind.jpg">
        全部 raw 材质供体</a> ·
        <a href="../../phase4_experiments/EXP-P4-20260729-001/review.json">
        被拒绝的错误输入评审</a> ·
        <a href="../../phase4_experiments/EXP-P4-20260729-002/review.json">
        修正后评审</a></p></section>"""
    if biology_review_path.is_file():
        review = load_json(biology_review_path)
        selected = review["selected_variant"]
        smoke_section += f"""<section><h2>第二项路线冒烟：BIO-02 非水材料</h2>
        <p>{html.escape(review['raw_donor_finding_zh'])}</p>
        <img class="overview" loading="lazy"
        src="experiments/EXP-P4-20260729-003/ensemble-material-comparison-gain_030.jpg"
        alt="保卫细胞程序底图、叶表皮 raw 供体和受限 composite">
        <p><b>{html.escape(review['verdict_zh'])}</b></p>
        <table><tbody>
        <tr><th>非允许区最大像素差</th><td>{selected['non_allowed_max_abs_difference_0_255']}</td></tr>
        <tr><th>低频结构 MSE</th><td>{selected['low_frequency_mse_vs_program']}</td></tr>
        <tr><th>留一法最大 MAE</th><td>{selected['leave_one_out_maximum_mae_0_255']}</td></tr>
        <tr><th>材质提升</th><td>{selected['material_improvement_5']}/5</td></tr>
        </tbody></table>
        <p><a href="experiments/EXP-P4-20260729-003/candidates-blind.jpg">
        全部 raw 供体</a> ·
        <a href="../../phase4_experiments/EXP-P4-20260729-003/review.json">
        评审原文</a></p></section>"""
    if chemistry_review_path.is_file():
        review = load_json(chemistry_review_path)
        selected = review["selected_variant"]
        smoke_section += f"""<section><h2>第三项路线冒烟：CHEM-02 安全不等于有效</h2>
        <p>{html.escape(review['raw_donor_finding_zh'])}</p>
        <img class="overview" loading="lazy"
        src="experiments/EXP-P4-20260729-004/ensemble-material-comparison-gain_060.jpg"
        alt="结晶程序底图、不合格的盐晶面供体和受限 composite">
        <p><b>{html.escape(review['verdict_zh'])}</b></p>
        <table><tbody>
        <tr><th>晶体允许区占全图</th><td>{selected['allowed_pixel_fraction'] * 100:.2f}%</td></tr>
        <tr><th>非允许区最大像素差</th><td>{selected['non_allowed_max_abs_difference_0_255']}</td></tr>
        <tr><th>允许区平均细节变化</th><td>{selected['allowed_mean_abs_detail_change_0_255']}/255</td></tr>
        <tr><th>材质提升</th><td>{selected['material_improvement_5']}/5（未达标）</td></tr>
        </tbody></table>
        <p>因此没有继续放大扩散残差。当前程序把每颗小晶体拆成四个确定性晶面：
        数量、轮廓与质量守恒仍由状态决定，同时在最终画面尺寸上能看清体积关系。
        若以后验证局部模型路线，必须另做“放大对象空间渲染”实验。</p>
        <p><a href="experiments/EXP-P4-20260729-004/candidates-blind.jpg">
        4 张失败 raw 供体</a> ·
        <a href="../../phase4_experiments/EXP-P4-20260729-004/review.json">
        评审原文</a> ·
        <a href="programs/CHEM-02/keyframes/03_end/clean.png">
        改用程序晶面的当前结果</a></p></section>"""
    if mathematics_review_path.is_file():
        review = load_json(mathematics_review_path)
        selected = review["selected_variant"]
        smoke_section += f"""<section><h2>第四项路线冒烟：MATH-01 不强制照片化</h2>
        <p>{html.escape(review['raw_donor_finding_zh'])}</p>
        <img class="overview" loading="lazy"
        src="experiments/EXP-P4-20260729-005/ensemble-material-comparison-gain_050.jpg"
        alt="单位圆程序底图、纸张供体和只修改空白底纸的 composite">
        <p><b>{html.escape(review['verdict_zh'])}</b></p>
        <table><tbody>
        <tr><th>底纸允许区占全图</th><td>{selected['allowed_pixel_fraction'] * 100:.2f}%</td></tr>
        <tr><th>数学图形区最大像素差</th><td>{selected['non_allowed_max_abs_difference_0_255']}</td></tr>
        <tr><th>允许区平均细节变化</th><td>{selected['allowed_mean_abs_detail_change_0_255']}/255</td></tr>
        <tr><th>定位</th><td>可选纸张风格；程序原图和模型关闭仍合格</td></tr>
        </tbody></table>
        <p><a href="experiments/EXP-P4-20260729-005/candidates-blind.jpg">
        4 张 raw 纸张供体</a> ·
        <a href="../../phase4_experiments/EXP-P4-20260729-005/review.json">
        评审原文</a></p></section>"""
    if physics_review_path.is_file():
        review = load_json(physics_review_path)
        selected = review["selected_variant"]
        smoke_section += f"""<section><h2>第五项路线冒烟：PHYS-02 精确器材表面</h2>
        <p>{html.escape(review['raw_donor_finding_zh'])}</p>
        <img class="overview" loading="lazy"
        src="experiments/EXP-P4-20260729-006/ensemble-material-comparison-gain_060.jpg"
        alt="电磁感应程序器材、哑光材料供体和只修改磁铁的 composite">
        <p><b>{html.escape(review['verdict_zh'])}</b></p>
        <table><tbody>
        <tr><th>磁铁允许区占全图</th><td>{selected['allowed_pixel_fraction'] * 100:.2f}%</td></tr>
        <tr><th>非磁铁区最大像素差</th><td>{selected['non_allowed_max_abs_difference_0_255']}</td></tr>
        <tr><th>允许区平均细节变化</th><td>{selected['allowed_mean_abs_detail_change_0_255']}/255</td></tr>
        <tr><th>留一法最大 MAE</th><td>{selected['leave_one_out_maximum_mae_0_255']}</td></tr>
        </tbody></table>
        <p><a href="experiments/EXP-P4-20260729-006/candidates-blind.jpg">
        4 张 raw 器材材质供体</a> ·
        <a href="../../phase4_experiments/EXP-P4-20260729-006/review.json">
        评审原文</a></p></section>"""
    REPORT_PATH.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 2 · Phase 4 五案例关键帧稳定报告</title>
<style>
:root{{--ink:#17363c;--deep:#0d2930;--paper:#f3efe4;--card:#fffdf7;--line:#c4d3ce;--teal:#167e75}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.65 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
header,main{{max-width:1180px;margin:auto;padding:36px 24px}}header{{padding-top:58px}}
h1{{font-size:clamp(2.2rem,6vw,4.7rem);line-height:1.04;margin:.15em 0}}
h2{{font-size:2rem;margin:.15em 0}}section{{border-top:3px solid var(--ink);padding:42px 0}}
.eyebrow{{color:var(--teal);font-weight:800;letter-spacing:.13em}}.overview,video{{width:100%;
border-radius:12px;background:var(--deep)}}.frames{{display:grid;grid-template-columns:repeat(2,1fr);
gap:10px;margin:18px 0}}figure{{margin:0;background:var(--card);padding:8px;border:1px solid var(--line);
border-radius:10px}}figure img{{display:block;width:100%;border-radius:7px}}figcaption{{padding:5px}}
table{{width:100%;border-collapse:collapse;background:var(--card);font-size:.9rem}}
th,td{{border:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}}
pre{{background:var(--deep);color:#eaf4f0;padding:14px;border-radius:9px;overflow:auto}}
a{{color:var(--teal)}}@media(max-width:650px){{.frames{{grid-template-columns:1fr}}
header,main{{padding-left:14px;padding-right:14px}}}}
</style></head><body><header><p class="eyebrow">LOOP ENGINEER · PHASE 4 PASSED</p>
<h1>五个新增概念，<br>关键帧策略已经稳定</h1>
<p>单位圆、电磁感应、蒸发结晶、气孔开闭和牛轭湖都由同一通用运行器生成 49 帧、
4 个机制关键帧、机器状态和语义层；这 245 张程序帧的模型调用为 0。随后六轮实验
共运行 24 张图片候选，其中包含一次修正后重跑。模型不是每例都必须用：结晶材质路线
被明确拒绝并改用程序晶面，Stage 2 视频模型调用仍为 0。</p></header><main>
<section><h2>五案例总览</h2><img class="overview" src="remaining-keyframes.jpg"
alt="剩余五个案例各四个关键帧"></section>
<section><h2>先看结论：每个案例现在走哪条路</h2>
<table><thead><tr><th>案例</th><th>当前关键帧策略</th><th>自评</th></tr></thead><tbody>
<tr><td><code>MATH-01</code></td><td>程序几何 + 可选底纸纹理 gain=0.5</td><td>通过；不照片化</td></tr>
<tr><td><code>PHYS-02</code></td><td>程序器材 + 磁铁内部哑光纹理 gain=0.6</td><td>通过；案例候选</td></tr>
<tr><td><code>CHEM-02</code></td><td>确定性四晶面着色，图像模型关闭</td><td>通过；模型路线拒绝</td></tr>
<tr><td><code>BIO-02</code></td><td>程序细胞 + 保卫细胞纹理 gain=0.3</td><td>通过；案例候选</td></tr>
<tr><td><code>GEO-01</code></td><td>程序水体拓扑 + 水纹 gain=0.4</td><td>通过；跨学科路线</td></tr>
</tbody></table></section>
<section><h2>十案例不再共用一条生图路线</h2>
<p>下面的选择器只读取能力标签和语义层类型，不读取案例 ID。精确计数、器材边界、
连续材质以及“静态环境 + 动态场”分别承担不同的模型职责。</p>
<table><thead><tr><th>案例</th><th>数据类型路线</th><th>图像模型职责</th></tr></thead>
<tbody>{route_rows}</tbody></table>
<p><a href="route-plan.json">查看每例硬门、必需输入和 Phase 3 证据</a></p></section>
{_program_sections(programs)}
{smoke_section}
<section><h2>当前自动判断</h2>
<p><b>Phase 4 通过，自动进入 Phase 5。</b>五个新增案例的机制断言、统一输出合同、
245 帧、固定种子稳定性和逐像素保护门禁均通过；有效、可选和被拒绝的模型路线都已
明确记录。下一阶段只让视频模型连接相邻的正确状态，不能让它补做概念规划。</p>
<pre>.venv/bin/python -m modules.video_model.stage2.phase4
.venv/bin/python -m modules.video_model.stage2.phase4 --check</pre>
<p><a href="phase4_program_manifest.json">机器可读里程碑清单</a> ·
<a href="../phase-3/report.html">Phase 3 路由依据</a></p></section>
</main></body></html>""",
        encoding="utf-8",
    )


def _missing_links() -> list[str]:
    text = REPORT_PATH.read_text(encoding="utf-8")
    targets = re.findall(r'(?:href|src|poster)="([^"]+)"', text)
    return [
        target
        for target in targets
        if not target.startswith(("#", "http://", "https://"))
        and not (REPORT_PATH.parent / target).resolve().exists()
    ]


def _update_ledger(smoke_experiments: list[dict[str, Any]]) -> None:
    ledger = load_json(LEDGER_PATH)
    experiment_ids = {
        item["experiment_id"] for item in smoke_experiments
    }
    retained = [
        item
        for item in ledger["experiments"]
        if item["experiment_id"] not in experiment_ids
    ]
    records = []
    for item in smoke_experiments:
        experiment_id = item["experiment_id"]
        generated = load_json(
            OUTPUT_ROOT
            / "experiments"
            / experiment_id
            / "_work"
            / "generate.json"
        )
        spec = load_json(
            STAGE2_ROOT
            / "phase4_experiments"
            / experiment_id
            / "spec.json"
        )
        records.append(
            {
                "experiment_id": experiment_id,
                "phase": 4,
                "case_id": item["case_id"],
                "primary_hypothesis": spec["hypothesis_zh"],
                "status": item["verdict"],
                "image_candidates": item["candidate_count"],
                "new_image_model_runs": item["new_image_model_runs"],
                "reused_candidates": int(generated["cache"]["reused"]),
                "video_candidates": 0,
                "model_ids": sorted(
                    value["model_id"]
                    for value in generated["models"].values()
                ),
                "output_manifest": (
                    f"output/phase-4/experiments/{experiment_id}/"
                    "_work/generate.json"
                ),
                "review": (
                    f"phase4_experiments/{experiment_id}/review.json"
                ),
            }
        )
    ledger["experiments"] = retained + records
    write_json(LEDGER_PATH, ledger)


def build_phase4_programs(*, check_only: bool = False) -> dict[str, Any]:
    if check_only:
        manifest = load_json(MANIFEST_PATH)
        build_route_plan(check_only=True)
        for case_id in CASE_IDS:
            validate_program_tree(PROGRAM_ROOT / case_id)
        for source in manifest["sources"].values():
            path = STAGE2_ROOT / source["path"]
            if sha256_path(path) != source["sha256"]:
                raise ValueError(f"Phase 4 program source changed: {path}")
        if sha256_path(REPORT_PATH) != manifest["report"]["sha256"]:
            raise ValueError("Phase 4 program report changed")
        if sha256_path(ROUTE_PLAN_PATH) != manifest["route_plan"]["sha256"]:
            raise ValueError("Phase 4 route plan changed")
        for experiment in manifest.get("route_smoke_experiments", []):
            for key, base in (
                ("metadata", OUTPUT_ROOT),
                ("review", STAGE2_ROOT),
            ):
                record = experiment[key]
                if sha256_path(base / record["path"]) != record["sha256"]:
                    raise ValueError(
                        f"Phase 4 smoke evidence changed: {record['path']}"
                    )
        missing = _missing_links()
        if missing:
            raise ValueError(f"Phase 4 report links missing: {missing}")
        return manifest

    PROGRAM_ROOT.mkdir(parents=True, exist_ok=True)
    programs = [
        build_program(PROGRAMS[case_id], PROGRAM_ROOT / case_id, phase=4)
        for case_id in CASE_IDS
    ]
    if not all(item["status"] == "passed" for item in programs):
        raise ValueError("a Phase 4 deterministic program failed")
    overview = _overview(programs)
    route_plan = build_route_plan()
    _write_report(programs, route_plan)
    smoke_experiments = []
    for experiment_id in (
        "EXP-P4-20260729-001",
        "EXP-P4-20260729-002",
        "EXP-P4-20260729-003",
        "EXP-P4-20260729-004",
        "EXP-P4-20260729-005",
        "EXP-P4-20260729-006",
    ):
        generated_path = (
            OUTPUT_ROOT
            / "experiments"
            / experiment_id
            / "_work"
            / "generate.json"
        )
        review_path = (
            STAGE2_ROOT
            / "phase4_experiments"
            / experiment_id
            / "review.json"
        )
        if generated_path.is_file() and review_path.is_file():
            generated = load_json(generated_path)
            review = load_json(review_path)
            smoke_experiments.append(
                {
                    "experiment_id": experiment_id,
                    "case_id": generated["case_id"],
                    "verdict": review["verdict"],
                    "new_image_model_runs": generated["cache"][
                        "generated"
                    ],
                    "candidate_count": len(generated["candidates"]),
                    "metadata": artifact_record(
                        generated_path, OUTPUT_ROOT
                    ),
                    "review": artifact_record(
                        review_path, STAGE2_ROOT
                    ),
                }
            )
    _update_ledger(smoke_experiments)
    sources = {
        name: {
            "path": path.relative_to(STAGE2_ROOT).as_posix(),
            "sha256": sha256_path(path),
        }
        for name, path in {
            "runner": Path(__file__),
            "generic_program_runner": STAGE2_ROOT
            / "framework/program_runner.py",
            "remaining_case_plugins": STAGE2_ROOT
            / "cases/remaining_programs.py",
            "case_registry": STAGE2_ROOT / "case_registry.json",
        }.items()
    }
    manifest = {
        "schema_version": "1.0",
        "phase": 4,
        "status": "passed",
        "phase_complete": True,
        "automatic_next_action": "run_phase5_motion_class_video_smokes",
        "program_count": 5,
        "frame_count": 245,
        "keyframe_count": 20,
        "program_model_runs": {"image": 0, "video": 0},
        "model_runs": {
            "image": sum(
                item["new_image_model_runs"]
                for item in smoke_experiments
            ),
            "video": 0,
        },
        "case_ids": list(CASE_IDS),
        "programs": [
            {
                "case_id": item["case_id"],
                "status": item["status"],
                "manifest": artifact_record(
                    PROGRAM_ROOT / item["case_id"] / "program_manifest.json",
                    OUTPUT_ROOT,
                ),
            }
            for item in programs
        ],
        "sources": sources,
        "overview": artifact_record(overview, OUTPUT_ROOT),
        "route_plan": artifact_record(ROUTE_PLAN_PATH, OUTPUT_ROOT),
        "route_smoke_experiments": smoke_experiments,
        "report": artifact_record(REPORT_PATH, OUTPUT_ROOT),
    }
    write_json(MANIFEST_PATH, manifest)
    missing = _missing_links()
    if missing:
        raise ValueError(f"Phase 4 report links missing: {missing}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_phase4_programs(check_only=args.check)
    print(
        f"Phase 4 programs: {manifest['status']} · "
        f"{manifest['program_count']} programs · "
        f"{manifest['frame_count']} frames · "
        f"next={manifest['automatic_next_action']}"
    )


if __name__ == "__main__":
    main()
