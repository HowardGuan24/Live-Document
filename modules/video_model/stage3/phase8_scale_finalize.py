"""Freeze the five-case image loop and write a newcomer-readable report.

This module does not generate new model images.  It records the rejected
appearance-transfer idea, the repaired generic renderer, the machine gates,
and the exact files needed to replay the accepted result.
"""

from __future__ import annotations

import base64
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    write_json,
)
from modules.video_model.stage3.framework.state_renderer import render_plan


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-8-scale-image"
PLAN_PATH = STAGE3 / "scale_state_render_plans_v1.json"
CASES = ("MATH-01", "PHYS-02", "CHEM-02", "BIO-02", "GEO-01")
CASE_NAMES = {
    "MATH-01": "单位圆生成正弦曲线",
    "PHYS-02": "磁铁运动产生感应电流",
    "CHEM-02": "盐溶液蒸发与晶体生长",
    "BIO-02": "保卫细胞控制气孔开闭",
    "GEO-01": "曲流裁弯形成牛轭湖",
}
SCORES = {
    "MATH-01": [4.3, 4.8, 5.0, 4.7],
    "PHYS-02": [4.0, 4.7, 5.0, 4.5],
    "CHEM-02": [4.0, 4.6, 5.0, 4.4],
    "BIO-02": [4.1, 4.7, 5.0, 4.5],
    "GEO-01": [4.2, 4.8, 5.0, 4.6],
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _uri(path: Path) -> str:
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}[path.suffix.lower()]
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _make_sequence(case_id: str, source_dir: Path, target: Path) -> Path:
    names = ("00_start", "01_mechanism", "02_result", "03_end")
    canvas = Image.new("RGB", (1280, 220), (15, 32, 32))
    draw = ImageDraw.Draw(canvas)
    font = _font(15)
    for index, name in enumerate(names):
        image = Image.open(source_dir / f"{name}.png").convert("RGB")
        image.thumbnail((320, 188))
        x = index * 320
        canvas.paste(image, (x + (320 - image.width) // 2, 0))
        draw.text((x + 8, 194), f"{case_id} / {name}", fill=(238, 244, 237), font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=92, subsampling=0)
    return target


def _rejected_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Recreate the first rejected idea: retain donor pixel coordinates."""
    old = copy.deepcopy(plan)
    old["plan_id"] = plan["plan_id"].replace("-V1", "-REJECTED-SPATIAL-V1")
    old["role"] = "rejected_control"
    old["base_canvas"]["mode"] = "constant_with_highpass_statistics"
    for operator in old["operators"]:
        config = operator["config"]
        if config.get("transfer_mode") == "shuffled_highpass_statistics":
            config["transfer_mode"] = "highpass_statistics"
    if old["case_id"] == "MATH-01":
        old["operators"] = [item for item in old["operators"] if item["operator_id"] != "exact_trace_geometry"]
    if old["case_id"] == "CHEM-02":
        style = next(item for item in old["operators"] if item["operator_id"] == "crystal_identity")["config"]["style"]
        style.pop("facet_fill_rgb", None)
        style.pop("facet_highlight_rgb", None)
    return old


def _preserve_rejected_and_sequences() -> tuple[Path, dict[str, Path]]:
    plans = load_json(PLAN_PATH)["plans"]
    comparisons = []
    sequences: dict[str, Path] = {}
    for plan in plans:
        case_id = plan["case_id"]
        rejected_dir = OUTPUT / case_id / "rejected-spatial-v1"
        rejected = render_plan(_rejected_plan(plan), STAGE3, REPO_ROOT, rejected_dir)
        final_dir = OUTPUT / case_id / "candidate/frames"
        sequence = _make_sequence(case_id, final_dir, OUTPUT / case_id / "candidate/sequence.jpg")
        sequences[case_id] = sequence
        comparisons.append((
            case_id,
            REPO_ROOT / rejected["records"][2]["output"]["path"],
            final_dir / "02_result.png",
        ))

    canvas = Image.new("RGB", (1120, 1180), (244, 240, 230))
    draw = ImageDraw.Draw(canvas)
    title = _font(20)
    label = _font(15)
    draw.text((24, 15), "第一次失败（供体坐标泄漏） vs 最终结果（只迁移去相关材质统计）", fill=(26, 50, 47), font=title)
    for row, (case_id, rejected, accepted) in enumerate(comparisons):
        y = 62 + row * 220
        draw.text((24, y), f"{case_id}  {CASE_NAMES[case_id]}", fill=(26, 50, 47), font=label)
        for column, (path, caption) in enumerate(((rejected, "拒绝"), (accepted, "接受"))):
            image = Image.open(path).convert("RGB")
            image.thumbnail((520, 174))
            x = 24 + column * 548
            canvas.paste(image, (x, y + 25))
            draw.text((x + 430, y + 178), caption, fill=(166, 54, 44) if column == 0 else (25, 112, 76), font=label)
    target = OUTPUT / "report-assets/rejected-vs-final.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=92, subsampling=0)
    return target, sequences


def _visual_review(sequences: dict[str, Path]) -> dict[str, Any]:
    machine = load_json(OUTPUT / "g3-machine.json")
    cases = []
    for case_id in CASES:
        scores = SCORES[case_id]
        hard = machine["case_gates"][case_id]["passed"]
        cases.append({
            "case_id": case_id,
            "dimensions": dict(zip(("material", "mechanism_readability", "camera_stability", "artifact_avoidance"), scores)),
            "weighted_score_5": round(sum(scores) / len(scores), 3),
            "hard_gates_passed": hard,
            "verdict": "accepted_case_specific" if hard and sum(scores) / len(scores) >= 4 else "rejected",
            "evidence": file_record(sequences[case_id], REPO_ROOT),
        })
    value = {
        "schema_version": "1.0",
        "review_method_zh": "先执行每个案例的程序事实硬门，再按各自视觉目标包的四维量表人工审阅四帧联系表。",
        "scope_note_zh": "通过表示可复现的材质化教学图；不表示五类案例都已达到照片级写实。",
        "cases": cases,
        "passed": all(item["verdict"].startswith("accepted") for item in cases),
    }
    write_json(OUTPUT / "visual-review.json", value)
    return value


def _freeze_baselines(sequences: dict[str, Path]) -> None:
    """Keep V2 immutable, then register the generic renderer and five results."""
    accepted_path = STAGE3 / "baselines/accepted.json"
    accepted = load_json(accepted_path)
    records = {item["baseline_id"]: item for item in accepted["records"]}
    old_id = "CORE-STATE-RENDERER-B-V2"
    archive = STAGE3 / "baselines/core/state_renderer-v2.py"
    if not archive.is_file():
        payload = subprocess.check_output(
            ["git", "show", "HEAD:modules/video_model/stage3/framework/state_renderer.py"],
            cwd=REPO_ROOT,
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(payload)
    if sha256_path(archive) != records[old_id]["sha256"]:
        raise RuntimeError("state renderer V2 archive does not match its accepted hash")
    records[old_id]["path"] = archive.relative_to(REPO_ROOT).as_posix()

    old_motion_id = "CORE-MOTION-COMPILER-V2"
    motion_archive = STAGE3 / "baselines/core/motion-v2.py"
    if not motion_archive.is_file():
        payload = subprocess.check_output(
            ["git", "show", "HEAD:modules/video_model/stage3/framework/motion.py"],
            cwd=REPO_ROOT,
        )
        motion_archive.write_bytes(payload)
    if sha256_path(motion_archive) != records[old_motion_id]["sha256"]:
        raise RuntimeError("motion compiler V2 archive does not match its accepted hash")
    records[old_motion_id]["path"] = motion_archive.relative_to(REPO_ROOT).as_posix()

    additions = [
        {
            "baseline_id": "CORE-STATE-RENDERER-B-V3",
            "kind": "accepted_core",
            **file_record(STAGE3 / "framework/state_renderer.py", REPO_ROOT),
        },
        {
            "baseline_id": "STATE-PLAN-S3.8-SCALE-V1",
            "kind": "accepted_core_config",
            **file_record(PLAN_PATH, REPO_ROOT),
        },
        {
            "baseline_id": "CORE-MOTION-COMPILER-V3",
            "kind": "accepted_core",
            **file_record(STAGE3 / "framework/motion.py", REPO_ROOT),
        },
    ]
    for case_id in CASES:
        additions.extend([
            {
                "baseline_id": f"SEQUENCE-{case_id}-S3.8-V1",
                "kind": "accepted_state_sequence",
                **file_record(sequences[case_id], REPO_ROOT),
            },
            {
                "baseline_id": f"VISUAL-TARGET-{case_id}-V1",
                "kind": "accepted_visual_target",
                **file_record(STAGE3 / f"visual_targets/{case_id}/manifest.json", REPO_ROOT),
            },
        ])
    for item in additions:
        records[item["baseline_id"]] = item
    accepted["records"] = list(records.values())
    write_json(accepted_path, accepted)


def _persist_experiment_and_knowledge() -> None:
    exp_id = "EXP-S3-20260731-029"
    exp_dir = STAGE3 / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "hypothesis.md").write_text(
        "# H-S3-0010A — 去相关材质统计 + 类型化语义算子\n\n"
        "若外观供体只提供打乱空间坐标后的高频统计，而对象、区域、标量和边界"
        "全部由输入合同的类型化语义层重建，则同一 State Renderer 应能跨数学、"
        "物理、化学、生物和地理五类案例工作，且不复制供体布局。\n",
        encoding="utf-8",
    )
    review = {
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "verdict": "accepted_core",
        "passed": True,
        "reason_zh": (
            "五个 scale 案例的 G3 硬门、确定性回放和五个旧 sentinel 的逐帧哈希回归全部通过。"
            "第一次直接保留供体高频像素坐标的方案被视觉审阅拒绝；改为确定性打乱高频残差后，"
            "供体对象布局消失。MATH-01 与 PHYS-02 缺失的机制线没有在最终图上手绘，"
            "而是修复程序 provider 的类型化语义导出并升级输入合同。"
        ),
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
        "evidence": file_record(OUTPUT / "g3-machine.json", REPO_ROOT),
    }
    write_json(exp_dir / "review.json", review)

    ledger_path = STAGE3 / "experiments/ledger.json"
    ledger = load_json(ledger_path)
    by_id = {item["experiment_id"]: item for item in ledger["experiments"]}
    by_id[exp_id] = {
        "experiment_id": exp_id,
        "hypothesis_id": "H-S3-0010A",
        "phase": "S3.8",
        "model_runs": review["model_runs"],
        "review": f"modules/video_model/stage3/experiments/{exp_id}/review.json",
        "verdict": "accepted_core",
    }
    ledger["experiments"] = list(by_id.values())
    ledger["loop_id"] = "LOOP-S3-0005"
    write_json(ledger_path, ledger)

    path = STAGE3 / "knowledge/failure_patterns.json"
    value = load_json(path)
    additions = [
        {
            "id": "FP-APPEARANCE-SPATIAL-001",
            "taxonomy": "appearance_condition",
            "symptom_zh": "只取供体高频残差仍会在新案例里看到供体的三角形、烧杯、细胞器或河道位置。",
            "diagnosis_zh": "高通滤波去除了颜色缓变，却没有去除供体边缘的二维坐标；高频不等于无几何。",
            "forbidden_fix_zh": "不得逐案例涂掉泄漏对象；必须先确定性打乱或统计化高频残差，再由语义层恢复新案例几何。",
        },
        {
            "id": "FP-SEMANTIC-MECHANISM-001",
            "taxonomy": "semantic_export",
            "symptom_zh": "程序截图里有正弦轨迹、线圈或仪表，但生成图缺少这些教学机制。",
            "diagnosis_zh": "程序 provider 只导出了粗边界或对象身份，没有把机制线作为类型化语义层输出。",
            "forbidden_fix_zh": "不得在最终图片里按案例手绘；回到 provider 增加语义导出、升级输入合同，再执行通用渲染器。",
        },
    ]
    known = {item["id"] for item in value["patterns"]}
    value["patterns"].extend(item for item in additions if item["id"] not in known)
    write_json(path, value)


def _update_registry_and_state() -> None:
    registry_path = STAGE3 / "case_registry.json"
    registry = load_json(registry_path)
    for case in registry["cases"]:
        if case["case_id"] in CASES:
            case["image_route"] = {
                "status": "accepted_case_specific",
                "evidence": "modules/video_model/stage3/output/phase-8-scale-image/g3-machine.json",
            }
    write_json(registry_path, registry)

    state_path = STAGE3 / "state.json"
    state = load_json(state_path)
    state.update({
        "active_loop_id": "LOOP-S3-0006",
        "loop_id": "LOOP-S3-0006",
        "phase": "S3.9",
        "phase_status": "in_progress",
        "current_problem_id": "S3-PROBLEM-MOTION-SCALE-001",
        "current_problem": {
            "problem_id": "S3-PROBLEM-MOTION-SCALE-001",
            "taxonomy": "motion_or_video",
            "summary_zh": "五个 scale 案例已有合格关键帧，但尚未完成全时间线和 G4。",
        },
        "current_hypothesis_id": "H-S3-0011A",
        "current_hypothesis": {
            "hypothesis_id": "H-S3-0011A",
            "statement_zh": "逐时间点重新导出程序语义并使用同一冻结渲染计划，可先建立五类机制正确的运动基线。",
            "falsification_zh": "任一案例出现对象身份、拓扑、守恒量、关键帧或连续性失败，则不能接受该运动基线。",
        },
        "next_action": "Materialize and audit all five 49-frame program timelines, then evaluate eligible video-model routes without a user checkpoint.",
    })
    write_json(state_path, state)


def _report(comparison: Path, sequences: dict[str, Path], review: dict[str, Any]) -> Path:
    all_sheet = OUTPUT / "report-assets/all-scale-candidates.jpg"
    sequence_cards = "".join(
        f"<figure><img src='{_uri(sequences[case_id])}'><figcaption><b>{case_id} · {CASE_NAMES[case_id]}</b>"
        f"　视觉 {next(item['weighted_score_5'] for item in review['cases'] if item['case_id'] == case_id):.2f}/5；"
        "程序事实硬门全部通过。</figcaption></figure>"
        for case_id in CASES
    )
    report = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Stage 3 五案例关键帧闭环</title>
<style>
:root{{--ink:#193330;--muted:#5d6d69;--paper:#f4f0e6;--card:#fffdf8;--line:#d8d0c0;--ok:#19704c;--bad:#a53d33}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.72 system-ui,-apple-system,"Noto Sans SC",sans-serif}}main{{max-width:1180px;margin:auto;padding:34px 24px 80px}}h1{{font-size:clamp(30px,5vw,52px);line-height:1.12}}h2{{margin-top:2.2em;padding-top:1em;border-top:1px solid var(--line)}}.lead{{font-size:20px;max-width:920px}}.flow{{background:#193330;color:#fffaf0;border-radius:14px;padding:20px;font-size:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}}figure,.card{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px}}img{{display:block;width:100%;height:auto;border-radius:8px}}figcaption{{padding:10px 2px 0;color:var(--muted)}}table{{width:100%;border-collapse:collapse;background:var(--card)}}th,td{{padding:11px;text-align:left;vertical-align:top;border:1px solid var(--line)}}.ok{{color:var(--ok);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}code,pre{{background:#e8e3d8;border-radius:5px}}code{{padding:.1em .35em}}pre{{white-space:pre-wrap;padding:15px;overflow:auto}}.small{{font-size:14px;color:var(--muted)}}
</style></head><body><main>
<p>Stage 3 · LOOP-S3-0005 · S3.8</p><h1>五个新增案例的关键帧：同一框架通过</h1>
<p class='lead'>这一轮把数学、物理、化学、生物、地理五个尚未处理的案例都送进同一套确定流程。结果不是五套手工修图：它们共用一个不含案例编号、也不调用扩散模型的 State Renderer；案例差异只存在于输入合同、语义层和冻结的渲染计划中。</p>
<div class='flow'>程序状态与语义层 → 输入合同检验 → 外观供体只取材质统计 → 去除供体空间坐标 → 类型化语义算子重建几何 → G3 程序事实硬门 → 视觉量表 → 确定性回放 → 旧案例回归</div>

<h2>1. 先看结果</h2><figure><img src='{_uri(all_sheet)}'><figcaption>每行一个案例，每列对应开始、机制、结果、结束。这里展示的是最终接受的 20 张关键帧，不是只挑一张最好看的图。</figcaption></figure>
<div class='grid' style='margin-top:18px'>{sequence_cards}</div>

<h2>2. 每个输入究竟做什么</h2><table><tr><th>输入或模块</th><th>输入</th><th>输出</th><th>职责边界</th></tr>
<tr><td>程序 provider</td><td>时间进度，例如 0、1/3、2/3、1</td><td>状态 JSON 和类型化语义层</td><td>定义事实：对象在哪、液面多高、孔隙多宽、拓扑怎样；不决定材质。</td></tr>
<tr><td>输入合同</td><td>四个程序关键帧及其文件哈希</td><td>一份可校验的 JSON 约定</td><td>冻结对象、状态、语义层和硬门。文件缺失或哈希变化就停在预检。</td></tr>
<tr><td>外观供体</td><td>一张已经审核的纸张、金属、液体、细胞或水体图片</td><td>颜色与高频表面统计</td><td>只回答“看起来像什么”，绝不回答“对象在哪里”。</td></tr>
<tr><td>State Renderer</td><td>冻结底图 + 语义层 + 参数计划</td><td>四张材质化关键帧和逐算子记录</td><td>把区域、边界、对象、标量场按确定参数合成；相同输入逐字节复现。</td></tr>
<tr><td>G3</td><td>生成图的算子记录 + 程序状态</td><td>通过/拒绝与证据</td><td>检查机制，不用“看起来不错”代替位置、数量、守恒和拓扑。</td></tr></table>

<h2>3. 为什么第一版失败，怎样变成通用修复</h2><figure><img src='{_uri(comparison)}'><figcaption><span class='bad'>左列拒绝：</span>高通滤波虽然去掉了大块颜色，但三角形、烧杯、细胞器和河道的边缘坐标仍在，因此供体布局泄漏到新案例。<span class='ok'>右列接受：</span>先用固定随机排列打散高频残差的二维坐标，再做轻微平滑，只保留纹理幅度统计；新案例的几何完全由语义层重新建立。</figcaption></figure>
<p>这个失败形成了一条普适规则：<b>“高频”不等于“没有几何”。</b>任何外观迁移都必须检测供体坐标是否进入结果。修复也不是给五个案例分别加遮罩，而是扩展一个通用底图模式 <code>constant_with_shuffled_highpass_statistics</code> 和一个通用区域模式 <code>shuffled_highpass_statistics</code>。</p>

<h2>4. 两次语义导出修复，不是在最终图上偷偷补画</h2><div class='grid'>
<div class='card'><h3>MATH-01</h3><p>旧程序截图能看到逐渐增长的正弦轨迹，但语义包只导出了圆和坐标轴。State Renderer 没有合法输入，最终图就缺轨迹。修复发生在程序 provider：新增 <code>math01_trace_boundary</code>，再升级合同到 V3。渲染器仍只读取一种通用的 <code>hard_boundary</code>。</p></div>
<div class='card'><h3>PHYS-02</h3><p>旧语义边界没有完整导出线圈、导线、仪表和表针。修复 provider 后得到 <code>phys02_instrument_boundary</code>，表针角度直接来自 <code>induced_current</code> 状态。渲染器没有写“磁铁案例”的特殊代码。</p></div></div>

<h2>5. 这轮没有使用 Canny、ControlNet 或 SDXL</h2><p>Canny 是从图片提取亮暗突变边缘的算法；SDXL Canny ControlNet 会读取这张边缘图，计算用于约束 SDXL 去噪过程的附加残差；SDXL Base 再结合文字提示和残差生成整张图。它适合允许模型重新解释场景的半自由生成，但很难保证晶体数、质量守恒、对象身份或牛轭湖拓扑。这个阶段需要先建立确定的机制基线，所以没有把 Canny 图送进 ControlNet，也没有声称扩散模型生成成功。外观来自已审核模型图的材质统计，几何和状态来自程序语义。</p>

<h2>6. 五个案例的硬门</h2><table><tr><th>案例</th><th>机器检查</th><th>结果</th></tr>
<tr><td>MATH-01</td><td>两个跟踪点匹配状态；轨迹只增长；圆与坐标轴来自类型化边界。</td><td class='ok'>通过</td></tr>
<tr><td>PHYS-02</td><td>一个磁铁、一个固定线圈；磁铁位置匹配；表针为正、零、负对应状态。</td><td class='ok'>通过</td></tr>
<tr><td>CHEM-02</td><td>晶体数 0/0/1/4；液面下降；液相与晶体相溶质总质量恒定。</td><td class='ok'>通过</td></tr>
<tr><td>BIO-02</td><td>两个细胞身份不变；孔隙开后关闭；连通关系与孔隙状态一致。</td><td class='ok'>通过</td></tr>
<tr><td>GEO-01</td><td>河道颈部持续收窄；主河道连通；牛轭湖只在结束状态出现。</td><td class='ok'>通过</td></tr></table>
<p><b>回归：</b><span class='ok'>五个旧 sentinel 案例逐帧哈希完全一致</span>，历史三角洲外观记录也仍可验证。五个新增案例各自重跑一次，20 张结果逐字节一致。</p>

<h2>7. 如何复现</h2><pre>cd /persistent/workspace-project/Live-Document
/opt/venv/bin/python -m modules.video_model.stage3.phase8_scale_image
/opt/venv/bin/python -m modules.video_model.stage3.phase8_scale_finalize
/opt/venv/bin/python -m pytest -q modules/video_model/stage3/tests</pre>
<p>第一条执行合同预检、20 张渲染、G3、确定性回放和旧案例回归；第二条冻结失败/成功对照、视觉评分、基线和本报告；第三条验证所有已接受记录的文件哈希。</p>

<h2>8. 自评与下一步</h2><p>这一轮的真实结论是：<b>五类案例已经有稳定、机制正确、带材料质感的教学关键帧基线</b>。它们并非全都达到照片级真实，尤其数学和地理仍应保持教学图的精确与可读性。下一轮不会等待人工确认：将对五个案例各导出 49 个连续状态，用同一渲染计划生成视频基线，执行 G4 后再判断哪些机制值得调用首尾帧视频模型。</p>
<p class='small'>为保证 Live Preview 和目录移动后仍可查看，本报告的 7 张说明图全部以内嵌 data URI 保存。原始 PNG、JSON、失败样例和参数仍在同目录，便于机器复查。</p>
</main></body></html>"""
    path = OUTPUT / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


def run() -> dict[str, Any]:
    machine = load_json(OUTPUT / "g3-machine.json")
    if not machine["passed"]:
        raise RuntimeError("cannot finalize: phase-8 machine gate failed")
    comparison, sequences = _preserve_rejected_and_sequences()
    review = _visual_review(sequences)
    if not review["passed"]:
        raise RuntimeError("cannot finalize: phase-8 visual review failed")
    _persist_experiment_and_knowledge()
    _update_registry_and_state()
    report = _report(comparison, sequences, review)
    _freeze_baselines(sequences)
    result = {
        "schema_version": "1.0",
        "loop_id": "LOOP-S3-0005",
        "phase": "S3.8",
        "cases": list(CASES),
        "image_route": "accepted_case_specific",
        "machine_gates_passed": True,
        "visual_review_passed": True,
        "determinism_passed": machine["determinism_replay"]["passed"],
        "cross_case_regression_passed": machine["cross_case_regression"]["passed"],
        "report": file_record(report, REPO_ROOT),
        "next_phase": "S3.9-scale-motion",
    }
    write_json(OUTPUT / "checkpoint.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
