"""Finalize the five-case motion loop and publish a self-contained report."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from modules.video_model.stage3.framework.contracts import file_record, load_json, write_json


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-9-scale-motion"
EXPERIMENTS = STAGE3 / "experiments"
CASES = ("MATH-01", "PHYS-02", "CHEM-02", "BIO-02", "GEO-01")
CASE_NAMES = {
    "MATH-01": "单位圆生成正弦曲线",
    "PHYS-02": "磁铁运动产生感应电流",
    "CHEM-02": "盐溶液蒸发与晶体生长",
    "BIO-02": "保卫细胞控制气孔开闭",
    "GEO-01": "曲流裁弯形成牛轭湖",
}


def _uri(path: Path) -> str:
    mime = {".jpg": "image/jpeg", ".png": "image/png", ".mp4": "video/mp4"}[path.suffix.lower()]
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _review(exp_id: str, verdict: str, reason: str, evidence: Path, video_runs: int) -> dict[str, Any]:
    value = {
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "verdict": verdict,
        "passed": verdict.startswith("accepted"),
        "reason_zh": reason,
        "model_runs": {"image_candidates": 0, "video_candidates": video_runs},
        "evidence": file_record(evidence, REPO_ROOT),
    }
    write_json(EXPERIMENTS / exp_id / "review.json", value)
    return value


def _persist_experiments() -> list[dict[str, Any]]:
    hypotheses = {
        "EXP-S3-20260731-030": (
            "H-S3-0011A",
            "逐时间点重新导出程序语义并使用同一冻结渲染计划，可以为五类机制建立可复现的运动基线。",
        ),
        "EXP-S3-20260731-031": (
            "H-S3-0011B",
            "首尾状态不同的结晶过程可由一段 LTX L1 在正确时间完成液面下降和 0→4 晶体生长。",
        ),
        "EXP-S3-20260731-032": (
            "H-S3-0011C",
            "首尾状态不同的曲流裁弯可由一段 LTX L1 保持主河道，并在后段产生一次拓扑分离。",
        ),
        "EXP-S3-20260731-033": (
            "H-S3-0011D",
            "给结晶过程增加两个已验收中间关键帧作为分段边界，可修复 L1 的过早成核和晶体身份波动。",
        ),
    }
    for exp_id, (hypothesis_id, statement) in hypotheses.items():
        root = EXPERIMENTS / exp_id
        root.mkdir(parents=True, exist_ok=True)
        path = root / "hypothesis.md"
        if not path.is_file():
            path.write_text(f"# {hypothesis_id}\n\n{statement}\n", encoding="utf-8")

    reviews = [
        _review(
            "EXP-S3-20260731-030",
            "accepted_core",
            "五案共 245 个程序状态全部完成类型化语义导出、State Renderer 重建、MP4 编解码与 G4；关键帧、连续性、身份、守恒和拓扑通过，旧运动基线未退化。",
            OUTPUT / "g4-machine.json",
            0,
        ),
        _review(
            "EXP-S3-20260731-031",
            "rejected",
            "CHEM-02 的 L1 首尾和连续性通过，液面也下降，但第 12/49 帧就出现晶体，且计数发生 1→5→3→2；扩散视频没有守住成核时刻和晶体身份。",
            OUTPUT / "CHEM-02/L1/g4.json",
            1,
        ),
        _review(
            "EXP-S3-20260731-032",
            "accepted_case_specific",
            "GEO-01 的 L1 在第 39/49 帧才从一个水体变为两个，之前持续保持单一河道，之后主河道和一个牛轭湖稳定存在；端点和连续性也通过。",
            OUTPUT / "GEO-01/L1/g4.json",
            1,
        ),
        _review(
            "EXP-S3-20260731-033",
            "rejected",
            "CHEM-02 的 L2 把首次成核推迟到第 21 帧并改善整体时序，但微小晶体仍在相邻帧消失重现，四晶体也短暂变为三个；稀疏边界没有提供跨段对象身份。",
            OUTPUT / "CHEM-02/L2/g4.json",
            3,
        ),
    ]
    ledger_path = EXPERIMENTS / "ledger.json"
    ledger = load_json(ledger_path)
    by_id = {item["experiment_id"]: item for item in ledger["experiments"]}
    for review in reviews:
        exp_id = review["experiment_id"]
        record = {
            "experiment_id": exp_id,
            "hypothesis_id": hypotheses[exp_id][0],
            "phase": "S3.9",
            "model_runs": review["model_runs"],
            "review": f"modules/video_model/stage3/experiments/{exp_id}/review.json",
            "verdict": review["verdict"],
        }
        if review["verdict"] == "rejected":
            record["failure_taxonomy"] = "motion_or_video"
        by_id[exp_id] = record
    ledger["experiments"] = list(by_id.values())
    ledger["loop_id"] = "LOOP-S3-0006"
    write_json(ledger_path, ledger)
    return reviews


def _update_knowledge() -> None:
    path = STAGE3 / "knowledge/failure_patterns.json"
    value = load_json(path)
    additions = [
        {
            "id": "FP-PROVIDER-TIMELINE-001",
            "taxonomy": "semantic_export",
            "symptom_zh": "四个关键帧都能生成，但第一次采样完整时间线时，结晶 provider 在中间帧因绘图坐标类型报错。",
            "diagnosis_zh": "关键帧 smoke 没有覆盖任意进度；Pillow 版本对混合整数/浮点多边形坐标的容忍度不同。",
            "forbidden_fix_zh": "不得跳过坏帧或用前后帧插值；所有 provider 必须通过固定 49 点采样，并把绘图坐标规范成普通 Python 数值。",
        },
        {
            "id": "FP-GATE-SPARSE-ASSUMPTION-001",
            "taxonomy": "metric_domain",
            "symptom_zh": "BIO-02 四个关键帧的开度为 10/49/49/10，旧 G4 因此误以为两个中间关键帧之间应保持 49。",
            "diagnosis_zh": "稀疏关键帧相同不代表区间内状态恒定；完整程序在正中间达到 62 的峰值。",
            "forbidden_fix_zh": "不得从四个样本猜连续函数；G4 必须读取完整 states.jsonl 或逐点调用 provider 后再冻结趋势。",
        },
        {
            "id": "FP-VIDEO-NUCLEATION-001",
            "taxonomy": "motion_or_video",
            "symptom_zh": "结晶视频端点正确，但晶体过早出现、数量先增后减，分段后仍有微小晶体消失重现。",
            "diagnosis_zh": "首尾扩散和独立 FLF 片段都没有晶体对象 ID；文字能提示大致事件，却不能守住成核阈值和跨帧身份。",
            "forbidden_fix_zh": "不得因最终四颗晶体正确就接受；必须检查首次成核帧和逐帧计数，失败后使用完整程序时间线回退。",
        },
    ]
    known = {item["id"] for item in value["patterns"]}
    value["patterns"].extend(item for item in additions if item["id"] not in known)
    write_json(path, value)


def _freeze_baselines() -> None:
    path = STAGE3 / "baselines/accepted.json"
    accepted = load_json(path)
    records = {item["baseline_id"]: item for item in accepted["records"]}
    additions = [
        {
            "baseline_id": "SCALE-PROGRAM-PROVIDER-S3.9-V1",
            "kind": "accepted_program_provider",
            **file_record(REPO_ROOT / "modules/video_model/stage2/cases/remaining_programs.py", REPO_ROOT),
        },
        {
            "baseline_id": "G4-SCALE-MOTION-S3.9-V1",
            "kind": "accepted_core_evidence",
            **file_record(OUTPUT / "g4-machine.json", REPO_ROOT),
        },
        {
            "baseline_id": "VIDEO-GEO-01-L1-S3.9-V1",
            "kind": "accepted_video_transition",
            **file_record(OUTPUT / "GEO-01/L1/transition.mp4", REPO_ROOT),
        },
    ]
    for case_id in CASES:
        additions.append({
            "baseline_id": f"VIDEO-{case_id}-PROGRAM-TIMELINE-S3.9-V1",
            "kind": "accepted_case_specific",
            **file_record(OUTPUT / f"{case_id}/deterministic/transition.mp4", REPO_ROOT),
        })
    for item in additions:
        records[item["baseline_id"]] = item
    accepted["records"] = list(records.values())
    write_json(path, accepted)


def _update_registry_and_state() -> None:
    path = STAGE3 / "case_registry.json"
    registry = load_json(path)
    routes = {
        "MATH-01": {
            "status": "accepted_deterministic_default",
            "evidence": "modules/video_model/stage3/output/phase-9-scale-motion/MATH-01/g4.json",
            "model_route": "not_run_first_equals_last_and_exact_trajectory_required",
        },
        "PHYS-02": {
            "status": "accepted_deterministic_default",
            "evidence": "modules/video_model/stage3/output/phase-9-scale-motion/PHYS-02/g4.json",
            "model_route": "not_run_cyclic_position_plus_exact_pointer_identity",
        },
        "CHEM-02": {
            "status": "accepted_deterministic_fallback",
            "evidence": "modules/video_model/stage3/output/phase-9-scale-motion/CHEM-02/g4.json",
            "rejected_model_routes": ["L1", "L2"],
        },
        "BIO-02": {
            "status": "accepted_deterministic_default",
            "evidence": "modules/video_model/stage3/output/phase-9-scale-motion/BIO-02/g4.json",
            "model_route": "not_run_first_equals_last_and_midpoint_peak_required",
        },
        "GEO-01": {
            "status": "L1_accepted",
            "evidence": "modules/video_model/stage3/output/phase-9-scale-motion/GEO-01/L1/g4.json",
            "deterministic_fallback": "modules/video_model/stage3/output/phase-9-scale-motion/GEO-01/g4.json",
        },
    }
    for case in registry["cases"]:
        if case["case_id"] in routes:
            case["motion_route"] = routes[case["case_id"]]
    write_json(path, registry)

    state_path = STAGE3 / "state.json"
    state = load_json(state_path)
    state.update({
        "active_loop_id": "LOOP-S3-0007",
        "loop_id": "LOOP-S3-0007",
        "phase": "S3.10",
        "phase_status": "in_progress",
        "current_problem_id": "S3-PROBLEM-FULL-REGRESSION-001",
        "current_problem": {
            "problem_id": "S3-PROBLEM-FULL-REGRESSION-001",
            "taxonomy": "release",
            "summary_zh": "十个正式案例已各有图像与运动路线，需要统一重放全部合同、基线和报告链接后更新发布结论。",
        },
        "current_hypothesis_id": "H-S3-0012A",
        "current_hypothesis": {
            "hypothesis_id": "H-S3-0012A",
            "statement_zh": "若十案 G0-G4、已接受文件哈希和历史回归一次通过，则 Stage 3 可形成确定流程候选版。",
            "falsification_zh": "任一案例缺路线、任一接受基线失配或任一新人报告资源失效都阻止阶段发布。",
        },
        "next_action": "Run the complete ten-case and historical release regression, repair only evidenced failures, then publish the Stage 3 release report.",
    })
    write_json(state_path, state)


def _report() -> Path:
    deterministic_cards = "".join(
        f"<figure><img src='{_uri(OUTPUT / f'{case_id}/deterministic/generated-frames.jpg')}'><figcaption><b>{case_id} · {CASE_NAMES[case_id]}</b>：49 个程序状态逐帧重建，G4 通过。</figcaption>"
        f"<video controls preload='metadata' src='{_uri(OUTPUT / f'{case_id}/deterministic/transition.mp4')}'></video></figure>"
        for case_id in CASES
    )
    chem_l1 = load_json(OUTPUT / "CHEM-02/L1/g4.json")
    chem_l2 = load_json(OUTPUT / "CHEM-02/L2/g4.json")
    geo_l1 = load_json(OUTPUT / "GEO-01/L1/g4.json")
    geo_cutoff = next(item for item in geo_l1["mechanism_checks"] if item["name"] == "topology_change_occurs_after_neck_narrowing")["evidence"]["first_two_component_frame"]
    report = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Stage 3 五案例运动闭环</title><style>
:root{{--ink:#193330;--muted:#5d6d69;--paper:#f4f0e6;--card:#fffdf8;--line:#d8d0c0;--ok:#19704c;--bad:#a53d33}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.72 system-ui,-apple-system,"Noto Sans SC",sans-serif}}main{{max-width:1180px;margin:auto;padding:34px 24px 80px}}h1{{font-size:clamp(30px,5vw,52px);line-height:1.12}}h2{{margin-top:2.2em;padding-top:1em;border-top:1px solid var(--line)}}.lead{{font-size:20px;max-width:930px}}.flow{{background:#193330;color:#fffaf0;border-radius:14px;padding:20px;font-size:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}}figure,.card{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px}}img,video{{display:block;width:100%;height:auto;border-radius:8px;background:#132b29}}figcaption{{padding:10px 2px;color:var(--muted)}}table{{width:100%;border-collapse:collapse;background:var(--card)}}th,td{{padding:11px;text-align:left;vertical-align:top;border:1px solid var(--line)}}.ok{{color:var(--ok);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}code,pre{{background:#e8e3d8;border-radius:5px}}code{{padding:.1em .35em}}pre{{white-space:pre-wrap;padding:15px;overflow:auto}}.small{{font-size:14px;color:var(--muted)}}</style></head><body><main>
<p>Stage 3 · LOOP-S3-0006 · S3.9</p><h1>五类运动都完成：一个模型路线通过，四个使用确定性路线</h1>
<p class='lead'>本轮不是把四张关键帧简单拼成 MP4。每个案例先生成 49 份完整程序状态，再用上一轮冻结的材质计划逐帧重建；全部通过后，才对适合当前首尾帧模型的两个非循环案例做有限 LTX 对照。</p>
<div class='flow'>程序 provider(49 次) → 每帧状态 JSON + 类型化语义层 → 同一 State Renderer → 49 张材质帧 → MP4 编码 → 再解码 → G4 关键帧/连续性/机制检查 → 可用时尝试 LTX L1 → 失败才升级 L2 → 接受或确定性回退</div>

<h2>1. 五个可复现的运动基线</h2><div class='grid'>{deterministic_cards}</div>
<p>这些视频的“运动”来自程序事实，外观来自已验收供体。它们不冒充视频模型输出。优点是对象身份、质量、状态峰值和拓扑可逐帧检查；限制是运动细节仍偏教学动画，而不是自由摄影。</p>

<h2>2. 为什么先导出完整时间线</h2><table><tr><th>只看四张关键帧会误判什么</th><th>完整 49 帧揭示的事实</th></tr>
<tr><td>BIO-02 的开度是 10 / 49 / 49 / 10，看起来像中间保持不变。</td><td>实际在第 24 帧达到 62 px，再对称关闭。旧门禁据四点猜“平台”而错杀，已修为读取完整状态。</td></tr>
<tr><td>CHEM-02 的四张关键帧都能生成，看起来 provider 已稳定。</td><td>第 27 个任意时间样本首次暴露 Pillow 混合坐标类型错误；修复 provider 后才允许继续。</td></tr>
<tr><td>首尾帧能告诉模型开始和结束。</td><td>不能告诉它阈值何时越过、对象是否消失重生，或中间是否还有更高峰值。</td></tr></table>

<h2>3. GEO-01：LTX L1 通过</h2><figure><img src='{_uri(OUTPUT / 'GEO-01/L1/generated-frames.jpg')}'><figcaption><span class='ok'>接受。</span>模型在第 {geo_cutoff}/49 帧才把一个连通水体变为两个；之前河颈持续变窄，之后主河道始终是最大的连通水体，且只多出一个牛轭湖。</figcaption><video controls preload='metadata' src='{_uri(OUTPUT / 'GEO-01/L1/transition.mp4')}'></video></figure>
<p>输入是开始图、结束图和结构化文字合同。LTX-2.3 从噪声生成中间帧；<code>guide_strength=0.7</code> 表示首尾图约束进入潜变量的权重，<code>cfg=1.0</code> 是文字条件引导尺度，固定 <code>seed=2026073132</code> 使同一噪声起点可复查。G4 不读取这些参数来“加分”，只检查解码后的像素与水体连通性。</p>

<h2>4. CHEM-02：L1 与 L2 都拒绝</h2><div class='grid'>
<figure><img src='{_uri(OUTPUT / 'CHEM-02/L1/generated-frames.jpg')}'><figcaption><span class='bad'>L1 拒绝。</span>首尾、连续性和液面下降都对，但首次晶体在第 12/49 帧出现；程序阈值约在后半段。晶体计数还经历 1→5→3→2。</figcaption><video controls preload='metadata' src='{_uri(OUTPUT / 'CHEM-02/L1/transition.mp4')}'></video></figure>
<figure><img src='{_uri(OUTPUT / 'CHEM-02/L2/generated-frames.jpg')}'><figcaption><span class='bad'>L2 拒绝。</span>三段分别使用 00→01、01→02、02→03，首次成核推迟到第 21 帧；但微小晶体仍在相邻帧消失重现，四颗也短暂变成三颗。</figcaption><video controls preload='metadata' src='{_uri(OUTPUT / 'CHEM-02/L2/transition.mp4')}'></video></figure></div>
<p>L2 不是模型原生接收四张图：当前运行时只支持首尾图，所以实际调用三次 FLF，再删除重复边界帧并拼接。片段共享边界像素，却不共享“这是同一颗晶体”的隐状态。最终采用上方的 49 帧确定性视频。</p>

<h2>5. 为什么另外三案没有浪费模型调用</h2><table><tr><th>案例</th><th>首尾帧接口缺少的信息</th><th>结论</th></tr>
<tr><td>MATH-01</td><td>首尾画面圆周点相同，但中间必须完成一整圈并严格绘出一周期正弦；还要求解析几何。</td><td>确定性默认；既有精确刚体实验已经证明 FLF 会闪回或改形。</td></tr>
<tr><td>PHYS-02</td><td>磁铁首尾位置相同，中间必须接近、停止、撤离，表针依次正/零/负。</td><td>确定性默认；首尾图不能承载停顿和符号反转的完整次序。</td></tr>
<tr><td>BIO-02</td><td>首尾气孔都关闭，但中间必须先开到隐藏峰值再关闭，并保持同一对细胞。</td><td>确定性默认；当前运行时不能输入完整开度曲线，BIO-01 的 L1/L2 已暴露身份失败。</td></tr></table>

<h2>6. G4 到底检查什么</h2><p>每个 MP4 都被重新解码为 49 张 RGB 图。通用部分检查：帧数、首尾相对输入的平均绝对像素误差、最大相邻帧跳变、四个关键时间点。案例部分再读程序状态：数学检查同步和轨迹单调，物理检查运动阶段与电流符号，化学检查溶剂/浓度/晶体/质量，生物检查身份与开度峰值，地理检查颈宽、主河连通和牛轭湖拓扑。任何机制硬门失败，美观不能抵消。</p>

<h2>7. 完整提示词与复现</h2><p>两条 LTX 的正负提示词、每段输入图、workflow API JSON、模型权重指纹、seed 和输出均保存在各自 <code>L1/inputs</code>、<code>L1/_work</code> 或 <code>L2/segments</code> 中。本报告不只列 seed，因为 seed 只是固定噪声起点；模型、首尾图、文本、采样 sigmas 和运行时都必须相同。</p>
<pre>cd /persistent/workspace-project/Live-Document
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase9_scale_motion
/persistent/ComfyUI/start-ltx2.3.sh
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase9_scale_video generate --case GEO-01
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase9_scale_video audit --case GEO-01
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase9_scale_video generate --case CHEM-02 --level L2
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase9_scale_video audit --case CHEM-02 --level L2
/opt/venv/bin/python -m modules.video_model.stage3.phase9_scale_finalize</pre>
<p class='small'>本报告内嵌 8 个 MP4 和 8 张九宫格预览，Live Preview 不依赖相对资源路径。原始 245 帧、语义 NPY/JSON、失败视频和逐帧证据未删除。</p>
</main></body></html>"""
    path = OUTPUT / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


def run() -> dict[str, Any]:
    machine = load_json(OUTPUT / "g4-machine.json")
    chem_l1 = load_json(OUTPUT / "CHEM-02/L1/g4.json")
    chem_l2 = load_json(OUTPUT / "CHEM-02/L2/g4.json")
    geo_l1 = load_json(OUTPUT / "GEO-01/L1/g4.json")
    if not machine["passed"] or chem_l1["passed"] or chem_l2["passed"] or not geo_l1["passed"]:
        raise RuntimeError("motion verdict inputs do not match the reviewed outcome")
    reviews = _persist_experiments()
    _update_knowledge()
    _update_registry_and_state()
    _freeze_baselines()
    report = _report()
    result = {
        "schema_version": "1.0",
        "loop_id": "LOOP-S3-0006",
        "phase": "S3.9",
        "deterministic_routes": {case_id: "passed" for case_id in CASES},
        "model_routes": {"GEO-01/L1": "accepted_case_specific", "CHEM-02/L1": "rejected", "CHEM-02/L2": "rejected"},
        "total_video_model_candidates": sum(item["model_runs"]["video_candidates"] for item in reviews),
        "cross_case_motion_regression_passed": machine["cross_case_motion_regression"]["passed"],
        "report": file_record(report, REPO_ROOT),
        "next_phase": "S3.10-full-release-regression",
    }
    write_json(OUTPUT / "checkpoint.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
