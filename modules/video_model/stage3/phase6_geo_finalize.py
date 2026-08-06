"""Finalize the GEO-02 image and motion loop without claiming model success.

The generated checkpoint is deliberately self-contained: every explanatory
image and all three short videos are embedded as data URIs so Live Preview does
not depend on the report's current directory.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    write_json,
)


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-6-rerun-2"
EXPERIMENTS = STAGE3 / "experiments"


def _uri(path: Path) -> str:
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp4": "video/mp4",
    }[path.suffix.lower()]
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _review(
    experiment_id: str,
    verdict: str,
    reason_zh: str,
    evidence: Path,
    *,
    image_runs: int = 0,
    video_runs: int = 0,
) -> dict[str, Any]:
    value = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "verdict": verdict,
        "passed": verdict.startswith("accepted"),
        "reason_zh": reason_zh,
        "model_runs": {
            "image_candidates": image_runs,
            "video_candidates": video_runs,
        },
        "evidence": file_record(evidence, REPO_ROOT),
    }
    write_json(EXPERIMENTS / experiment_id / "review.json", value)
    return value


def _persist_reviews() -> list[dict[str, Any]]:
    reviews = [
        _review(
            "EXP-S3-20260731-024",
            "accepted_case_specific",
            (
                "同一冻结山体外观上，程序导出的空气团身份和云雨标量场使四帧的"
                "越山、迎风降雨、背风变干全部通过 G3；PHYS-01、CHEM-01 和三角洲"
                "历史回归通过。新 scalar_field_overlay 尚未在第二个使用该算子的"
                "学科中通过，因此暂不晋升通用核心。"
            ),
            OUTPUT / "GEO-02/g3-machine.json",
        ),
        _review(
            "EXP-S3-20260731-025",
            "rejected",
            (
                "LTX 首尾帧和像素连续性通过，但把蓝色降雨解释成贴着山坡移动的"
                "实体水带；空气团身份门禁也失败。端点相似不能替代机制正确。"
            ),
            OUTPUT / "GEO-02/video/L1/g4.json",
            video_runs=1,
        ),
        _review(
            "EXP-S3-20260731-026",
            "rejected",
            (
                "三段 LTX 提高了关键帧边界贴合，但各片段不共享对象隐状态，"
                "空气团出现反向、消失和重现，身份与轨迹门禁失败。"
            ),
            OUTPUT / "GEO-02/video/L2/g4.json",
            video_runs=3,
        ),
    ]
    fallback_id = "EXP-S3-20260731-027"
    fallback_dir = EXPERIMENTS / fallback_id
    fallback_dir.mkdir(parents=True, exist_ok=True)
    (fallback_dir / "hypothesis.md").write_text(
        "# H-S3-0008D — GEO-02 完整程序时间线回退\n\n"
        "如果首尾帧视频模型不能守恒空气团身份和降雨事件，就把程序的 49 个"
        "时间点逐帧导出到同一 State Renderer。预期地形与外观冻结，空气团"
        "持续向右，迎风降雨先升后降。\n",
        encoding="utf-8",
    )
    reviews.append(
        _review(
            fallback_id,
            "accepted_case_specific",
            (
                "49 个时间点都由 GEO-02 的确定性 provider 重新计算语义层，再经"
                "同一外观供体和 State Renderer 重建；G4 的端点、连续性、空气团"
                "身份、降雨峰值与四个稀疏关键帧全部通过。它是明确的程序回退，"
                "不冒充视频模型生成成功。"
            ),
            OUTPUT / "GEO-02/video/deterministic/g4.json",
        )
    )
    return reviews


def _update_ledger(reviews: list[dict[str, Any]]) -> None:
    ledger_path = EXPERIMENTS / "ledger.json"
    ledger = load_json(ledger_path)
    by_id = {item["experiment_id"]: item for item in ledger["experiments"]}
    hypotheses = {
        "EXP-S3-20260731-024": "H-S3-0008B",
        "EXP-S3-20260731-025": "H-S3-0008C",
        "EXP-S3-20260731-026": "H-S3-0008C",
        "EXP-S3-20260731-027": "H-S3-0008D",
    }
    for review in reviews:
        experiment_id = review["experiment_id"]
        record = {
            "experiment_id": experiment_id,
            "hypothesis_id": hypotheses[experiment_id],
            "phase": "S3.6-rerun-2",
            "model_runs": review["model_runs"],
            "review": (
                f"modules/video_model/stage3/experiments/{experiment_id}/review.json"
            ),
            "verdict": review["verdict"],
        }
        if review["verdict"] == "rejected":
            record["failure_taxonomy"] = "motion_or_video"
        by_id[experiment_id] = record
    ledger["experiments"] = list(by_id.values())
    ledger["loop_id"] = "LOOP-S3-0003"
    write_json(ledger_path, ledger)


def _update_knowledge() -> None:
    path = STAGE3 / "knowledge/failure_patterns.json"
    value = load_json(path)
    additions = [
        {
            "id": "FP-VIDEO-FIELD-001",
            "taxonomy": "motion_or_video",
            "symptom_zh": "稀疏雨线在生成视频中变成沿地形爬行的实心水带或冰带。",
            "diagnosis_zh": "首尾帧模型把局部标量事件当成具有实体表面的对象。",
            "forbidden_fix_zh": "不得因端点和连续性通过就接受；必须检测事件的时间峰值、空间侧别和材料类别。",
        },
        {
            "id": "FP-VIDEO-IDENTITY-002",
            "taxonomy": "motion_or_video",
            "symptom_zh": "分段视频在接缝处的空气团消失、反向或以新身份重现。",
            "diagnosis_zh": "相邻片段共享边界图像，却不共享对象身份、速度或内部生成状态。",
            "forbidden_fix_zh": "不得只测接缝关键帧 MAE；必须跨全片追踪同一对象身份与方向。",
        },
        {
            "id": "FP-REGRESSION-NUMERIC-001",
            "taxonomy": "metric_domain",
            "symptom_zh": "确定性回放只有一个像素的一个通道相差 1/255，却被纯文件哈希判失败。",
            "diagnosis_zh": "编码或连续采样的最低有效位差异不等于视觉或语义回归。",
            "forbidden_fix_zh": "不得取消哈希；先测哈希，失配时只允许预冻结的单像素单 LSB 数值等价规则。",
        },
    ]
    existing = {item["id"] for item in value["patterns"]}
    value["patterns"].extend(item for item in additions if item["id"] not in existing)
    write_json(path, value)

    open_path = STAGE3 / "knowledge/open_problems.json"
    problems = load_json(open_path)
    problems["problems"] = [
        item
        for item in problems["problems"]
        if item["problem_id"] not in {
            "S3-PROBLEM-VISUAL-001",
            "S3-PROBLEM-RELEASE-GEO-001",
        }
    ]
    write_json(open_path, problems)


def _update_registry_and_state() -> None:
    registry_path = STAGE3 / "case_registry.json"
    registry = load_json(registry_path)
    geo = next(item for item in registry["cases"] if item["case_id"] == "GEO-02")
    geo["visual_target_status"] = "accepted_project_baseline"
    geo["known_gaps"] = []
    geo["completeness"]["visual_target_status"] = "accepted_project_baseline"
    geo["completeness"]["known_gaps"] = []
    geo["image_route"] = {
        "status": "accepted_case_specific",
        "evidence": "modules/video_model/stage3/output/phase-6-rerun-2/GEO-02/g3-machine.json",
    }
    geo["motion_route"] = {
        "status": "accepted_deterministic_fallback",
        "evidence": "modules/video_model/stage3/output/phase-6-rerun-2/GEO-02/video/deterministic/g4.json",
        "rejected_model_routes": ["L1", "L2"],
    }
    write_json(registry_path, registry)

    state_path = STAGE3 / "state.json"
    state = load_json(state_path)
    state.update(
        {
            "active_loop_id": "LOOP-S3-0004",
            "loop_id": "LOOP-S3-0004",
            "phase": "S3.7",
            "phase_status": "in_progress",
            "current_problem_id": "S3-PROBLEM-VTP-SCALE-001",
            "current_problem": {
                "problem_id": "S3-PROBLEM-VTP-SCALE-001",
                "taxonomy": "visual_target",
                "summary_zh": "五个 scale Case 仍缺 Visual Target Package。",
            },
            "current_hypothesis_id": "H-S3-0009A",
            "current_hypothesis": {
                "hypothesis_id": "H-S3-0009A",
                "statement_zh": "外观供体可以跨案例复用，但每个案例仍必须有独立的机制正例、反例和硬门。",
                "falsification_zh": "若共享外观把供体的对象布局带入新案例，或案例机制无法用独立硬门描述，则拒绝该视觉包。",
            },
            "current_case_cohort": {
                "target": "MATH-01",
                "route_regressions": ["BIO-02", "GEO-01"],
                "historical_regression": "GEO-HIST-DELTA-01",
            },
            "current_cohort": {
                "target": "MATH-01",
                "regressions": ["BIO-02", "GEO-01"],
                "historical": "GEO-HIST-DELTA-01",
            },
            "next_action": "Build and validate Visual Target Packages for all five scale cases, then continue without a user checkpoint.",
        }
    )
    write_json(state_path, state)


def _report() -> Path:
    target = OUTPUT / "report-assets/target-comparison.jpg"
    blind = OUTPUT / "report-assets/blind-geo-comparison.jpg"
    l1_preview = OUTPUT / "GEO-02/video/L1/generated-frames.jpg"
    l2_preview = OUTPUT / "GEO-02/video/L2/generated-frames.jpg"
    fallback_preview = OUTPUT / "GEO-02/video/deterministic/generated-frames.jpg"
    l1_video = OUTPUT / "GEO-02/video/L1/transition.mp4"
    l2_video = OUTPUT / "GEO-02/video/L2/transition.mp4"
    fallback_video = OUTPUT / "GEO-02/video/deterministic/transition.mp4"
    report = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Stage 3 GEO-02 闭环报告</title>
<style>
:root{{--ink:#18312f;--muted:#566a67;--paper:#f5f1e7;--card:#fffdf8;--ok:#19704c;--bad:#a83c34;--line:#d9d1c1}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.72 system-ui,-apple-system,"Noto Sans SC",sans-serif}}main{{max-width:1180px;margin:auto;padding:34px 24px 80px}}h1{{font-size:clamp(30px,5vw,54px);line-height:1.1;margin:.2em 0}}h2{{margin-top:2.2em;border-top:1px solid var(--line);padding-top:1em}}h3{{margin-bottom:.3em}}.lead{{font-size:20px;max-width:900px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}}.card,figure{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:0}}img,video{{display:block;width:100%;height:auto;border-radius:8px;background:#102522}}figcaption{{padding:10px 2px 0;color:var(--muted)}}.ok{{color:var(--ok);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}code{{background:#e7e2d7;padding:.12em .4em;border-radius:4px}}table{{width:100%;border-collapse:collapse;background:var(--card)}}th,td{{text-align:left;vertical-align:top;padding:12px;border:1px solid var(--line)}}.flow{{font-size:18px;background:#18312f;color:#f7f2e8;border-radius:14px;padding:20px}}.small{{color:var(--muted);font-size:14px}}
</style></head><body><main>
<p>Stage 3 · LOOP-S3-0003 · GEO-02</p><h1>地形雨案例：图像通过，视频模型失败，确定性时间线通过</h1>
<p class='lead'>本轮不是“生成一张山景图”。目标是让同一座山在四个程序状态中，稳定讲清楚湿空气越山、迎风坡降雨、背风坡变干。图像流程通过；两种 LTX 视频路线都没有守住机制，因此最终采用可复现的程序时间线回退。</p>
<div class='flow'>程序状态 JSON + 语义层 → 固定山体外观 → State Renderer 写回空气团与云雨 → G3 图像门禁 → LTX L1 → 失败 → LTX L2 → 失败 → 49 帧程序时间线回退 → G4 通过</div>

<h2>1. 输入分别负责什么</h2><table><tr><th>输入</th><th>作用</th><th>不能决定什么</th></tr>
<tr><td>程序状态 JSON</td><td>空气团位置、湿度、降雨强度、时间顺序</td><td>山体照片质感</td></tr>
<tr><td>语义层</td><td>告诉渲染器“哪一块是云雨标量场、哪一个是空气团身份”</td><td>不把整张截图的文字和箭头当边缘</td></tr>
<tr><td>冻结的 SDXL 山体图</td><td>山体材质、阴天光照、空气透视</td><td>不负责降雨在哪一侧，也不允许改变程序机制</td></tr>
<tr><td>语言提示词</td><td>向视频模型说明地形、空气团和降雨事件</td><td>不能提供逐帧对象身份守恒</td></tr></table>
<h2>2. 图像怎样生成</h2><figure><img src='{_uri(target)}'><figcaption>从左到右展示程序事实、外观供体、旧 ControlNet 失败和本轮结果。旧方案把云雨边缘当成山脊；新方案不让云雨进入山体几何生成，而是在固定山体上从语义层确定性写回。</figcaption></figure>
<div class='grid' style='margin-top:18px'><div class='card'><h3>新增通用算子</h3><p><code>scalar_field_overlay</code> 读取一张数值场：<code>soft_tint</code> 表示云或湿度体，<code>streaks</code> 表示局部降雨。它只在声明区域工作，强度可由状态 JSON 的字段调制。</p></div><div class='card'><h3>图像硬门</h3><p class='ok'>6/6 通过：</p><p>山体来源固定；空气团只向右；第三帧降雨达峰；主降雨在迎风侧；终帧湿度和降雨下降；声明范围外像素不变。</p></div></div>
<h2>3. 盲评与对照</h2><figure><img src='{_uri(blind)}'><figcaption>盲评 A 是带程序状态的候选，B 只是重复山体外观。A 得分 4.52/5 且硬门全过；B 虽然山体好看，但没有机制，因此不能通过。</figcaption></figure>

<h2>4. 三条视频路线的实际结果</h2><div class='grid'>
<figure><img src='{_uri(l1_preview)}'><figcaption><span class='bad'>L1 整段生成：拒绝。</span>首尾和像素连续性通过，但稀疏蓝雨被解释成沿坡爬动的实心水带，空气团可见性也失败。</figcaption><video controls preload='metadata' src='{_uri(l1_video)}'></video></figure>
<figure><img src='{_uri(l2_preview)}'><figcaption><span class='bad'>L2 三段生成：拒绝。</span>中间关键帧更贴合，但三个片段不共享身份，空气团反向、消失、再出现。</figcaption><video controls preload='metadata' src='{_uri(l2_video)}'></video></figure>
<figure><img src='{_uri(fallback_preview)}'><figcaption><span class='ok'>49 帧确定性回退：通过。</span>每个时间点重新调用程序 provider 导出状态和语义层，再用同一外观渲染，不从截图猜颜色。</figcaption><video controls preload='metadata' src='{_uri(fallback_video)}'></video></figure></div>

<h2>5. 为什么最终路线通过</h2><table><tr><th>检查</th><th>结果</th></tr>
<tr><td>首帧/尾帧相对输入平均像素误差</td><td class='ok'>1.3749 / 1.4089，阈值 15</td></tr>
<tr><td>最大相邻帧跳变</td><td class='ok'>1.7397，阈值 12</td></tr>
<tr><td>空气团身份与方向</td><td class='ok'>49 帧持续可见，向右净移动 674.987 px</td></tr>
<tr><td>降雨事件</td><td class='ok'>第 32 帧达峰（全程 2/3），终帧降至峰值的 0</td></tr>
<tr><td>四个程序关键帧</td><td class='ok'>全部在预冻结误差内</td></tr>
<tr><td>跨案例回归</td><td class='ok'>PHYS-01、CHEM-01、三角洲历史基线通过</td></tr></table>

<h2>6. 本轮沉淀成什么通用认知</h2><p>第一，外观图只能提供“看起来像什么”，标量事件必须由程序状态重新写回。第二，首尾帧视频模型能保持画面相似，不等于懂“降雨是短暂标量事件”。第三，分段视频共享图像却不共享对象身份，因此精确教学动画必须检查完整轨迹。第四，模型路线失败后，应自动退回完整程序时间线，而不是停下来等人工确认。</p>
<p class='small'>本文件内嵌 5 张说明图和 3 个 MP4；移动目录或通过 Live Preview 打开时不依赖相对图片路径。详细机器证据保留在同目录 JSON 中。</p>
</main></body></html>"""
    path = OUTPUT / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


def run() -> dict[str, Any]:
    reviews = _persist_reviews()
    _update_ledger(reviews)
    _update_knowledge()
    _update_registry_and_state()
    report = _report()
    checkpoint = {
        "schema_version": "1.0",
        "loop_id": "LOOP-S3-0003",
        "case_id": "GEO-02",
        "image_route": "accepted_case_specific",
        "video_model_routes": {"L1": "rejected", "L2": "rejected"},
        "motion_route": "accepted_deterministic_fallback",
        "cross_case_regression_passed": load_json(
            OUTPUT / "cross-case-regression.json"
        )["passed"],
        "report": file_record(report, REPO_ROOT),
        "next_loop": "LOOP-S3-0004",
        "remaining_scale_cases": [
            "MATH-01",
            "PHYS-02",
            "CHEM-02",
            "BIO-02",
            "GEO-01",
        ],
    }
    write_json(OUTPUT / "checkpoint.json", checkpoint)
    return checkpoint


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
