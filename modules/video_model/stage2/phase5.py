"""Build the cumulative beginner-readable Phase 5 motion report."""

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
SOURCE_ROOT = STAGE2_ROOT / "phase5_experiments"
OUTPUT_ROOT = STAGE2_ROOT / "output" / "phase-5"
EXPERIMENT_OUTPUT_ROOT = OUTPUT_ROOT / "experiments"
REPORT_PATH = OUTPUT_ROOT / "report.html"
MANIFEST_PATH = OUTPUT_ROOT / "phase5_manifest.json"
LEDGER_PATH = STAGE2_ROOT / "experiments" / "ledger.json"
POLICY_PATH = STAGE2_ROOT / "protocols" / "regression_policy.json"
MOTION_ORDER = (
    "rigid_motion",
    "continuous_field_propagation",
    "liquid_mixing",
    "object_division",
    "boundary_topology_change",
)


def _experiments() -> list[dict[str, Any]]:
    records = []
    if not SOURCE_ROOT.is_dir():
        return records
    for source_dir in sorted(SOURCE_ROOT.glob("EXP-P5-*")):
        spec_path = source_dir / "spec.json"
        review_path = source_dir / "review.json"
        run_path = (
            EXPERIMENT_OUTPUT_ROOT
            / source_dir.name
            / "_work"
            / "run.json"
        )
        if not (
            spec_path.is_file()
            and review_path.is_file()
            and run_path.is_file()
        ):
            continue
        records.append(
            {
                "experiment_id": source_dir.name,
                "spec_path": spec_path,
                "review_path": review_path,
                "run_path": run_path,
                "spec": load_json(spec_path),
                "review": load_json(review_path),
                "run": load_json(run_path),
            }
        )
    return records


def _motion_status(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policy = load_json(POLICY_PATH)["video_motion_representatives"]
    completed = {
        record["run"]["motion_class"]
        for record in records
        if record["review"].get("motion_class_passed", False)
    }
    return [
        {
            "motion_class": motion,
            "case_id": policy[motion],
            "status": "passed" if motion in completed else "pending",
        }
        for motion in MOTION_ORDER
    ]


def _experiment_section(record: dict[str, Any]) -> str:
    experiment_id = record["experiment_id"]
    spec = record["spec"]
    review = record["review"]
    run = record["run"]
    relative = f"experiments/{experiment_id}"
    is_model_video = run["model_runs"]["video"] > 0
    poster_index = int(run["sample_indices"][4])
    checks = "".join(
        f"<li><code>{html.escape(check['name'])}</code>："
        f"{'通过' if check['passed'] else '失败'}</li>"
        for check in run["hard_checks"]
    )
    scores = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value}/5</td></tr>"
        for name, value in review["scores_5"].items()
    )
    evidence = ""
    radial = run.get("radial_propagation_audit")
    if radial:
        traces = "；".join(
            " → ".join(str(value) for value in values)
            for values in radial["extent_by_center_px"]
        )
        evidence = (
            "<p><b>方向审计：</b>最外波纹半径（px）为 "
            + html.escape(traces)
            + "。这证明总体向外，不只是低闪烁。</p>"
        )
    identity = run.get("color_identity_reference_audit")
    if identity:
        evidence += (
            "<p><b>对象身份审计：</b>面积比 "
            f"{identity['minimum_area_ratio']}–{identity['maximum_area_ratio']}，"
            "形状特征比 "
            f"{identity['minimum_shape_eigenvalue_ratio']}–"
            f"{identity['maximum_shape_eigenvalue_ratio']}，"
            "最大质心路径误差 "
            f"{identity['maximum_centroid_error_px']} px。</p>"
        )
    color_mass = run.get("color_mass_trend_audit")
    if color_mass:
        evidence += (
            "<p><b>颜色量审计：</b>局部显色积分强度峰值为首帧的 "
            f"{color_mass['maximum_score_fraction_of_initial']:.3f} 倍，"
            "尾帧为 "
            f"{color_mass['final_score_fraction_of_initial']:.3f} 倍。"
            "这个数值门只检查颜色量，仍必须人工确认是在扩散，"
            "不是实心色块缩小。</p>"
        )
    division = run.get("color_component_division_audit")
    if division:
        evidence += (
            "<p><b>对象分裂审计：</b>紫色组件从 "
            f"{division['initial_component_count']} 个变为 "
            f"{division['final_component_count']} 个，尾帧左右为 "
            f"{division['final_left_component_count']} + "
            f"{division['final_right_component_count']}。"
            "组件计数能检查可见数量与分配，但不能恢复程序父子 ID。</p>"
        )
    topology = run.get("color_region_topology_audit")
    if topology:
        evidence += (
            "<p><b>区域拓扑审计：</b>水色组件从 "
            f"{topology['initial_component_count']} 个变为 "
            f"{topology['final_component_count']} 个；每一帧都保留一个"
            "左右贯通主通道，尾帧独立水体面积为 "
            f"{topology['final_largest_isolated_area_px']} px。"
            "这些值直接从输出像素连通域计算，不读取程序自报布尔值。</p>"
        )
    if is_model_video:
        positive = html.escape(spec["prompt"]["positive"])
        negative = html.escape(spec["prompt"]["negative"])
        prompt_block = f"""<h3>实际提示词</h3>
        <details><summary>正向提示词</summary><pre>{positive}</pre></details>
        <details><summary>负向提示词</summary><pre>{negative}</pre></details>"""
        evidence_links = (
            f'<a href="{relative}/_work/workflow_api.json">'
            "完整 ComfyUI 工作流</a> · "
        )
        output_class = "LTX-2.3 首尾帧生成"
    else:
        prompt_block = """<h3>为什么没有提示词</h3>
        <p>该结果由确定性程序逐帧重采样，视频模型关闭；状态和对象轨迹直接来自
        program plugin，因此没有正负向生成提示词。</p>"""
        evidence_links = (
            f'<a href="{relative}/_work/program_validation.json">'
            "程序机制与对象身份验证</a> · "
        )
        output_class = "确定性程序回退（模型关闭）"
    return f"""<section id="{experiment_id}">
    <p class="eyebrow">{html.escape(spec['motion_class'])} · {html.escape(spec['case_id'])}</p>
    <h2>{html.escape(review['verdict_zh'])}</h2>
    <video controls muted loop preload="metadata"
    poster="{relative}/samples/frame_{poster_index:03d}.png"
    src="{relative}/transition.mp4"></video>
    <p><b>输出类型：</b>{output_class}</p>
    <p>{html.escape(review['visual_finding_zh'])}</p>
    <div class="pair">
      <figure><img src="{relative}/inputs/first.png" alt="首帧输入">
      <figcaption>首帧：{html.escape(spec['source']['first_keyframe_id'])}</figcaption></figure>
      <figure><img src="{relative}/inputs/last.png" alt="尾帧输入">
      <figcaption>尾帧：{html.escape(spec['source']['last_keyframe_id'])}</figcaption></figure>
    </div>
    <h3>实际输出的 9 个时间点</h3>
    <a href="{relative}/generated-frames.jpg"><img class="wide"
    src="{relative}/generated-frames.jpg" alt="九个按时间排序的视频抽帧"></a>
    <p>{html.escape(review['machine_evidence_zh'])}</p>{evidence}
    <h3>硬门与人工评分</h3><ul>{checks}</ul>
    <div class="pair"><table><tbody>{scores}
    <tr><th>加权总分</th><td>{review['weighted_score_100']}/100</td></tr>
    </tbody></table>
    <div class="note"><b>本轮结论：</b>
    {html.escape(review['verdict_zh'])}<br><br>
    <b>通用认识：</b>{html.escape(review['generalization_zh'])}</div></div>
    {prompt_block}
    <p>{evidence_links}
    <a href="{relative}/_work/run.json">逐帧机器审计</a> ·
    <a href="../../phase5_experiments/{experiment_id}/spec.json">实验规格</a> ·
    <a href="../../phase5_experiments/{experiment_id}/review.json">评审原文</a></p>
    </section>"""


def _write_report(records: list[dict[str, Any]]) -> None:
    status = _motion_status(records)
    phase_complete = all(
        item["status"] == "passed" for item in status
    )
    rows = "".join(
        f"<tr><td><code>{html.escape(item['motion_class'])}</code></td>"
        f"<td>{html.escape(item['case_id'])}</td>"
        f"<td>{'通过' if item['status'] == 'passed' else '待运行'}</td></tr>"
        for item in status
    )
    next_item = next(
        (item for item in status if item["status"] == "pending"),
        None,
    )
    next_text = (
        f"{next_item['motion_class']} / {next_item['case_id']}"
        if next_item
        else "Phase 6 release regression"
    )
    sections = "".join(_experiment_section(record) for record in records)
    video_model_runs = sum(
        record["run"]["model_runs"]["video"] for record in records
    )
    completion_note = (
        "Phase 5 已覆盖全部五种运动；下一步是 Phase 6 全量发布回归。"
        if phase_complete
        else "Phase 5 仍只覆盖部分运动类型，不能称作阶段完成。"
    )
    phase_label = "PASSED" if phase_complete else "IN PROGRESS"
    REPORT_PATH.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 2 · Phase 5 首尾帧视频工作流</title>
<style>
:root{{--ink:#18343b;--deep:#0d2830;--paper:#f3efe4;--card:#fffdf7;
--line:#bdcec9;--teal:#167d74;--orange:#d87636}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.68 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
header,main{{max-width:1160px;margin:auto;padding:38px 24px}}header{{padding-top:62px}}
h1{{font-size:clamp(2.2rem,6vw,4.8rem);line-height:1.03;margin:.12em 0}}
h2{{font-size:2rem;line-height:1.25}}h3{{margin-top:28px}}section{{
border-top:3px solid var(--ink);padding:44px 0}}.eyebrow{{color:var(--teal);
font-weight:800;letter-spacing:.11em}}video,.wide{{display:block;width:100%;
border-radius:13px;background:var(--deep)}}.pair{{display:grid;
grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}}figure{{margin:0;
background:var(--card);border:1px solid var(--line);border-radius:11px;overflow:hidden}}
figure img{{display:block;width:100%}}figcaption{{padding:10px 13px}}table{{width:100%;
border-collapse:collapse;background:var(--card)}}th,td{{border:1px solid var(--line);
padding:9px;text-align:left}}.note{{background:#e3f0eb;border-left:5px solid var(--teal);
padding:16px;border-radius:8px}}details{{background:var(--card);border:1px solid var(--line);
border-radius:8px;margin:10px 0}}summary{{padding:12px;font-weight:700}}pre{{margin:0;
padding:14px;white-space:pre-wrap;background:var(--deep);color:#edf7f3;overflow:auto}}
a{{color:var(--teal)}}@media(max-width:700px){{.pair{{grid-template-columns:1fr}}
header,main{{padding-left:14px;padding-right:14px}}}}
</style></head><body><header><p class="eyebrow">LOOP ENGINEER · PHASE 5 {phase_label}</p>
<h1>首尾帧只定状态，<br>视频模型只补中间运动</h1>
<p>本报告每完成一种运动类型就累积更新。当前已运行 {video_model_runs} 次 LTX-2.3
视频调用，覆盖 {sum(item['status'] == 'passed' for item in status)}/5 种运动。
首轮不是纯文本视频：程序生成的相邻 K0/K1 分别接入模型的第一帧和最后一帧。</p>
</header><main>
<section><h2>第一次看的人先理解这五步</h2>
<ol><li>程序先计算两个机制正确的相邻状态；</li>
<li>去掉箭头和文字，只把 clean 图作为首尾帧；</li>
<li>语言提示词只描述中间怎么动，以及什么必须不动；</li>
<li>LTX-2.3 生成中间帧，固定种子只用于复现运动；</li>
<li>端点、固定锚点、时间连续性和运动方向分别验收。</li></ol>
<p>低闪烁并不等于机制正确。例如波纹平滑地向内收缩仍然是失败，所以每类运动必须
有与其数据形态匹配的 evaluator。</p></section>
<section><h2>五种运动覆盖状态</h2><table><thead><tr><th>运动类型</th>
<th>代表案例</th><th>状态</th></tr></thead><tbody>{rows}</tbody></table>
<p>自动下一项：<b>{html.escape(next_text)}</b>。</p></section>
{sections}
<section><h2>模型、参数与已知未通过项</h2>
<table><tbody><tr><th>视频模型</th><td>LTX-2.3 22B dev FP8 +
distilled 1.1 LoRA</td></tr><tr><th>文本编码器</th><td>Gemma 3 12B IT FP4 mixed</td></tr>
<tr><th>首轮尺寸</th><td>576×320，24 fps，49 帧，约 2 秒</td></tr>
<tr><th>本阶段图片调用</th><td>0</td></tr><tr><th>当前视频调用</th>
<td>{video_model_runs}</td></tr></tbody></table>
<div class="note"><b>尚未通过发布门：</b>ComfyUI API 没有暴露 LTX 文本编码器的实际
token 数，因此目前只记录提示词字符数，不能声称 prompt token integrity 已通过。
{completion_note}</div>
<pre>/persistent/ComfyUI/start-ltx2.3.sh
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage2.phase5_experiment \\
  --experiment EXP-P5-20260729-001 --prepare
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage2.phase5_experiment \\
  --experiment EXP-P5-20260729-001 --generate
.venv/bin/python -m modules.video_model.stage2.phase5
.venv/bin/python -m modules.video_model.stage2.phase5 --check</pre>
<p><a href="phase5_manifest.json">机器可读 Phase 5 清单</a> ·
<a href="../phase-4/program-report.html">Phase 4 关键帧报告</a></p></section>
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


def _update_ledger(records: list[dict[str, Any]]) -> None:
    ledger = load_json(LEDGER_PATH)
    ids = {record["experiment_id"] for record in records}
    retained = [
        item
        for item in ledger["experiments"]
        if item["experiment_id"] not in ids
    ]
    additions = []
    for record in records:
        run = record["run"]
        additions.append(
            {
                "experiment_id": record["experiment_id"],
                "phase": 5,
                "case_id": run["case_id"],
                "primary_hypothesis": record["spec"]["hypothesis_zh"],
                "status": record["review"]["verdict"],
                "image_candidates": 0,
                "new_image_model_runs": 0,
                "reused_candidates": 0,
                "video_candidates": run["model_runs"]["video"],
                "model_ids": (
                    [
                        "LTX-2.3-22B-dev-FP8",
                        "LTX-2.3-distilled-1.1-LoRA",
                        "Gemma-3-12B-IT-FP4-mixed",
                    ]
                    if run["model_runs"]["video"]
                    else ["deterministic-program-plugin"]
                ),
                "output_manifest": (
                    f"output/phase-5/experiments/{record['experiment_id']}/"
                    "_work/run.json"
                ),
                "review": (
                    f"phase5_experiments/{record['experiment_id']}/"
                    "review.json"
                ),
            }
        )
    ledger["experiments"] = retained + additions
    write_json(LEDGER_PATH, ledger)


def build_phase5(*, check_only: bool = False) -> dict[str, Any]:
    if check_only:
        manifest = load_json(MANIFEST_PATH)
        for source in manifest["sources"].values():
            path = STAGE2_ROOT / source["path"]
            if sha256_path(path) != source["sha256"]:
                raise ValueError(f"Phase 5 source changed: {path}")
        for record in manifest["experiments"]:
            for key, base in (
                ("spec", STAGE2_ROOT),
                ("review", STAGE2_ROOT),
                ("run", STAGE2_ROOT),
            ):
                artifact = record[key]
                if sha256_path(base / artifact["path"]) != artifact["sha256"]:
                    raise ValueError(
                        f"Phase 5 evidence changed: {artifact['path']}"
                    )
        if sha256_path(REPORT_PATH) != manifest["report"]["sha256"]:
            raise ValueError("Phase 5 report changed")
        missing = _missing_links()
        if missing:
            raise ValueError(f"Phase 5 report links missing: {missing}")
        return manifest

    records = _experiments()
    if not records:
        raise ValueError("no reviewed Phase 5 experiment")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_report(records)
    _update_ledger(records)
    status = _motion_status(records)
    pending = [item for item in status if item["status"] == "pending"]
    sources = {
        name: {
            "path": path.relative_to(STAGE2_ROOT).as_posix(),
            "sha256": sha256_path(path),
        }
        for name, path in {
            "phase5_builder": Path(__file__),
            "generic_ltx_runner": STAGE2_ROOT / "framework/ltx_flf.py",
            "experiment_cli": STAGE2_ROOT / "phase5_experiment.py",
            "program_video_fallback": STAGE2_ROOT
            / "framework/program_video.py",
            "program_fallback_cli": STAGE2_ROOT
            / "phase5_program_fallback.py",
            "sentinel_program_plugins": STAGE2_ROOT
            / "cases/sentinel_programs.py",
            "remaining_program_plugins": STAGE2_ROOT
            / "cases/remaining_programs.py",
            "regression_policy": POLICY_PATH,
        }.items()
    }
    manifest = {
        "schema_version": "1.0",
        "phase": 5,
        "status": (
            "passed" if not pending else "motion_smoke_in_progress"
        ),
        "phase_complete": not pending,
        "motion_classes_passed": len(status) - len(pending),
        "motion_classes_total": len(status),
        "motion_status": status,
        "model_runs": {
            "image": 0,
            "video": sum(
                record["run"]["model_runs"]["video"]
                for record in records
            ),
        },
        "automatic_next_action": (
            "run_phase6_release_regression"
            if not pending
            else (
                "run_motion_smoke:"
                + pending[0]["motion_class"]
                + ":"
                + pending[0]["case_id"]
            )
        ),
        "known_release_blockers": [
            "video_prompt_token_integrity_not_exposed_by_comfyui_api"
        ],
        "sources": sources,
        "experiments": [
            {
                "experiment_id": record["experiment_id"],
                "case_id": record["run"]["case_id"],
                "motion_class": record["run"]["motion_class"],
                "verdict": record["review"]["verdict"],
                "spec": artifact_record(
                    record["spec_path"], STAGE2_ROOT
                ),
                "review": artifact_record(
                    record["review_path"], STAGE2_ROOT
                ),
                "run": artifact_record(
                    record["run_path"], STAGE2_ROOT
                ),
            }
            for record in records
        ],
        "report": artifact_record(REPORT_PATH, OUTPUT_ROOT),
    }
    write_json(MANIFEST_PATH, manifest)
    missing = _missing_links()
    if missing:
        raise ValueError(f"Phase 5 report links missing: {missing}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_phase5(check_only=args.check)
    print(
        f"Phase 5: {manifest['status']} · "
        f"{manifest['motion_classes_passed']}/"
        f"{manifest['motion_classes_total']} motion classes · "
        f"video_runs={manifest['model_runs']['video']} · "
        f"next={manifest['automatic_next_action']}"
    )


if __name__ == "__main__":
    main()
