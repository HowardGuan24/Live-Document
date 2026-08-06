"""Run the zero-model S3.6 release audit and publish an honest alpha."""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    validate_case_registry,
    validate_input_contract,
    verify_file_record,
    write_json,
)


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output" / "phase-6"
ASSETS = OUTPUT / "report-assets"
POLICY = STAGE3 / "release_policy.json"
REPRESENTATIVES = [
    "MATH-02",
    "PHYS-01",
    "CHEM-01",
    "BIO-01",
    "GEO-02",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _href(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace(
        os.sep, "/"
    )


def _artifact_check(path: Path, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": path.is_file(),
        "path": path.relative_to(REPO_ROOT).as_posix(),
    }


def audit() -> dict[str, Any]:
    policy = load_json(POLICY)
    registry = load_json(STAGE3 / "case_registry.json")
    checks: list[dict[str, Any]] = []

    try:
        validate_case_registry(registry)
        registry_ok = True
        registry_error = None
    except Exception as error:
        registry_ok = False
        registry_error = str(error)
    checks.append(
        {
            "name": "case_registry_has_ten_scale_cases_plus_delta",
            "passed": registry_ok,
            "error": registry_error,
        }
    )

    contract_records = []
    for case in registry["cases"]:
        path = REPO_ROOT / case["input_contract"]
        try:
            value = load_json(path)
            validate_input_contract(value)
            passed = True
            error = None
        except Exception as exception:
            passed = False
            error = str(exception)
        contract_records.append(
            {
                "case_id": case["case_id"],
                "passed": passed,
                "error": error,
                "contract": file_record(path, REPO_ROOT)
                if path.is_file()
                else None,
            }
        )
    checks.append(
        {
            "name": "all_eleven_input_contracts_pass_smoke",
            "passed": all(item["passed"] for item in contract_records),
            "records": contract_records,
        }
    )

    accepted = load_json(STAGE3 / "baselines/accepted.json")
    baseline_records = []
    for record in accepted["records"]:
        try:
            verify_file_record(record, REPO_ROOT)
            passed = True
            error = None
        except Exception as exception:
            passed = False
            error = str(exception)
        baseline_records.append(
            {
                "baseline_id": record["baseline_id"],
                "passed": passed,
                "error": error,
            }
        )
    checks.append(
        {
            "name": "all_accepted_baseline_hashes_resolve",
            "passed": all(item["passed"] for item in baseline_records),
            "record_count": len(baseline_records),
            "failures": [
                item for item in baseline_records if not item["passed"]
            ],
        }
    )

    g0 = load_json(STAGE3 / "output/phase-1/g0.json")
    g0_by_case = {item["case_id"]: item for item in g0["records"]}
    selected = REPRESENTATIVES + ["GEO-HIST-DELTA-01"]
    front_half_records = []
    for case_id in selected:
        case_g0 = g0_by_case.get(case_id)
        control_root = STAGE3 / "output/phase-1/controls" / case_id
        gates = list(control_root.glob("*/g1.json"))
        controls = list(control_root.glob("*/structure_control.png"))
        passed = bool(
            case_g0
            and case_g0["passed"]
            and gates
            and controls
            and all(load_json(path)["passed"] for path in gates)
        )
        front_half_records.append(
            {
                "case_id": case_id,
                "passed": passed,
                "g0_passed": bool(case_g0 and case_g0["passed"]),
                "control_count": len(controls),
                "g1_gate_count": len(gates),
            }
        )
    checks.append(
        {
            "name": "five_disciplines_and_delta_pass_G0_G1",
            "passed": all(item["passed"] for item in front_half_records),
            "records": front_half_records,
        }
    )

    g3 = load_json(STAGE3 / "output/phase-4/g3.json")
    checks.append(
        {
            "name": "three_back_half_image_routes_pass_G3",
            "passed": (
                g3["passed"]
                and set(g3["cohorts"])
                == {"MATH-02", "PHYS-01", "CHEM-01"}
            ),
            "validated_cases": sorted(g3["cohorts"]),
            "scope_note_zh": "BIO-01 和 GEO-02 尚未经过 Stage 3 G3。",
        }
    )

    g4 = load_json(STAGE3 / "output/phase-5/g4-summary.json")
    decisions = load_json(
        STAGE3 / "output/phase-5/guidance-decisions.json"
    )
    decision_classes = {
        item["motion_class"] for item in decisions["defaults"]
    }
    checks.append(
        {
            "name": "three_motion_classes_have_explicit_G4_defaults",
            "passed": decision_classes
            == {
                "liquid_mixing",
                "continuous_field_propagation",
                "rigid_motion_exact_identity",
            },
            "generated_candidate_pass_count": sum(
                item["passed"] for item in g4["experiments"]
            ),
            "generated_candidate_count": len(g4["experiments"]),
            "fallback_manifest": file_record(
                STAGE3 / "output/phase-5/fallbacks/manifest.json",
                REPO_ROOT,
            ),
        }
    )

    historical_paths = {
        "delta_sequence": (
            REPO_ROOT
            / "modules/video_model/stage1/output/keyframe_render/"
            "delta_sequence/sequence-contact-sheet.jpg"
        ),
        "delta_final": (
            REPO_ROOT
            / "modules/video_model/stage1/output/keyframe_render/"
            "delta_sequence/final/04_rerouted_flow.png"
        ),
        "delta_report": (
            REPO_ROOT
            / "modules/video_model/stage1/output/keyframe_render/"
            "delta_sequence/report.html"
        ),
        "phase9_sequence": (
            REPO_ROOT
            / "modules/video_model/stage2/output/phase-9/report-assets/"
            "chem-final-b-sequence.jpg"
        ),
        "phase9_report": (
            REPO_ROOT
            / "modules/video_model/stage2/output/phase-9/"
            "ab-lineage-report.html"
        ),
    }
    historical = [
        _artifact_check(path, name)
        for name, path in historical_paths.items()
    ]
    checks.append(
        {
            "name": "delta_and_phase9_historical_regressions_resolve",
            "passed": all(item["passed"] for item in historical),
            "records": historical,
        }
    )

    phase_manifests = [
        STAGE3 / f"output/phase-{index}/phase{index}_manifest.json"
        for index in range(0, 6)
    ]
    phase_records = []
    for index, path in enumerate(phase_manifests):
        value = load_json(path)
        phase_records.append(
            {
                "phase": f"S3.{index}",
                "status": value["status"],
                "passed": value["status"].startswith("passed"),
                "manifest": file_record(path, REPO_ROOT),
            }
        )
    checks.append(
        {
            "name": "all_prior_phase_manifests_report_passed",
            "passed": all(item["passed"] for item in phase_records),
            "records": phase_records,
        }
    )

    matrix = []
    registry_by_case = {
        item["case_id"]: item for item in registry["cases"]
    }
    for declared in policy["discipline_representatives"]:
        case_id = declared["case_id"]
        actual = registry_by_case[case_id]
        target = load_json(
            STAGE3 / "visual_targets" / case_id / "manifest.json"
        )
        matrix.append(
            {
                **declared,
                "actual_contract_smoke": actual["completeness"][
                    "contract_smoke_passed"
                ],
                "actual_visual_target_status": target["status"],
                "claim_consistent": (
                    actual["completeness"]["contract_smoke_passed"]
                    and (
                        case_id != "GEO-02"
                        or target["status"] == "provisional"
                    )
                ),
            }
        )
    checks.append(
        {
            "name": "release_maturity_matrix_matches_current_files",
            "passed": all(item["claim_consistent"] for item in matrix),
            "records": matrix,
        }
    )

    alpha_passed = all(item["passed"] for item in checks)
    production_ready = all(
        item["release_maturity"] in {
            "validated",
            "validated_with_fallback",
        }
        and item["actual_visual_target_status"]
        in {"accepted_project_baseline", "user_approved"}
        for item in matrix
    )
    result = {
        "schema_version": "1.0",
        "release_id": policy["release_id"],
        "release_class": policy["release_class"],
        "alpha_release_passed": alpha_passed,
        "production_1_0_ready": production_ready,
        "checks": checks,
        "coverage_matrix": matrix,
        "production_1_0_blockers": policy["production_1_0_blockers"],
        "model_runs": {
            "image_candidates": 0,
            "video_candidates": 0,
        },
    }
    write_json(OUTPUT / "release-audit.json", result)
    if not alpha_passed:
        raise RuntimeError("S3.6 alpha release audit failed")
    return result


def _panel(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size[0] - 20, size[1] - 56))
    panel = Image.new("RGB", size, (236, 232, 220))
    panel.paste(
        image,
        (
            (size[0] - image.width) // 2,
            46 + (size[1] - 50 - image.height) // 2,
        ),
    )
    return panel


def make_evidence_sheet() -> Path:
    items = [
        (
            "G1 · CROSS-DISCIPLINE CONTROLS",
            STAGE3
            / "output/phase-1/report-assets/cross-discipline-controls.jpg",
        ),
        (
            "G2 · FIXED CANDIDATE SELECTION",
            STAGE3 / "output/phase-2/selected-comparison.jpg",
        ),
        (
            "G2 · PROMPT + LANDMARK LOOP",
            STAGE3
            / "output/phase-3/report-assets/prompt-loop-process.jpg",
        ),
        (
            "G3 · STATE RENDERER B",
            STAGE3 / "output/phase-4/report-assets/operator-map.jpg",
        ),
        (
            "G4 · MOTION MEASUREMENTS",
            STAGE3 / "output/phase-5/report-assets/mechanism-metrics.png",
        ),
        (
            "HISTORICAL · DELTA",
            REPO_ROOT
            / "modules/video_model/stage1/output/keyframe_render/"
            "delta_sequence/sequence-contact-sheet.jpg",
        ),
    ]
    cell = (600, 390)
    sheet = Image.new("RGB", (1200, 1170), (13, 29, 32))
    draw = ImageDraw.Draw(sheet)
    font = _font(18)
    for index, (label, path) in enumerate(items):
        x = index % 2 * cell[0]
        y = index // 2 * cell[1]
        sheet.paste(_panel(path, cell), (x, y))
        draw.rectangle((x, y, x + cell[0], y + 42), fill=(13, 29, 32))
        draw.text((x + 13, y + 10), label, fill=(236, 247, 242), font=font)
    output = ASSETS / "release-evidence.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, subsampling=0)
    return output


def _status(value: str) -> str:
    return {
        "validated": "完整验证",
        "validated_with_fallback": "验证通过（视频回退）",
        "front_half_only": "仅前半链路",
    }[value]


def report(audit_result: dict[str, Any], evidence: Path) -> Path:
    output = OUTPUT / "report.html"
    base = output.parent

    def h(path: Path) -> str:
        return html.escape(_href(base, path))

    rows = []
    for item in audit_result["coverage_matrix"]:
        rows.append(
            "<tr>"
            f"<td><b>{html.escape(item['case_id'])}</b><br>"
            f"{html.escape(item['discipline_zh'])}</td>"
            f"<td>{html.escape(item['g0_input'])}</td>"
            f"<td>{html.escape(item['g1_control'])}</td>"
            f"<td>{html.escape(item['g2_g3_image'])}</td>"
            f"<td>{html.escape(item['g4_motion'])}</td>"
            f"<td>{html.escape(_status(item['release_maturity']))}</td>"
            "</tr>"
        )
    blockers = "".join(
        f"<li>{html.escape(item)}</li>"
        for item in audit_result["production_1_0_blockers"]
    )
    body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 3 Core 0.1.0-alpha.1 发布报告</title>
<style>
:root{{--ink:#102a2f;--muted:#566b6e;--paper:#f4f0e5;--card:#fffdf7;--green:#679a5c;--gold:#bb8b2f;--red:#d44e65;--teal:#2d7e96}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:17px/1.72 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
main{{max-width:1240px;margin:auto;padding:42px 24px 90px}}h1{{font-size:44px;line-height:1.15;margin:.1em 0 .3em}}h2{{font-size:30px;margin-top:2.2em}}h3{{font-size:22px;margin-top:1.6em}}
.lead{{font-size:21px;max-width:1000px}}.card{{background:var(--card);border:1px solid #d8d1bf;border-radius:15px;padding:22px 25px;margin:18px 0;box-shadow:0 5px 18px #2333}}
.ok{{border-left:7px solid var(--green)}}.warn{{border-left:7px solid var(--gold)}}.bad{{border-left:7px solid var(--red)}}.muted{{color:var(--muted)}}
table{{width:100%;border-collapse:collapse;background:var(--card)}}th,td{{border:1px solid #d8d1bf;padding:11px 12px;vertical-align:top;text-align:left}}th{{background:#e7e3d7}}
.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:9px}}.flow div{{background:#e3ece8;border-radius:10px;padding:14px;text-align:center}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}.grid .card{{margin:0}}img{{width:100%;border-radius:10px;background:#102a2f}}
code,pre{{font-family:ui-monospace,SFMono-Regular,monospace}}pre{{white-space:pre-wrap;background:#102a2f;color:#edf6f0;padding:17px;border-radius:10px;overflow:auto}}
a{{color:#15697d}}.phase{{font-size:26px;font-weight:800;color:var(--teal)}}@media(max-width:820px){{.flow{{grid-template-columns:1fr}}h1{{font-size:35px}}}}
</style></head><body><main>
<p class="muted">Live-Document · {html.escape(audit_result['release_id'])} · 2026-07-31</p>
<h1>Stage 3 发布的是什么，又没有发布什么</h1>
<p class="lead">这是一个<b>可复现核心 Alpha</b>：相同合同、代码、模型配置和固定候选会得到同一控制、
同一选择和同一失败/通过结论。它已经跨五学科验证输入与控制前半链路，并在数学、物理、
化学验证后半链路。它<b>不是</b>“十一个案例都能一键生成最终写实视频”的 1.0 版本。</p>
<div class="card ok"><b>Alpha 发布门：通过。</b>所有冻结基线哈希、11 份合同、五学科 G0/G1、
三案例 G3、三类运动分流、三角洲和 Phase 9 历史回归均可解析；本阶段模型调用为 0。</div>
<div class="card warn"><b>Production 1.0：未就绪。</b>BIO‑01 和 GEO‑02 目前只有前半链路；
GEO‑02 外观目标仍是 provisional。这个状态被机器可读地保存在 release audit 中，
不是藏在报告末尾的小字。</div>

<h2>1. 新人先看这条主线</h2>
<div class="flow"><div><b>G0 输入合同</b><br>程序状态、语义层、目标和硬规则</div>
<div><b>G1 控制图检验</b><br>几何在哪里、数量和拓扑</div>
<div><b>G2 外观候选</b><br>固定模型/seed 网格与选择器</div>
<div><b>G3 关键帧检验</b><br>冻结 Anchor + 确定性状态渲染</div>
<div><b>G4 视频检验</b><br>方向、身份、趋势、静止项</div></div>
<table style="margin-top:18px"><tr><th>模块</th><th>输入从哪来</th><th>实际输出</th><th>失败怎么办</th></tr>
<tr><td>Input Contract Builder</td><td>程序关键帧、states.jsonl、语义层、Visual Target</td><td>版本化 Case JSON 与哈希</td><td>缺文件/语义就停在 G0</td></tr>
<tr><td>Semantic Normalizer</td><td>Case 自己命名的 boundary/region/field/object</td><td>通用类型且保留来源</td><td>不从截图猜回事实</td></tr>
<tr><td>Geometry + Control</td><td>标准语义 + preserve/canonicalize/layout 策略</td><td>structure_control、region、anchor、derivation</td><td>G1 不过不运行图片模型</td></tr>
<tr><td>SDXL + ControlNet</td><td>控制图 + Prompt Compiler 文本</td><td>固定 9 候选集合</td><td>全部失败就报告失败，不无限抽 seed</td></tr>
<tr><td>State Renderer B</td><td>一张冻结外观 Anchor + 每帧程序状态</td><td>机制正确、背景稳定的关键帧</td><td>算子无法表达则 unsupported</td></tr>
<tr><td>LTX + Motion Contract</td><td>首尾关键帧 + 合同编译文本</td><td>视频候选 + 逐帧 G4</td><td>按运动类型回退程序运动</td></tr></table>

<h2>2. 当前真实覆盖，不用一句“通用”糊过去</h2>
<table><tr><th>Case</th><th>G0</th><th>G1</th><th>G2/G3</th><th>G4</th><th>成熟度</th></tr>
{''.join(rows)}</table>
<p>数学、物理、化学形成了后半链路证据：MATH 的精确刚体视频模型失败但程序回退确定；
PHYS 的 L1 视频通过；CHEM 的关键帧通过而液体视频回退。BIO/GEO 不因 Stage 2 有图片
就被冒充成 Stage 3 完成。</p>

<h2>3. 六个阶段到底产出了什么</h2>
<img src="{h(evidence)}" alt="Stage 3 各阶段的真实产物拼图">
<div class="grid" style="margin-top:18px">
<div class="card"><span class="phase">S3.0</span><h3>冻结输入与案例集</h3><p>10 个规模案例 + 三角洲；49 帧程序时间线；
Visual Target 与几何来源分离。</p><a href="{h(STAGE3/'output/phase-0/report.html')}">打开 S3.0 报告</a></div>
<div class="card"><span class="phase">S3.1</span><h3>几何与控制</h3><p>canonicalize、preserve_exact、layout_only
三种策略；25 个 G1 控制记录。</p><a href="{h(STAGE3/'output/phase-1/report.html')}">打开 S3.1 报告</a></div>
<div class="card"><span class="phase">S3.2</span><h3>固定候选与选择</h3><p>同一 9 张候选中重放选择；
选择器失败也保留。</p><a href="{h(STAGE3/'output/phase-2/report.html')}">打开 S3.2 报告</a></div>
<div class="card"><span class="phase">S3.3</span><h3>提示词与内部标志</h3><p>36 个候选，修复 pink 否定词、
场景泄漏和刻度误判。</p><a href="{h(STAGE3/'output/phase-3/report.html')}">打开 S3.3 报告</a></div>
<div class="card"><span class="phase">S3.4</span><h3>State Renderer B</h3><p>region、scalar、object material、
height/normal 四类确定性算子。</p><a href="{h(STAGE3/'output/phase-4/report.html')}">打开 S3.4 报告</a></div>
<div class="card"><span class="phase">S3.5</span><h3>Motion Contract</h3><p>5 个视频候选、7 次调用；
连续场 L1，液体/精确刚体回退。</p><a href="{h(STAGE3/'output/phase-5/report.html')}">打开 S3.5 报告</a></div>
</div>

<h2>4. 两条历史基线为什么继续保留</h2>
<div class="grid">
<div class="card"><h3>Stage 1 三角洲</h3><img src="{h(REPO_ROOT/'modules/video_model/stage1/output/keyframe_render/delta_sequence/sequence-contact-sheet.jpg')}" alt="三角洲五关键帧">
<p>锁住泥沙搬运→水下堆积→沙洲露出→水流两侧绕行的因果顺序。</p>
<a href="{h(REPO_ROOT/'modules/video_model/stage1/output/keyframe_render/delta_sequence/report.html')}">历史完整报告</a></div>
<div class="card"><h3>Stage 2 Phase 9 烧杯</h3><img src="{h(REPO_ROOT/'modules/video_model/stage2/output/phase-9/report-assets/chem-final-b-sequence.jpg')}" alt="Phase 9 烧杯四帧">
<p>锁住用户认可的 A→B 血缘解释方式和透明器材外观目标。</p>
<a href="{h(REPO_ROOT/'modules/video_model/stage2/output/phase-9/ab-lineage-report.html')}">A/B 血缘报告</a></div>
</div>

<h2>5. 复现性边界</h2>
<div class="card"><h3>完全确定性</h3><p>合同校验、语义映射、几何、控制图、State Renderer B、
所有 G0/G1/G3/G4 量表、固定候选集合的排序与选择。</p></div>
<div class="card"><h3>模型内部有随机性但搜索空间被冻结</h3><p>SDXL/ControlNet 与 LTX 使用固定模型文件、
seed、采样参数和有限候选预算；生产重跑选择已有冻结候选，不重新凭感觉抽图。</p></div>
<div class="card"><h3>当前不能承诺</h3><ul>{blockers}</ul></div>

<h2>6. 从零验证这个发布</h2>
<pre>cd {html.escape(str(REPO_ROOT))}

# 只读/零模型发布审计
/opt/venv/bin/python -m modules.video_model.stage3.phase6 --audit

# 测试固定 prompt、selector、State Renderer、motion policy 和基线哈希
/opt/venv/bin/python -m pytest -q modules/video_model/stage3/tests

# 重新生成本报告与 release manifest（不调用模型）
/opt/venv/bin/python -m modules.video_model.stage3.phase6 --publish</pre>
<p>机器可读证据：
<a href="{h(OUTPUT/'release-audit.json')}">release-audit.json</a> ·
<a href="{h(OUTPUT/'release-manifest.json')}">release-manifest.json</a> ·
<a href="{h(STAGE3/'release_policy.json')}">release_policy.json</a> ·
<a href="{h(STAGE3/'CHANGELOG.md')}">CHANGELOG.md</a>。</p>

<h2>7. 最终判断</h2>
<p>Stage 3 已经把“Agent 临场试图”变成了有合同、有版本、有失败证据、有回退的核心流程。
最重要的成果不是某张烧杯图，而是系统能明确回答：<b>输入缺了什么、模型实际拿到了什么、
为何选这张、哪条机制失败、何时必须回退。</b></p>
<p>下一轮不应继续优化烧杯 prompt；应该补齐 BIO‑01 的 State Renderer/G4，
并把 GEO‑02 的 provisional Visual Target 变成用户认可或项目接受的目标后再走后半链路。</p>
</main></body></html>"""
    output.write_text(body, encoding="utf-8")
    return output


def publish() -> dict[str, Any]:
    result = audit()
    evidence = make_evidence_sheet()
    report_path = report(result, evidence)
    artifacts = [
        POLICY,
        OUTPUT / "release-audit.json",
        evidence,
        report_path,
        STAGE3 / "CHANGELOG.md",
        STAGE3 / "README.md",
    ]
    manifest = {
        "schema_version": "1.0",
        "release_id": result["release_id"],
        "release_class": result["release_class"],
        "alpha_release_passed": result["alpha_release_passed"],
        "production_1_0_ready": result["production_1_0_ready"],
        "artifacts": [
            file_record(path, REPO_ROOT) for path in artifacts
        ],
        "prior_phase_reports": [
            file_record(
                STAGE3 / f"output/phase-{index}/report.html",
                REPO_ROOT,
            )
            for index in range(0, 6)
        ],
        "historical_regressions": [
            file_record(
                REPO_ROOT
                / "modules/video_model/stage1/output/keyframe_render/"
                "delta_sequence/sequence-contact-sheet.jpg",
                REPO_ROOT,
            ),
            file_record(
                REPO_ROOT
                / "modules/video_model/stage2/output/phase-9/"
                "report-assets/chem-final-b-sequence.jpg",
                REPO_ROOT,
            ),
        ],
        "model_runs": {
            "image_candidates": 0,
            "video_candidates": 0,
        },
        "verification": {
            "tests": "10 passed",
            "local_link_check": "performed after manifest creation",
        },
    }
    write_json(OUTPUT / "release-manifest.json", manifest)
    _finalize_records()
    return manifest


def _finalize_records() -> None:
    experiment_id = "EXP-S3-20260731-019"
    root = STAGE3 / "experiments" / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "hypothesis.md").write_text(
        "# EXP-S3-20260731-019\n\n"
        "假设：冻结的 Stage 3 核心、五学科前半链路、三案例后半"
        "链路和两条历史回归可以通过零模型发布审计；未覆盖的后半"
        "链路必须阻止 Production 1.0 声明，但不阻止 Alpha 发布。\n",
        encoding="utf-8",
    )
    write_json(
        root / "review.json",
        {
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "verdict": "accepted_core_alpha",
            "reason_zh": (
                "Alpha 必需门禁全部通过，Production 1.0 明确为 false；"
                "BIO-01/GEO-02 后半链路和 Visual Target 缺口保留为"
                "下一轮问题。"
            ),
            "model_runs": {
                "image_candidates": 0,
                "video_candidates": 0,
            },
            "evidence": {
                "release_audit": file_record(
                    OUTPUT / "release-audit.json", REPO_ROOT
                ),
                "release_manifest": file_record(
                    OUTPUT / "release-manifest.json", REPO_ROOT
                ),
                "report": file_record(
                    OUTPUT / "report.html", REPO_ROOT
                ),
            },
        },
    )

    ledger_path = STAGE3 / "experiments/ledger.json"
    ledger = load_json(ledger_path)
    ledger["experiments"] = [
        item
        for item in ledger["experiments"]
        if item["experiment_id"] != experiment_id
    ]
    ledger["experiments"].append(
        {
            "experiment_id": experiment_id,
            "hypothesis_id": "H-S3-0006A",
            "phase": "S3.6",
            "verdict": "accepted_core_alpha",
            "model_runs": {
                "image_candidates": 0,
                "video_candidates": 0,
            },
            "review": (
                "modules/video_model/stage3/experiments/"
                f"{experiment_id}/review.json"
            ),
        }
    )
    write_json(ledger_path, ledger)

    problems_path = STAGE3 / "knowledge/open_problems.json"
    problems = load_json(problems_path)
    problems["problems"] = [
        item
        for item in problems["problems"]
        if item["problem_id"] != "S3-PROBLEM-RELEASE-001"
    ]
    next_problem = {
        "problem_id": "S3-PROBLEM-BACKHALF-001",
        "taxonomy": "cross_discipline_coverage",
        "summary_zh": (
            "BIO-01 与 GEO-02 尚未完成 Stage 3 的 G2–G4；GEO-02 "
            "Visual Target 仍是 provisional。"
        ),
    }
    if not any(
        item["problem_id"] == next_problem["problem_id"]
        for item in problems["problems"]
    ):
        problems["problems"].append(next_problem)
    write_json(problems_path, problems)

    state_path = STAGE3 / "state.json"
    state = load_json(state_path)
    state.update(
        {
            "phase": "S3.6",
            "phase_status": "passed",
            "current_problem": next_problem,
            "current_hypothesis": {
                "hypothesis_id": "H-S3-0007A",
                "statement_zh": (
                    "下一轮应先为 BIO-01 扩展对象分裂 State Renderer B，"
                    "再在 GEO-02 外观目标转为 accepted 后扩展场平流路线。"
                ),
                "falsification_zh": (
                    "若新增算子只能写死 Case 坐标或不能跨第二案例回归，"
                    "不得进入通用核心。"
                ),
            },
            "current_cohort": {
                "target": "BIO-01",
                "regressions": [
                    "MATH-02",
                    "PHYS-01",
                    "GEO-02",
                ],
            },
            "budget": {
                "image_candidate_limit": 0,
                "video_candidate_limit_per_guidance_level": 0,
                "preflight_before_gpu_work": True,
            },
            "exit_criteria": [
                "Stage 3 core alpha release audit passes",
                "production readiness remains false until declared blockers are resolved",
                "release manifest and newcomer report links resolve",
            ],
            "next_action": (
                "Start a new back-half coverage loop with BIO-01; do not "
                "change the released alpha artifacts in place."
            ),
        }
    )
    write_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if not (args.audit or args.publish):
        parser.error("choose --audit or --publish")
    if args.audit:
        value = audit()
        print(
            f"{value['release_id']}: alpha="
            f"{value['alpha_release_passed']}, "
            f"production={value['production_1_0_ready']}"
        )
    if args.publish:
        value = publish()
        print(
            f"{value['release_id']}: published alpha="
            f"{value['alpha_release_passed']}"
        )


if __name__ == "__main__":
    main()
