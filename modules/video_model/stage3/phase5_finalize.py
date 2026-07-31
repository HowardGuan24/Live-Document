"""Finalize S3.5 decisions, durable records and newcomer-readable report."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    write_json,
)


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output" / "phase-5"
ASSETS = OUTPUT / "report-assets"
EXPERIMENTS = STAGE3 / "experiments"


def _href(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace(
        os.sep, "/"
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _gate(experiment_id: str) -> dict[str, Any]:
    return load_json(
        OUTPUT / "experiments" / experiment_id / "g4.json"
    )


def _check(gate: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in gate["checks"] if item["name"] == name)


def _controlled_variable_audit() -> dict[str, Any]:
    left = load_json(EXPERIMENTS / "EXP-S3-20260731-013/spec.json")
    right = load_json(EXPERIMENTS / "EXP-S3-20260731-014/spec.json")
    same_fields = [
        "case_id",
        "motion_class",
        "source",
        "model",
        "settings",
        "audit",
    ]
    checks = [
        {
            "name": f"{name}_identical",
            "passed": left[name] == right[name],
        }
        for name in same_fields
    ]
    checks.append(
        {
            "name": "prompt_is_only_semantic_input_difference",
            "passed": left["prompt"] != right["prompt"],
        }
    )
    value = {
        "schema_version": "1.0",
        "comparison": [
            "EXP-S3-20260731-013",
            "EXP-S3-20260731-014",
        ],
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "left_prompt_characters": {
            key: len(value) for key, value in left["prompt"].items()
        },
        "right_prompt_characters": {
            key: len(value) for key, value in right["prompt"].items()
        },
        "note_zh": (
            "experiment_id 和输出文件名前缀只用于隔离产物；两条模型调用的"
            "图像内容、模型、seed、采样、帧数和门禁相同。"
        ),
    }
    write_json(OUTPUT / "controlled-variable-audit.json", value)
    return value


def _chart() -> Path:
    chem_l0 = _check(
        _gate("EXP-S3-20260731-013"),
        "localized_scalar_mass_decays_without_regrowth",
    )
    chem_l1 = _check(
        _gate("EXP-S3-20260731-014"),
        "localized_scalar_mass_decays_without_regrowth",
    )
    phys_l1 = _check(
        _gate("EXP-S3-20260731-015"),
        "declared_sparse_checkpoint_frames_are_followed",
    )
    phys_l2 = _check(
        _gate("EXP-S3-20260731-016"),
        "declared_sparse_checkpoint_frames_are_followed",
    )
    width, height = 1500, 720
    image = Image.new("RGB", (width, height), (245, 242, 232))
    draw = ImageDraw.Draw(image)
    title = _font(32)
    label = _font(22)
    small = _font(18)
    dark = (18, 42, 46)
    draw.text((50, 36), "S3.5 mechanism measurements", fill=dark, font=title)

    graph_left, graph_top = 80, 125
    graph_width, graph_height = 600, 460
    draw.text(
        (graph_left, 86),
        "CHEM: pink pixels / initial pixels",
        fill=dark,
        font=label,
    )
    series = [
        (
            "L0 brief",
            chem_l0["colored_pixel_counts"],
            (45, 126, 150),
        ),
        (
            "L1 contract",
            chem_l1["colored_pixel_counts"],
            (220, 79, 105),
        ),
    ]
    max_value = max(max(values) for _, values, _ in series)
    for y_tick in (0.0, 0.5, 1.0, 1.5):
        y = graph_top + graph_height - y_tick / 1.55 * graph_height
        draw.line(
            (graph_left, y, graph_left + graph_width, y),
            fill=(205, 201, 190),
            width=1,
        )
        draw.text(
            (25, y - 10), f"{y_tick:.1f}", fill=dark, font=small
        )
    for name, values, color in series:
        initial = max(values[0], 1)
        points = []
        for index, value in enumerate(values):
            x = graph_left + index / (len(values) - 1) * graph_width
            y = (
                graph_top
                + graph_height
                - value / initial / 1.55 * graph_height
            )
            points.append((x, y))
        draw.line(points, fill=color, width=5)
    draw.line(
        (
            graph_left,
            graph_top + graph_height - 1.1 / 1.55 * graph_height,
            graph_left + graph_width,
            graph_top + graph_height - 1.1 / 1.55 * graph_height,
        ),
        fill=(187, 139, 47),
        width=3,
    )
    draw.text(
        (graph_left, 610),
        "Gate: never exceed 1.10× initial",
        fill=(145, 99, 20),
        font=small,
    )
    for index, (name, _, color) in enumerate(series):
        x = graph_left + index * 235
        draw.rectangle((x, 655, x + 28, 675), fill=color)
        draw.text((x + 38, 652), name, fill=dark, font=small)

    right = 820
    draw.text(
        (right, 86),
        "PHYS: error at accepted checkpoints (lower is better)",
        fill=dark,
        font=label,
    )
    l1 = [
        item["mean_absolute_pixel_error_0_255"]
        for item in phys_l1["records"]
    ]
    l2 = [
        item["mean_absolute_pixel_error_0_255"]
        for item in phys_l2["records"]
    ]
    labels = ["start", "mechanism", "result", "end"]
    max_bar = 15.0
    for index, (name, a, b) in enumerate(zip(labels, l1, l2)):
        y = 150 + index * 115
        draw.text((right, y + 15), name, fill=dark, font=small)
        x0 = right + 140
        draw.rectangle(
            (x0, y, x0 + a / max_bar * 430, y + 34),
            fill=(45, 126, 150),
        )
        draw.rectangle(
            (x0, y + 44, x0 + b / max_bar * 430, y + 78),
            fill=(103, 154, 92),
        )
        draw.text(
            (x0 + a / max_bar * 430 + 8, y + 7),
            f"{a:.2f}",
            fill=dark,
            font=small,
        )
        draw.text(
            (x0 + b / max_bar * 430 + 8, y + 50),
            f"{b:.2f}",
            fill=dark,
            font=small,
        )
    draw.rectangle((right, 655, right + 28, 675), fill=(45, 126, 150))
    draw.text((right + 38, 652), "L1 one span", fill=dark, font=small)
    draw.rectangle(
        (right + 250, 655, right + 278, 675), fill=(103, 154, 92)
    )
    draw.text(
        (right + 288, 652), "L2 segmented", fill=dark, font=small
    )
    path = ASSETS / "mechanism-metrics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _write_reviews() -> None:
    reviews = {
        "EXP-S3-20260731-013": {
            "verdict": "rejected",
            "reason_zh": (
                "L0 首尾和连续性通过，但粉色像素先增长到初始值的 "
                "1.135922 倍，超过预冻结 1.10 门禁。"
            ),
        },
        "EXP-S3-20260731-014": {
            "verdict": "rejected",
            "reason_zh": (
                "L1 结构化文本没有修复液体扩散，峰值反而达到 "
                "1.465116 倍；不能把更长提示词当作更强运动控制。"
            ),
        },
        "EXP-S3-20260731-015": {
            "verdict": "accepted_motion_default",
            "reason_zh": (
                "L1 保持两个固定点源，波纹从两中心向外扩展，首尾、"
                "连续性和中间合同帧误差门禁全部通过。"
            ),
        },
        "EXP-S3-20260731-016": {
            "verdict": "accepted_optional_not_default",
            "reason_zh": (
                "L2 把中间两帧 MAE 从 8.946/12.5005 降到 "
                "5.232/6.7277，但没有增加机制门禁通过数，并在分段"
                "边界出现更大的 4.668/5.5116 像素跳变；不升级默认。"
            ),
        },
        "EXP-S3-20260731-017": {
            "verdict": "rejected",
            "reason_zh": (
                "首尾虽然吻合，但中间帧让红、蓝、青三角形面积最大偏差"
                "达到 1.0/0.890255/0.260371，视觉上出现消失、融合和"
                "整套布局闪回，违反刚体身份合同。"
            ),
        },
    }
    for experiment_id, review in reviews.items():
        gate_path = (
            OUTPUT / "experiments" / experiment_id / "g4.json"
        )
        write_json(
            EXPERIMENTS / experiment_id / "review.json",
            {
                "schema_version": "1.0",
                "experiment_id": experiment_id,
                **review,
                "model_runs": {
                    "image_candidates": 0,
                    "video_candidates": 1,
                },
                "evidence": {
                    "g4": file_record(gate_path, REPO_ROOT),
                },
            },
        )

    final_id = "EXP-S3-20260731-018"
    root = EXPERIMENTS / final_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "hypothesis.md").write_text(
        "# EXP-S3-20260731-018\n\n"
        "假设：按运动类型选择最低充分引导，并在失败时显式回退，"
        "可以形成不依赖临场抽卡的通用 S3.5 决策表。\n",
        encoding="utf-8",
    )
    write_json(
        root / "review.json",
        {
            "schema_version": "1.0",
            "experiment_id": final_id,
            "verdict": "accepted_core",
            "reason_zh": (
                "三种运动类型均得到确定默认：连续场 L1；液体混合和"
                "精确刚体在本模型失败时走确定性程序回退；L2 只作为"
                "中间合同帧收益明确时的可选级别；L3 在当前接口禁用。"
            ),
            "model_runs": {
                "image_candidates": 0,
                "video_candidates": 0,
            },
            "evidence": {
                "g4_summary": file_record(
                    OUTPUT / "g4-summary.json", REPO_ROOT
                ),
                "decisions": file_record(
                    OUTPUT / "guidance-decisions.json", REPO_ROOT
                ),
            },
        },
    )


def _decisions() -> dict[str, Any]:
    value = {
        "schema_version": "1.0",
        "policy_id": "S3.5-MOTION-DEFAULTS-V1",
        "selection_rule": (
            "Choose the lowest level that passes every G4 hard gate. "
            "Promote a higher level only when it adds a mechanism benefit "
            "without a quality regression."
        ),
        "defaults": [
            {
                "motion_class": "liquid_mixing",
                "default": "deterministic_program_animation",
                "reason_zh": (
                    "L0 和 L1 都先放大显色量再消散；文本条件不足以控制"
                    "逐帧浓度单调性。"
                ),
                "evidence": [
                    "EXP-S3-20260731-013",
                    "EXP-S3-20260731-014",
                ],
            },
            {
                "motion_class": "continuous_field_propagation",
                "default": "L1",
                "optional_upgrade": (
                    "L2 only when accepted intermediate checkpoints are "
                    "contract-critical and seam checks pass"
                ),
                "reason_zh": (
                    "L1 已通过全部机制门禁；L2 改善中间帧误差但没有"
                    "新增机制通过项，并增加接缝跳变。"
                ),
                "evidence": [
                    "EXP-S3-20260731-015",
                    "EXP-S3-20260731-016",
                ],
            },
            {
                "motion_class": "rigid_motion_exact_identity",
                "default": "deterministic_program_animation",
                "reason_zh": (
                    "L1 中间过程丢失或融合对象；首尾帧相似不能证明"
                    "对象身份和面积守恒。"
                ),
                "evidence": [
                    "EXP-S3-20260731-017",
                    "EXP-P5-20260729-002",
                    "EXP-P5-20260729-003",
                ],
            },
        ],
        "L3": {
            "status": "unsupported",
            "reason_zh": (
                "当前 ComfyUI 工作流只在 frame_idx=0 和 frame_idx=-1 "
                "加入图像 Guide；没有程序视频、轨迹、mask 或运动场接口。"
            ),
        },
    }
    write_json(OUTPUT / "guidance-decisions.json", value)
    return value


def _update_durable_records() -> None:
    ledger = load_json(EXPERIMENTS / "ledger.json")
    new_ids = {
        f"EXP-S3-20260731-{index:03d}" for index in range(13, 19)
    }
    ledger["experiments"] = [
        item
        for item in ledger["experiments"]
        if item["experiment_id"] not in new_ids
    ]
    verdicts = {
        "013": "rejected",
        "014": "rejected",
        "015": "accepted_motion_default",
        "016": "accepted_optional_not_default",
        "017": "rejected",
        "018": "accepted_core",
    }
    for suffix, verdict in verdicts.items():
        experiment_id = f"EXP-S3-20260731-{suffix}"
        ledger["experiments"].append(
            {
                "experiment_id": experiment_id,
                "hypothesis_id": "H-S3-0005A",
                "phase": "S3.5",
                "verdict": verdict,
                "model_runs": {
                    "image_candidates": 0,
                    "video_candidates": (
                        0 if suffix == "018" else 1
                    ),
                },
                "review": (
                    f"modules/video_model/stage3/experiments/"
                    f"{experiment_id}/review.json"
                ),
            }
        )
    write_json(EXPERIMENTS / "ledger.json", ledger)

    patterns_path = STAGE3 / "knowledge/failure_patterns.json"
    patterns = load_json(patterns_path)
    additions = [
        {
            "id": "FP-VIDEO-LIQUID-001",
            "taxonomy": "motion_guidance",
            "symptom_zh": "液体显色最后消失，但中途先变大或分裂成额外色块。",
            "diagnosis_zh": (
                "FLF 模型满足首尾像素，不等于遵守浓度或质量的逐帧单调约束。"
            ),
            "forbidden_fix_zh": (
                "不得只看尾帧或放宽峰值门禁；用逐帧标量质量检查，"
                "失败则回退确定性状态渲染。"
            ),
        },
        {
            "id": "FP-VIDEO-RIGID-001",
            "taxonomy": "motion_guidance",
            "symptom_zh": "首尾四块拼图正确，中间却闪回、融合、消失或重生。",
            "diagnosis_zh": (
                "首尾帧扩散视频没有原生对象 ID、刚体面积或轨迹约束。"
            ),
            "forbidden_fix_zh": (
                "不得以端点 MAE 通过替代对象身份门禁；精确刚体失败时"
                "保留程序动画。"
            ),
        },
        {
            "id": "FP-VIDEO-SEGMENT-001",
            "taxonomy": "motion_guidance",
            "symptom_zh": "稀疏分段提高中间帧忠实度，但边界出现速度或相位接缝。",
            "diagnosis_zh": "多个独立 FLF 片段共享关键帧像素，却不共享隐状态或速度。",
            "forbidden_fix_zh": (
                "不得只报告中间帧误差；同时检查边界相邻帧跳变，"
                "没有机制增益时维持较低指导级别。"
            ),
        },
    ]
    ids = {item["id"] for item in patterns["patterns"]}
    patterns["patterns"].extend(
        item for item in additions if item["id"] not in ids
    )
    write_json(patterns_path, patterns)

    hypotheses = STAGE3 / "knowledge/hypotheses.jsonl"
    existing = hypotheses.read_text(encoding="utf-8")
    hypothesis_id = "H-S3-0005A"
    if hypothesis_id not in existing:
        record = {
            "hypothesis_id": hypothesis_id,
            "phase": "S3.5",
            "status": "partially_supported",
            "statement_zh": (
                "结构化 motion contract 对连续场足够；但它没有改善液体"
                "标量单调性和精确刚体身份，稀疏分段也不是自动默认。"
            ),
            "evidence": "modules/video_model/stage3/output/phase-5/guidance-decisions.json",
        }
        with hypotheses.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    problems_path = STAGE3 / "knowledge/open_problems.json"
    problems = load_json(problems_path)
    problems["problems"] = [
        item
        for item in problems["problems"]
        if item["problem_id"] != "S3-PROBLEM-MOTION-001"
    ]
    release_problem = {
        "problem_id": "S3-PROBLEM-RELEASE-001",
        "taxonomy": "cross_discipline_release",
        "summary_zh": (
            "S3.6 仍需把已冻结的 G0–G4 流程跑过五学科代表和历史回归，"
            "并发布版本 manifest。"
        ),
    }
    if not any(
        item["problem_id"] == release_problem["problem_id"]
        for item in problems["problems"]
    ):
        problems["problems"].append(release_problem)
    write_json(problems_path, problems)

    accepted_path = STAGE3 / "baselines/accepted.json"
    accepted = load_json(accepted_path)
    new_records = [
        {
            "baseline_id": "CORE-MOTION-COMPILER-V1",
            "kind": "accepted_core",
            **file_record(STAGE3 / "framework/motion.py", REPO_ROOT),
        },
        {
            "baseline_id": "MOTION-GUIDANCE-POLICY-V1",
            "kind": "accepted_core_config",
            **file_record(STAGE3 / "motion_guidance.json", REPO_ROOT),
        },
        {
            "baseline_id": "MOTION-DEFAULTS-S3.5-V1",
            "kind": "accepted_core_config",
            **file_record(
                OUTPUT / "guidance-decisions.json", REPO_ROOT
            ),
        },
        {
            "baseline_id": "VIDEO-PHYS-01-L1-S3.5-V1",
            "kind": "accepted_video_transition",
            **file_record(
                OUTPUT
                / "experiments/EXP-S3-20260731-015/transition.mp4",
                REPO_ROOT,
            ),
        },
    ]
    ids = {item["baseline_id"] for item in accepted["records"]}
    accepted["records"].extend(
        item for item in new_records if item["baseline_id"] not in ids
    )
    write_json(accepted_path, accepted)

    state = load_json(STAGE3 / "state.json")
    state.update(
        {
            "phase": "S3.6",
            "phase_status": "in_progress",
            "current_problem": {
                "problem_id": "S3-PROBLEM-RELEASE-001",
                "taxonomy": "cross_discipline_release",
                "summary_zh": (
                    "运行五学科、三角洲和 Phase 9 烧杯回归，并发布"
                    "可复现版本。"
                ),
            },
            "current_hypothesis": {
                "hypothesis_id": "H-S3-0006A",
                "statement_zh": (
                    "冻结的合同、几何、候选选择、State Renderer B 和"
                    "按运动类型分流策略可以通过跨学科发布回归。"
                ),
                "falsification_zh": (
                    "任何代表案例或历史基线哈希/硬门禁失败，都停止发布"
                    "并回到对应模块，而不是修报告。"
                ),
            },
            "current_cohort": {
                "target": "cross_discipline_release",
                "regressions": [
                    "MATH-02",
                    "PHYS-01",
                    "CHEM-01",
                    "BIO-01",
                    "GEO-02",
                    "GEO-HIST-DELTA-01",
                    "CHEM-01-PHASE9",
                ],
            },
            "budget": {
                "image_candidate_limit": 0,
                "video_candidate_limit_per_guidance_level": 0,
                "preflight_before_gpu_work": True,
            },
            "exit_criteria": [
                "all frozen baseline records resolve",
                "five discipline representatives pass contract smoke",
                "delta and Phase 9 visual lineage regressions resolve",
                "version manifest, changelog and newcomer report are published",
            ],
            "next_action": (
                "Run S3.6 zero-model release regression before deciding "
                "whether any new model call is justified."
            ),
        }
    )
    write_json(STAGE3 / "state.json", state)


def _report(controlled: dict[str, Any], metrics_chart: Path) -> Path:
    report = OUTPUT / "report.html"
    out = report.parent

    def h(path: Path) -> str:
        return html.escape(_href(out, path))

    ids = [
        "EXP-S3-20260731-013",
        "EXP-S3-20260731-014",
        "EXP-S3-20260731-015",
        "EXP-S3-20260731-016",
        "EXP-S3-20260731-017",
    ]
    previews = {
        item: OUTPUT / "experiments" / item / "generated-frames.jpg"
        for item in ids
    }
    videos = {
        item: OUTPUT / "experiments" / item / "transition.mp4"
        for item in ids
    }
    config = load_json(STAGE3 / "motion_guidance.json")
    chem_l0 = _check(
        _gate(ids[0]), "localized_scalar_mass_decays_without_regrowth"
    )
    chem_l1 = _check(
        _gate(ids[1]), "localized_scalar_mass_decays_without_regrowth"
    )
    phys_l1 = _check(
        _gate(ids[2]), "declared_sparse_checkpoint_frames_are_followed"
    )
    phys_l2 = _check(
        _gate(ids[3]), "declared_sparse_checkpoint_frames_are_followed"
    )
    math = _check(
        _gate(ids[4]), "four_colored_rigid_identities_preserve_area"
    )
    chem_contract = (
        OUTPUT / "contracts/CHEM-01/01_mechanism__02_result.json"
    )
    phys_contract = OUTPUT / "contracts/PHYS-01/00_start__03_end.json"
    math_contract = (
        OUTPUT / "contracts/MATH-02/00_start__01_mechanism.json"
    )
    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 3 · S3.5 Motion Contract 与视频引导</title>
<style>
:root{{--ink:#102a2f;--muted:#52676b;--paper:#f4f0e5;--card:#fffdf7;--teal:#2d7e96;--green:#679a5c;--red:#dc4f69;--gold:#bb8b2f;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:17px/1.7 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
main{{max-width:1220px;margin:auto;padding:42px 24px 90px}} h1{{font-size:42px;line-height:1.15;margin:.1em 0 .3em}} h2{{margin-top:2.3em;font-size:29px}} h3{{font-size:22px;margin-top:1.8em}}
.lead{{font-size:21px;max-width:980px}} .card{{background:var(--card);border:1px solid #d8d1bf;border-radius:15px;padding:22px 25px;margin:18px 0;box-shadow:0 5px 18px #2333}}
.ok{{border-left:7px solid var(--green)}} .bad{{border-left:7px solid var(--red)}} .warn{{border-left:7px solid var(--gold)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}} .grid .card{{margin:0}}
table{{width:100%;border-collapse:collapse;background:var(--card)}} th,td{{padding:12px 13px;border:1px solid #d8d1bf;vertical-align:top;text-align:left}} th{{background:#e8e3d5}}
img,video{{width:100%;border-radius:10px;background:#14292d}} video{{aspect-ratio:16/9}} code,pre{{font-family:ui-monospace,SFMono-Regular,monospace}} pre{{white-space:pre-wrap;background:#102a2f;color:#edf6f0;padding:17px;border-radius:10px;overflow:auto}}
.flow{{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:10px;align-items:center;text-align:center}} .flow div{{background:#e2ece8;padding:16px 10px;border-radius:10px}} .arrow{{font-size:28px;color:var(--teal)}}
.metric{{font-size:27px;font-weight:750}} .muted{{color:var(--muted)}} a{{color:#16677d}} .tag{{display:inline-block;padding:2px 9px;border-radius:20px;background:#dce9e5;margin-right:6px}}
@media(max-width:760px){{.flow{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg)}}h1{{font-size:34px}}}}
</style></head><body><main>
<p class="muted">Live-Document · Stage 3 · S3.5 · 2026-07-31</p>
<h1>什么运动能交给视频模型，什么不能</h1>
<p class="lead">本阶段不是挑一条“看着顺”的视频，而是用同一套逐帧门禁确定最低充分引导。
结论是：<b>连续水波用 L1 就够；液体浓度单调变化和精确刚体身份不能只靠当前首尾帧模型，
必须回退确定性程序运动。</b> L2 能提高中间帧吻合度，但不是自动默认。</p>

<div class="card ok"><b>阶段结果：通过。</b>5 个固定视频候选、7 次模型调用（L2 由 3 段组成）；
2 个通过全部 G4，3 个作为失败证据保留；8 项仓库测试通过。通过指“默认分流规则已确定”，
不是说所有视频模型候选都成功。</div>

<h2>1. 先讲清模型实际接收什么</h2>
<p>当前部署的是 <b>LTX‑2.3 22B First/Last‑Frame to Video</b>。22B 表示约 220 亿参数；
FP8 checkpoint 是较低精度的模型权重文件。Gemma 文本编码器把提示词变成模型可用的向量；
LTX 的两个 <code>LTXVAddGuide</code> 节点只在 <code>frame_idx=0</code> 和
<code>frame_idx=-1</code> 放入首帧和尾帧。它没有原生的程序视频、对象轨迹、mask、
运动场或中间帧入口。</p>
<div class="flow">
<div>程序 states.jsonl<br><small>运动事实</small></div><span class="arrow">→</span>
<div>Motion Contract<br><small>对象、顺序、趋势、静止项</small></div><span class="arrow">→</span>
<div>Prompt Compiler<br><small>变成正/负文本</small></div>
</div>
<div class="flow" style="margin-top:12px">
<div>首帧 PNG<br><small>Guide at frame 0</small></div><span class="arrow">→</span>
<div>LTX‑2.3 去噪采样<br><small>文本 + 首尾图</small></div><span class="arrow">←</span>
<div>尾帧 PNG<br><small>Guide at last frame</small></div>
</div>
<p>所以 Motion Contract <b>不是</b>一种隐藏的 ControlNet，也不会返回逐帧残差。它有两个用途：
一是编译更明确的语言；二是生成模型输出后逐帧验收。模型能忽略语言中的“单调”
或“对象身份”，G4 就必须拒绝。</p>

<h3>四个引导等级</h3>
<table><tr><th>等级</th><th>模型拿到的输入</th><th>本阶段含义</th></tr>
<tr><td>L0</td><td>首帧 + 尾帧 + 一句简短语言</td><td>语言基线</td></tr>
<tr><td>L1</td><td>同一首尾帧 + 合同编译文本</td><td>对象清单、事件顺序、趋势、静止项</td></tr>
<tr><td>L2</td><td>多个相邻 L1 首尾帧片段再拼接</td><td>这是分段 FLF，不冒充原生中间帧控制</td></tr>
<tr><td>L3</td><td>程序视频/轨迹/运动场</td><td><b>当前接口不支持，禁用</b></td></tr></table>

<h2>2. 实验如何做到可复现</h2>
<p>所有规格在模型调用前写入 <a href="{h(STAGE3/'motion_guidance.json')}">motion_guidance.json</a>，
预检见 <a href="{h(OUTPUT/'preflight.json')}">preflight.json</a>。CHEM 的 L0/L1
严格单变量检查为 <b>{str(controlled['passed']).lower()}</b>：
输入图片、模型、seed、采样器、分辨率、帧数和门禁逐项相同，只改变提示词。</p>
<table><tr><th>固定项</th><th>值</th><th>它控制什么</th></tr>
<tr><td>模型</td><td>LTX‑2.3 22B FP8 + distilled LoRA 0.5</td><td>视频生成器及加速适配权重</td></tr>
<tr><td>画面</td><td>576×320，24 fps</td><td>输出像素尺寸与播放帧率</td></tr>
<tr><td>guide_strength</td><td>0.7</td><td>首尾图片对首尾 latent 的约束强度</td></tr>
<tr><td>CFG</td><td>1.0</td><td>文本条件的 classifier-free guidance 比例</td></tr>
<tr><td>image_compression</td><td>25</td><td>输入图在 LTX 预处理中的压缩级别</td></tr>
<tr><td>noise seed</td><td>2026073105；L2 后两段依次 +1</td><td>固定扩散噪声起点；不是事后抽卡</td></tr>
<tr><td>候选预算</td><td>每级 1 个</td><td>失败后不追加 seed 直到碰巧通过</td></tr></table>
<p>完整 ComfyUI API 工作流、提示词和模型指纹都在每个实验目录的 <code>_work/</code>。
模型文件指纹不是模型名称猜测，而是实际部署文件的 SHA‑256 记录。</p>

<h2>3. CHEM‑01：语言没有控制住浓度单调性</h2>
<p>相邻帧从“一个粉色液滴和局部粉色羽流”到“无色溶液”。硬门禁不只看最后是否无色，
还要求粉色质量不能先增长。合同见 <a href="{h(chem_contract)}">01_mechanism__02_result.json</a>。</p>
<div class="grid">
<div class="card bad"><h3>L0 简短语言</h3><video controls muted loop preload="metadata" poster="{h(previews[ids[0]])}"><source src="{h(videos[ids[0]])}" type="video/mp4"></video>
<p><span class="metric">{chem_l0['peak_fraction_of_initial']:.3f}×</span><br>中途粉色峰值；门禁 ≤ 1.10×。</p>
<p><a href="{h(previews[ids[0]])}">九帧过程图</a> · <a href="{h(OUTPUT/'experiments'/ids[0]/'g4.json')}">逐帧 G4</a></p></div>
<div class="card bad"><h3>L1 结构化合同</h3><video controls muted loop preload="metadata" poster="{h(previews[ids[1]])}"><source src="{h(videos[ids[1]])}" type="video/mp4"></video>
<p><span class="metric">{chem_l1['peak_fraction_of_initial']:.3f}×</span><br>更长提示词反而形成更大的竖直粉色柱。</p>
<p><a href="{h(previews[ids[1]])}">九帧过程图</a> · <a href="{h(OUTPUT/'experiments'/ids[1]/'g4.json')}">逐帧 G4</a></p></div>
</div>
<div class="card warn"><b>决定：</b>当前 LTX 路线不发布。首尾 MAE 和连续性都通过，但标量质量门禁失败；
默认使用 <a href="{h(OUTPUT/'fallbacks/CHEM-01/program-motion.mp4')}">确定性程序运动回退</a>。
回退保机制、不是 S3.4 写实外观，报告不把它包装成最终成片。</div>

<h2>4. PHYS‑01：L1 已足够，L2 更准但有接缝成本</h2>
<p>长段从 60 px 波前到 165 px 波前，两个橙色点源必须固定。合同见
<a href="{h(phys_contract)}">00_start__03_end.json</a>。L1 一次生成 73 帧；L2 用
00→01、01→02、02→03 三个 25 帧片段，删除重复边界帧后仍为 73 帧。</p>
<div class="grid">
<div class="card ok"><h3>L1 单段（默认）</h3><video controls muted loop preload="metadata" poster="{h(previews[ids[2]])}"><source src="{h(videos[ids[2]])}" type="video/mp4"></video>
<p>点源最大漂移 0.877 px；两个波前测得向外进展 68/69 px。</p>
<p><a href="{h(previews[ids[2]])}">九帧过程图</a> · <a href="{h(OUTPUT/'experiments'/ids[2]/'g4.json')}">G4</a></p></div>
<div class="card warn"><h3>L2 稀疏分段（可选）</h3><video controls muted loop preload="metadata" poster="{h(previews[ids[3]])}"><source src="{h(videos[ids[3]])}" type="video/mp4"></video>
<p>中间帧 MAE：{phys_l2['records'][1]['mean_absolute_pixel_error_0_255']:.2f} /
{phys_l2['records'][2]['mean_absolute_pixel_error_0_255']:.2f}，优于 L1 的
{phys_l1['records'][1]['mean_absolute_pixel_error_0_255']:.2f} /
{phys_l1['records'][2]['mean_absolute_pixel_error_0_255']:.2f}；但边界跳变为 4.668/5.512。</p>
<p><a href="{h(previews[ids[3]])}">九帧过程图</a> · <a href="{h(OUTPUT/'experiments'/ids[3]/'g4.json')}">G4</a> ·
<a href="{h(OUTPUT/'experiments'/ids[3]/'_work/assembly.json')}">拼接记录</a></p></div>
</div>
<div class="card ok"><b>决定：</b>两者都通过机制门禁，但按“最低充分”规则选择 L1。
只有教学合同要求精确经过中间状态、且接缝门禁仍通过时才升级 L2。</div>

<h2>5. MATH‑02：首尾正确不代表四块刚体真的移动</h2>
<p>合同要求四块三角形的颜色、面积、身份和对象局部木纹贯穿运动。合同见
<a href="{h(math_contract)}">00_start__01_mechanism.json</a>。</p>
<div class="card bad"><video controls muted loop preload="metadata" poster="{h(previews[ids[4]])}"><source src="{h(videos[ids[4]])}" type="video/mp4"></video>
<p>模型在中间反复把“左侧四块”和“右侧完整拼图”闪回/融合。四种颜色最大面积偏差为
{html.escape(str(math['maximum_area_deviation_fraction_by_identity']))}，门禁是每块 ≤ 0.12。</p>
<p><a href="{h(previews[ids[4]])}">九帧过程图</a> · <a href="{h(OUTPUT/'experiments'/ids[4]/'g4.json')}">G4</a> ·
<a href="{h(OUTPUT/'fallbacks/MATH-02/program-motion.mp4')}">确定性程序回退</a></p></div>
<p>这也复现了 Stage 2 两次 LTX 刚体失败：扩散视频模型有像素连续性，但当前接口没有对象 ID、
解析面积和刚体轨迹条件。默认回退不是“模型不够好看”，而是模型没有该类硬约束。</p>

<h2>6. 一张图看完量化结果</h2>
<img src="{h(metrics_chart)}" alt="CHEM 粉色质量曲线与 PHYS 中间帧误差对比">

<h2>7. 发布给后续 Case 的确定规则</h2>
<table><tr><th>运动类型</th><th>默认</th><th>为什么</th></tr>
<tr><td>连续场传播</td><td><b>L1</b></td><td>水波方向、固定点源和中间状态门禁全部通过；L2 没有新增机制通过项。</td></tr>
<tr><td>复杂路径且中间状态是硬合同</td><td>L2 可选</td><td>只在中间帧误差显著改善、且边界跳变仍合格时升级。</td></tr>
<tr><td>液体混合/浓度单调性</td><td>确定性程序运动</td><td>L0/L1 均出现中途标量反向增长。</td></tr>
<tr><td>精确刚体对象身份</td><td>确定性程序运动</td><td>模型中间丢失、融合或重生对象。</td></tr>
<tr><td>程序视频/运动场直连</td><td>unsupported</td><td>部署工作流没有这一输入，不能伪造。</td></tr></table>
<p>机器可读版本：<a href="{h(OUTPUT/'guidance-decisions.json')}">guidance-decisions.json</a>；
失败回退来源与哈希：<a href="{h(OUTPUT/'fallbacks/manifest.json')}">fallbacks/manifest.json</a>。</p>

<h2>8. 从零复现</h2>
<pre>cd {html.escape(str(REPO_ROOT))}

# 1) 只做规格、输入、模型文件和工作流预检；不调用模型
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase5 --prepare

# 2) 启动已部署的 ComfyUI
/persistent/ComfyUI/start-ltx2.3.sh

# 3) 运行五个冻结候选；已存在的视频会保留，不重复调用
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase5 --generate

# 4) 逐帧 G4、回退清单和报告
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase5 --audit --fallbacks
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase5_finalize

# 5) 不依赖视频环境的仓库测试
/opt/venv/bin/python -m pytest -q modules/video_model/stage3/tests</pre>

<h2>9. 没有隐藏掉的限制</h2>
<ul>
<li>G4 的水波“向外传播”量表检查总体半径和固定源，不宣称逐像素恢复完整波相位。</li>
<li>L2 三段使用不同固定 seed（主 seed 依次 +0/+1/+2）；这是预先冻结的分段规格，不是失败后抽卡。</li>
<li>确定性回退目前来自程序视觉，不继承 S3.4 的完整写实材质；这是后续“全状态 B 渲染视频”的明确工程缺口。</li>
<li>本阶段只覆盖三种代表运动类型；未把结论偷换成所有十一个 Case 都已发布。</li>
</ul>
<p class="muted">下一步已自动进入 S3.6：先做零模型跨学科发布回归，再决定是否存在新模型调用的正当理由。</p>
</main></body></html>"""
    report.write_text(body, encoding="utf-8")
    return report


def _manifest(report: Path, chart: Path) -> None:
    artifacts = [
        report,
        chart,
        OUTPUT / "preflight.json",
        OUTPUT / "controlled-variable-audit.json",
        OUTPUT / "g4-summary.json",
        OUTPUT / "guidance-decisions.json",
        OUTPUT / "fallbacks/manifest.json",
    ]
    for experiment_id in (
        "EXP-S3-20260731-013",
        "EXP-S3-20260731-014",
        "EXP-S3-20260731-015",
        "EXP-S3-20260731-016",
        "EXP-S3-20260731-017",
    ):
        artifacts.extend(
            [
                OUTPUT / "experiments" / experiment_id / "transition.mp4",
                OUTPUT / "experiments" / experiment_id / "g4.json",
                OUTPUT
                / "experiments"
                / experiment_id
                / "generated-frames.jpg",
            ]
        )
    write_json(
        OUTPUT / "phase5_manifest.json",
        {
            "schema_version": "1.0",
            "phase": "S3.5",
            "status": "passed_with_motion_class_routing",
            "artifacts": [
                file_record(path, REPO_ROOT) for path in artifacts
            ],
            "model_runs": {
                "image_candidates": 0,
                "video_candidates": 5,
                "video_model_calls": 7,
            },
            "tests": {
                "command": (
                    "/opt/venv/bin/python -m pytest -q "
                    "modules/video_model/stage3/tests"
                ),
                "result": "8 passed",
            },
        },
    )


def main() -> None:
    controlled = _controlled_variable_audit()
    _decisions()
    _write_reviews()
    _update_durable_records()
    chart = _chart()
    report = _report(controlled, chart)
    _manifest(report, chart)
    print(f"S3.5 finalized: {report}")


if __name__ == "__main__":
    main()
