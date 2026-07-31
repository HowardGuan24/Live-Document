"""Finalize S3.3 evidence, state, knowledge and beginner-readable report."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage2.framework.image_experiment import (
    _token_preflight,
)
from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    write_json,
)
from modules.video_model.stage3.framework.prompt import compile_prompt


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output" / "phase-3"
ASSETS = OUTPUT / "report-assets"
EXPERIMENTS = STAGE3 / "experiments"
MODEL_ROOT = Path(
    "/workspace/ai-concept-animator/.cache/models/sdxl-base-1.0"
)


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def href(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace(
        os.sep, "/"
    )


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size[0] - 24, size[1] - 64))
    panel = Image.new("RGB", size, (235, 232, 221))
    panel.paste(
        image,
        ((size[0] - image.width) // 2, 48 + (size[1] - 60 - image.height) // 2),
    )
    return panel


def contact_sheet(
    panels: list[tuple[str, Path]], output: Path, columns: int = 3
) -> None:
    size = (520, 350)
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (size[0] * columns, size[1] * rows), (14, 29, 32)
    )
    draw = ImageDraw.Draw(sheet)
    label_font = font(18)
    for index, (label, path) in enumerate(panels):
        x = (index % columns) * size[0]
        y = (index // columns) * size[1]
        panel = thumbnail(path, size)
        sheet.paste(panel, (x, y))
        draw.rectangle((x, y, x + size[0], y + 42), fill=(14, 29, 32))
        draw.text((x + 14, y + 11), label, fill=(235, 247, 242), font=label_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, subsampling=0)


def landmark_panel(path: Path, expected_y: list[int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((483, 39, 512, 161), outline=(255, 187, 51), width=3)
    for y in expected_y:
        draw.line((483, y, 512, y), fill=(255, 82, 82), width=2)
    return image


def make_assets() -> dict[str, Path]:
    clean = (
        REPO_ROOT
        / "modules/video_model/stage2/output/phase-2/CHEM-01/"
        "keyframes/00_start/clean.png"
    )
    control = (
        OUTPUT.parent
        / "phase-1/controls/CHEM-01/00_start/structure_control.png"
    )
    paths = {
        "process": ASSETS / "prompt-loop-process.jpg",
        "landmarks": ASSETS / "landmark-audit.jpg",
    }
    contact_sheet(
        [
            ("1 PROGRAM FRAME", clean),
            ("2 STRUCTURE CONTROL", control),
            (
                "3 V1 REJECTED: COLOR LEAK",
                OUTPUT / "selection/selected.png",
            ),
            (
                "4 V2 REJECTED: NO GATE PASS",
                OUTPUT
                / "experiments/EXP-S3-20260730-006/raw/"
                "auto_control_080/seed_7103.png",
            ),
            (
                "5 V3 REJECTED: TICKS LOST",
                OUTPUT / "selection-v3/selected.png",
            ),
            (
                "6 V4 ACCEPTED CASE ANCHOR",
                OUTPUT / "selection-v4-position-audit/selected.png",
            ),
        ],
        paths["process"],
    )
    gate = load_json(
        OUTPUT.parent
        / "phase-1/controls/CHEM-01/00_start/g1.json"
    )
    expected_y = gate["required_internal_landmarks"][0][
        "expected_group_center_y"
    ]
    panels = [
        (
            "CONTROL: EXPECTED POSITIONS",
            landmark_panel(control, expected_y),
        ),
        (
            "FALSE PASS: REFLECTION BANDS",
            landmark_panel(
                OUTPUT
                / "experiments/EXP-S3-20260730-008/raw/"
                "auto_control_080/seed_7102.png",
                expected_y,
            ),
        ),
        (
            "FINAL: 5 POSITION MATCHES",
            landmark_panel(
                OUTPUT / "selection-v4-position-audit/selected.png",
                expected_y,
            ),
        ),
    ]
    size = (520, 350)
    sheet = Image.new("RGB", (1560, 350), (14, 29, 32))
    draw = ImageDraw.Draw(sheet)
    label_font = font(18)
    for index, (label, image) in enumerate(panels):
        image.thumbnail((496, 286))
        x = index * 520
        panel = Image.new("RGB", size, (235, 232, 221))
        panel.paste(image, ((520 - image.width) // 2, 52))
        sheet.paste(panel, (x, 0))
        draw.rectangle((x, 0, x + 520, 42), fill=(14, 29, 32))
        draw.text((x + 14, 11), label, fill=(235, 247, 242), font=label_font)
    sheet.save(paths["landmarks"], quality=92, subsampling=0)
    return paths


def compile_v5_cross_case() -> dict[str, Any]:
    records = {}
    for case_id in ("CHEM-01", "MATH-02", "PHYS-01"):
        prompt = compile_prompt(
            STAGE3 / "contracts" / f"{case_id}.json",
            STAGE3 / "visual_targets" / case_id / "manifest.json",
            STAGE3 / "prompt_lexicon_v5.json",
            OUTPUT / "prompts-v5" / case_id,
        )
        tokens = _token_preflight(
            MODEL_ROOT,
            prompt["positive_prompt"],
            prompt["negative_prompt"],
        )
        prompt["token_preflight"] = tokens
        write_json(
            OUTPUT / "prompts-v5" / case_id / "prompt_manifest.json",
            prompt,
        )
        records[case_id] = {
            "compiler_id": prompt["compiler_id"],
            "appearance_profile": prompt["appearance_profile"],
            "positive_tokens": tokens["positive"][
                "counts_including_special_tokens"
            ],
            "negative_tokens": tokens["negative"][
                "counts_including_special_tokens"
            ],
            "would_truncate": any(
                item["would_truncate"] for item in tokens.values()
            ),
            "provenance_slots": sorted(prompt["provenance"]),
        }
    result = {
        "schema_version": "1.0",
        "passed": all(
            item["compiler_id"] == "stage3_prompt_compiler_v2"
            and not item["would_truncate"]
            and len(item["provenance_slots"]) == 6
            for item in records.values()
        ),
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
        "records": records,
        "scope_zh": (
            "证明编译器接口和来源追踪跨三种几何/材质 profile 工作；"
            "不声称 MATH-02、PHYS-01 的扩散图像质量已经验证。"
        ),
    }
    write_json(OUTPUT / "prompt-v5-cross-case-compile.json", result)
    return result


def write_review(
    experiment_id: str,
    hypothesis_zh: str,
    verdict: str,
    reason_zh: str,
    model_runs: dict[str, int],
    evidence: dict[str, Any],
    failure_taxonomy: str | None = None,
) -> None:
    root = EXPERIMENTS / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "hypothesis.md").write_text(
        f"# {experiment_id}\n\n假设：{hypothesis_zh}\n",
        encoding="utf-8",
    )
    review = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "verdict": verdict,
        "reason_zh": reason_zh,
        "model_runs": model_runs,
        "evidence": evidence,
    }
    if failure_taxonomy:
        review["failure_taxonomy"] = failure_taxonomy
    write_json(root / "review.json", review)


def update_records(cross_case: dict[str, Any]) -> None:
    v1_audit = {
        "passed": False,
        "reason_zh": (
            "正向 prompt 出现 no pink；扩散模型仍会注意 pink 这个 token，"
            "九张候选全部产生粉色液体/光照。"
        ),
    }
    write_json(OUTPUT / "visual-audit-v1.json", v1_audit)
    v2_selection = load_json(OUTPUT / "selection-v2-summary.json")
    v3_selection = load_json(OUTPUT / "selection-v3-summary.json")
    v4_selection = load_json(
        OUTPUT / "selection-v4-position-audit/selection.json"
    )
    final_audit = {
        "schema_version": "1.0",
        "passed": True,
        "candidate_id": v4_selection["selected_candidate_id"],
        "checks": [
            {
                "name": "neutral_initial_state",
                "passed": True,
                "evidence_zh": "液体和背景不再被禁用颜色词染粉。",
            },
            {
                "name": "one_separate_beaker_and_burette",
                "passed": True,
                "evidence_zh": "两个器材分离，滴定管尖端与杯沿保留空气间隙。",
            },
            {
                "name": "identity_landmark_visible",
                "passed": True,
                "evidence_zh": "管身可见至少五组与 primitive 预期位置匹配的刻度。",
            },
            {
                "name": "empty_stable_background",
                "passed": True,
                "evidence_zh": "没有额外烧瓶、标签、箭头或连接杆。",
            },
        ],
        "limitations_zh": [
            "滴定管的整体比例仍受 S3.1 canonical primitive 限制，不是完整实验室长滴定管。",
            "v4 九张中只有一张通过最终刻度门禁；因此该词表只作为 CHEM-01 案例配置，不升级为所有 Case 的通用视觉结论。",
            "S3.3 验证的是 Prompt Compiler 的确定性与可追溯性；多关键帧状态一致性留给 S3.4。",
        ],
    }
    write_json(OUTPUT / "visual-audit-final.json", final_audit)

    write_review(
        "EXP-S3-20260730-005",
        "可追溯 Prompt Compiler 在不改控制和 seed 时能直接改善外观。",
        "rejected",
        "否定颜色词被写入正向条件，九张图全部发粉。",
        {"image_candidates": 9, "video_candidates": 0},
        {"visual_audit": file_record(OUTPUT / "visual-audit-v1.json", REPO_ROOT)},
        "prompt_polarity",
    )
    write_review(
        "EXP-S3-20260730-006",
        "禁用颜色只进入 negative conditioning 后可恢复中性初始态。",
        "rejected",
        "颜色污染消失，但冻结 selector 的 eligible_count 为 0；不得事后放宽阈值。",
        {"image_candidates": 9, "video_candidates": 0},
        {
            "eligible_count": v2_selection["eligible_count"],
            "selection": file_record(
                OUTPUT / "selection-v2-summary.json", REPO_ROOT
            ),
        },
        "no_candidate_passed",
    )
    write_review(
        "EXP-S3-20260730-007",
        "收窄为无杂物摄影背景可减少额外器材并增加玻璃对比。",
        "rejected",
        "旧 selector 选出一张干净图，但独立语义复核发现滴定管刻度丢失。",
        {"image_candidates": 9, "video_candidates": 0},
        {
            "old_selected": v3_selection["selected_candidate_id"],
            "landmark_replay": file_record(
                OUTPUT
                / "selection-v3-landmark-audit/selection.json",
                REPO_ROOT,
            ),
        },
        "semantic_landmark_missing",
    )
    write_review(
        "EXP-S3-20260730-008",
        "把 provider 声明的刻度部件写入正向 prompt，固定集合中可出现身份完整候选。",
        "accepted_case_specific",
        "最终门禁在九张中保留一张；适合作为 CHEM-01 Anchor，但不足以证明跨案例通用提升。",
        {"image_candidates": 9, "video_candidates": 0},
        {
            "selected_candidate_id": v4_selection[
                "selected_candidate_id"
            ],
            "final_audit": file_record(
                OUTPUT / "visual-audit-final.json", REPO_ROOT
            ),
        },
    )
    write_review(
        "EXP-S3-20260730-009",
        "内部标志必须匹配 primitive 声明的位置，不能只统计 ROI 中的横线数量。",
        "accepted_core",
        "同一 EXP-008 候选回放中，反光带误判被拒绝，真实刻度候选成为唯一 eligible；未新增模型图。",
        {"image_candidates": 0, "video_candidates": 0},
        {
            "selection": file_record(
                OUTPUT
                / "selection-v4-position-audit/selection.json",
                REPO_ROOT,
            ),
            "selector": file_record(
                STAGE3 / "selector_v4.json", REPO_ROOT
            ),
        },
    )
    write_review(
        "EXP-S3-20260730-010",
        "Prompt Compiler 的 profile 路由和保留条件可完全来自版本数据，而不是 Case 分支。",
        "accepted_core",
        "v5 在 CHEM-01、MATH-02、PHYS-01 编译出六槽来源记录并通过双 tokenizer 截断预检；本实验不运行模型。",
        {"image_candidates": 0, "video_candidates": 0},
        {"cross_case_compile": cross_case},
    )

    ledger = load_json(EXPERIMENTS / "ledger.json")
    ledger["experiments"] = [
        item
        for item in ledger["experiments"]
        if item["experiment_id"]
        not in {
            "EXP-S3-20260730-005",
            "EXP-S3-20260730-006",
            "EXP-S3-20260730-007",
            "EXP-S3-20260730-008",
            "EXP-S3-20260730-009",
            "EXP-S3-20260730-010",
        }
    ]
    verdicts = {
        "005": ("H-S3-0003A", "rejected", 9, "prompt_polarity"),
        "006": ("H-S3-0003B", "rejected", 9, "no_candidate_passed"),
        "007": ("H-S3-0003C", "rejected", 9, "semantic_landmark_missing"),
        "008": ("H-S3-0003D", "accepted_case_specific", 9, None),
        "009": ("H-S3-0003E", "accepted_core", 0, None),
        "010": ("H-S3-0003F", "accepted_core", 0, None),
    }
    for suffix, (hypothesis_id, verdict, runs, failure) in verdicts.items():
        item = {
            "experiment_id": f"EXP-S3-20260730-{suffix}",
            "hypothesis_id": hypothesis_id,
            "phase": "S3.3",
            "verdict": verdict,
            "model_runs": {
                "image_candidates": runs,
                "video_candidates": 0,
            },
            "review": (
                "modules/video_model/stage3/experiments/"
                f"EXP-S3-20260730-{suffix}/review.json"
            ),
        }
        if failure:
            item["failure_taxonomy"] = failure
        ledger["experiments"].append(item)
    write_json(EXPERIMENTS / "ledger.json", ledger)

    hypotheses = [
        {
            "hypothesis_id": values[0],
            "experiment_id": f"EXP-S3-20260730-{suffix}",
            "verdict": values[1],
            "learning_zh": learning,
        }
        for suffix, values, learning in [
            (
                "005",
                verdicts["005"],
                "negative phrase 放在 positive prompt 里仍会激活其中的名词；颜色禁区必须由字段所有权控制。",
            ),
            (
                "006",
                verdicts["006"],
                "修正颜色词不足以保证几何通过，冻结门禁无候选时必须报告失败。",
            ),
            (
                "007",
                verdicts["007"],
                "外轮廓覆盖不能证明对象身份部件仍存在。",
            ),
            (
                "008",
                verdicts["008"],
                "身份部件既应进入结构 provider，也可进入 traceable prompt slot；目前证据只支持 CHEM-01。",
            ),
            (
                "009",
                verdicts["009"],
                "重复标志要匹配 provider 声明的位置；只数边缘组会把反光误判为刻度。",
            ),
            (
                "010",
                verdicts["010"],
                "visual package → profile 与 preservation wording 已从代码分支迁到版本数据。",
            ),
        ]
    ]
    hypothesis_path = STAGE3 / "knowledge/hypotheses.jsonl"
    old_lines = [
        line
        for line in hypothesis_path.read_text(encoding="utf-8").splitlines()
        if not any(
            f'"experiment_id": "EXP-S3-20260730-{suffix}"' in line
            for suffix in verdicts
        )
    ]
    with hypothesis_path.open("w", encoding="utf-8") as handle:
        for line in old_lines:
            handle.write(line + "\n")
        for item in hypotheses:
            handle.write(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            )

    failures = load_json(STAGE3 / "knowledge/failure_patterns.json")
    failures["patterns"] = [
        item
        for item in failures["patterns"]
        if not item["id"].startswith("FP-PROMPT-")
        and not item["id"].startswith("FP-LANDMARK-")
    ]
    failures["patterns"].extend(
        [
            {
                "id": "FP-PROMPT-001",
                "taxonomy": "prompt_polarity",
                "symptom_zh": "要求“不要粉色”却生成整批粉色图。",
                "diagnosis_zh": "禁用名词 pink 出现在正向 conditioning，模型仍注意该 token。",
                "forbidden_fix_zh": "不得只换 seed；把禁用概念移入 negative slot 并冻结词段所有权。",
            },
            {
                "id": "FP-PROMPT-002",
                "taxonomy": "prompt_scene_leakage",
                "symptom_zh": "目标器材之外出现烧瓶、瓶子或实验室杂物。",
                "diagnosis_zh": "宽泛的 laboratory/apparatus 场景词诱发模型补全常见共现物。",
                "forbidden_fix_zh": "不得逐图修补；使用最小场景词和明确空背景。",
            },
            {
                "id": "FP-LANDMARK-001",
                "taxonomy": "semantic_landmark",
                "symptom_zh": "外轮廓正确，但器材刻度等身份部件消失。",
                "diagnosis_zh": "selector 只测轮廓覆盖，没有读取 primitive 的内部标志合同。",
                "forbidden_fix_zh": "不得靠人工记住每个 Case；provider 必须输出可计算 landmark 声明。",
            },
            {
                "id": "FP-LANDMARK-002",
                "taxonomy": "gate_false_positive",
                "symptom_zh": "玻璃反光横带被误判成多条刻度。",
                "diagnosis_zh": "只数 ROI 内边缘组，没有与 provider 的预期位置对齐。",
                "forbidden_fix_zh": "不得事后挑 seed；先冻结 expected positions 和容差，再回放完整集合。",
            },
        ]
    )
    write_json(STAGE3 / "knowledge/failure_patterns.json", failures)

    write_json(
        STAGE3 / "knowledge/open_problems.json",
        {
            "schema_version": "1.0",
            "problems": [
                {
                    "problem_id": "S3-PROBLEM-STATE-001",
                    "taxonomy": "state_renderer",
                    "summary_zh": "已有冻结外观 Anchor，但区域、标量场、对象和 height/normal 尚未通过统一 B 接口生成多关键帧。",
                },
                {
                    "problem_id": "S3-PROBLEM-VISUAL-001",
                    "taxonomy": "visual_target",
                    "summary_zh": "GEO-02 外观目标仍为 provisional。",
                },
            ],
        },
    )

    accepted = load_json(STAGE3 / "baselines/accepted.json")
    replace_ids = {
        "CORE-GEOMETRY-COMPILER-V1",
        "CORE-GEOMETRY-COMPILER-V2",
        "CORE-PROMPT-COMPILER-V2",
        "CORE-CANDIDATE-SELECTOR-V4",
        "CASE-PROMPT-CHEM-01-V4",
        "ANCHOR-CHEM-01-S3.3-V1",
    }
    accepted["records"] = [
        item
        for item in accepted["records"]
        if item["baseline_id"] not in replace_ids
    ]
    additions = [
        ("CORE-GEOMETRY-COMPILER-V2", "accepted_core", STAGE3 / "framework/geometry.py"),
        ("CORE-PROMPT-COMPILER-V2", "accepted_core", STAGE3 / "framework/prompt.py"),
        ("CORE-CANDIDATE-SELECTOR-V4", "accepted_core", STAGE3 / "selector_v4.json"),
        ("CASE-PROMPT-CHEM-01-V4", "accepted_case_specific", STAGE3 / "prompt_lexicon_v4.json"),
        (
            "ANCHOR-CHEM-01-S3.3-V1",
            "accepted_model_anchor",
            OUTPUT / "selection-v4-position-audit/selected.png",
        ),
    ]
    for baseline_id, kind, path in additions:
        accepted["records"].append(
            {"baseline_id": baseline_id, "kind": kind, **file_record(path, REPO_ROOT)}
        )
    write_json(STAGE3 / "baselines/accepted.json", accepted)

    write_json(
        STAGE3 / "state.json",
        {
            "schema_version": "1.0",
            "loop_id": "LOOP-S3-0001",
            "phase": "S3.4",
            "phase_status": "in_progress",
            "current_problem": {
                "problem_id": "S3-PROBLEM-STATE-001",
                "taxonomy": "state_renderer",
                "summary_zh": "把冻结外观 Anchor 与程序的区域、标量、对象和高度/法线数据确定性合成多关键帧。",
            },
            "current_hypothesis": {
                "hypothesis_id": "H-S3-0004A",
                "statement_zh": "四类通用 B 算子可只改变合同声明的状态区域，同时保持 Anchor 背景和稳定器材逐像素不动。",
                "falsification_zh": "稳定区变化、状态单调性错误、对象身份丢失或跨案例接口需要 Case 坐标即失败。",
            },
            "current_cohort": {
                "target": "CHEM-01",
                "regressions": ["PHYS-01", "MATH-02"],
                "geometry_anchor": "modules/video_model/stage3/output/phase-3/selection-v4-position-audit/selected.png",
            },
            "exit_criteria": [
                "region, scalar, object and height_or_normal operators have versioned data contracts",
                "CHEM-01 produces all contracted keyframes from one frozen Anchor",
                "stable background and apparatus pass pixel-difference gates",
                "PHYS-01 and MATH-02 operator regressions pass without Case-specific coordinates",
            ],
            "budget": {
                "preflight_before_gpu_work": True,
                "s3_4_new_image_model_candidates": 0,
                "s3_4_video_candidates": 0,
            },
            "next_action": "Implement model-free State Renderer B operators, starting with CHEM-01 scalar/region composition and cross-case interface regressions.",
        },
    )


def report_html(assets: dict[str, Path], cross_case: dict[str, Any]) -> str:
    report_dir = OUTPUT
    prompts = {}
    for version, folder in [
        ("v1", "prompts"),
        ("v2", "prompts-v2"),
        ("v3", "prompts-v3"),
        ("v4", "prompts-v4"),
    ]:
        prompts[version] = {
            "positive": (
                OUTPUT / folder / "CHEM-01/positive_prompt.txt"
            ).read_text(encoding="utf-8").strip(),
            "negative": (
                OUTPUT / folder / "CHEM-01/negative_prompt.txt"
            ).read_text(encoding="utf-8").strip(),
        }
    rows = "\n".join(
        f"<tr><td><code>{case_id}</code></td>"
        f"<td>{item['appearance_profile']}</td>"
        f"<td>{item['positive_tokens']['tokenizer']} / "
        f"{item['negative_tokens']['tokenizer']}</td>"
        f"<td>{'通过' if not item['would_truncate'] else '失败'}</td></tr>"
        for case_id, item in cross_case["records"].items()
    )
    candidate_sections = []
    rounds = [
        (
            "v1 / EXP-005：失败——颜色词极性写反",
            "experiments/EXP-S3-20260730-005/candidates-labeled.jpg",
            "正向句子里写了 “no pink”。对扩散模型来说，pink 仍是一个被关注的词，所以九张都发粉。",
        ),
        (
            "v2 / EXP-006：失败——颜色正确，但 0 张通过固定门禁",
            "experiments/EXP-S3-20260730-006/candidates-labeled.jpg",
            "把 pink 移入负向提示后颜色恢复中性；没有候选同时满足轮廓精度、额外边缘和对比度，因此没有事后降门槛。",
        ),
        (
            "v3 / EXP-007：失败——外观干净，但刻度消失",
            "experiments/EXP-S3-20260730-007/candidates-labeled.jpg",
            "删除宽泛实验室语境后，8/9 通过旧门禁；人工语义复核发现所谓“滴定管”丢了刻度，暴露出 selector 只看外轮廓。",
        ),
        (
            "v4 / EXP-008：案例级成功——刻度进入 prompt 与门禁",
            "experiments/EXP-S3-20260730-008/candidates-labeled.jpg",
            "最终门禁只保留 1/9：scale 0.8、seed 7101。它成为 CHEM-01 Anchor，但单个通过样本不足以宣称所有 Case 都会改善。",
        ),
    ]
    for title, image, text in rounds:
        candidate_sections.append(
            f"<article class='round'><h3>{html.escape(title)}</h3>"
            f"<p>{html.escape(text)}</p>"
            f"<a href='{image}'><img src='{image}' alt='{html.escape(title)}'></a></article>"
        )
    prompt_blocks = "\n".join(
        f"<details {'open' if version == 'v4' else ''}>"
        f"<summary>{version} 完整提示词</summary>"
        f"<p><strong>正向 conditioning：</strong></p>"
        f"<pre>{html.escape(value['positive'])}</pre>"
        f"<p><strong>负向 conditioning：</strong></p>"
        f"<pre>{html.escape(value['negative'])}</pre></details>"
        for version, value in prompts.items()
    )
    positive_ref = (
        REPO_ROOT
        / "modules/video_model/stage2/output/phase-7/route-a/"
        "experiments/EXP-P7-A-chem-01-00_start/raw/"
        "semantic_control_065/seed_7101.png"
    )
    negative_ref = (
        REPO_ROOT
        / "modules/video_model/stage2/output/phase-9/"
        "report-assets/chem-a-rejected-direct-sequence.jpg"
    )
    model_meta = load_json(
        OUTPUT
        / "experiments/EXP-S3-20260730-008/_work/generate.json"
    )
    selection = load_json(
        OUTPUT / "selection-v4-position-audit/selection.json"
    )
    selected = selection["selected_candidate"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 3 · S3.3 Prompt Compiler 阶段报告</title>
<style>
:root{{--ink:#172321;--muted:#5b6864;--paper:#f4f0e6;--card:#fffdf7;--green:#0d6d59;--red:#a13c32;--gold:#b48120;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.68 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
main{{max-width:1180px;margin:auto;padding:42px 24px 90px}} h1{{font-size:clamp(34px,5vw,64px);line-height:1.05;margin:.2em 0}}
h2{{margin-top:52px;border-top:1px solid #c9c4b7;padding-top:28px}} h3{{margin-bottom:6px}} p{{max-width:86ch}}
.lead{{font-size:20px;color:#31413d}} .status{{display:inline-block;background:#d7eee7;color:#075342;padding:7px 12px;border-radius:999px;font-weight:750}}
.warning{{background:#fff0cf;border-left:5px solid var(--gold);padding:14px 18px}} .bad{{color:var(--red)}} .good{{color:var(--green)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}} .card,.round,details{{background:var(--card);border:1px solid #d9d3c5;border-radius:14px;padding:18px}}
img{{display:block;max-width:100%;height:auto;border-radius:10px;border:1px solid #cfc9bb}} .hero{{width:100%;margin:20px 0}}
table{{width:100%;border-collapse:collapse;background:var(--card)}} th,td{{padding:11px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #ddd7ca}} th{{background:#e7eee9}}
code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#142422;color:#e9f3ef;padding:15px;border-radius:9px}}
.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;align-items:stretch}} .flow div{{background:#e5eee9;border-radius:10px;padding:12px}} .flow b{{display:block;color:var(--green)}}
@media(max-width:800px){{.flow{{grid-template-columns:1fr}}}} a{{color:#086b58}} .small{{font-size:14px;color:var(--muted)}}
</style></head><body><main>
<span class="status">S3.3 已通过 · 已进入 S3.4</span>
<h1>Prompt Compiler：从“凭感觉写提示词”到可追溯、可失败、可复现</h1>
<p class="lead">本阶段不是为了证明一条神奇 prompt 可以解决所有案例，而是固定一套编译和验收方法。
同一合同、视觉目标、词表、控制图、模型、seed 和参数，会得到同一批候选与同一选择结论；
固定集合没有合格图时，系统明确失败。</p>

<h2>先看结果</h2>
<a href="{href(report_dir, assets['process'])}"><img class="hero" src="{href(report_dir, assets['process'])}" alt="四轮提示词实验的完整过程"></a>
<div class="grid">
<div class="card"><h3 class="good">编译器结论：通过</h3><p>词段只来自 Input Contract、Visual Target Package 和版本词表；
CHEM-01、MATH-02、PHYS-01 均完成来源追踪和 token 预检。</p></div>
<div class="card"><h3 class="good">CHEM-01 Anchor：案例级接受</h3><p>最终选中 <code>{selected['candidate_id']}</code>，
ControlNet scale {selected['configuration_id'].split('_')[-1]}，seed {selected['seed']}。九张中只有一张通过，未推广成全局风格。</p></div>
<div class="card"><h3 class="bad">保留了三类失败</h3><p>颜色词污染、无候选通过、外轮廓正确但内部刻度丢失。失败图没有删除，也没有包装成成功。</p></div>
</div>

<h2>输入怎样一步步变成图片</h2>
<div class="flow">
<div><b>1. 输入合同</b>程序关键帧、对象类别/数量、几何策略和必须保持的关系。</div>
<div><b>2. 几何控制</b>S3.1 根据程序语义重建黑底白线；它只决定“东西在哪里”。</div>
<div><b>3. Prompt Compiler</b>把对象、相机、材质/光、当前状态、保留项和反例编成六个可追溯槽位。</div>
<div><b>4. SDXL + ControlNet</b>SDXL 负责合成像素；ControlNet 在每个去噪步骤把结构线转换成额外引导残差，约束 SDXL 的布局。</div>
<div><b>5. 门禁与排序</b>先拒绝几何、身份部件、曝光或禁区失败的图；只在合格图内按冻结权重排序。</div>
</div>
<p><strong>Canny 是什么：</strong>它是一种传统边缘检测算法，把亮暗突变变成线。本案例喂给模型的控制图不是把整张截图做
dense Canny，而是 Geometry Resolver 画出的干净语义线；“SDXL Canny ControlNet”之所以叫 Canny，
是因为它训练时学会理解这类边缘图。selector 另外使用 Canny 测量输出图的边缘是否贴近控制。</p>
<p><strong>conditioning 是什么：</strong>它不是最后的图，而是影响扩散去噪过程的条件。正向文字描述想出现什么，
负向文字描述要压制什么；ControlNet conditioning 是结构线。ControlNet 返回中间特征残差，SDXL 把这些残差加进
自己的去噪网络后输出最终 RGB 图片。</p>

<h2>几何与外观没有混在一起</h2>
<div class="grid">
<div class="card"><h3>几何端口</h3><img src="../phase-1/controls/CHEM-01/00_start/structure_control.png" alt="实际结构控制图">
<p>来源：程序 object identity + hard boundary + canonical primitive。决定一个烧杯、一个上方滴定管、两者分离和内部刻度位置；
不读取外观参考图。</p></div>
<div class="card"><h3>外观正例</h3><img src="{href(report_dir, positive_ref)}" alt="透明玻璃外观正例">
<p>只提供玻璃、光照、色调和真实感目标；它的位置不能覆盖程序几何。</p></div>
<div class="card"><h3>外观反例</h3><img src="{href(report_dir, negative_ref)}" alt="逐帧自由生成反例">
<p>告诉词表和量表要避免器材漂移、背景变化、颜色泄漏；反例不作为结构输入。</p></div>
</div>

<h2>四轮自迭代，不是看图后无限抽卡</h2>
{''.join(candidate_sections)}

<h2>为什么“数到横线”仍然会错</h2>
<a href="{href(report_dir, assets['landmarks'])}"><img class="hero" src="{href(report_dir, assets['landmarks'])}" alt="刻度位置门禁解释"></a>
<p>v3 selector 只数黄色 ROI 里有多少组横向边缘，所以玻璃顶部的反光带也能凑够数量。
v4 改为由 <code>canonical_graduated_tube_v1</code> provider 输出七个预期纵向位置和 ±3 px 容差，
候选至少匹配五个才通过。selector 代码只实现“重复水平标志”的通用测量，不包含
<code>CHEM-01</code> 或烧杯坐标；新 primitive 若有身份部件，也必须自己声明。</p>

<h2>完整提示词与每次变化</h2>
<p>正向 prompt 不是一整段自由作文，而是：
<code>scene identity + camera + material/light + state + must preserve</code>。
negative prompt 是独立的反例/伪影槽。v1-v4 文件均冻结并记录 SHA-256。</p>
{prompt_blocks}

<h2>参数不是暗号：每个值控制什么</h2>
<table><thead><tr><th>参数</th><th>本阶段值</th><th>普通话解释</th><th>是否参与本路线</th></tr></thead><tbody>
<tr><td>SDXL Base 1.0 FP16</td><td><code>stabilityai/stable-diffusion-xl-base-1.0</code></td><td>主图片生成模型；FP16 是半精度权重，降低显存占用。</td><td>是</td></tr>
<tr><td>SDXL Canny ControlNet FP16</td><td><code>diffusers/controlnet-canny-sdxl-1.0</code></td><td>读取结构边缘并在 SDXL 去噪过程中提供空间残差。</td><td>是</td></tr>
<tr><td>ControlNet scale</td><td>0.50 / 0.65 / 0.80</td><td>结构残差的权重；越高通常越贴线，也可能更僵硬。三档在看图前冻结。</td><td>是</td></tr>
<tr><td>seed</td><td>7101 / 7102 / 7103</td><td>初始噪声编号；相同输入与运行时使用相同 seed 可重放同一候选。它不是质量分。</td><td>是</td></tr>
<tr><td>steps</td><td>30</td><td>去噪迭代次数；不是帧数。</td><td>是</td></tr>
<tr><td>guidance scale</td><td>6.0</td><td>文字条件相对无条件预测的影响强度。</td><td>是</td></tr>
<tr><td>strength</td><td>配置中保留 0.5</td><td>只有 Img2Img 才决定保留输入 RGB 的程度。本阶段实际为 <code>controlnet_t2i</code>，代码不会把 strength 传给 pipeline，因此它不影响结果。</td><td><strong>否</strong></td></tr>
<tr><td>resolution / dtype</td><td>1024×576 / float16</td><td>输出尺寸与模型计算精度。</td><td>是</td></tr>
<tr><td>token preflight</td><td>v4 正 74、负 69；上限 77</td><td>SDXL 有两个 tokenizer；任一会截断就停止，避免尾部“必须保留”静默丢失。</td><td>是</td></tr>
</tbody></table>

<h2>跨案例证明了什么，没证明什么</h2>
<table><thead><tr><th>Case</th><th>外观 profile</th><th>正/负 token</th><th>编译</th></tr></thead><tbody>{rows}</tbody></table>
<p class="warning"><strong>边界：</strong>这三例证明 v5 编译器能够从数据配置路由 profile、生成六槽 prompt、
追踪来源并避免截断；S3.3 没有为 MATH-02 和 PHYS-01 新跑扩散候选，所以不宣称它们的图片质量已经提升。
CHEM-01 的 v4 词表只登记为案例级配置。</p>

<h2>模型、运行时与复现</h2>
<p>EXP-008 实际运行：<code>{model_meta['runtime']['gpu']}</code>；
PyTorch <code>{model_meta['runtime']['torch']}</code>，diffusers
<code>{model_meta['package_versions']['diffusers']}</code>，scheduler
<code>{model_meta['scheduler']}</code>。本阶段共新生成 36 张图片候选、0 个视频候选。</p>
<p>权重指纹、每张图片哈希、生成耗时和显存记录在
<a href="experiments/EXP-S3-20260730-008/_work/generate.json"><code>generate.json</code></a>；
最终门禁明细在 <a href="selection-v4-position-audit/selection.json"><code>selection.json</code></a>。</p>
<pre>cd /workspace/Live-Document
/opt/venv/bin/python -m modules.video_model.stage3.phase3_trial preflight \
  --plan modules/video_model/stage3/prompt_trials/EXP-S3-20260730-008.json
/opt/venv/bin/python -m modules.video_model.stage3.phase3_trial generate \
  --plan modules/video_model/stage3/prompt_trials/EXP-S3-20260730-008.json
/opt/venv/bin/python -m modules.video_model.stage3.phase3_trial select \
  --plan modules/video_model/stage3/prompt_trials/EXP-S3-20260730-008.json
/opt/venv/bin/python -m modules.video_model.stage3.phase3_finalize</pre>
<p class="small">注意：上面 trial plan 冻结的是 selector_v3，忠实重放 EXP-008 的第一次选择。
最终 position-matching selector_v4 是同一九张候选上的零模型回放，见 EXP-009 与
<code>selection-v4-position-audit</code>；没有追加 seed。</p>

<h2>阶段结论与下一步</h2>
<ul>
<li><strong>进入核心：</strong>Prompt Compiler v2 的数据驱动 profile/preservation 路由；
primitive 声明内部身份标志、selector 按预期位置验收的接口。</li>
<li><strong>只进入 CHEM-01 配置：</strong>prompt lexicon v4 与选中的透明玻璃 Anchor。</li>
<li><strong>没有声称：</strong>任意概念都能一次生成好图，或 v4 是所有学科的共同风格。</li>
<li><strong>下一阶段 S3.4：</strong>不再重新抽 Anchor；从同一冻结底图出发，用程序 region、
scalar field、object 和 height/normal 数据确定性生成多个机制关键帧，并检查稳定区域是否变化。</li>
</ul>
</main></body></html>"""


def main() -> None:
    frozen_v4_hash = (
        "ac19526bb5bbedf51f58d6e1eb83926a78d45e43022b2583552c58041ac59bf6"
    )
    if sha256_path(STAGE3 / "prompt_lexicon_v4.json") != frozen_v4_hash:
        raise RuntimeError("frozen v4 lexicon changed after EXP-008")
    assets = make_assets()
    cross_case = compile_v5_cross_case()
    if not cross_case["passed"]:
        raise RuntimeError("v5 cross-case compiler checks failed")
    update_records(cross_case)
    report_path = OUTPUT / "report.html"
    report_path.write_text(
        report_html(assets, cross_case), encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "phase": "S3.3",
        "status": "passed",
        "model_runs": {
            "image_candidates": 36,
            "video_candidates": 0,
        },
        "accepted_core": [
            "stage3_prompt_compiler_v2",
            "primitive-declared internal landmark gate",
            "position-matching candidate selector v4",
        ],
        "accepted_case_specific": [
            "CHEM-01 prompt lexicon v4",
            "CHEM-01 anchor auto_control_080-s7101",
        ],
        "report": file_record(report_path, REPO_ROOT),
        "assets": {
            name: file_record(path, REPO_ROOT)
            for name, path in assets.items()
        },
        "selected_anchor": file_record(
            OUTPUT / "selection-v4-position-audit/selected.png",
            REPO_ROOT,
        ),
        "cross_case_compile": file_record(
            OUTPUT / "prompt-v5-cross-case-compile.json",
            REPO_ROOT,
        ),
        "next_phase": "S3.4",
    }
    write_json(OUTPUT / "phase3_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "passed",
                "report": str(report_path),
                "selected": manifest["selected_anchor"],
                "next_phase": "S3.4",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
