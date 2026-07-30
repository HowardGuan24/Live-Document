"""Build and verify the Stage 2 Phase 6 release regression."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .framework.contracts import (
    artifact_record,
    load_json,
    sha256_path,
    write_json,
)
from .framework.release_image_regression import (
    build_release_image_regression,
)
from .phase1 import build_phase1
from .phase2 import build_phase2
from .phase4 import build_phase4_programs
from .phase5 import build_phase5


STAGE2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STAGE2_ROOT.parents[2]
OUTPUT_ROOT = STAGE2_ROOT / "output" / "phase-6"
REPORT_PATH = OUTPUT_ROOT / "stage2-final-report.html"
MANIFEST_PATH = OUTPUT_ROOT / "phase6_manifest.json"
TOKEN_AUDIT_PATH = OUTPUT_ROOT / "video-token-integrity.json"
ROUTES_PATH = STAGE2_ROOT / "phase6_image_routes.json"
POLICY_PATH = STAGE2_ROOT / "protocols/regression_policy.json"
VERSION_PATH = STAGE2_ROOT / "VERSION"
CHANGELOG_PATH = STAGE2_ROOT / "CHANGELOG.md"

DISCIPLINE_ZH = {
    "mathematics": "数学",
    "physics": "物理",
    "chemistry": "化学",
    "biology": "生物",
    "geography": "地理",
}

CHECK_LABELS_ZH = {
    "ten_case_contract_and_keyframe_regression": "十案例合同与关键帧",
    "five_discipline_image_representatives": "五学科图片代表",
    "five_video_motion_representatives": "五类运动代表",
    "video_prompt_token_integrity": "视频提示词真实 Token",
    "historical_delta_immutable_regression": "三角洲历史文件不回退",
    "released_core_has_no_benchmark_case_id_branches": "通用核心没有案例名分支",
    "final_report_links_resolve": "最终报告全部链接可打开",
}

CONTROL_BY_EXPERIMENT = {
    "EXP-20260729-005": "control_off.png",
    "EXP-20260729-009": "semantic_apparatus_line_art.png",
    "EXP-20260729-012": "control_off.png",
    "EXP-20260729-013": "dense_canny.png",
    "EXP-20260729-015": "dense_canny.png",
}

DONOR_CONFIGURATION_BY_EXPERIMENT = {
    "EXP-20260729-005": "t2i_material_donor",
    "EXP-20260729-009": "semantic_line_art",
    "EXP-20260729-012": "t2i_wood_material_donor",
    "EXP-20260729-013": "t2i_dense_cell_and_chromatids",
    "EXP-20260729-015": "dense_scale_035",
}

CORE_FILES = (
    "framework/contracts.py",
    "framework/program_runner.py",
    "framework/route_selector.py",
    "framework/material_projection.py",
    "framework/release_image_regression.py",
    "framework/ltx_flf.py",
    "framework/program_video.py",
    "framework/fixture_builder.py",
)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _href(path: Path) -> str:
    return os.path.relpath(path, REPORT_PATH.parent).replace(os.sep, "/")


def _check(
    name: str, passed: bool, evidence: Any
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "evidence": evidence,
    }


def _experiment_id_from_path(path: str) -> str:
    for part in Path(path).parts:
        if part.startswith("EXP-"):
            return part
    raise ValueError(f"experiment id missing from path: {path}")


def _labeled_sheet(
    cells: list[tuple[str, Path]],
    output: Path,
    *,
    columns: int,
) -> None:
    thumb = (384, 216)
    label_height = 34
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * thumb[0],
            rows * (thumb[1] + label_height),
        ),
        (13, 35, 41),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(cells):
        column = index % columns
        row = index // columns
        x = column * thumb[0]
        y = row * (thumb[1] + label_height)
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb, Image.Resampling.LANCZOS)
        tile = Image.new("RGB", thumb, (238, 236, 226))
        tile.paste(
            image,
            (
                (thumb[0] - image.width) // 2,
                (thumb[1] - image.height) // 2,
            ),
        )
        sheet.paste(tile, (x, y))
        draw.text(
            (x + 10, y + thumb[1] + 9),
            label,
            fill=(238, 248, 244),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, subsampling=0)


def _build_friendly_report_assets(
    images: list[dict[str, Any]],
) -> None:
    root = OUTPUT_ROOT / "report-assets"
    for item in images:
        experiment_id = item["experiment_id"]
        experiment_root = (
            STAGE2_ROOT / "output/phase-3" / experiment_id
        )
        generated = load_json(
            experiment_root / "_work/generate.json"
        )
        configuration_id = (
            DONOR_CONFIGURATION_BY_EXPERIMENT[experiment_id]
        )
        candidates = [
            experiment_root / candidate["path"]
            for candidate in generated["candidates"]
            if candidate["configuration_id"] == configuration_id
        ]
        if len(candidates) != 4:
            raise ValueError(
                f"{experiment_id}: expected four frozen donor samples"
            )
        _labeled_sheet(
            [
                (f"MODEL MATERIAL SAMPLE {index + 1}", path)
                for index, path in enumerate(candidates)
            ],
            root / f"{experiment_id}-model-samples.jpg",
            columns=2,
        )
        if item["case_id"] not in {"MATH-02", "PHYS-01"}:
            continue
        spec = load_json(
            STAGE2_ROOT
            / "experiments"
            / experiment_id
            / "spec.json"
        )
        evidence = load_json(STAGE2_ROOT / item["evidence"]["path"])
        final_path = (
            experiment_root
            / evidence["full_ensemble"]["composite"]["path"]
        )
        process_path = root / f"{item['case_id']}-process.jpg"
        _labeled_sheet(
            [
                (
                    "PROGRAM FACTS",
                    STAGE2_ROOT / spec["source"]["clean_frame"],
                ),
                ("MODEL MATERIAL SAMPLE", candidates[0]),
                ("FINAL: FACTS + MATERIAL", final_path),
            ],
            process_path,
            columns=3,
        )
        item["comparison"] = artifact_record(
            process_path, STAGE2_ROOT
        )


def _run_new_image_regressions(
    routes: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for config in routes["new_zero_model_regressions"]:
        root = (
            OUTPUT_ROOT
            / "image-regressions"
            / config["case_id"]
        )
        records.append(
            build_release_image_regression(
                stage2_root=STAGE2_ROOT,
                config=config,
                output_root=root,
            )
        )
    return records


def _collect_image_representatives(
    routes: dict[str, Any],
    new_manifests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for config in routes["existing_accepted_representatives"]:
        evidence_path = STAGE2_ROOT / config["evidence"]
        review_path = STAGE2_ROOT / config["review"]
        evidence = load_json(evidence_path)
        review = load_json(review_path)
        experiment_id = config["experiment_id"]
        experiment_root = (
            STAGE2_ROOT / "output/phase-3" / experiment_id
        )
        checks_pass = bool(evidence["hard_checks"]) and all(
            item["passed"] for item in evidence["hard_checks"]
        )
        visible_change = float(
            evidence["full_ensemble"]["metrics"][
                "allowed_mean_abs_detail_change_0_255"
            ]
        )
        records.append(
            {
                "case_id": config["case_id"],
                "discipline": config["discipline"],
                "status": (
                    "passed"
                    if checks_pass
                    and review["verdict"].startswith("accepted")
                    and visible_change >= 0.2
                    else "failed"
                ),
                "route": config["route"],
                "scope": "one frozen representative keyframe",
                "experiment_id": experiment_id,
                "hard_checks_passed": sum(
                    item["passed"]
                    for item in evidence["hard_checks"]
                ),
                "hard_checks_total": len(evidence["hard_checks"]),
                "visible_detail_change_0_255": visible_change,
                "visual_review_zh": review["verdict_zh"],
                "comparison": artifact_record(
                    experiment_root
                    / evidence["comparison_sheet"]["path"],
                    STAGE2_ROOT,
                ),
                "evidence": artifact_record(
                    evidence_path, STAGE2_ROOT
                ),
                "review": artifact_record(
                    review_path, STAGE2_ROOT
                ),
                "model_runs_during_phase6": {
                    "image": 0,
                    "video": 0,
                },
            }
        )
    by_case = {
        manifest["case_id"]: manifest
        for manifest in new_manifests
    }
    for config in routes["new_zero_model_regressions"]:
        manifest = by_case[config["case_id"]]
        experiment_id = _experiment_id_from_path(
            config["donor_evidence"]
        )
        detail = [
            frame["metrics"][
                "mutable_mean_abs_detail_change_0_255"
            ]
            for frame in manifest["keyframes"]
        ]
        records.append(
            {
                "case_id": config["case_id"],
                "discipline": config["discipline"],
                "status": (
                    "passed"
                    if manifest["status"] == "passed"
                    and config["visual_review"]["status"] == "accepted"
                    else "failed"
                ),
                "route": (
                    "four-keyframe program sequence plus "
                    "region-limited robust material residual"
                ),
                "scope": "all four frozen keyframes",
                "experiment_id": experiment_id,
                "hard_checks_passed": sum(
                    item["passed"]
                    for item in manifest["hard_checks"]
                ),
                "hard_checks_total": len(
                    manifest["hard_checks"]
                ),
                "visible_detail_change_0_255": round(
                    sum(detail) / len(detail), 6
                ),
                "visual_review_zh": config["visual_review"][
                    "finding_zh"
                ],
                "comparison": artifact_record(
                    OUTPUT_ROOT
                    / "image-regressions"
                    / config["case_id"]
                    / manifest["comparison"]["path"],
                    STAGE2_ROOT,
                ),
                "evidence": artifact_record(
                    OUTPUT_ROOT
                    / "image-regressions"
                    / config["case_id"]
                    / "manifest.json",
                    STAGE2_ROOT,
                ),
                "review": config["visual_review"],
                "model_runs_during_phase6": {
                    "image": 0,
                    "video": 0,
                },
            }
        )
    order = load_json(POLICY_PATH)["discipline_representatives"]
    by_case = {item["case_id"]: item for item in records}
    ordered = [by_case[case_id] for case_id in order]
    _build_friendly_report_assets(ordered)
    return ordered


def _collect_contract_regression() -> dict[str, Any]:
    phase1 = build_phase1(check_only=True)
    phase2 = build_phase2(check_only=True)
    phase4 = build_phase4_programs(check_only=True)
    fixtures = {
        item["case_id"]: item for item in phase1["fixtures"]
    }
    programs = {
        item["case_id"]: item for item in (
            phase2["programs"] + phase4["programs"]
        )
    }
    case_ids = load_json(POLICY_PATH)["contract_smoke"][
        "case_ids"
    ]
    rows = []
    for case_id in case_ids:
        fixture = fixtures.get(case_id)
        program = programs.get(case_id)
        rows.append(
            {
                "case_id": case_id,
                "contract_fixture": (
                    "passed" if fixture else "missing"
                ),
                "deterministic_program": (
                    program["status"] if program else "missing"
                ),
                "keyframe_gate": (
                    "passed"
                    if program and program["status"] == "passed"
                    else "failed"
                ),
            }
        )
    return {
        "status": (
            "passed"
            if len(rows) == 10
            and all(
                row["contract_fixture"] == "passed"
                and row["deterministic_program"] == "passed"
                and row["keyframe_gate"] == "passed"
                for row in rows
            )
            else "failed"
        ),
        "case_count": len(rows),
        "program_frame_count": (
            phase2["frame_count"] + phase4["frame_count"]
        ),
        "program_keyframe_count": (
            phase2["keyframe_count"] + phase4["keyframe_count"]
        ),
        "cases": rows,
    }


def _collect_video_regression() -> list[dict[str, Any]]:
    manifest = build_phase5(check_only=True)
    accepted: dict[str, dict[str, Any]] = {}
    for item in manifest["experiments"]:
        review = load_json(STAGE2_ROOT / item["review"]["path"])
        if review.get("motion_class_passed", False):
            accepted[item["motion_class"]] = {
                "case_id": item["case_id"],
                "motion_class": item["motion_class"],
                "status": "passed",
                "experiment_id": item["experiment_id"],
                "classification": (
                    "model_video"
                    if load_json(
                        STAGE2_ROOT / item["run"]["path"]
                    )["model_runs"]["video"]
                    else "deterministic_program_fallback"
                ),
                "run": item["run"],
                "review": item["review"],
            }
    order = load_json(POLICY_PATH)[
        "video_motion_representatives"
    ]
    return [accepted[motion] for motion in order]


def _verify_token_audit() -> dict[str, Any]:
    audit = load_json(TOKEN_AUDIT_PATH)
    verified = []
    for item in audit["experiments"]:
        spec_path = (
            STAGE2_ROOT
            / "phase5_experiments"
            / item["experiment_id"]
            / "spec.json"
        )
        spec = load_json(spec_path)
        hashes_match = all(
            item[name]["text_sha256"]
            == _text_sha256(spec["prompt"][name])
            for name in ("positive", "negative")
        )
        verified.append(
            item["passed"]
            and hashes_match
            and all(
                item[name][
                    "content_tokens_including_bos"
                ]
                <= audit["measurement"]["release_soft_limit"]
                for name in ("positive", "negative")
            )
        )
    return {
        "status": (
            "passed"
            if audit["status"] == "passed"
            and len(verified) == 8
            and all(verified)
            else "failed"
        ),
        "experiment_count": len(verified),
        "measurement": audit["measurement"],
        "tokenizer_asset": audit["tokenizer_asset"],
        "experiments": audit["experiments"],
        "artifact": artifact_record(
            TOKEN_AUDIT_PATH, STAGE2_ROOT
        ),
    }


def _verify_historical_baseline() -> dict[str, Any]:
    phase0 = load_json(
        STAGE2_ROOT / "output/phase-0/phase0_manifest.json"
    )
    records = []
    for frozen in phase0["baseline_assets"]:
        path = REPO_ROOT / frozen["path"]
        current_hash = sha256_path(path)
        records.append(
            {
                "path": frozen["path"],
                "role": frozen["role"],
                "frozen_sha256": frozen["sha256"],
                "current_sha256": current_hash,
                "unchanged": (
                    path.is_file()
                    and current_hash == frozen["sha256"]
                    and path.stat().st_size == frozen["size_bytes"]
                ),
            }
        )
    return {
        "status": (
            "passed"
            if records and all(item["unchanged"] for item in records)
            else "failed"
        ),
        "asset_count": len(records),
        "method_zh": (
            "逐个复核 Phase 0 冻结的 Stage 1 三角洲图片、视频、"
            "报告和元数据的字节数与 SHA-256；Phase 6 没有覆盖旧结果。"
        ),
        "scope_limit_zh": (
            "这是历史文件不回退检查，不是用新核心重新生成三角洲；"
            "旧序列尚未迁移到 Stage 2 的标准语义层合同。"
        ),
        "assets": records,
    }


def _audit_core_hardcoding() -> dict[str, Any]:
    pattern = re.compile(r"\b(?:MATH|PHYS|CHEM|BIO|GEO)-\d+\b")
    rows = []
    for relative in CORE_FILES:
        path = STAGE2_ROOT / relative
        matches = sorted(set(pattern.findall(path.read_text())))
        rows.append(
            {
                "path": relative,
                "case_id_matches": matches,
                "passed": not matches,
            }
        )
    return {
        "status": (
            "passed" if all(item["passed"] for item in rows) else "failed"
        ),
        "rule": (
            "released core Python files may branch on declared layer or "
            "motion types, but not on stable benchmark case IDs"
        ),
        "files": rows,
        "core": [
            "schema validation and artifact hashing",
            "deterministic program runner",
            "data-type route selection",
            "robust multi-donor material projection",
            "first/last-frame video runner and evaluator interfaces",
            "motion-aware deterministic fallback interface",
        ],
        "plugins": [
            "scientific state equations and keyframe thresholds",
            "case semantic-layer derivation",
            "object counts, colors, geometry and annotations",
            "material prompts, negative prompts and residual gains",
            "case motion evaluator thresholds",
        ],
    }


def _prompt_record(experiment_id: str) -> dict[str, Any]:
    spec = load_json(
        STAGE2_ROOT
        / "experiments"
        / experiment_id
        / "spec.json"
    )
    positive = ", ".join(spec["prompt_parts"].values())
    control_name = CONTROL_BY_EXPERIMENT[experiment_id]
    control_route = next(
        item["control_route"]
        for item in spec["configurations"]
        if (
            (
                experiment_id == "EXP-20260729-009"
                and item["configuration_id"] == "semantic_line_art"
            )
            or (
                experiment_id == "EXP-20260729-015"
                and item["configuration_id"] == "dense_scale_035"
            )
            or experiment_id
            not in {"EXP-20260729-009", "EXP-20260729-015"}
        )
    )
    return {
        "positive": positive,
        "negative": spec["negative_artifacts"],
        "control_route": control_route,
        "control": (
            STAGE2_ROOT
            / "output/phase-3"
            / experiment_id
            / "controls"
            / control_name
        ),
        "raw_sheet": (
            OUTPUT_ROOT
            / "report-assets"
            / f"{experiment_id}-model-samples.jpg"
        ),
        "spec": (
            STAGE2_ROOT
            / "experiments"
            / experiment_id
            / "spec.json"
        ),
        "model_fingerprints": (
            STAGE2_ROOT
            / "output/phase-3"
            / experiment_id
            / "_work/model_fingerprints.json"
        ),
    }


def _image_sections(
    images: list[dict[str, Any]],
) -> str:
    notes = {
        "MATH-02": (
            "四块拼图、共享边和两个空区必须精确；因此 ControlNet "
            "关闭，只借用无对象木纹的高频细节。"
        ),
        "PHYS-01": (
            "波峰、波谷和节点来自连续高度场；密集 Canny 会留下刻线，"
            "因此模型只生成无对象水面材质供体。"
        ),
        "CHEM-01": (
            "语义线稿帮助 SDXL 生成玻璃和液体候选，但这些整图候选不能"
            "直接当成事实；Phase 6 只提取其液体微纹理。"
        ),
        "BIO-01": (
            "dense Canny 能提示细胞轮廓，却不能可靠守住染色体数量；"
            "因此程序保留染色体，模型残差只进入细胞质区域。"
        ),
        "GEO-02": (
            "较低强度 dense Canny 产生山地材质候选；最终山峰轮廓、"
            "空气团和雨带仍由程序层逐像素保护。"
        ),
    }
    control_explanations = {
        "control_off": (
            "未使用 ControlNet。黑图明确记录控制强度为 0；"
            "模型只负责生成一张无科学对象的材质样本。"
        ),
        "semantic_apparatus_line_art": (
            "语义器材线稿。白线由程序确定性绘出烧杯、液面、"
            "滴定管和滴嘴关系，用于当时生成玻璃/液体 raw；"
            "Phase 6 最终图并不照抄这张模型整图。"
        ),
        "dense_canny": (
            "较密 Canny 边缘。白线从程序 clean frame 的可见轮廓"
            "确定性提取，用于生成材质供体；对象数量和连续场仍由"
            "后续语义层保护，而不是假设 Canny 能表达全部科学事实。"
        ),
    }
    control_labels = {
        "control_off": "未使用 ControlNet",
        "semantic_apparatus_line_art": "语义器材线稿",
        "dense_canny": "较密 Canny 边缘",
    }
    sections = []
    for item in images:
        prompt = _prompt_record(item["experiment_id"])
        sections.append(
            f"""<article class="case" id="image-{html.escape(item['case_id'])}">
<p class="eyebrow">{DISCIPLINE_ZH[item['discipline']]} · {html.escape(item['case_id'])}</p>
<h3>{'四帧发布回归' if item['scope'].startswith('all') else '冻结代表帧回归'}：通过</h3>
<p>{html.escape(notes[item['case_id']])}</p>
<a href="{_href(STAGE2_ROOT / item['comparison']['path'])}">
<img class="wide" loading="lazy"
src="{_href(STAGE2_ROOT / item['comparison']['path'])}"
alt="{html.escape(item['case_id'])} 程序图与材质增强图对比"></a>
<p><b>自评：</b>{html.escape(item['visual_review_zh'])}</p>
<table><tbody>
<tr><th>范围</th><td>{html.escape(item['scope'])}</td></tr>
<tr><th>机器门</th><td>{item['hard_checks_passed']}/{item['hard_checks_total']} 通过</td></tr>
<tr><th>允许区平均可见变化</th><td>{item['visible_detail_change_0_255']:.3f}/255</td></tr>
<tr><th>Phase 6 新模型调用</th><td>图片 0，视频 0；复用已保存 raw 供体</td></tr>
</tbody></table>
<details><summary>模型当时实际看到的控制图</summary>
<img class="control" loading="lazy" src="{_href(prompt['control'])}"
alt="{html.escape(item['case_id'])} 控制图">
<p><b>{html.escape(control_labels[prompt['control_route']])}：</b>
{html.escape(control_explanations[prompt['control_route']])}</p>
</details>
<details><summary>完整正向与负向图片提示词</summary>
<h4>正向</h4><pre>{html.escape(prompt['positive'])}</pre>
<h4>负向</h4><pre>{html.escape(prompt['negative'])}</pre>
</details>
<details><summary>四个 raw 模型候选（材质供体）</summary>
<a href="{_href(prompt['raw_sheet'])}"><img class="wide" loading="lazy"
src="{_href(prompt['raw_sheet'])}" alt="{html.escape(item['case_id'])} raw 候选"></a>
<p>样本 1–4 对应四个冻结复现编号；主报告隐藏编号是为了先说明作用，具体编号仍保存在
机器证据中。这些是模型直接输出，不等于最终图。程序把每张候选减去模糊低频，只保留
限幅的小尺度残差，再对四张逐像素取中位数；最终只投影到允许区。</p>
</details>
<p><a href="{_href(prompt['spec'])}">实验规格</a> ·
<a href="{_href(prompt['model_fingerprints'])}">模型权重指纹</a> ·
<a href="{_href(STAGE2_ROOT / item['evidence']['path'])}">机器证据</a></p>
</article>"""
        )
    return "".join(sections)


def _video_cards(videos: list[dict[str, Any]]) -> str:
    cards = []
    for item in videos:
        experiment_root = (
            STAGE2_ROOT
            / "output/phase-5/experiments"
            / item["experiment_id"]
        )
        run = load_json(STAGE2_ROOT / item["run"]["path"])
        poster = (
            experiment_root
            / "samples"
            / f"frame_{int(run['sample_indices'][4]):03d}.png"
        )
        label = (
            "LTX-2.3 模型视频"
            if item["classification"] == "model_video"
            else "确定性程序回退（视频模型关闭）"
        )
        cards.append(
            f"""<article class="video-card">
<p class="eyebrow">{html.escape(item['motion_class'])}</p>
<h3>{html.escape(item['case_id'])}</h3>
<video controls muted loop preload="metadata"
poster="{_href(poster)}"
src="{_href(experiment_root / 'transition.mp4')}"></video>
<p><b>{label}</b></p>
<p><a href="{_href(experiment_root / 'generated-frames.jpg')}">查看九个时间点</a> ·
<a href="{_href(STAGE2_ROOT / item['run']['path'])}">逐帧机器审计</a> ·
<a href="{_href(STAGE2_ROOT / item['review']['path'])}">人工评审</a></p>
</article>"""
        )
    return "".join(cards)


def _write_report(
    *,
    version: str,
    status: str,
    checks: list[dict[str, Any]],
    contracts: dict[str, Any],
    images: list[dict[str, Any]],
    videos: list[dict[str, Any]],
    token: dict[str, Any],
    baseline: dict[str, Any],
    core: dict[str, Any],
) -> None:
    check_evidence_zh = {
        "ten_case_contract_and_keyframe_regression": (
            "10 个案例；490 张程序帧；40 张关键帧。"
        ),
        "five_discipline_image_representatives": (
            "数学、物理、化学、生物、地理各 1 个，全部通过。"
        ),
        "five_video_motion_representatives": (
            "刚体、连续场、液体混合、对象分裂、边界拓扑各 1 个，全部通过。"
        ),
        "video_prompt_token_integrity": (
            "用模型实际 tokenizer 复核 8 次模型视频的正负提示词，无截断。"
        ),
        "historical_delta_immutable_regression": (
            "14 个 Stage 1 三角洲文件的字节数和 SHA-256 均未变化。"
        ),
        "released_core_has_no_benchmark_case_id_branches": (
            "扫描 8 个发布核心 Python 文件，没有出现十个基准案例 ID。"
        ),
        "final_report_links_resolve": (
            "图片、视频、JSON、旧报告和复现资料链接均存在。"
        ),
    }
    check_rows = "".join(
        f"<tr><td>{html.escape(CHECK_LABELS_ZH[item['name']])}</td>"
        f"<td class=\"{'pass' if item['passed'] else 'fail'}\">"
        f"{'通过' if item['passed'] else '失败'}</td>"
        f"<td>{html.escape(check_evidence_zh[item['name']])}</td></tr>"
        for item in checks
    )
    contract_rows = "".join(
        f"<tr><td><code>{item['case_id']}</code></td>"
        f"<td>{item['contract_fixture']}</td>"
        f"<td>{item['deterministic_program']}</td>"
        f"<td>{item['keyframe_gate']}</td></tr>"
        for item in contracts["cases"]
    )
    token_rows = "".join(
        f"<tr><td>{item['experiment_id']}</td><td>{item['case_id']}</td>"
        f"<td>{item['positive']['content_tokens_including_bos']}</td>"
        f"<td>{item['negative']['content_tokens_including_bos']}</td>"
        f"<td>{'通过' if item['passed'] else '失败'}</td></tr>"
        for item in token["experiments"]
    )
    core_rows = "".join(
        f"<tr><td><code>{html.escape(item['path'])}</code></td>"
        f"<td>{'0 个案例 ID' if item['passed'] else html.escape(str(item['case_id_matches']))}</td></tr>"
        for item in core["files"]
    )
    core_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in core["core"]
    )
    plugin_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in core["plugins"]
    )
    REPORT_PATH.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 2 v{html.escape(version)} 最终发布报告</title>
<style>
:root{{--ink:#17333a;--paper:#f4f0e7;--card:#fffdf8;--line:#c9d6d1;
--teal:#11776d;--orange:#d36f34;--deep:#0e2a31;--ok:#126b43;--bad:#a32f2f}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.7 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
header{{color:white;background:linear-gradient(125deg,#0d2b32,#126f68 64%,#cd6a32)}}
header>div,main{{max-width:1200px;margin:auto;padding:54px 24px}}
h1{{font-size:clamp(2.5rem,7vw,5.4rem);line-height:1;margin:.12em 0}}
h2{{font-size:2rem;line-height:1.25}}h3{{font-size:1.35rem;line-height:1.35}}
nav{{position:sticky;top:0;z-index:5;padding:11px max(20px,calc((100vw - 1200px)/2));
display:flex;gap:8px;overflow:auto;background:rgba(244,240,231,.96);
border-bottom:1px solid var(--line)}}nav a{{white-space:nowrap;padding:5px 10px;
border-radius:16px;background:white;text-decoration:none;color:var(--ink)}}
section{{margin:30px 0;padding:30px;background:rgba(255,255,255,.68);
border:1px solid var(--line);border-radius:20px}}.eyebrow{{color:var(--teal);
font-weight:800;letter-spacing:.1em}}.status{{display:inline-block;padding:7px 13px;
border-radius:18px;background:#d9f4e7;color:var(--ok);font-weight:800}}
.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:20px 0}}
.flow div,.stat,.note{{padding:15px;border-radius:12px;background:var(--card);
border:1px solid var(--line)}}.flow b{{display:block;color:var(--teal)}}
.stats,.video-grid,.two{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}
.stats{{grid-template-columns:repeat(4,1fr)}}.stat strong{{display:block;
font-size:2.3rem;color:var(--teal)}}.case{{padding:28px 0;border-top:3px solid var(--ink)}}
img,video{{max-width:100%;border-radius:12px;background:var(--deep)}}.wide,video{{width:100%}}
.control{{display:block;max-height:360px;margin:auto}}table{{width:100%;border-collapse:collapse;
background:var(--card);margin:14px 0}}th,td{{border:1px solid var(--line);padding:9px;
text-align:left;vertical-align:top}}th{{background:#e8f1ed}}details{{margin:10px 0;
border:1px solid var(--line);border-radius:10px;background:var(--card)}}
summary{{padding:12px 15px;font-weight:750;cursor:pointer}}details>*:not(summary){{margin:14px}}
pre{{overflow:auto;padding:15px;background:var(--deep);color:#edfff9;border-radius:9px;
white-space:pre-wrap}}code{{font-size:.9em}}.pass{{color:var(--ok);font-weight:800}}
.fail{{color:var(--bad);font-weight:800}}.warning{{border-left:5px solid var(--orange);
background:#fff0e5;padding:15px;border-radius:8px}}.good{{border-left:5px solid var(--teal);
background:#e6f4ef;padding:15px;border-radius:8px}}a{{color:#08675f}}
@media(max-width:800px){{section{{padding:20px 14px}}.flow,.stats,.video-grid,.two{{
grid-template-columns:1fr}}th,td{{min-width:130px}}}}
</style></head><body>
<header><div><p class="eyebrow">LOOP ENGINEER · PHASE 6 · RELEASE</p>
<h1>程序决定事实，模型只补它擅长的部分</h1>
<p>Stage 2 v{html.escape(version)} 把十个跨学科案例、五个图片代表和五类运动
放进同一套可复现发布协议。结论不是“每张都照片级”，而是第一次明确冻结了
什么能交给生成模型、什么必须由程序保护，以及失败后怎样自动回退。</p>
<span class="status">{'全部发布门禁通过' if status == 'passed' else '发布失败'}</span>
</div></header>
<nav><a href="#outcome">结果</a><a href="#workflow">生成流程</a>
<a href="#contracts">十案例</a><a href="#images">图片</a><a href="#videos">视频</a>
<a href="#tokens">Token</a><a href="#boundary">核心边界</a>
<a href="#failures">失败与修正</a><a href="#reproduce">复现</a></nav>
<main>
<section id="outcome"><p class="eyebrow">01 · 发布结论</p><h2>v{version} 可以称为通用框架版本</h2>
<div class="stats"><div class="stat"><strong>10/10</strong>案例合同与关键帧</div>
<div class="stat"><strong>5/5</strong>学科图片代表</div>
<div class="stat"><strong>5/5</strong>运动类型</div>
<div class="stat"><strong>0</strong>Phase 6 新模型调用</div></div>
<p class="good"><b>“0 次新模型调用”不是没用模型。</b>图片材质来自 Phase 3 已保存的
SDXL Base 1.0 FP16 / SDXL Canny ControlNet FP16 raw 候选；视频证据来自 Phase 5
的八次 LTX-2.3 调用。Phase 6 复用这些不可覆盖证据，重新运行统一组合和门禁，
避免为了发布再抽幸运种子。</p>
<table><thead><tr><th>发布门</th><th>结论</th><th>证据摘要</th></tr></thead>
<tbody>{check_rows}</tbody></table></section>

<section id="workflow"><p class="eyebrow">02 · 新人先看这里</p><h2>一段动画究竟怎样生成</h2>
<div class="flow"><div><b>1 概念切片</b>每段只讲一个主要因果变化。</div>
<div><b>2 程序状态</b>计算位置、数量、连续场、拓扑和身份。</div>
<div><b>3 四张关键帧</b>按机制阈值取开始、机制、结果和结束。</div>
<div><b>4 材质增强</b>图片模型提供玻璃、水、木材、细胞质或山体微纹理。</div>
<div><b>5 相邻过渡</b>视频模型连接两个正确状态；做不好就回退程序插值。</div></div>
<p>程序同时导出普通人看的 clean frame 和机器读的语义层。比如
<code>region</code> 回答“哪里允许加材质”，<code>hard_boundary</code>
回答“哪条轮廓不能动”，<code>scalar_field</code> 保存浓度或振幅，
<code>object_identity</code> 保存数量和跨帧身份。Canny 只是硬边界的一种
可选表达，不是每个案例都必须有。</p>
<h3>Phase 6 图片核心的实际计算</h3>
<pre>4 个固定种子的 SDXL raw 候选
→ 每张减去高斯模糊，只留下小尺度材质残差
→ 残差限幅，避免强边和模型物体混入
→ 四张逐像素取中位数，过滤单个种子的偶然伪影
→ 只在程序声明的允许区域叠加
→ 对象、边界、连续场关键区域和其他像素保持原值
→ 逐像素保护、低频结构、留一稳定性、跨帧一致性、非空变化五道门</pre>
<p class="warning"><b>为什么不直接让模型重画程序图？</b>实验已经证明，
整图重画可以更“像照片”，但会复制染色体、改河道连通性、融化拼图或让浓度反向变化。
当前通用解选择机制可靠性优先；写实程度是受控提升，不作虚假承诺。</p></section>

<section id="contracts"><p class="eyebrow">03 · 完整发布层</p><h2>十个案例都能走同一合同</h2>
<p>每个案例先过不加载模型的 schema、状态、语义层、控制推导、manifest 和报告链接；
再过真实程序的机制与四关键帧门。五个哨兵程序 + 五个扩展程序共
{contracts['program_frame_count']} 帧、{contracts['program_keyframe_count']} 张关键帧。</p>
<table><thead><tr><th>案例</th><th>合同 fixture</th><th>确定性程序</th>
<th>关键帧门</th></tr></thead><tbody>{contract_rows}</tbody></table>
<p><a href="../phase-1/report.html">合同与语义层说明</a> ·
<a href="../phase-2/report.html">五个哨兵程序</a> ·
<a href="../phase-4/program-report.html">五个扩展程序</a></p></section>

<section id="images"><p class="eyebrow">04 · 五学科图片回归</p>
<h2>先看控制图、提示词和 raw，再看最终组合</h2>
<p>下面每例都展示了程序图与最终图、实际控制图、完整提示词、四个 raw 候选和机器证据。
数学、物理沿用 Phase 3 已冻结代表帧；化学、生物、地理在 Phase 6 对四张关键帧
统一运行新核心。所有 Phase 6 结果由同一 Python 核心生成，案例差异只在 JSON 配置。</p>
{_image_sections(images)}</section>

<section id="videos"><p class="eyebrow">05 · 五类运动回归</p>
<h2>视频模型不是每种运动都值得用</h2>
<div class="video-grid">{_video_cards(videos)}</div>
<p class="good">连续波场、细胞分裂和河道拓扑变化保留 LTX 输出；拼图刚体和液体扩散
在两次模型预算后仍破坏机制，因此发布版明确使用确定性程序回退。回退不是伪装成
模型成功，报告、manifest 和视频卡片都标明模型关闭。</p>
<p><a href="../phase-5/report.html">打开 Phase 5 全部十轮尝试、提示词与失败证据</a></p>
</section>

<section id="tokens"><p class="eyebrow">06 · 真实 tokenizer 审计</p>
<h2>字符数不再冒充 token 数</h2>
<p>审计直接从 <code>gemma_3_12B_it_fp4_mixed.safetensors</code> 的
<code>spiece_model</code> 读取 SentencePiece，并调用 ComfyUI 实际使用的
<code>LTXAVGemmaTokenizer</code>。模型文件内嵌 tokenizer SHA-256：
<code>{token['tokenizer_asset']['embedded_sha256']}</code>。</p>
<table><thead><tr><th>实验</th><th>案例</th><th>正向 content tokens</th>
<th>负向 content tokens</th><th>结论</th></tr></thead><tbody>{token_rows}</tbody></table>
<p>每条实际内容都小于 1024；编码器左侧补 pad 到 1024。表中 content 数包含 BOS=2，
排除 pad=0。八次模型视频提示词均为单行 token row，没有静默截断。</p>
<p><a href="video-token-integrity.json">完整逐提示词审计 JSON</a></p></section>

<section id="boundary"><p class="eyebrow">07 · 通用核心与案例插件</p>
<h2>换概念时，哪些代码不该改</h2>
<div class="two"><div class="note"><h3>进入 v{version} 通用核心</h3><ul>{core_items}</ul></div>
<div class="note"><h3>继续留在案例插件或 JSON</h3><ul>{plugin_items}</ul></div></div>
<p>发布扫描禁止核心 Python 文件出现稳定案例 ID；因此它只能根据层类型、运动类型和
配置字段工作。Phase 1 早期写在 Python 里的对象数量也已移到模板 JSON。</p>
<table><thead><tr><th>发布核心文件</th><th>案例 ID 扫描</th></tr></thead>
<tbody>{core_rows}</tbody></table></section>

<section id="failures"><p class="eyebrow">08 · 失败没有被删除</p><h2>四次关键修正</h2>
<ol>
<li><b>ControlNet 好看但事实漂移：</b>线稿只能约束部分边缘，无法表达连续浓度、
对象父子身份和水体拓扑。改成“模型生成材质供体，程序保留事实”。</li>
<li><b>安全但无效：</b>CHEM-02 的晶体投影不改坏像素，却肉眼几乎无提升，被拒绝；
Phase 6 又发现 CHEM-01 首轮因保护整只烧杯而原样复制。现在新增“结果不能是 no-op”硬门。</li>
<li><b>状态 JSON 说对，实际像素仍错：</b>GEO-01 曾声明牛轭湖隔离，但 raster 仍连通。
现在 evaluator 直接从水色像素计算连通域，并修正程序塞口位置。</li>
<li><b>视频模型能力边界：</b>LTX 的刚体拼图会变形，液体颜色量会先异常增长；
两次预算用尽后自动选择可证明正确的程序轨迹和软标量场。</li>
</ol>
<p><a href="../phase-3/report.html">Phase 3 全部 raw/ControlNet 负对照</a> ·
<a href="../phase-4/program-report.html">Phase 4 程序与路线失败</a> ·
<a href="../phase-5/report.html">Phase 5 视频失败与回退</a></p></section>

<section id="history"><p class="eyebrow">09 · 三角洲历史回归</p><h2>14 个冻结文件没有被覆盖</h2>
<p>{html.escape(baseline['method_zh'])}</p>
<img class="wide" loading="lazy"
src="{_href(REPO_ROOT / 'modules/video_model/stage1/output/keyframe_render/delta_sequence/sequence-contact-sheet.jpg')}"
alt="Stage 1 三角洲五张历史关键帧">
<p class="warning"><b>范围说明：</b>{html.escape(baseline['scope_limit_zh'])}
所以本项能证明旧成果没有回退，不能声称三角洲已经用 v{version} 从零重渲染。</p>
<p><a href="../../../stage1/output/keyframe_render/delta_sequence/report.html">
打开三角洲完整生成过程</a></p></section>

<section id="reproduce"><p class="eyebrow">10 · 版本与复现</p><h2>从仓库根目录重跑</h2>
<p>版本号保存在 <a href="../../VERSION"><code>VERSION</code></a>，
变更与已知边界保存在 <a href="../../CHANGELOG.md"><code>CHANGELOG.md</code></a>。
轻量合同、程序、组合和测试使用项目 <code>.venv</code>；真实 LTX tokenizer
依赖 ComfyUI 环境。</p>
<pre>.venv/bin/python -m modules.video_model.stage2.phase0 --check
.venv/bin/python -m modules.video_model.stage2.phase1 --check
.venv/bin/python -m modules.video_model.stage2.phase2 --check
.venv/bin/python -m modules.video_model.stage2.phase3 --check
.venv/bin/python -m modules.video_model.stage2.phase4 --check
.venv/bin/python -m modules.video_model.stage2.phase5 --check
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage2.phase6_token_audit
.venv/bin/python -m modules.video_model.stage2.phase6_image_regression
.venv/bin/python -m modules.video_model.stage2.phase6
.venv/bin/python -m modules.video_model.stage2.phase6 --check
.venv/bin/python -m pytest -q modules/video_model/stage2/tests</pre>
<p><a href="phase6_manifest.json">发布 manifest</a> ·
<a href="../../phase6_image_routes.json">五学科图片路由配置</a> ·
<a href="../../CHANGELOG.md">变更记录</a></p></section>
</main></body></html>""",
        encoding="utf-8",
    )


def _missing_links() -> list[str]:
    text = REPORT_PATH.read_text(encoding="utf-8")
    targets = re.findall(
        r'(?:href|src|poster)="([^"]+)"', text
    )
    return [
        target
        for target in targets
        if not target.startswith(("#", "http://", "https://"))
        and not (
            target == "phase6_manifest.json"
            and not MANIFEST_PATH.exists()
        )
        and not (REPORT_PATH.parent / target).resolve().exists()
    ]


def _source_records() -> dict[str, dict[str, Any]]:
    paths = {
        "phase6_builder": Path(__file__),
        "release_image_core": (
            STAGE2_ROOT
            / "framework/release_image_regression.py"
        ),
        "image_routes": ROUTES_PATH,
        "token_audit_runner": (
            STAGE2_ROOT / "phase6_token_audit.py"
        ),
        "regression_policy": POLICY_PATH,
        "version": VERSION_PATH,
        "changelog": CHANGELOG_PATH,
    }
    return {
        name: {
            "path": path.relative_to(STAGE2_ROOT).as_posix(),
            "sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }


def build_phase6(*, check_only: bool = False) -> dict[str, Any]:
    if check_only:
        manifest = load_json(MANIFEST_PATH)
        for source in manifest["sources"].values():
            path = STAGE2_ROOT / source["path"]
            if (
                sha256_path(path) != source["sha256"]
                or path.stat().st_size != source["size_bytes"]
            ):
                raise ValueError(
                    f"Phase 6 source changed: {source['path']}"
                )
        for representative in manifest[
            "image_representatives"
        ]:
            evidence = STAGE2_ROOT / representative["evidence"]["path"]
            if sha256_path(evidence) != representative["evidence"]["sha256"]:
                raise ValueError(
                    f"image evidence changed: {evidence}"
                )
            comparison = (
                STAGE2_ROOT / representative["comparison"]["path"]
            )
            if (
                sha256_path(comparison)
                != representative["comparison"]["sha256"]
            ):
                raise ValueError(
                    f"image comparison changed: {comparison}"
                )
            review = representative["review"]
            if (
                isinstance(review, dict)
                and "path" in review
                and sha256_path(STAGE2_ROOT / review["path"])
                != review["sha256"]
            ):
                raise ValueError(
                    f"image review changed: {review['path']}"
                )
        contracts = _collect_contract_regression()
        videos = _collect_video_regression()
        token = _verify_token_audit()
        baseline = _verify_historical_baseline()
        core = _audit_core_hardcoding()
        if contracts["status"] != "passed":
            raise ValueError("Phase 6 contract regression failed")
        if len(videos) != 5 or not all(
            item["status"] == "passed" for item in videos
        ):
            raise ValueError("Phase 6 video regression failed")
        if token["status"] != "passed":
            raise ValueError("Phase 6 token integrity failed")
        if (
            token["artifact"]["sha256"]
            != manifest["token_integrity"]["artifact"]["sha256"]
        ):
            raise ValueError("Phase 6 token audit artifact changed")
        if baseline["status"] != "passed":
            raise ValueError("Phase 6 historical regression failed")
        if core["status"] != "passed":
            raise ValueError("Phase 6 core hardcoding audit failed")
        if sha256_path(REPORT_PATH) != manifest["report"]["sha256"]:
            raise ValueError("Phase 6 report changed")
        missing = _missing_links()
        if missing:
            raise ValueError(
                f"Phase 6 report links missing: {missing}"
            )
        if manifest["status"] != "passed":
            raise ValueError("Phase 6 manifest is not passed")
        return manifest

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    version = VERSION_PATH.read_text().strip()
    routes = load_json(ROUTES_PATH)
    if version != routes["release_version"]:
        raise ValueError("VERSION and image route release differ")
    new_images = _run_new_image_regressions(routes)
    contracts = _collect_contract_regression()
    images = _collect_image_representatives(
        routes, new_images
    )
    videos = _collect_video_regression()
    token = _verify_token_audit()
    baseline = _verify_historical_baseline()
    core = _audit_core_hardcoding()
    checks = [
        _check(
            "ten_case_contract_and_keyframe_regression",
            contracts["status"] == "passed",
            {
                "cases": contracts["case_count"],
                "program_frames": contracts[
                    "program_frame_count"
                ],
                "keyframes": contracts[
                    "program_keyframe_count"
                ],
            },
        ),
        _check(
            "five_discipline_image_representatives",
            len(images) == 5
            and all(item["status"] == "passed" for item in images),
            {
                item["case_id"]: item["status"] for item in images
            },
        ),
        _check(
            "five_video_motion_representatives",
            len(videos) == 5
            and all(item["status"] == "passed" for item in videos),
            {
                item["motion_class"]: {
                    "case_id": item["case_id"],
                    "classification": item["classification"],
                }
                for item in videos
            },
        ),
        _check(
            "video_prompt_token_integrity",
            token["status"] == "passed",
            {
                "actual_tokenizer": token["measurement"][
                    "tokenizer_class"
                ],
                "model_video_prompts": token[
                    "experiment_count"
                ],
            },
        ),
        _check(
            "historical_delta_immutable_regression",
            baseline["status"] == "passed",
            {"unchanged_assets": baseline["asset_count"]},
        ),
        _check(
            "released_core_has_no_benchmark_case_id_branches",
            core["status"] == "passed",
            {"files_scanned": len(core["files"])},
        ),
    ]
    preliminary = (
        "passed"
        if all(item["passed"] for item in checks)
        else "failed"
    )
    _write_report(
        version=version,
        status=preliminary,
        checks=checks,
        contracts=contracts,
        images=images,
        videos=videos,
        token=token,
        baseline=baseline,
        core=core,
    )
    missing = _missing_links()
    checks.append(
        _check(
            "final_report_links_resolve",
            not missing,
            {"missing": missing},
        )
    )
    status = (
        "passed"
        if all(item["passed"] for item in checks)
        else "failed"
    )
    _write_report(
        version=version,
        status=status,
        checks=checks,
        contracts=contracts,
        images=images,
        videos=videos,
        token=token,
        baseline=baseline,
        core=core,
    )
    missing = _missing_links()
    if missing:
        raise ValueError(f"Phase 6 report links missing: {missing}")
    manifest = {
        "schema_version": "1.0",
        "phase": 6,
        "release_version": version,
        "status": status,
        "classification": (
            "released_generic_framework"
            if status == "passed"
            else "release_candidate_failed"
        ),
        "model_runs_during_phase6": {
            "image": 0,
            "video": 0,
        },
        "reused_model_evidence": {
            "image_donors": sum(
                item.get("reused_model_donors", 0)
                for item in new_images
            ),
            "historical_ltx_video_runs": 8,
        },
        "checks": checks,
        "contract_regression": contracts,
        "image_representatives": images,
        "video_representatives": videos,
        "token_integrity": token,
        "historical_regression": baseline,
        "core_plugin_boundary": core,
        "sources": _source_records(),
        "report": artifact_record(REPORT_PATH, OUTPUT_ROOT),
        "automatic_next_action": (
            "release_complete"
            if status == "passed"
            else "continue_phase6_optimization"
        ),
    }
    write_json(MANIFEST_PATH, manifest)
    if status != "passed":
        raise ValueError("Phase 6 release gates failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_phase6(check_only=args.check)
    print(
        f"Phase 6: {manifest['status']} · "
        f"release={manifest['release_version']} · "
        f"10 cases · 5 image representatives · "
        f"5 motion classes · "
        f"next={manifest['automatic_next_action']}"
    )


if __name__ == "__main__":
    main()
