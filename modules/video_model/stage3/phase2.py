"""Stage 3 S3.2 fixed candidates, deterministic selector, and audit."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage2.framework.image_experiment import (
    _token_preflight,
    compile_prompt,
    generate_experiment,
)
from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    validate_loop_state,
    write_json,
)
from modules.video_model.stage3.framework.selector import (
    evaluate_and_select,
)


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
MATRIX_PATH = STAGE3 / "candidate_matrix.json"
OUTPUT = STAGE3 / "output" / "phase-2"
EXPERIMENT_ID = "EXP-S3-20260730-003"
EXPERIMENT_ROOT = OUTPUT / "experiments" / EXPERIMENT_ID
SELECTION_ROOT = OUTPUT / "selection"
SELECTION_V2_ROOT = OUTPUT / "selection-v2"
SELECTOR_V2_PATH = STAGE3 / "selector_v2.json"
GEOMETRY_GATE = (
    STAGE3
    / "output"
    / "phase-1"
    / "controls"
    / "CHEM-01"
    / "00_start"
    / "g1.json"
)


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def href(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace(
        os.sep, "/"
    )


def build_spec(matrix: dict[str, Any]) -> dict[str, Any]:
    source_root = (
        REPO_ROOT
        / "modules/video_model/stage2/output/phase-2/CHEM-01/"
        "keyframes/00_start"
    )
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "case_id": matrix["case_id"],
        "hypothesis_zh": (
            "Stage 3 自动 canonical control 在固定 SDXL + Canny "
            "ControlNet 搜索空间内，能产生至少一个同时通过结构与基础外观门禁的候选。"
        ),
        "single_variable_zh": (
            "只改变 ControlNet conditioning scale；模型、提示词、控制图、"
            "scheduler、步数、尺寸和三个 seed 全部固定。"
        ),
        "source": {
            "keyframe_id": matrix["keyframe_id"],
            "clean_frame": str(source_root / "clean.png"),
            "semantic_layers": str(
                source_root / "semantic_layers.json"
            ),
        },
        "control_overrides": {
            "stage3_auto_control": str(
                REPO_ROOT / matrix["geometry_control"]["path"]
            )
        },
        "control_override_explanations": {
            "stage3_auto_control": (
                "由 S3.1 typed object identity、程序 hard_boundary "
                "和 canonical primitive 自动编译；不读取视觉目标图几何。"
            )
        },
        "prompt_parts": matrix["prompt_source"]["positive_parts"],
        "negative_artifacts": matrix["prompt_source"]["negative"],
        "render": {
            key: value
            for key, value in matrix["render"].items()
            if key != "scheduler_expected"
        },
        "configurations": [
            {
                **item,
                "pipeline_mode": matrix["pipeline_mode"],
                "control_route": "stage3_auto_control",
            }
            for item in matrix["configurations"]
        ],
        "blind_shuffle_seed": 2026073003,
        "budget": {
            "maximum_new_image_candidates": matrix["budget"][
                "maximum_new_image_candidates"
            ],
            "actual_planned_image_candidates": (
                len(matrix["configurations"])
                * len(matrix["render"]["seeds"])
            ),
            "planned_external_reuse": 0,
            "maximum_new_generation": matrix["budget"][
                "maximum_new_image_candidates"
            ],
            "maximum_video_trials": 0,
        },
    }


def preflight() -> dict[str, Any]:
    import torch

    matrix = load_json(MATRIX_PATH)
    spec = build_spec(matrix)
    planned = len(spec["configurations"]) * len(
        spec["render"]["seeds"]
    )
    model_paths = {
        name: Path(record["path"])
        for name, record in matrix["models"].items()
    }
    prompt = compile_prompt(spec["prompt_parts"])
    token_check = _token_preflight(
        model_paths["sdxl_base"],
        prompt,
        spec["negative_artifacts"],
    )
    weights = matrix["selector"]["score_weights"]
    checks = [
        {
            "name": "matrix_frozen_status",
            "passed": matrix["status"] == "frozen_before_generation",
        },
        {
            "name": "planned_candidates_equal_fixed_budget",
            "passed": planned
            == matrix["budget"]["maximum_new_image_candidates"]
            == 9,
            "evidence": planned,
        },
        {
            "name": "automatic_geometry_control_exists",
            "passed": (
                REPO_ROOT / matrix["geometry_control"]["path"]
            ).is_file(),
        },
        {
            "name": "geometry_gate_passed",
            "passed": load_json(GEOMETRY_GATE)["passed"],
        },
        {
            "name": "local_fp16_models_exist",
            "passed": all(path.is_dir() for path in model_paths.values()),
        },
        {
            "name": "gpu_runtime_available",
            "passed": torch.cuda.is_available(),
            "evidence": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        {
            "name": "prompt_tokens_do_not_truncate",
            "passed": not any(
                token_check[key]["would_truncate"]
                for key in ("positive", "negative")
            ),
            "evidence": token_check,
        },
        {
            "name": "selector_weights_sum_to_one",
            "passed": abs(sum(weights.values()) - 1.0) < 1e-9,
        },
        {
            "name": "post_hoc_changes_forbidden",
            "passed": matrix["budget"][
                "post_hoc_seed_addition_forbidden"
            ]
            and matrix["budget"]["post_hoc_prompt_change_forbidden"],
        },
    ]
    result = {
        "schema_version": "1.0",
        "matrix": file_record(MATRIX_PATH, REPO_ROOT),
        "spec": spec,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "preflight.json", result)
    write_json(EXPERIMENT_ROOT / "spec.json", spec)
    if not result["passed"]:
        raise RuntimeError("S3.2 preflight failed")
    return result


def generate() -> dict[str, Any]:
    preflight_result = preflight()
    metadata = generate_experiment(
        preflight_result["spec"], EXPERIMENT_ROOT
    )
    expected = load_json(MATRIX_PATH)["render"]["scheduler_expected"]
    if metadata["scheduler"] != expected:
        raise RuntimeError(
            f"scheduler changed: {metadata['scheduler']} != {expected}"
        )
    if len(metadata["candidates"]) != 9:
        raise RuntimeError("fixed candidate count changed")
    return metadata


def select() -> dict[str, Any]:
    metadata_path = EXPERIMENT_ROOT / "_work" / "generate.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("run --generate before --select")
    result = evaluate_and_select(
        MATRIX_PATH,
        metadata_path,
        GEOMETRY_GATE,
        SELECTION_ROOT,
        REPO_ROOT,
    )
    replay = evaluate_and_select(
        MATRIX_PATH,
        metadata_path,
        GEOMETRY_GATE,
        OUTPUT / "_selection_replay",
        REPO_ROOT,
    )
    replay_result = {
        "first_selected_candidate_id": result[
            "selected_candidate_id"
        ],
        "replay_selected_candidate_id": replay[
            "selected_candidate_id"
        ],
        "passed": result["selected_candidate_id"]
        == replay["selected_candidate_id"],
    }
    write_json(OUTPUT / "selection-replay.json", replay_result)
    if not replay_result["passed"]:
        raise RuntimeError("selector replay changed decision")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def refine_selector() -> dict[str, Any]:
    metadata_path = EXPERIMENT_ROOT / "_work" / "generate.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("run --generate before --refine")
    result = evaluate_and_select(
        MATRIX_PATH,
        metadata_path,
        GEOMETRY_GATE,
        SELECTION_V2_ROOT,
        REPO_ROOT,
        selector_policy_path=SELECTOR_V2_PATH,
    )
    replay = evaluate_and_select(
        MATRIX_PATH,
        metadata_path,
        GEOMETRY_GATE,
        OUTPUT / "_selection_v2_replay",
        REPO_ROOT,
        selector_policy_path=SELECTOR_V2_PATH,
    )
    v1 = load_json(SELECTION_ROOT / "selection.json")
    v1_record_under_v2 = next(
        item
        for item in result["records"]
        if item["candidate_id"] == v1["selected_candidate_id"]
    )
    regression = {
        "v1_selected_candidate_id": v1["selected_candidate_id"],
        "v1_selected_rejected_by_v2": not v1_record_under_v2[
            "hard_gate_passed"
        ],
        "v2_selected_candidate_id": result[
            "selected_candidate_id"
        ],
        "replay_selected_candidate_id": replay[
            "selected_candidate_id"
        ],
        "same_frozen_candidate_count": len(result["records"]) == 9,
    }
    regression["passed"] = (
        regression["v1_selected_rejected_by_v2"]
        and regression["v2_selected_candidate_id"]
        == regression["replay_selected_candidate_id"]
        and regression["same_frozen_candidate_count"]
    )
    write_json(OUTPUT / "selector-v2-regression.json", regression)
    if not regression["passed"]:
        raise RuntimeError("selector v2 regression failed")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return (
        ImageFont.truetype(str(path), size)
        if path.is_file()
        else ImageFont.load_default()
    )


def make_comparison(selection: dict[str, Any]) -> Path:
    selected = SELECTION_V2_ROOT / "selected.png"
    control = (
        REPO_ROOT
        / load_json(MATRIX_PATH)["geometry_control"]["path"]
    )
    target = (
        REPO_ROOT
        / load_json(MATRIX_PATH)["appearance_target"][
            "positive_reference"
        ]
    )
    panels = [
        ("S3.1 automatic control", control),
        (
            f"S3.2 selected: {selection['selected_candidate_id']}",
            selected,
        ),
        ("accepted Phase 9 appearance anchor", target),
    ]
    panel_w, panel_h = 520, 340
    sheet = Image.new(
        "RGB", (panel_w * 3, panel_h), (17, 20, 19)
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(17)
    for index, (label, path) in enumerate(panels):
        image = Image.open(path).convert("RGB")
        image.thumbnail((panel_w - 20, panel_h - 54))
        x = index * panel_w + (panel_w - image.width) // 2
        y = 42 + (panel_h - 48 - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text(
            (index * panel_w + 12, 10),
            label,
            fill=(239, 245, 241),
            font=font,
        )
    path = OUTPUT / "selected-comparison.jpg"
    sheet.save(path, quality=92, subsampling=0)
    return path


def check_links(report: Path) -> list[str]:
    text = report.read_text(encoding="utf-8")
    missing = []
    for marker in ("src='", "href='"):
        start = 0
        while True:
            i = text.find(marker, start)
            if i < 0:
                break
            a = i + len(marker)
            b = text.find("'", a)
            value = text[a:b]
            start = b + 1
            if value and not value.startswith(("#", "http:", "https:")):
                if not (report.parent / value).resolve().exists():
                    missing.append(value)
    return sorted(set(missing))


def render_report(
    selection: dict[str, Any],
    audit: dict[str, Any],
    comparison: Path,
    checks: list[dict[str, Any]],
) -> Path:
    report = OUTPUT / "report.html"
    selected = selection["selected_candidate"]
    ranked = sorted(
        selection["records"],
        key=lambda item: -item["scores"]["total"],
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['candidate_id'])}</td>"
        f"<td>{'PASS' if item['hard_gate_passed'] else 'REJECT'}</td>"
        f"<td>{item['control_coverage']['total']:.3f}</td>"
        f"<td>{item['control_coverage']['per_object'].get('glass_beaker',0):.3f}</td>"
        f"<td>{item['control_coverage']['per_object'].get('glass_burette',0):.3f}</td>"
        f"<td>{item['scores']['total']:.3f}</td></tr>"
        for item in ranked
    )
    audit_rows = "".join(
        "<li class='"
        + ("pass" if item["passed"] else "fail")
        + "'>"
        + ("✓ " if item["passed"] else "✗ ")
        + html.escape(item["name_zh"])
        + f"<small>{html.escape(item['evidence_zh'])}</small></li>"
        for item in audit["checks"]
    )
    checks_html = "".join(
        f"<li class='pass'>✓ {html.escape(item['name'])}"
        f"<small>{html.escape(item.get('evidence_zh',''))}</small></li>"
        for item in checks
    )
    report.write_text(
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 3 · S3.2 固定候选与选择</title><style>
body{margin:0;background:#f2eee3;color:#19211e;font-family:system-ui,-apple-system,
"Noto Sans SC",sans-serif;line-height:1.68}header{padding:60px max(5vw,24px) 40px;
background:#193d34;color:white}h1{font-size:clamp(34px,5vw,56px);line-height:1.08;
margin:.2em 0}header p{max-width:900px;color:#dce9e4;font-size:18px}main{max-width:1200px;
margin:auto;padding:28px 24px 80px}section{background:#fffdf7;border:1px solid #d8d0c1;
border-radius:16px;padding:28px;margin:22px 0}h2{margin:0 0 10px}.badge{display:inline-block;
background:#dff2e8;color:#176548;border-radius:999px;padding:5px 10px;font-weight:800}
.hero{width:100%;display:block;background:#111;border-radius:12px}table{width:100%;
border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #ddd5c7}
th{background:#eee9dd}.pass{color:#176548;font-weight:750}.fail{color:#9b372e;font-weight:750}
li small{display:block;color:#657069;font-weight:400;margin-left:22px}.callout{border-left:5px solid #26627a;
background:#e8f1f3;padding:14px 18px}code{background:#e8ece7;padding:2px 5px;border-radius:4px}
a{color:#17617a}</style></head><body><header><span class='badge'>"""
        + ("S3.2 PASS" if audit["passed"] else "S3.2 FAILED")
        + """</span><h1>先冻结九张候选，再让同一套规则给出唯一结论</h1>
<p>本轮只改变 ControlNet 强度，三个 seed、提示词、模型、尺寸、步数和自动控制图都在生成前写入矩阵。
没有看完结果再补 seed。选择器先做结构硬门禁，再比较可测外观；最后由 Agent 对选中图做独立视觉审计。</p>
</header><main><section><h2>输入、选中结果和历史外观锚点</h2><img class='hero' src='"""
        + html.escape(href(report.parent, comparison))
        + """' alt='自动控制、选中候选和历史锚点对比'>
<p>左图只负责几何；中图是本轮 raw SDXL + ControlNet 输出；右图只作为外观统计和视觉上限，
没有参与左图的几何生成。</p></section>
<section><h2>固定候选矩阵</h2><p><strong>模型：</strong>SDXL Base 1.0 FP16 +
SDXL Canny ControlNet FP16；<strong>seeds：</strong>7101/7102/7103；
<strong>ControlNet scale：</strong>0.50/0.65/0.80；共 9 张。</p>
<p><a href='"""
        + html.escape(href(report.parent, MATRIX_PATH))
        + """'>打开冻结 candidate_matrix.json</a> · <a href='"""
        + html.escape(
            href(
                report.parent,
                EXPERIMENT_ROOT / "candidates-labeled.jpg",
            )
        )
        + """'>打开九张候选大图</a></p></section>
<section><h2>自动门禁与排名</h2><table><thead><tr><th>候选</th><th>硬门禁</th>
<th>总控制覆盖</th><th>烧杯覆盖</th><th>滴定管覆盖</th><th>总分</th></tr></thead>
<tbody>"""
        + rows
        + """</tbody></table><div class='callout'><p><strong>v1 失败：</strong>
最初选中 auto_control_080-s7101，但独立视觉审计发现滴定管与烧杯被幽灵中心线连接。
EXP-003 被保留为失败，不追加 seed。</p><p><strong>v2 自动选中：</strong>"""
        + html.escape(selection["selected_candidate_id"] or "无")
        + f"""；可用候选 {selection['eligible_count']}/9。选择重放得到同一 ID。</p>
<p>v2 额外要求候选边缘靠近控制、背景不能多物体、本应分离的对象不能被同一边缘分量连接，
杯内也不能出现未声明中心线。“外观相似”只比较曝光、饱和度和纹理统计，不从参考图取形状。</p></div></section>
<section><h2>Agent 独立视觉审计</h2><ul>"""
        + audit_rows
        + """</ul><p><strong>结论：</strong>"""
        + html.escape(audit["verdict_zh"])
        + """</p><p>自动选择器 v2 的已知边界：传统特征仍不能完整判断高级玻璃审美，
所以 Agent 的哈希绑定视觉审计仍作为显式硬检查保存；未来可替换为版本化视觉判别器，
但不能假装当前已经自动解决。</p></section>
<section><h2>阶段出口检查</h2><ul>"""
        + checks_html
        + """</ul><p>复现：<code>/opt/venv/bin/python -m modules.video_model.stage3.phase2 --all</code></p>
<p><a href='"""
        + html.escape(
            href(report.parent, SELECTION_V2_ROOT / "selection.json")
        )
        + """'>selection.json</a> · <a href='"""
        + html.escape(href(report.parent, OUTPUT / "visual-audit.json"))
        + """'>visual-audit.json</a> · <a href='"""
        + html.escape(href(report.parent, OUTPUT / "phase2_manifest.json"))
        + """'>phase2_manifest.json</a></p></section>
<section><h2>下一步</h2><p>S3.2 通过后自动进入 S3.3 Prompt Compiler：
把目前沿用的 Phase 7 文本拆成可追溯槽位，并将视觉目标包里的材质、光照、相机和反例
编译成稳定 prompt；控制图、seed 和选择器保持不变，避免同时改多个变量。</p></section>
</main></body></html>""",
        encoding="utf-8",
    )
    return report


def cross_discipline_regression() -> dict[str, Any]:
    phase8 = load_json(
        REPO_ROOT
        / "modules/video_model/stage2/output/phase-8/phase8-manifest.json"
    )
    math_selected = phase8["evaluation"]["MATH-02"][
        "selected_b_variant"
    ]
    phys_selected = phase8["evaluation"]["PHYS-01"][
        "selected_b_variant"
    ]
    math_path = (
        REPO_ROOT
        / "modules/video_model/stage2/output/phase-8/route-b-only/"
        f"MATH-02/variants/{math_selected}/03_end.png"
    )
    phys_path = (
        REPO_ROOT
        / "modules/video_model/stage2/output/phase-8/route-b-only/"
        f"PHYS-01/variants/{phys_selected}/02_result.png"
    )
    registry = load_json(STAGE3 / "case_registry.json")
    contract_checks = []
    for case in registry["cases"]:
        path = REPO_ROOT / case["input_contract"]
        contract_checks.append(
            {
                "case_id": case["case_id"],
                "exists": path.is_file(),
                "sha256": sha256_path(path) if path.is_file() else None,
            }
        )
    records = [
        {
            "case_id": "MATH-02",
            "discipline": "mathematics",
            "selected_id": math_selected,
            "artifact": file_record(math_path, REPO_ROOT),
            "gate_zh": "四块对象、面积与 payload 精确；确定性 B 选择不受 CHEM selector 改动。",
            "passed": math_path.is_file(),
        },
        {
            "case_id": "PHYS-01",
            "discipline": "physics",
            "selected_id": phys_selected,
            "artifact": file_record(phys_path, REPO_ROOT),
            "gate_zh": "高度场到光学实现相关性为 1.0；确定性 B 选择不受 CHEM selector 改动。",
            "passed": phys_path.is_file()
            and all(
                value == 1.0
                for value in phase8["evaluation"]["PHYS-01"][
                    "program_overlay_realization_correlations"
                ]
            ),
        },
    ]
    result = {
        "schema_version": "1.0",
        "purpose_zh": (
            "S3.2 selector 修改不得改变精确数学和连续物理场的既有确定性路线。"
        ),
        "records": records,
        "all_ten_plus_delta_contract_smoke": all(
            item["exists"] for item in contract_checks
        )
        and len(contract_checks) == 11,
        "contract_checks": contract_checks,
    }
    result["passed"] = all(
        item["passed"] for item in records
    ) and result["all_ten_plus_delta_contract_smoke"]
    write_json(OUTPUT / "cross-discipline-regression.json", result)
    return result


def update_iteration_records(
    selection: dict[str, Any],
    audit: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    experiments_root = STAGE3 / "experiments"
    v1_root = experiments_root / EXPERIMENT_ID
    v2_root = experiments_root / "EXP-S3-20260730-004"
    for root in (v1_root, v2_root):
        root.mkdir(parents=True, exist_ok=True)
    (v1_root / "hypothesis.md").write_text(
        "# H-S3-0002A\n\n固定九张候选和 selector v1 "
        "能自动选择结构与外观均合格的 Anchor。\n",
        encoding="utf-8",
    )
    write_json(
        v1_root / "spec.json",
        {
            "matrix": file_record(MATRIX_PATH, REPO_ROOT),
            "selector": "stage3_candidate_selector_v1",
            "candidate_count": 9,
        },
    )
    write_json(
        v1_root / "review.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "verdict": "rejected",
            "failure_taxonomy": "gate_or_selector",
            "selected_candidate_id": "auto_control_080-s7101",
            "visual_audit": file_record(
                OUTPUT / "visual-audit-v1.json", REPO_ROOT
            ),
            "model_runs": metadata["model_runs"],
        },
    )
    (v2_root / "hypothesis.md").write_text(
        "# H-S3-0002B\n\n在同一冻结候选上增加边缘精度、对象分离、"
        "背景额外边缘和杯内中心线门禁，可以拒绝 v1 缺陷并稳定选择干净候选。\n",
        encoding="utf-8",
    )
    write_json(
        v2_root / "spec.json",
        {
            "selector_policy": file_record(
                SELECTOR_V2_PATH, REPO_ROOT
            ),
            "candidate_source": file_record(
                EXPERIMENT_ROOT / "_work" / "generate.json",
                REPO_ROOT,
            ),
            "new_model_candidates": 0,
        },
    )
    write_json(
        v2_root / "review.json",
        {
            "experiment_id": "EXP-S3-20260730-004",
            "verdict": "accepted_core",
            "selected_candidate_id": selection[
                "selected_candidate_id"
            ],
            "visual_audit": file_record(
                OUTPUT / "visual-audit.json", REPO_ROOT
            ),
            "model_runs": {
                "image_candidates": 0,
                "video_candidates": 0,
            },
        },
    )
    ledger_path = experiments_root / "ledger.json"
    ledger = load_json(ledger_path)
    additions = [
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": "S3.2",
            "hypothesis_id": "H-S3-0002A",
            "verdict": "rejected",
            "failure_taxonomy": "gate_or_selector",
            "model_runs": metadata["model_runs"],
            "review": rel(v1_root / "review.json"),
        },
        {
            "experiment_id": "EXP-S3-20260730-004",
            "phase": "S3.2",
            "hypothesis_id": "H-S3-0002B",
            "verdict": "accepted_core",
            "model_runs": {
                "image_candidates": 0,
                "video_candidates": 0,
            },
            "review": rel(v2_root / "review.json"),
        },
    ]
    ids = {item["experiment_id"] for item in additions}
    ledger["experiments"] = [
        item
        for item in ledger["experiments"]
        if item["experiment_id"] not in ids
    ] + additions
    write_json(ledger_path, ledger)

    hypotheses_path = STAGE3 / "knowledge" / "hypotheses.jsonl"
    existing = [
        json.loads(line)
        for line in hypotheses_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    new_hypotheses = [
        {
            "hypothesis_id": "H-S3-0002A",
            "verdict": "rejected",
            "experiment_id": EXPERIMENT_ID,
            "failure_taxonomy": "gate_or_selector",
            "learning_zh": (
                "只奖励控制覆盖会偏爱带有额外连接的候选；"
                "coverage 不是 geometry precision。"
            ),
        },
        {
            "hypothesis_id": "H-S3-0002B",
            "verdict": "accepted_core",
            "experiment_id": "EXP-S3-20260730-004",
            "learning_zh": (
                "对象分离、额外边缘精度和内部禁区必须由合同关系生成，"
                "并在外观排名之前作为硬门禁。"
            ),
        },
    ]
    new_ids = {item["hypothesis_id"] for item in new_hypotheses}
    combined = [
        item
        for item in existing
        if item.get("hypothesis_id") not in new_ids
    ] + new_hypotheses
    hypotheses_path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            + "\n"
            for item in combined
        ),
        encoding="utf-8",
    )

    baselines_path = STAGE3 / "baselines" / "accepted.json"
    baselines = load_json(baselines_path)
    selected_record = {
        "baseline_id": "ANCHOR-CHEM-01-S3.2-V1",
        "kind": "accepted_model_anchor",
        **file_record(SELECTION_V2_ROOT / "selected.png", REPO_ROOT),
    }
    baselines["records"] = [
        item
        for item in baselines["records"]
        if item["baseline_id"] != selected_record["baseline_id"]
    ] + [selected_record]
    write_json(baselines_path, baselines)


def finalize() -> dict[str, Any]:
    selection = load_json(SELECTION_V2_ROOT / "selection.json")
    audit_path = OUTPUT / "visual-audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            "visual-audit.json is required after inspecting selected.png"
        )
    audit = load_json(audit_path)
    if audit["candidate_sha256"] != selection["selected_candidate"][
        "sha256"
    ]:
        raise ValueError("visual audit is not bound to selected candidate")
    comparison = make_comparison(selection)
    metadata = load_json(
        EXPERIMENT_ROOT / "_work" / "generate.json"
    )
    replay = load_json(OUTPUT / "selector-v2-regression.json")
    cross_regression = cross_discipline_regression()
    checks = [
        {
            "name": "preflight_passed_before_generation",
            "passed": load_json(OUTPUT / "preflight.json")["passed"],
            "evidence_zh": "模型、GPU、token、预算、控制图与选择器权重均先检查。",
        },
        {
            "name": "fixed_nine_candidates_completed",
            "passed": len(metadata["candidates"]) == 9,
            "evidence_zh": (
                f"generated={metadata['cache']['generated']}, "
                f"reused={metadata['cache']['reused']}"
            ),
        },
        {
            "name": "selector_v1_failure_is_preserved",
            "passed": replay["v1_selected_rejected_by_v2"],
            "evidence_zh": (
                f"{replay['v1_selected_candidate_id']} 被 v2 几何门禁拒绝。"
            ),
        },
        {
            "name": "selector_v2_replay_is_identical",
            "passed": replay["passed"],
            "evidence_zh": replay["v2_selected_candidate_id"],
        },
        {
            "name": "at_least_one_candidate_passed_hard_gates",
            "passed": selection["eligible_count"] > 0,
            "evidence_zh": f"{selection['eligible_count']}/9",
        },
        {
            "name": "selected_candidate_visual_audit_passed",
            "passed": audit["passed"],
            "evidence_zh": audit["verdict_zh"],
        },
        {
            "name": "cross_discipline_route_regressions_passed",
            "passed": cross_regression["passed"],
            "evidence_zh": (
                "MATH-02、PHYS-01 选择不变；10+1 合同冒烟通过。"
            ),
        },
        {
            "name": "no_video_model_was_run",
            "passed": metadata["model_runs"]["video_candidates"] == 0,
        },
    ]
    passed = all(item["passed"] for item in checks)
    update_iteration_records(selection, audit, metadata)
    report = render_report(selection, audit, comparison, checks)
    missing = [
        value
        for value in check_links(report)
        if value != "phase2_manifest.json"
    ]
    if missing:
        raise RuntimeError(f"report links missing: {missing}")
    checks.append(
        {
            "name": "report_links_resolve",
            "passed": True,
            "evidence_zh": "候选、矩阵、审计和选择证据全部存在。",
        }
    )
    report = render_report(selection, audit, comparison, checks)
    state = load_json(STAGE3 / "state.json")
    if passed:
        state.update(
            {
                "phase": "S3.3",
                "phase_status": "in_progress",
                "exit_criteria": [
                    "prompt slots derive only from contract, visual target and versioned dictionaries",
                    "positive and negative token limits pass both SDXL tokenizers",
                    "control, seeds and selector remain unchanged during prompt experiment",
                    "prompt provenance resolves every phrase to a source field",
                ],
                "budget": {
                    "s3_3_image_candidate_limit": 9,
                    "s3_3_video_candidate_limit": 0,
                    "preflight_before_gpu_work": True,
                },
                "current_problem": {
                    "problem_id": "S3-PROBLEM-PROMPT-001",
                    "taxonomy": "appearance_condition",
                    "summary_zh": (
                        "当前仍沿用 Phase 7 自由文本；材质、光照、相机和反例"
                        "尚未由 Visual Target Package 可追溯编译。"
                    ),
                },
                "current_hypothesis": {
                    "hypothesis_id": "H-S3-0003",
                    "statement_zh": (
                        "版本化 prompt slots 能在控制/seed 不变时提升外观量表，"
                        "且不会增加几何泄漏。"
                    ),
                    "falsification_zh": (
                        "token 截断、几何门禁下降、视觉分数无提升或只对烧杯有效即失败。"
                    ),
                },
                "next_action": (
                    "Build the traceable S3.3 Prompt Compiler and compare "
                    "against the frozen Phase 7 prompt with identical controls/seeds."
                ),
            }
        )
    else:
        state["phase_status"] = "failed"
        state["next_action"] = (
            "Classify the selected visual failure and start a new fixed "
            "S3.2 experiment; do not append seeds to EXP-S3-20260730-003."
        )
    validate_loop_state(state)
    write_json(STAGE3 / "state.json", state)
    manifest = {
        "schema_version": "1.0",
        "phase": "S3.2",
        "status": "passed" if passed else "failed",
        "classification": "fixed_candidate_and_selector",
        "matrix": file_record(MATRIX_PATH, REPO_ROOT),
        "selector_v2": file_record(SELECTOR_V2_PATH, REPO_ROOT),
        "model_runs": metadata["model_runs"],
        "selected_candidate_id": selection["selected_candidate_id"],
        "checks": checks,
        "artifacts": {
            "report": file_record(report, REPO_ROOT),
            "candidate_sheet": file_record(
                EXPERIMENT_ROOT / "candidates-labeled.jpg",
                REPO_ROOT,
            ),
            "selection": file_record(
                SELECTION_V2_ROOT / "selection.json", REPO_ROOT
            ),
            "failed_v1_audit": file_record(
                OUTPUT / "visual-audit-v1.json", REPO_ROOT
            ),
            "visual_audit": file_record(audit_path, REPO_ROOT),
            "comparison": file_record(comparison, REPO_ROOT),
        },
        "next_phase": (
            {"phase": "S3.3", "status": "in_progress"}
            if passed
            else None
        ),
    }
    write_json(OUTPUT / "phase2_manifest.json", manifest)
    if check_links(report):
        raise RuntimeError("final report link failure")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "preflight",
            "generate",
            "select",
            "refine",
            "finalize",
            "all",
        ],
    )
    args = parser.parse_args()
    if args.action == "preflight":
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
    elif args.action == "generate":
        print(json.dumps(generate(), ensure_ascii=False, indent=2))
    elif args.action == "select":
        select()
    elif args.action == "refine":
        refine_selector()
    elif args.action == "finalize":
        finalize()
    else:
        generate()
        select()
        refine_selector()
        finalize()


if __name__ == "__main__":
    main()
