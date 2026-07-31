"""Run Stage 3 S3.1 route-aware geometry/control experiments.

S3.1 is deliberately model-free. It validates all three geometry policies
before prompt or diffusion tuning:

- canonicalize: CHEM-01 target, PHYS-02 regression, MATH-02 isolation;
- preserve_exact: MATH-02 target, PHYS-01 and CHEM-01 regressions;
- layout_only: GEO-02 target, BIO-01 and PHYS-01 regressions, delta history.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    validate_loop_state,
    verify_file_record,
    write_json,
)
from modules.video_model.stage3.framework.geometry import (
    compile_control,
    compile_legacy_delta_layout,
    identity_preflight,
)


STAGE3 = Path(__file__).resolve().parent
VIDEO_MODEL = STAGE3.parent
STAGE2 = VIDEO_MODEL / "stage2"
STAGE1 = VIDEO_MODEL / "stage1"
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output" / "phase-1"
EXPERIMENTS = STAGE3 / "experiments"
KNOWLEDGE = STAGE3 / "knowledge"


COHORTS = {
    "canonicalize": {
        "target": "CHEM-01",
        "regressions": ["PHYS-02", "MATH-02"],
        "reason_zh": (
            "烧杯有人工上限却与程序断链；PHYS-02 检查器材库跨学科，"
            "MATH-02 检查该路线不会污染精确几何。"
        ),
    },
    "preserve_exact": {
        "target": "MATH-02",
        "regressions": ["PHYS-01", "CHEM-01"],
        "reason_zh": (
            "拼图的点、面积和对象身份不能近似；PHYS-01 检查连续高度场，"
            "CHEM-01 检查策略隔离。"
        ),
    },
    "layout_only": {
        "target": "GEO-02",
        "regressions": ["BIO-01", "PHYS-01"],
        "historical": ["GEO-HIST-DELTA-01"],
        "reason_zh": (
            "山地与有机细胞不应被粗程序轮廓锁死；只保留区域拓扑、"
            "对象锚点和场接口，并回归历史三角洲。"
        ),
    },
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def href(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace(
        os.sep, "/"
    )


def contract(case_id: str) -> dict[str, Any]:
    return load_json(STAGE3 / "contracts" / f"{case_id}.json")


def verify_g0(case_id: str) -> dict[str, Any]:
    value = contract(case_id)
    checks = []
    for item in value["keyframes"]:
        frame_pass = True
        for field in (
            "state",
            "clean_program_frame",
            "annotated_program_frame",
            "semantic_layers",
        ):
            try:
                verify_file_record(item[field], REPO_ROOT)
            except (ValueError, FileNotFoundError):
                frame_pass = False
        checks.append(
            {
                "name": f"{item['keyframe_id']}_artifact_hashes",
                "passed": frame_pass,
            }
        )
    policy_match = value["geometry_policy"] in {
        "preserve_exact",
        "canonicalize",
        "layout_only",
    }
    checks.append(
        {
            "name": "geometry_policy_supported",
            "passed": policy_match,
            "evidence": value["geometry_policy"],
        }
    )
    checks.append(
        {
            "name": "semantic_coordinate_source_declared",
            "passed": (
                "not inferred from the rendered screenshot"
                in value["semantic_exports"]["source_policy"]
            ),
        }
    )
    return {
        "case_id": case_id,
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def _thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", size, (16, 18, 18))
    image.thumbnail((size[0] - 20, size[1] - 54))
    x = (size[0] - image.width) // 2
    y = 38 + (size[1] - 48 - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_sheet(
    panels: list[tuple[str, Path]],
    output: Path,
    columns: int = 3,
) -> None:
    panel_size = (520, 340)
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * panel_size[0], rows * panel_size[1]),
        (236, 232, 221),
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(19)
    for index, (label, path) in enumerate(panels):
        panel = _thumbnail(path, panel_size)
        x = (index % columns) * panel_size[0]
        y = (index // columns) * panel_size[1]
        sheet.paste(panel, (x, y))
        draw.text((x + 14, y + 9), label, fill=(238, 245, 242), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=91, subsampling=0)


def line_proximity(
    automatic_path: Path, upper_bound_path: Path
) -> dict[str, float]:
    automatic = Image.open(automatic_path).convert("L")
    upper = Image.open(upper_bound_path).convert("L")
    if automatic.size != upper.size:
        upper = upper.resize(automatic.size, Image.Resampling.NEAREST)
    auto = np.asarray(automatic) > 0
    manual = np.asarray(upper) > 0
    auto_near = np.asarray(
        Image.fromarray(np.uint8(auto) * 255).filter(
            ImageFilter.MaxFilter(25)
        )
    ) > 0
    manual_near = np.asarray(
        Image.fromarray(np.uint8(manual) * 255).filter(
            ImageFilter.MaxFilter(25)
        )
    ) > 0
    return {
        "automatic_edge_density": round(float(auto.mean()), 6),
        "manual_edge_density": round(float(manual.mean()), 6),
        "automatic_within_12px_of_manual": round(
            float(manual_near[auto].mean()), 6
        ),
        "manual_within_12px_of_automatic": round(
            float(auto_near[manual].mean()), 6
        ),
    }


def write_experiment(
    experiment_id: str,
    hypothesis: dict[str, Any],
    spec: dict[str, Any],
    review: dict[str, Any],
) -> None:
    root = EXPERIMENTS / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    hypothesis_text = (
        f"# {experiment_id}\n\n"
        f"假设：{hypothesis['statement_zh']}\n\n"
        f"证伪条件：{hypothesis['falsification_zh']}\n"
    )
    (root / "hypothesis.md").write_text(
        hypothesis_text, encoding="utf-8"
    )
    write_json(root / "spec.json", spec)
    write_json(root / "review.json", review)


def append_hypothesis(record: dict[str, Any]) -> None:
    path = KNOWLEDGE / "hypotheses.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        )


def run_controls() -> tuple[
    list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]
]:
    all_case_ids = [
        "CHEM-01",
        "PHYS-02",
        "MATH-02",
        "PHYS-01",
        "GEO-02",
        "BIO-01",
        "GEO-HIST-DELTA-01",
    ]
    g0 = [verify_g0(case_id) for case_id in all_case_ids]
    if not all(item["passed"] for item in g0):
        raise RuntimeError("G0 failure")
    write_json(
        OUTPUT / "g0.json",
        {
            "schema_version": "1.0",
            "passed": True,
            "records": g0,
        },
    )

    # H1 is intentionally tested before the legacy adapter. It must fail
    # because CHEM-01 does not identify the burette as a typed object.
    direct = identity_preflight(contract("CHEM-01"), REPO_ROOT)
    h1 = {
        "hypothesis_id": "H-S3-0001A",
        "statement_zh": (
            "现有 object_identity 已足够让规范几何库直接重建烧杯与滴定管。"
        ),
        "falsification_zh": (
            "任何稳定器材在任一关键帧缺少 typed identity 即失败。"
        ),
    }
    direct_review = {
        "schema_version": "1.0",
        "experiment_id": "EXP-S3-20260730-001",
        "verdict": "rejected",
        "model_runs": {"image": 0, "video": 0},
        "reason_zh": (
            "四个 CHEM-01 关键帧都只有 glass_beaker 身份；"
            "硬边界里可见的滴定管没有 glass_burette 身份。"
        ),
        "evidence": direct,
    }
    write_experiment(
        "EXP-S3-20260730-001",
        h1,
        {
            "change": "none; direct identity coverage preflight",
            "target": "CHEM-01",
            "image_candidate_budget": 0,
        },
        direct_review,
    )
    append_hypothesis(
        {
            **h1,
            "verdict": "rejected",
            "experiment_id": "EXP-S3-20260730-001",
            "failure_taxonomy": "semantic_export",
        }
    )
    if direct["passed"]:
        raise RuntimeError("H1 unexpectedly passed; fixture changed")

    h2 = {
        "hypothesis_id": "H-S3-0001B",
        "statement_zh": (
            "合同声明缺失类型与空间关系后，Semantic Normalizer 可从未归属的"
            " hard_boundary 分量恢复 bbox，再由通用 typed primitive 重建器出图；"
            "全程不读取 RGB 程序截图或外观参考。"
        ),
        "falsification_zh": (
            "若恢复对象数量错误、越界、控制图密度异常、跨学科器材失败，"
            "或 preserve_exact payload 改变任一字节，即失败。"
        ),
    }

    gates: list[dict[str, Any]] = []
    compile_sets = {
        "CHEM-01": ["00_start", "01_mechanism", "02_result", "03_end"],
        "PHYS-02": ["00_start", "01_mechanism", "02_result", "03_end"],
        "MATH-02": ["00_start", "01_mechanism", "02_result", "03_end"],
        "PHYS-01": ["00_start", "01_mechanism", "02_result", "03_end"],
        "GEO-02": ["00_start", "01_mechanism", "02_result", "03_end"],
        "BIO-01": ["00_start", "01_mechanism", "02_result", "03_end"],
    }
    for case_id, keyframes in compile_sets.items():
        value = contract(case_id)
        for keyframe_id in keyframes:
            gate = compile_control(
                value,
                keyframe_id,
                OUTPUT
                / "controls"
                / case_id
                / keyframe_id,
                REPO_ROOT,
            )
            gates.append(gate)
    delta_gate = compile_legacy_delta_layout(
        contract("GEO-HIST-DELTA-01"),
        119,
        OUTPUT / "controls" / "GEO-HIST-DELTA-01" / "119_end",
        REPO_ROOT,
    )
    gates.append(delta_gate)
    if not all(gate["passed"] for gate in gates):
        failures = [
            (gate["case_id"], gate.get("keyframe_id"))
            for gate in gates
            if not gate["passed"]
        ]
        raise RuntimeError(f"G1 failure: {failures}")

    auto_control = (
        OUTPUT
        / "controls"
        / "CHEM-01"
        / "00_start"
        / "structure_control.png"
    )
    manual_control = (
        STAGE2
        / "experiments"
        / "EXP-20260729-009"
        / "semantic_apparatus_line_art.png"
    )
    proximity = line_proximity(auto_control, manual_control)
    write_json(OUTPUT / "manual-upper-bound-comparison.json", proximity)

    # Freeze one representative compiled artifact per policy and verify a
    # second compile is byte-identical.
    replay_dir = OUTPUT / "_replay"
    replay_specs = [
        ("CHEM-01", "00_start"),
        ("MATH-02", "01_mechanism"),
        ("GEO-02", "02_result"),
    ]
    replay_records = []
    for case_id, keyframe_id in replay_specs:
        first = (
            OUTPUT
            / "controls"
            / case_id
            / keyframe_id
            / "structure_control.png"
        )
        replay_gate = compile_control(
            contract(case_id),
            keyframe_id,
            replay_dir / case_id / keyframe_id,
            REPO_ROOT,
        )
        second = (
            replay_dir
            / case_id
            / keyframe_id
            / "structure_control.png"
        )
        replay_records.append(
            {
                "case_id": case_id,
                "keyframe_id": keyframe_id,
                "first_sha256": sha256_path(first),
                "replay_sha256": sha256_path(second),
                "passed": (
                    replay_gate["passed"]
                    and sha256_path(first) == sha256_path(second)
                ),
            }
        )
    if not all(item["passed"] for item in replay_records):
        raise RuntimeError("determinism replay failure")

    cohort_results = []
    for policy, cohort in COHORTS.items():
        relevant = {
            cohort["target"],
            *cohort["regressions"],
            *cohort.get("historical", []),
        }
        records = [
            gate for gate in gates if gate["case_id"] in relevant
        ]
        cohort_results.append(
            {
                "policy": policy,
                **cohort,
                "gate_record_count": len(records),
                "passed": all(item["passed"] for item in records),
            }
        )
    phase_passed = all(item["passed"] for item in cohort_results)
    review = {
        "schema_version": "1.0",
        "experiment_id": "EXP-S3-20260730-002",
        "verdict": "accepted_core" if phase_passed else "rejected",
        "model_runs": {"image": 0, "video": 0},
        "cohorts": cohort_results,
        "g1_record_count": len(gates),
        "determinism_replay": replay_records,
        "manual_upper_bound_diagnostic": proximity,
        "reason_zh": (
            "typed primitive、精确 payload 透传和 layout interface "
            "均通过跨案例 G1；自动烧杯线稿不复制人工模板，"
            "但器材类别、数量、关系与稀疏结构已达到进入 S3.2 的条件。"
        ),
    }
    write_experiment(
        "EXP-S3-20260730-002",
        h2,
        {
            "single_core_change": (
                "route-aware geometry resolver + typed semantic normalizer"
            ),
            "cohorts": COHORTS,
            "fixed_output_size": [1024, 576],
            "image_candidate_budget": 0,
            "video_candidate_budget": 0,
            "appearance_inputs_forbidden": True,
            "dense_canny_is_negative_control_only": True,
        },
        review,
    )
    append_hypothesis(
        {
            **h2,
            "verdict": review["verdict"],
            "experiment_id": "EXP-S3-20260730-002",
            "scope": (
                "core geometry/control compiler; appearance quality not tested"
            ),
        }
    )
    return gates, review, g0


def build_report_assets() -> dict[str, Path]:
    assets = OUTPUT / "report-assets"
    assets.mkdir(parents=True, exist_ok=True)
    chem = STAGE2 / "output" / "phase-2" / "CHEM-01"
    make_sheet(
        [
            (
                "1 program frame",
                chem / "keyframes" / "00_start" / "clean.png",
            ),
            (
                "2 object identity (missing burette)",
                chem
                / "keyframes"
                / "00_start"
                / "layers"
                / "chem01_object_identity_preview.png",
            ),
            (
                "3 semantic hard boundary",
                chem
                / "keyframes"
                / "00_start"
                / "layers"
                / "chem01_apparatus_boundary_preview.png",
            ),
            (
                "4 dense Canny (negative)",
                STAGE2
                / "output"
                / "phase-7"
                / "route-a"
                / "experiments"
                / "EXP-P7-A-chem-01-00_start"
                / "controls"
                / "dense_canny.png",
            ),
            (
                "5 manual upper bound",
                STAGE2
                / "experiments"
                / "EXP-20260729-009"
                / "semantic_apparatus_line_art.png",
            ),
            (
                "6 Stage 3 automatic control",
                OUTPUT
                / "controls"
                / "CHEM-01"
                / "00_start"
                / "structure_control.png",
            ),
        ],
        assets / "canonicalize-process.jpg",
        columns=3,
    )
    make_sheet(
        [
            (
                "PHYS-02 program",
                STAGE2
                / "output"
                / "phase-4"
                / "programs"
                / "PHYS-02"
                / "keyframes"
                / "01_mechanism"
                / "clean.png",
            ),
            (
                "PHYS-02 canonical control",
                OUTPUT
                / "controls"
                / "PHYS-02"
                / "01_mechanism"
                / "structure_control.png",
            ),
            (
                "MATH-02 program",
                STAGE2
                / "output"
                / "phase-2"
                / "MATH-02"
                / "keyframes"
                / "01_mechanism"
                / "clean.png",
            ),
            (
                "MATH-02 exact control",
                OUTPUT
                / "controls"
                / "MATH-02"
                / "01_mechanism"
                / "structure_control.png",
            ),
            (
                "GEO-02 program",
                STAGE2
                / "output"
                / "phase-2"
                / "GEO-02"
                / "keyframes"
                / "02_result"
                / "clean.png",
            ),
            (
                "GEO-02 sparse layout",
                OUTPUT
                / "controls"
                / "GEO-02"
                / "02_result"
                / "structure_control.png",
            ),
            (
                "BIO-01 program",
                STAGE2
                / "output"
                / "phase-2"
                / "BIO-01"
                / "keyframes"
                / "02_result"
                / "clean.png",
            ),
            (
                "BIO-01 region + anchors",
                OUTPUT
                / "controls"
                / "BIO-01"
                / "02_result"
                / "structure_control.png",
            ),
        ],
        assets / "cross-discipline-controls.jpg",
        columns=2,
    )
    delta = STAGE1 / "output" / "causal_delta"
    make_sheet(
        [
            ("legacy program frame", delta / "frames" / "0104.png"),
            (
                "state-derived shoreline",
                OUTPUT
                / "controls"
                / "GEO-HIST-DELTA-01"
                / "119_end"
                / "structure_control.png",
            ),
            (
                "land / deposit / sandbar regions",
                OUTPUT
                / "controls"
                / "GEO-HIST-DELTA-01"
                / "119_end"
                / "regions.png",
            ),
            (
                "separate sparse flow vectors",
                OUTPUT
                / "controls"
                / "GEO-HIST-DELTA-01"
                / "119_end"
                / "flow_anchors.png",
            ),
        ],
        assets / "delta-state-adapter.jpg",
        columns=2,
    )
    return {
        "canonicalize_process": assets / "canonicalize-process.jpg",
        "cross_discipline": assets / "cross-discipline-controls.jpg",
        "delta_adapter": assets / "delta-state-adapter.jpg",
    }


def check_report_links(report: Path) -> list[str]:
    text = report.read_text(encoding="utf-8")
    missing = []
    for marker in ("src='", "href='"):
        start = 0
        while True:
            index = text.find(marker, start)
            if index < 0:
                break
            begin = index + len(marker)
            end = text.find("'", begin)
            value = text[begin:end]
            start = end + 1
            if (
                not value
                or value.startswith(("#", "http:", "https:", "data:"))
            ):
                continue
            if not (report.parent / value).resolve().exists():
                missing.append(value)
    return sorted(set(missing))


def render_report(
    assets: dict[str, Path],
    review: dict[str, Any],
    checks: list[dict[str, Any]],
) -> Path:
    report = OUTPUT / "report.html"
    proximity = review["manual_upper_bound_diagnostic"]
    cohort_cards = []
    for item in review["cohorts"]:
        cohort_cards.append(
            "<article><span class='pass'>PASS</span>"
            f"<h3>{html.escape(item['policy'])}</h3>"
            f"<p>目标：<strong>{item['target']}</strong><br>"
            f"回归：{html.escape(', '.join(item['regressions']))}</p>"
            f"<p>{html.escape(item['reason_zh'])}</p></article>"
        )
    check_items = "".join(
        f"<li>✓ {html.escape(item['name'])}"
        f"<small>{html.escape(item.get('evidence_zh', ''))}</small></li>"
        for item in checks
    )
    report.write_text(
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 3 · S3.1 几何重建与控制编译</title><style>
:root{--ink:#19221f;--paper:#f3efe4;--card:#fffdf7;--line:#d7cfc0;
--green:#176548;--red:#a13b31;--blue:#225d75;--muted:#65716b}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:Inter,system-ui,-apple-system,"Noto Sans SC",sans-serif;line-height:1.7}
header{padding:60px max(5vw,24px) 42px;background:#173c33;color:white}
h1{font-size:clamp(34px,5vw,58px);line-height:1.08;margin:.15em 0}
header p{max-width:950px;font-size:18px;color:#dcebe5}main{max-width:1220px;
margin:auto;padding:30px 24px 80px}section{background:var(--card);border:1px solid var(--line);
border-radius:17px;padding:28px;margin:22px 0}h2{font-size:28px;margin:0 0 10px}
h3{margin:.45em 0}.status,.pass{display:inline-block;border-radius:999px;padding:4px 10px;
font-weight:800;background:#dff2e8;color:var(--green)}.rejected{background:#f7ded9;color:var(--red)}
.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin:22px 0}
.flow div{background:#e7eef0;border-top:4px solid var(--blue);padding:12px;font-size:14px}
.flow strong{display:block;font-size:18px}.heroimg{width:100%;display:block;border-radius:12px;
background:#111}.caption{color:var(--muted);font-size:14px}.cards{display:grid;
grid-template-columns:repeat(3,1fr);gap:14px}.cards article{border:1px solid var(--line);
border-radius:12px;padding:18px;background:#faf8f1}.callout{border-left:6px solid var(--blue);
padding:14px 18px;background:#eaf2f4}.fail{border-left-color:var(--red);background:#fae8e3}
table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top}code{background:#e8ece8;padding:2px 5px;border-radius:4px}
li{margin:8px 0;color:var(--green);font-weight:700}li small{display:block;
color:var(--muted);font-weight:400;margin-left:22px}a{color:#16627d}
@media(max-width:850px){.flow{grid-template-columns:1fr 1fr}.cards{grid-template-columns:1fr}}
</style></head><body><header><span class="status">S3.1 PASS · 0 model runs</span>
<h1>程序形状不好时，不再手画一张“只对这个案例有效”的线稿</h1>
<p>本阶段把几何处理固定成三条可复现策略：精确几何原样通过、规范器材按类型重建、
自然与有机场景只保留布局和拓扑。所有控制图都由程序语义生成，没有读取最终好图的几何。</p>
</header><main>
<section><h2>自动流程实际做了什么</h2><div class="flow">
<div><strong>1 读合同</strong>确认 Case、画布、对象类型、区域与几何策略。</div>
<div><strong>2 语义预检</strong>检查每件稳定物体是否有类型、位置和稳定 ID。</div>
<div><strong>3 几何分流</strong>exact 透传；器材 canonicalize；自然形态 layout_only。</div>
<div><strong>4 编译控制</strong>分别输出 structure、regions、anchors 和 derivation。</div>
<div><strong>5 G1 检验</strong>查数量、范围、关系、拓扑、密度、文字泄漏和来源。</div>
</div></section>
<section><h2>烧杯：从一次性模板变成可追溯自动路线</h2>
<img class='heroimg' src='"""
        + html.escape(href(report.parent, assets["canonicalize_process"]))
        + """' alt='烧杯控制图完整生成过程'>
<p class='caption'>从左到右：程序图 → 对象身份 → 程序硬边界 → dense Canny 负对照 →
人工上限 → Stage 3 自动控制。黑白图不是渲染结果，而是告诉后续 ControlNet 稳定器材在哪里。</p>
<div class="callout fail"><span class="status rejected">H1 REJECTED</span>
<p><strong>第一次预检真的失败了：</strong>现有 identity 只有 <code>glass_beaker</code>，
没有 <code>glass_burette</code>。如果直接写几何重建器，滴定管就会消失。本 Loop 没有把失败藏掉，
也没有立刻调用模型。</p></div>
<div class="callout"><span class="status">H2 ACCEPTED CORE</span>
<p>合同补的是“缺少一个 glass_burette，且它由烧杯上方一对未归属平行硬边界表示”；
没有补最终像素坐标。Normalizer 从 hard_boundary 量出 bbox，typed primitive 库再画规范滴定管。
这解释了 Semantic Normalizer 的真实上下文：它位于旧程序导出与通用几何库之间，只修接口，不负责外观。</p></div>
<h3>与人工上限的诊断对照</h3>
<table><tr><th>指标</th><th>自动路线</th><th>含义</th></tr>
<tr><td>边缘密度</td><td>"""
        + str(proximity["automatic_edge_density"])
        + """</td><td>保持稀疏，不把整张程序图变成线。</td></tr>
<tr><td>自动线在人工线 12px 邻域内</td><td>"""
        + f"{proximity['automatic_within_12px_of_manual']:.1%}"
        + """</td><td>只用于评价接近程度；人工线从未作为自动路线输入。</td></tr>
<tr><td>人工线在自动线 12px 邻域内</td><td>"""
        + f"{proximity['manual_within_12px_of_automatic']:.1%}"
        + """</td><td>差异主要来自杯体比例和滴定管细节，不影响类别/数量/关系门禁。</td></tr>
</table></section>
<section><h2>不是只为烧杯：三个 cohort 全部完成</h2>
<div class="cards">"""
        + "".join(cohort_cards)
        + """</div><img class='heroimg' src='"""
        + html.escape(href(report.parent, assets["cross_discipline"]))
        + """' alt='跨学科三类控制策略'>
<p class='caption'>PHYS-02 把粗线圈重建为规范线圈；MATH-02 的三角形 payload 一字节不改；
GEO-02 只保留山地边界和空气团锚点，不把云雨标量场硬转成 Canny；BIO-01 保留细胞拓扑与染色体锚点。</p></section>
<section><h2>连续程序与历史三角洲没有被丢掉</h2>
<img class='heroimg' src='"""
        + html.escape(href(report.parent, assets["delta_adapter"]))
        + """' alt='三角洲状态网格适配'>
<p>历史三角洲尚无 Stage 3 标准 semantic_layers.json，但原程序的 120 行状态仍保存了
<code>land</code>、<code>new_land</code>、沉积厚度、粒子和 flow samples。legacy adapter
直接读取这些数组，分别输出岸线、区域和稀疏流向；没有从写实终帧反推。最终状态检查到
一个连通沙洲和两条绕流路径。</p></section>
<section><h2>G0 / G1 与复现</h2><ul>"""
        + check_items
        + """</ul><p>固定入口：<code>.venv/bin/python -m modules.video_model.stage3.phase1</code></p>
<p>机器可读证据：<a href='"""
        + html.escape(
            href(report.parent, EXPERIMENTS / "EXP-S3-20260730-001")
        )
        + """'>失败实验 001</a> · <a href='"""
        + html.escape(
            href(report.parent, EXPERIMENTS / "EXP-S3-20260730-002")
        )
        + """'>接受实验 002</a> · <a href='"""
        + html.escape(href(report.parent, OUTPUT / "phase1_manifest.json"))
        + """'>phase1_manifest.json</a></p></section>
<section><h2>自评与下一阶段</h2><p><strong>S3.1 通过。</strong>通过的含义仅是：
自动控制来源清楚、三类策略都有跨案例回归、输出可重放；它不表示 SDXL 图片已经变好，
因为本阶段刻意没有运行模型。</p><p><strong>自动进入 S3.2：</strong>冻结 SDXL 候选矩阵、
seed、ControlNet 强度、G2/G3 选择与 tie-break。下一阶段才会检验自动线稿能否稳定生成
接近视觉目标包的真实场景。</p></section>
</main></body></html>""",
        encoding="utf-8",
    )
    return report


def run() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gates, review, g0 = run_controls()
    assets = build_report_assets()
    checks = [
        {
            "name": "G0_all_selected_inputs_and_hashes_pass",
            "passed": all(item["passed"] for item in g0),
            "evidence_zh": "7 个 target/regression Case 的合同和关键帧文件签名均有效。",
        },
        {
            "name": "direct_identity_hypothesis_was_falsified",
            "passed": not identity_preflight(
                contract("CHEM-01"), REPO_ROOT
            )["passed"],
            "evidence_zh": "缺失 glass_burette 被归类为 semantic_export，而非静默手画。",
        },
        {
            "name": "all_three_geometry_policy_cohorts_pass",
            "passed": all(
                item["passed"] for item in review["cohorts"]
            ),
            "evidence_zh": "canonicalize / preserve_exact / layout_only 均有 target 和跨学科回归。",
        },
        {
            "name": "twenty_five_G1_records_pass",
            "passed": len(gates) == 25
            and all(item["passed"] for item in gates),
            "evidence_zh": "6 个正式案例各 4 帧 + 三角洲终态。",
        },
        {
            "name": "representative_controls_replay_byte_identically",
            "passed": all(
                item["passed"]
                for item in review["determinism_replay"]
            ),
            "evidence_zh": "三类策略各抽一例重新编译，PNG SHA-256 相同。",
        },
        {
            "name": "S3_1_used_no_image_or_video_model",
            "passed": True,
            "evidence_zh": "model_runs: image=0, video=0。",
        },
    ]
    if not all(item["passed"] for item in checks):
        raise RuntimeError("S3.1 exit check failure")

    report = render_report(assets, review, checks)
    # The manifest is a forward link until the final write.
    missing = [
        value
        for value in check_report_links(report)
        if value != "phase1_manifest.json"
    ]
    if missing:
        raise RuntimeError(f"report missing links: {missing}")
    checks.append(
        {
            "name": "report_links_resolve",
            "passed": True,
            "evidence_zh": "所有过程图、实验记录和本地证据可直接打开。",
        }
    )
    report = render_report(assets, review, checks)

    ledger_path = EXPERIMENTS / "ledger.json"
    ledger = load_json(ledger_path)
    ledger["experiments"] = [
        {
            "experiment_id": "EXP-S3-20260730-001",
            "phase": "S3.1",
            "hypothesis_id": "H-S3-0001A",
            "verdict": "rejected",
            "failure_taxonomy": "semantic_export",
            "model_runs": {"image": 0, "video": 0},
            "review": rel(
                EXPERIMENTS
                / "EXP-S3-20260730-001"
                / "review.json"
            ),
        },
        {
            "experiment_id": "EXP-S3-20260730-002",
            "phase": "S3.1",
            "hypothesis_id": "H-S3-0001B",
            "verdict": "accepted_core",
            "model_runs": {"image": 0, "video": 0},
            "review": rel(
                EXPERIMENTS
                / "EXP-S3-20260730-002"
                / "review.json"
            ),
        },
    ]
    write_json(ledger_path, ledger)

    baselines_path = STAGE3 / "baselines" / "accepted.json"
    baselines = load_json(baselines_path)
    new_baselines = [
        {
            "baseline_id": "CORE-GEOMETRY-COMPILER-V1",
            "kind": "accepted_core",
            **file_record(
                STAGE3 / "framework" / "geometry.py", REPO_ROOT
            ),
        }
    ]
    for case_id, keyframe_id in (
        ("CHEM-01", "00_start"),
        ("PHYS-02", "01_mechanism"),
        ("MATH-02", "01_mechanism"),
        ("GEO-02", "02_result"),
        ("BIO-01", "02_result"),
        ("GEO-HIST-DELTA-01", "119_end"),
    ):
        path = (
            OUTPUT
            / "controls"
            / case_id
            / keyframe_id
            / "structure_control.png"
        )
        new_baselines.append(
            {
                "baseline_id": (
                    f"CONTROL-{case_id}-{keyframe_id}-V1"
                ),
                "kind": "accepted_control",
                **file_record(path, REPO_ROOT),
            }
        )
    old_ids = {
        item["baseline_id"] for item in new_baselines
    }
    baselines["records"] = [
        item
        for item in baselines["records"]
        if item["baseline_id"] not in old_ids
    ] + new_baselines
    write_json(baselines_path, baselines)

    state = {
        "schema_version": "1.0",
        "loop_id": "LOOP-S3-0001",
        "phase": "S3.2",
        "phase_status": "in_progress",
        "exit_criteria": [
            "candidate matrix freezes models, scheduler, seeds and control scales",
            "G2/G3 reject geometry or mechanism failures before appearance rank",
            "same inputs select same candidate or same failure conclusion",
            "appearance_to_geometry_leakage hard gate passes",
            "no post-hoc seed or prompt additions",
        ],
        "budget": {
            "s3_2_image_candidate_limit": 18,
            "s3_2_video_candidate_limit": 0,
            "preflight_before_gpu_work": True,
        },
        "current_problem": {
            "problem_id": "S3-PROBLEM-SELECTOR-001",
            "taxonomy": "gate_or_selector",
            "summary_zh": (
                "几何控制已稳定，但现有模型候选仍依赖人工看图挑选；"
                "候选矩阵与选择规则尚未冻结。"
            ),
        },
        "current_hypothesis": {
            "hypothesis_id": "H-S3-0002",
            "statement_zh": (
                "固定模型、seed、控制强度、硬门禁、视觉量表与 tie-break "
                "后，同一输入能复现同一选中 ID 或同一失败结论。"
            ),
            "falsification_zh": (
                "重放选择不同、需要临时加 seed/prompt，"
                "或外观参考改变几何即失败。"
            ),
        },
        "current_cohort": {
            "target": "CHEM-01",
            "regressions": ["PHYS-02", "MATH-02"],
            "geometry_baseline": rel(
                OUTPUT / "controls" / "CHEM-01" / "00_start"
            ),
        },
        "next_action": (
            "Freeze the S3.2 candidate matrix and selectors; run token/model "
            "preflight before using the accepted automatic controls with SDXL."
        ),
    }
    validate_loop_state(state)
    write_json(STAGE3 / "state.json", state)
    write_json(
        KNOWLEDGE / "open_problems.json",
        {
            "schema_version": "1.0",
            "problems": [
                state["current_problem"],
                {
                    "problem_id": "S3-PROBLEM-VISUAL-001",
                    "taxonomy": "visual_target",
                    "summary_zh": "GEO-02 外观目标仍为 provisional。",
                },
            ],
        },
    )

    artifacts = {
        "report": file_record(report, REPO_ROOT),
        "g0": file_record(OUTPUT / "g0.json", REPO_ROOT),
        "canonicalize_process": file_record(
            assets["canonicalize_process"], REPO_ROOT
        ),
        "cross_discipline": file_record(
            assets["cross_discipline"], REPO_ROOT
        ),
        "delta_adapter": file_record(
            assets["delta_adapter"], REPO_ROOT
        ),
        "experiment_001": file_record(
            EXPERIMENTS
            / "EXP-S3-20260730-001"
            / "review.json",
            REPO_ROOT,
        ),
        "experiment_002": file_record(
            EXPERIMENTS
            / "EXP-S3-20260730-002"
            / "review.json",
            REPO_ROOT,
        ),
    }
    manifest = {
        "schema_version": "1.0",
        "phase": "S3.1",
        "status": "passed",
        "classification": "geometry_resolver_and_control_compiler",
        "model_runs": {"image": 0, "video": 0},
        "cohorts": review["cohorts"],
        "checks": checks,
        "g1_record_count": len(gates),
        "artifacts": artifacts,
        "next_phase": {
            "phase": "S3.2",
            "status": "in_progress",
            "target": "CHEM-01",
        },
    }
    write_json(OUTPUT / "phase1_manifest.json", manifest)
    missing = check_report_links(report)
    if missing:
        raise RuntimeError(f"final report missing links: {missing}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    run()
