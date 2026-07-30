"""Build and validate the deterministic Stage 2 Phase 0 audit.

Phase 0 deliberately performs no image- or video-model inference. It freezes
the case registry, scoring protocol, regression policy, compute budget, and
Stage 1 reference hashes, then renders a beginner-readable HTML report.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


STAGE2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STAGE2_ROOT.parents[2]
OUTPUT_ROOT = STAGE2_ROOT / "output" / "phase-0"

CONFIG_PATHS = {
    "case_registry": STAGE2_ROOT / "case_registry.json",
    "scoring_protocol": STAGE2_ROOT
    / "protocols"
    / "scoring_protocol.json",
    "regression_policy": STAGE2_ROOT
    / "protocols"
    / "regression_policy.json",
    "compute_budget": STAGE2_ROOT / "protocols" / "compute_budget.json",
    "stage1_baseline": STAGE2_ROOT
    / "benchmarks"
    / "stage1_baseline.json",
    "score_record_schema": STAGE2_ROOT
    / "framework"
    / "schemas"
    / "score_record.schema.json",
}

DISCIPLINES = {
    "mathematics": "数学",
    "physics": "物理",
    "chemistry": "化学",
    "biology": "生物",
    "geography": "地理",
}

LAYER_LABELS = {
    "hard_boundary": "硬边界",
    "region": "对象或允许修改区域",
    "scalar_field": "连续标量场",
    "vector_field": "矢量场",
    "height_or_normal": "高度或法线",
    "object_identity": "对象身份",
    "annotation": "教学叠加",
}

MOTION_LABELS = {
    "rigid_motion": "刚体运动",
    "continuous_field_propagation": "连续场传播",
    "liquid_mixing": "液体混合",
    "object_division": "对象分裂",
    "boundary_topology_change": "边界拓扑变化",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path = REPO_ROOT) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes repository: {value}") from error
    return path


def validate_registry(
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    cases = registry.get("cases", [])
    case_ids = [case.get("case_id") for case in cases]
    if len(cases) != 10 or len(case_ids) != len(set(case_ids)):
        raise ValueError("case registry must contain 10 unique cases")

    expected_disciplines = set(DISCIPLINES)
    actual_disciplines = {case.get("discipline") for case in cases}
    if actual_disciplines != expected_disciplines:
        raise ValueError(
            f"discipline mismatch: {sorted(actual_disciplines)}"
        )
    counts = Counter(case["discipline"] for case in cases)
    if any(counts[item] != 2 for item in expected_disciplines):
        raise ValueError(f"each discipline requires two cases: {counts}")

    sentinels = [case for case in cases if case.get("sentinel")]
    sentinel_counts = Counter(case["discipline"] for case in sentinels)
    if len(sentinels) != 5 or any(
        sentinel_counts[item] != 1 for item in expected_disciplines
    ):
        raise ValueError("one sentinel is required for every discipline")

    allowed_layers = set(registry.get("semantic_layer_types", []))
    if allowed_layers != set(LAYER_LABELS):
        raise ValueError("semantic layer type registry is incomplete")
    for case in cases:
        layers = case.get("primary_layer_types", [])
        if not layers or not set(layers) <= allowed_layers:
            raise ValueError(
                f"{case['case_id']} uses invalid semantic layer types"
            )
        if not case.get("motion_classes"):
            raise ValueError(f"{case['case_id']} has no motion class")
        if not case.get("case_hard_gates"):
            raise ValueError(f"{case['case_id']} has no case hard gates")

    source_path = _repo_path(registry["case_source"])
    source = source_path.read_text(encoding="utf-8")
    headings = re.findall(
        r"^(数学|物理|化学|生物|地理) [12]｜([A-Z]+-\d+)｜(.+)$",
        source,
        re.MULTILINE,
    )
    heading_ids = {case_id for _, case_id, _ in headings}
    if len(headings) != 10 or heading_ids != set(case_ids):
        raise ValueError("case.txt headings do not match case_registry.json")

    historical = registry.get("historical_regressions", [])
    if [item.get("case_id") for item in historical] != [
        "GEO-HIST-DELTA-01"
    ]:
        raise ValueError("the Stage 1 delta historical regression is missing")

    return [
        {
            "name": "ten_unique_cases",
            "passed": True,
            "evidence": {"count": len(cases), "case_ids": case_ids},
        },
        {
            "name": "two_cases_per_discipline",
            "passed": True,
            "evidence": dict(sorted(counts.items())),
        },
        {
            "name": "one_sentinel_per_discipline",
            "passed": True,
            "evidence": [case["case_id"] for case in sentinels],
        },
        {
            "name": "case_text_matches_registry",
            "passed": True,
            "evidence": str(source_path.relative_to(REPO_ROOT)),
        },
    ]


def validate_scoring(
    scoring: dict[str, Any], score_schema: dict[str, Any]
) -> list[dict[str, Any]]:
    if scoring.get("protocol_id") != "stage2_scoring_v1":
        raise ValueError("unexpected scoring protocol ID")
    totals: dict[str, int] = {}
    for key in ("image_and_sequence_dimensions", "video_dimensions"):
        dimensions = scoring.get(key, [])
        ids = [item["id"] for item in dimensions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate scoring dimension under {key}")
        totals[key] = sum(int(item["weight"]) for item in dimensions)
        if totals[key] != 100:
            raise ValueError(f"{key} weights sum to {totals[key]}, not 100")
    gates = scoring.get("hard_gates", [])
    gate_ids = [gate["id"] for gate in gates]
    if len(gates) < 8 or len(gate_ids) != len(set(gate_ids)):
        raise ValueError("hard gates are missing or duplicated")
    thresholds = scoring.get("promotion_thresholds", {})
    if thresholds.get("minimum_cross_discipline_route_cases") != 2:
        raise ValueError("route promotion requires two other disciplines")
    if thresholds.get("minimum_seed_success_fraction") != 0.75:
        raise ValueError("seed robustness threshold must stay frozen at 0.75")

    required_score_fields = set(score_schema.get("required", []))
    if required_score_fields != {
        "protocol_id",
        "subject_id",
        "hard_gates",
        "dimensions",
        "total_points",
        "verdict",
    }:
        raise ValueError("score record schema required fields changed")

    return [
        {
            "name": "image_score_weights",
            "passed": True,
            "evidence": {"total": totals["image_and_sequence_dimensions"]},
        },
        {
            "name": "video_score_weights",
            "passed": True,
            "evidence": {"total": totals["video_dimensions"]},
        },
        {
            "name": "hard_gates_defined",
            "passed": True,
            "evidence": gate_ids,
        },
        {
            "name": "score_record_schema",
            "passed": True,
            "evidence": sorted(required_score_fields),
        },
    ]


def validate_regression(
    registry: dict[str, Any], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    cases = {case["case_id"]: case for case in registry["cases"]}
    all_ids = set(cases)
    smoke_ids = policy["contract_smoke"]["case_ids"]
    if len(smoke_ids) != 10 or set(smoke_ids) != all_ids:
        raise ValueError("contract smoke must contain every Stage 2 case")
    if policy["contract_smoke"].get("requires_model"):
        raise ValueError("contract smoke must not require a model")

    representatives = policy.get("discipline_representatives", [])
    if len(representatives) != 5 or not set(representatives) <= all_ids:
        raise ValueError("five valid discipline representatives are required")
    rep_disciplines = [cases[item]["discipline"] for item in representatives]
    if set(rep_disciplines) != set(DISCIPLINES):
        raise ValueError("discipline representatives do not cover all fields")
    sentinel_ids = {
        case["case_id"] for case in cases.values() if case["sentinel"]
    }
    if sentinel_ids != set(representatives):
        raise ValueError("registry sentinels and policy representatives differ")

    video_reps = policy.get("video_motion_representatives", {})
    if set(video_reps) != set(MOTION_LABELS):
        raise ValueError("video motion representatives are incomplete")
    for motion, case_id in video_reps.items():
        if case_id not in cases:
            raise ValueError(f"unknown video representative: {case_id}")
        if motion not in cases[case_id]["motion_classes"]:
            raise ValueError(
                f"{case_id} does not declare motion class {motion}"
            )
    if policy.get("historical_regression") != "GEO-HIST-DELTA-01":
        raise ValueError("historical delta regression is not frozen")

    return [
        {
            "name": "all_case_contract_smoke",
            "passed": True,
            "evidence": smoke_ids,
        },
        {
            "name": "five_discipline_representatives",
            "passed": True,
            "evidence": representatives,
        },
        {
            "name": "five_video_motion_representatives",
            "passed": True,
            "evidence": video_reps,
        },
    ]


def validate_budget(budget: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = budget.get("fixed_seed_set", [])
    per_iteration = budget.get("per_iteration", {})
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError("compute budget requires four unique fixed seeds")
    if per_iteration.get("maximum_primary_hypotheses") != 1:
        raise ValueError("each loop iteration must test one main hypothesis")
    expected_candidates = (
        len(seeds)
        * per_iteration["maximum_candidate_configurations_including_baseline"]
    )
    if per_iteration["maximum_new_image_candidates"] != expected_candidates:
        raise ValueError("image candidate budget does not match configs × seeds")
    if per_iteration.get("maximum_video_trials_after_image_gates") != 2:
        raise ValueError("video trial budget must remain two")
    if budget.get("phase_0") != {
        "image_model_runs": 0,
        "video_model_runs": 0,
    }:
        raise ValueError("Phase 0 must not run a generative model")
    return [
        {
            "name": "fixed_seed_budget",
            "passed": True,
            "evidence": seeds,
        },
        {
            "name": "single_hypothesis_budget",
            "passed": True,
            "evidence": per_iteration,
        },
        {
            "name": "phase_0_zero_model_runs",
            "passed": True,
            "evidence": budget["phase_0"],
        },
    ]


def validate_baseline(
    baseline: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assets = baseline.get("assets", [])
    if len(assets) != 14:
        raise ValueError("Stage 1 baseline must freeze 14 reference assets")
    records: list[dict[str, Any]] = []
    roles: set[str] = set()
    for asset in assets:
        role = asset["role"]
        if role in roles:
            raise ValueError(f"duplicate baseline role: {role}")
        roles.add(role)
        if not re.fullmatch(r"[0-9a-f]{64}", asset.get("sha256", "")):
            raise ValueError(f"invalid SHA-256 for baseline role: {role}")
        path = _repo_path(asset["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_path(path)
        if actual != asset["sha256"]:
            raise ValueError(
                f"baseline changed for {asset['path']}: "
                f"expected {asset['sha256']}, got {actual}"
            )
        records.append(
            {
                **asset,
                "size_bytes": path.stat().st_size,
                "verified": True,
            }
        )
    return (
        [
            {
                "name": "stage1_baseline_hashes",
                "passed": True,
                "evidence": {
                    "asset_count": len(records),
                    "roles": sorted(roles),
                },
            }
        ],
        records,
    )


def validate_all() -> dict[str, Any]:
    configs = {name: load_json(path) for name, path in CONFIG_PATHS.items()}
    checks: list[dict[str, Any]] = []
    checks.extend(validate_registry(configs["case_registry"]))
    checks.extend(
        validate_scoring(
            configs["scoring_protocol"],
            configs["score_record_schema"],
        )
    )
    checks.extend(
        validate_regression(
            configs["case_registry"],
            configs["regression_policy"],
        )
    )
    checks.extend(validate_budget(configs["compute_budget"]))
    baseline_checks, baseline_records = validate_baseline(
        configs["stage1_baseline"]
    )
    checks.extend(baseline_checks)
    return {
        "configs": configs,
        "checks": checks,
        "baseline_records": baseline_records,
    }


def _href(repo_relative_path: str) -> str:
    path = _repo_path(repo_relative_path)
    return os.path.relpath(path, OUTPUT_ROOT).replace(os.sep, "/")


def _pills(values: list[str], labels: dict[str, str]) -> str:
    return "".join(
        f'<span class="pill">{html.escape(labels.get(value, value))}</span>'
        for value in values
    )


def render_report(state: dict[str, Any]) -> str:
    configs = state["configs"]
    registry = configs["case_registry"]
    scoring = configs["scoring_protocol"]
    regression = configs["regression_policy"]
    budget = configs["compute_budget"]
    cases = registry["cases"]
    sentinel_ids = set(regression["discipline_representatives"])
    check_rows = "".join(
        "<tr>"
        f"<td>{html.escape(check['name'])}</td>"
        '<td class="pass">通过</td>'
        f"<td><code>{html.escape(json.dumps(check['evidence'], ensure_ascii=False))}</code></td>"
        "</tr>"
        for check in state["checks"]
    )
    case_rows = "".join(
        "<tr>"
        f"<td><code>{case['case_id']}</code></td>"
        f"<td>{html.escape(case['discipline_zh'])}</td>"
        f"<td>{html.escape(case['title_zh'])}</td>"
        f"<td>{_pills(case['primary_layer_types'], LAYER_LABELS)}</td>"
        f"<td>{'是' if case['case_id'] in sentinel_ids else '否'}</td>"
        "</tr>"
        for case in cases
    )
    image_score_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['label_zh'])}</td>"
        f"<td>{item['weight']}</td>"
        f"<td>{html.escape(item['question_zh'])}</td>"
        "</tr>"
        for item in scoring["image_and_sequence_dimensions"]
    )
    video_score_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['label_zh'])}</td>"
        f"<td>{item['weight']}</td>"
        f"<td>{html.escape(item['question_zh'])}</td>"
        "</tr>"
        for item in scoring["video_dimensions"]
    )
    gate_cards = "".join(
        "<article>"
        f"<h3>{html.escape(gate['id'])}</h3>"
        f"<p>{html.escape(gate['evidence'])}</p>"
        "</article>"
        for gate in scoring["hard_gates"]
    )
    sentinel_cards = "".join(
        "<article>"
        f"<p class=\"eyebrow\">{html.escape(case['discipline_zh'])}</p>"
        f"<h3><code>{case['case_id']}</code> {html.escape(case['title_zh'])}</h3>"
        f"<p>主要验证：{html.escape('、'.join(case['capability_tags']))}</p>"
        "</article>"
        for case in cases
        if case["case_id"] in sentinel_ids
    )
    fingerprint_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['role'])}</td>"
        f"<td><a href=\"{html.escape(_href(item['path']))}\"><code>{html.escape(item['path'])}</code></a></td>"
        f"<td><code>{item['sha256'][:16]}…</code></td>"
        f"<td>{item['size_bytes']:,}</td>"
        "</tr>"
        for item in state["baseline_records"]
    )
    motion_rows = "".join(
        "<tr>"
        f"<td>{html.escape(MOTION_LABELS[motion])}</td>"
        f"<td><code>{case_id}</code></td>"
        "</tr>"
        for motion, case_id in regression[
            "video_motion_representatives"
        ].items()
    )
    seed_text = "、".join(str(seed) for seed in budget["fixed_seed_set"])
    config_links = "".join(
        f'<li><a href="../../{html.escape(path.relative_to(STAGE2_ROOT).as_posix())}">'
        f"{html.escape(name)}</a></li>"
        for name, path in CONFIG_PATHS.items()
    )
    selected_comparison = _href(
        "modules/video_model/stage1/output/keyframe_render/final/comparison.jpg"
    )
    delta_sheet = _href(
        "modules/video_model/stage1/output/keyframe_render/"
        "delta_sequence/sequence-contact-sheet.jpg"
    )
    stage1_report = _href(
        "modules/video_model/stage1/output/keyframe_render/report.html"
    )
    delta_report = _href(
        "modules/video_model/stage1/output/keyframe_render/"
        "delta_sequence/report.html"
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage 2 Phase 0｜Loop Engineer 基线与协议</title>
<style>
:root {{ --ink:#173038; --muted:#60767b; --paper:#f7f5ee; --card:#fff;
  --line:#d9e0dc; --accent:#126e67; --accent2:#db7c3b; --ok:#167249; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--paper);
  font:16px/1.72 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ padding:64px 24px 48px; color:white;
  background:linear-gradient(125deg,#11373b,#176d67 62%,#d3763c); }}
header>div, main {{ max-width:1180px; margin:auto; }}
h1 {{ font-size:clamp(34px,6vw,66px); line-height:1.06; margin:8px 0 20px; }}
h2 {{ font-size:30px; line-height:1.2; margin:0 0 14px; }}
h3 {{ margin:0 0 8px; line-height:1.35; }}
p {{ margin:8px 0 14px; }}
.lede {{ max-width:850px; font-size:19px; opacity:.94; }}
.eyebrow {{ font-size:13px; font-weight:800; letter-spacing:.12em;
  text-transform:uppercase; color:var(--accent); }}
header .eyebrow {{ color:#c8fff3; }}
nav {{ position:sticky; top:0; z-index:4; display:flex; gap:10px;
  overflow:auto; padding:12px max(24px,calc((100vw - 1180px)/2));
  background:rgba(247,245,238,.96); border-bottom:1px solid var(--line); }}
nav a {{ white-space:nowrap; text-decoration:none; color:var(--ink);
  padding:5px 10px; border-radius:18px; background:white; }}
main {{ padding:28px 24px 80px; }}
section {{ margin:28px 0; padding:30px; border:1px solid var(--line);
  border-radius:20px; background:rgba(255,255,255,.76); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
  gap:16px; }}
.cards article, .stat {{ padding:18px; border-radius:15px; background:var(--card);
  border:1px solid var(--line); }}
.stat strong {{ display:block; font-size:34px; color:var(--accent); }}
.status {{ display:inline-block; padding:7px 12px; border-radius:20px;
  color:#0b5534; background:#d8f5e6; font-weight:800; }}
.warning {{ border-left:5px solid var(--accent2); padding:13px 17px;
  background:#fff2e7; border-radius:9px; }}
.plain {{ border-left:5px solid var(--accent); padding:13px 17px;
  background:#eaf6f4; border-radius:9px; }}
table {{ width:100%; border-collapse:collapse; margin-top:16px; background:white; }}
th,td {{ padding:11px 12px; border:1px solid var(--line); text-align:left;
  vertical-align:top; }}
th {{ background:#eaf2ef; }}
.table-wrap {{ overflow:auto; }}
.pill {{ display:inline-block; padding:2px 8px; margin:2px 4px 2px 0;
  border-radius:12px; background:#e7f2f0; font-size:13px; white-space:nowrap; }}
figure {{ margin:12px 0; padding:10px; border:1px solid var(--line);
  border-radius:14px; background:white; }}
figure img {{ display:block; width:100%; height:auto; border-radius:9px; }}
figcaption {{ margin-top:9px; color:var(--muted); }}
code {{ padding:2px 5px; border-radius:4px; background:#edf1ef; font-size:.91em; }}
pre {{ overflow:auto; padding:16px; border-radius:12px; background:#112c31;
  color:#e6fffa; }}
a {{ color:#0a6761; }}
.pass {{ color:var(--ok); font-weight:800; }}
.small {{ color:var(--muted); font-size:14px; }}
@media(max-width:700px) {{ section {{ padding:20px 15px; }} th,td {{ min-width:130px; }} }}
</style>
</head>
<body>
<header><div>
<p class="eyebrow">Stage 2 · Phase 0 · Loop 正式启动</p>
<h1>先冻结尺子，再让 Agent 自我迭代</h1>
<p class="lede">这一阶段没有生成新图片。它把十个跨学科案例、Stage 1 参考结果、
评分办法、回归层级和每轮预算固定下来。后面的 Agent 每次声称“变好了”，都必须拿
这些不随实验改变的证据来验证。</p>
<span class="status">Phase 0 自动检查全部通过</span>
</div></header>
<nav>
<a href="#outcome">结果</a><a href="#baseline">历史基线</a>
<a href="#cases">十个案例</a><a href="#sentinels">哨兵案例</a>
<a href="#score">怎么评分</a><a href="#regression">怎么回归</a>
<a href="#budget">预算</a><a href="#audit">审计与复现</a>
</nav>
<main>
<section id="outcome">
<p class="eyebrow">01 · 本阶段结果</p><h2>Phase 0 冻结了什么</h2>
<div class="grid">
<div class="stat"><strong>10</strong>个 Stage 2 新案例</div>
<div class="stat"><strong>5</strong>个学科，每科两个</div>
<div class="stat"><strong>5</strong>个首批哨兵案例</div>
<div class="stat"><strong>14</strong>个 Stage 1 基线文件已验哈希</div>
<div class="stat"><strong>100</strong>图片评分满分</div>
<div class="stat"><strong>0</strong>次图片或视频模型调用</div>
</div>
<p class="plain"><strong>“冻结”是什么意思？</strong> 这些案例 ID、参考文件哈希、
评分权重和种子组成为版本化输入。Agent 可以提出新策略，但不能为了让分数好看而
悄悄换关键帧、换参考图或换评分办法。</p>
</section>

<section id="baseline">
<p class="eyebrow">02 · 历史基线</p><h2>Stage 1 提供起跑线，不是像素答案</h2>
<p>跨概念图片不能直接计算像素相似度。下面的三角洲结果只用来比较四件事：材料是否
自然、固定场景是否稳定、机制变化是否清楚、模型伪影是否更少。</p>
<div class="grid">
<figure><a href="{selected_comparison}"><img src="{selected_comparison}"
alt="Stage 1 选定首尾关键帧"></a><figcaption>Stage 1 最初选定的首尾关键帧。
<a href="{stage1_report}">打开当时的完整报告</a></figcaption></figure>
<figure><a href="{delta_sheet}"><img src="{delta_sheet}"
alt="Stage 1 三角洲五张连续关键帧"></a><figcaption>后续五张机制关键帧，用于回归
固定背景、沉积增长和分流。<a href="{delta_report}">打开完整过程报告</a></figcaption>
</figure></div>
<p class="warning"><strong>诚实的基线说明：</strong>Stage 1 的最终图大量依靠程序区域
和固定背景合成来守住机制；SDXL 原始全图候选没有直接成为最终像素。因此 Stage 2
既要保住机制一致性，也要单独证明图像模型确实改善了材料。</p>
</section>

<section id="cases">
<p class="eyebrow">03 · 案例注册表</p><h2>十个案例覆盖哪些程序数据</h2>
<p>“主要数据层”决定生成路线。例如连续浓度不应该被强行变成线稿；精确拼图也不能
只靠一句提示词维持数量和面积。</p>
<div class="table-wrap"><table><thead><tr><th>ID</th><th>学科</th><th>概念</th>
<th>主要程序数据层</th><th>首批哨兵</th></tr></thead><tbody>{case_rows}</tbody>
</table></div>
<p class="small">完整的教学片段、模型职责和案例硬门禁见
<a href="../../case.txt">case.txt</a>；机器读取的稳定 ID 与能力标签见
<a href="../../case_registry.json">case_registry.json</a>。</p>
</section>

<section id="sentinels">
<p class="eyebrow">04 · 第一批实现顺序</p><h2>先用五个“哨兵”覆盖五种学科</h2>
<p>哨兵不是最容易的五个，而是用较少案例尽早暴露不同失败方式。通过它们之后，再把
路线扩展到剩余五个案例。</p>
<div class="grid cards">{sentinel_cards}</div>
</section>

<section id="score">
<p class="eyebrow">05 · 固定评分尺</p><h2>先过硬门禁，再谈好不好看</h2>
<p>每个软评分维度由评审给 0–5 分，再按
<code>该维度权重 × 评分 ÷ 5</code> 换算。硬门禁失败时，不能用材质分抵消机制错误。</p>
<h3>图片与关键帧序列：满分 100</h3>
<div class="table-wrap"><table><thead><tr><th>维度</th><th>权重</th><th>普通话问题</th>
</tr></thead><tbody>{image_score_rows}</tbody></table></div>
<h3 style="margin-top:24px">视频：满分 100</h3>
<div class="table-wrap"><table><thead><tr><th>维度</th><th>权重</th><th>普通话问题</th>
</tr></thead><tbody>{video_score_rows}</tbody></table></div>
<h3 style="margin-top:24px">任何一个失败都要拒绝的门禁</h3>
<div class="grid cards">{gate_cards}</div>
</section>

<section id="regression">
<p class="eyebrow">06 · 分层回归</p><h2>不是每改一处都跑十套昂贵模型</h2>
<div class="grid cards">
<article><h3>1. 契约冒烟</h3><p>十个案例全部检查 schema、状态、语义层、控制来源、
manifest 和报告链接；不加载模型。</p></article>
<article><h3>2. 路线回归</h3><p>当前案例 + 至少两个其他学科的同数据类型案例 +
Stage 1 三角洲。用于判断策略能否成为通用候选。</p></article>
<article><h3>3. 学科代表</h3><p>数学、物理、化学、生物、地理各一个固定哨兵，
用于阶段版本的图片回归。</p></article>
<article><h3>4. 完整发布</h3><p>十案例契约全部通过；五学科图片代表和五类运动视频
代表全部通过；三角洲不退化。</p></article>
</div>
<h3 style="margin-top:24px">五类视频运动代表</h3>
<div class="table-wrap"><table><thead><tr><th>运动类型</th><th>固定代表案例</th></tr>
</thead><tbody>{motion_rows}</tbody></table></div>
</section>

<section id="budget">
<p class="eyebrow">07 · 单轮预算</p><h2>Loop 不是无限抽卡</h2>
<div class="grid">
<div class="stat"><strong>1</strong>个主要假设</div>
<div class="stat"><strong>1</strong>张代表帧先试</div>
<div class="stat"><strong>3</strong>种配置，包含基线</div>
<div class="stat"><strong>12</strong>张新候选上限</div>
<div class="stat"><strong>2</strong>次视频试验上限</div>
</div>
<p>固定复现编号为：<code>{html.escape(seed_text)}</code>。它们只是模型噪声起点的编号，
不是材料名，也不代表某个案例。固定四个编号是为了计算成功率；至少 3/4 有效，才达到
当前协议的 75% 种子稳健性阈值。</p>
<p class="warning">连续三轮目标维度提升不足 2 分、硬门禁失败、输入基线变化、
模型身份不符或只有一个种子成功时，当前 loop 必须停止并记录证据。</p>
</section>

<section id="audit">
<p class="eyebrow">08 · 自动验证与复现</p><h2>本报告不是手工写的结论页</h2>
<p>报告由同一份机器配置生成。下面的检查在生成前执行；任一失败，报告构建命令会以
错误退出。</p>
<div class="table-wrap"><table><thead><tr><th>检查</th><th>结果</th><th>证据</th>
</tr></thead><tbody>{check_rows}</tbody></table></div>
<h3 style="margin-top:24px">冻结文件指纹</h3>
<div class="table-wrap"><table><thead><tr><th>用途</th><th>文件</th><th>SHA-256</th>
<th>字节</th></tr></thead><tbody>{fingerprint_rows}</tbody></table></div>
<h3 style="margin-top:24px">从仓库根目录复现</h3>
<pre>.venv/bin/python -m modules.video_model.stage2.phase0
.venv/bin/python -m modules.video_model.stage2.phase0 --check
.venv/bin/python -m pytest -q modules/video_model/stage2/tests</pre>
<p>Phase 0 的配置文件：</p><ul>{config_links}
<li><a href="phase0_manifest.json">本次构建 manifest</a></li></ul>
</section>

<section>
<p class="eyebrow">09 · 下一检查点</p><h2>Phase 1：通用契约与无需模型的骨架</h2>
<p>下一步会实现统一的概念规格、序列规格和七类语义层接口，并为十个案例建立最小
fixture。仍然先证明状态、控制和报告可复现，再进入五个哨兵案例的程序动画。</p>
</section>
</main></body></html>
"""


def build_manifest(
    state: dict[str, Any], report_path: Path
) -> dict[str, Any]:
    tracked_sources = [
        STAGE2_ROOT / "case.txt",
        STAGE2_ROOT / "loop.md",
        *CONFIG_PATHS.values(),
        Path(__file__).resolve(),
    ]
    return {
        "schema_version": "1.0",
        "phase": "phase-0",
        "status": "passed",
        "classification": "deterministic protocol and baseline audit",
        "model_runs": {"image": 0, "video": 0},
        "case_summary": {
            "new_case_count": len(
                state["configs"]["case_registry"]["cases"]
            ),
            "discipline_count": len(DISCIPLINES),
            "sentinel_count": len(
                state["configs"]["regression_policy"][
                    "discipline_representatives"
                ]
            ),
            "historical_regression_count": 1,
        },
        "checks": state["checks"],
        "source_files": [file_record(path) for path in tracked_sources],
        "baseline_assets": state["baseline_records"],
        "report": file_record(report_path),
        "next_phase": "phase-1 generic contracts and model-free fixtures",
    }


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"


def build_phase0(*, check_only: bool = False) -> dict[str, Any]:
    state = validate_all()
    report_text = render_report(state)
    report_path = OUTPUT_ROOT / "report.html"
    manifest_path = OUTPUT_ROOT / "phase0_manifest.json"
    if check_only:
        if not report_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                "Phase 0 output missing; run without --check first"
            )
        if report_path.read_text(encoding="utf-8") != report_text:
            raise ValueError("Phase 0 report is stale")
        expected_manifest = build_manifest(state, report_path)
        actual_manifest = load_json(manifest_path)
        if actual_manifest != expected_manifest:
            raise ValueError("Phase 0 manifest is stale")
        return actual_manifest

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    manifest = build_manifest(state, report_path)
    manifest_path.write_text(_json_text(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or check the deterministic Stage 2 Phase 0 audit."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate that committed Phase 0 outputs are current",
    )
    args = parser.parse_args()
    manifest = build_phase0(check_only=args.check)
    mode = "checked" if args.check else "built"
    print(
        f"Phase 0 {mode}: {len(manifest['checks'])} checks passed; "
        "image model runs=0; video model runs=0"
    )


if __name__ == "__main__":
    main()
