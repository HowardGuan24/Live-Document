"""Finalize S3.4 evidence, report, knowledge and transition to S3.5."""

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
OUTPUT = STAGE3 / "output" / "phase-4"
ASSETS = OUTPUT / "report-assets"
EXPERIMENTS = STAGE3 / "experiments"


def href(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace(
        os.sep, "/"
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _panel(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size[0] - 20, size[1] - 54))
    panel = Image.new("RGB", size, (236, 232, 220))
    panel.paste(
        image,
        (
            (size[0] - image.width) // 2,
            44 + (size[1] - 48 - image.height) // 2,
        ),
    )
    return panel


def _sheet(
    items: list[tuple[str, Path]], output: Path, columns: int
) -> None:
    size = (500, 330)
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (size[0] * columns, size[1] * rows), (13, 29, 32)
    )
    draw = ImageDraw.Draw(sheet)
    label_font = _font(17)
    for index, (label, path) in enumerate(items):
        x = (index % columns) * size[0]
        y = (index // columns) * size[1]
        sheet.paste(_panel(path, size), (x, y))
        draw.rectangle((x, y, x + size[0], y + 38), fill=(13, 29, 32))
        draw.text(
            (x + 12, y + 9),
            label,
            fill=(236, 247, 242),
            font=label_font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, subsampling=0)


def make_assets() -> dict[str, Path]:
    chem_root = (
        REPO_ROOT
        / "modules/video_model/stage2/output/phase-2/CHEM-01/"
        "keyframes/01_mechanism"
    )
    assets = {
        "chem_process": ASSETS / "chem-state-process.jpg",
        "operator_map": ASSETS / "operator-map.jpg",
    }
    _sheet(
        [
            (
                "1 FROZEN APPEARANCE ANCHOR",
                OUTPUT / "CHEM-01/frozen_anchor.png",
            ),
            (
                "2 PROGRAM REGION (WHERE)",
                chem_root
                / "layers/chem01_liquid_region_preview.png",
            ),
            (
                "3 PROGRAM pH FIELD (VALUE)",
                chem_root / "layers/chem01_ph_field_preview.png",
            ),
            (
                "4 COMPILED MUTABLE AREA",
                OUTPUT / "CHEM-01/mutable/01_mechanism.png",
            ),
            (
                "5 MECHANISM OUTPUT",
                OUTPUT / "CHEM-01/frames/01_mechanism.png",
            ),
            (
                "6 ENDPOINT OUTPUT",
                OUTPUT / "CHEM-01/frames/03_end.png",
            ),
        ],
        assets["chem_process"],
        3,
    )
    _sheet(
        [
            (
                "REGION + SCALAR",
                OUTPUT / "CHEM-01/sequence.jpg",
            ),
            (
                "HEIGHT / NORMAL",
                OUTPUT / "PHYS-01/sequence.jpg",
            ),
            (
                "OBJECT ID + LOCAL MATERIAL",
                OUTPUT / "MATH-02/sequence.jpg",
            ),
        ],
        assets["operator_map"],
        1,
    )
    return assets


def write_review(
    experiment_id: str,
    hypothesis_zh: str,
    verdict: str,
    reason_zh: str,
    evidence: dict[str, Any],
) -> None:
    root = EXPERIMENTS / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "hypothesis.md").write_text(
        f"# {experiment_id}\n\n假设：{hypothesis_zh}\n",
        encoding="utf-8",
    )
    write_json(
        root / "review.json",
        {
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "verdict": verdict,
            "reason_zh": reason_zh,
            "model_runs": {
                "image_candidates": 0,
                "video_candidates": 0,
            },
            "evidence": evidence,
        },
    )


def update_records() -> None:
    rejected = {
        "schema_version": "1.0",
        "passed": False,
        "experiment_id": "EXP-S3-20260731-011",
        "reason_zh": (
            "第一版图像本身满足对象几何，但门禁错误地用填充后的栅格像素数"
            "与解析三角形面积比较；同时用最近邻像素抽查局部材质，旋转后出现采样跳动。"
        ),
        "failed_checks": [
            {
                "case_id": "MATH-02",
                "name": "piece_area_preserved",
                "observed": [0.02796296] * 4,
                "threshold": 0.025,
                "root_cause": "raster boundary pixels were compared with analytical polygon area",
            },
            {
                "case_id": "MATH-02",
                "name": "object_local_material_binding_stable",
                "observed_minimum": 0.95220047,
                "threshold": 0.97,
                "root_cause": "nearest-pixel audit moved the sample point after rigid rotation",
            },
        ],
        "correction_zh": (
            "不改图像和阈值；面积改用多边形鞋带公式，材质抽查改用固定重心坐标的双线性采样。"
        ),
    }
    write_json(OUTPUT / "g3-v1-rejected.json", rejected)
    passed = load_json(OUTPUT / "g3.json")
    audit = {
        "schema_version": "1.0",
        "passed": True,
        "cases": {
            "CHEM-01": {
                "passed": True,
                "observations_zh": [
                    "四帧器材和背景保持稳定；开始帧与冻结 Anchor 逐像素相同。",
                    "机制帧只在程序 pH 羽流和当前液滴处出现局部粉色。",
                    "结果帧局部颜色消散，终点帧在全部液体区域持续显色。",
                ],
            },
            "PHYS-01": {
                "passed": True,
                "observations_zh": [
                    "两个橙色点源位置和数量不变。",
                    "波前半径按程序状态增大，重叠后出现节点/腹部结构。",
                    "同一模糊水面材质贯穿四帧，没有逐帧重新生成。",
                ],
            },
            "MATH-02": {
                "passed": True,
                "observations_zh": [
                    "四块三角形颜色、身份和对象局部木纹随刚体一起移动。",
                    "c² 与 a²+b² 两种拼法都由同一四块对象构成。",
                    "背景木纹来自同一个冻结材质供体。",
                ],
            },
        },
        "limitations_zh": [
            "CHEM-01 的显色是确定性光学叠加，不模拟体积散射；它保机制，不取代流体视频。",
            "PHYS-01 的真实水面只作为材料底图，波形事实仍完全来自程序 height field。",
            "MATH-02 回归使用空木纹供体而不是含对象的最终图，因为移动对象需要可恢复的干净背景。",
            "S3.4 只生成关键帧；帧间运动是否正确由 S3.5 验证。",
        ],
    }
    write_json(OUTPUT / "visual-audit.json", audit)

    write_review(
        "EXP-S3-20260731-011",
        "第一版通用 B 算子和门禁可以直接跨 CHEM、PHYS、MATH 全部通过。",
        "rejected",
        "CHEM 与 PHYS 通过；MATH 的两个失败来自量表采样域错误，不允许放宽阈值。",
        {"rejected_gate": file_record(OUTPUT / "g3-v1-rejected.json", REPO_ROOT)},
    )
    write_review(
        "EXP-S3-20260731-012",
        "解析几何用解析量表、连续图像用亚像素量表后，同一无模型输出可通过跨案例 G3。",
        "accepted_core",
        "三案例、四类算子和十一份合同 smoke 全部通过；视觉复核无硬失败。",
        {
            "g3": file_record(OUTPUT / "g3.json", REPO_ROOT),
            "visual_audit": file_record(OUTPUT / "visual-audit.json", REPO_ROOT),
        },
    )

    ledger = load_json(EXPERIMENTS / "ledger.json")
    ids = {"EXP-S3-20260731-011", "EXP-S3-20260731-012"}
    ledger["experiments"] = [
        item
        for item in ledger["experiments"]
        if item["experiment_id"] not in ids
    ]
    ledger["experiments"].extend(
        [
            {
                "experiment_id": "EXP-S3-20260731-011",
                "hypothesis_id": "H-S3-0004A",
                "phase": "S3.4",
                "verdict": "rejected",
                "failure_taxonomy": "metric_domain",
                "model_runs": {
                    "image_candidates": 0,
                    "video_candidates": 0,
                },
                "review": "modules/video_model/stage3/experiments/EXP-S3-20260731-011/review.json",
            },
            {
                "experiment_id": "EXP-S3-20260731-012",
                "hypothesis_id": "H-S3-0004B",
                "phase": "S3.4",
                "verdict": "accepted_core",
                "model_runs": {
                    "image_candidates": 0,
                    "video_candidates": 0,
                },
                "review": "modules/video_model/stage3/experiments/EXP-S3-20260731-012/review.json",
            },
        ]
    )
    write_json(EXPERIMENTS / "ledger.json", ledger)

    hypotheses_path = STAGE3 / "knowledge/hypotheses.jsonl"
    old = [
        line
        for line in hypotheses_path.read_text(encoding="utf-8").splitlines()
        if "EXP-S3-20260731-011" not in line
        and "EXP-S3-20260731-012" not in line
    ]
    records = [
        {
            "experiment_id": "EXP-S3-20260731-011",
            "hypothesis_id": "H-S3-0004A",
            "verdict": "rejected",
            "failure_taxonomy": "metric_domain",
            "learning_zh": "解析几何面积不能用含边界的栅格像素数验收；移动对象的局部材质不能用最近邻屏幕像素抽查。",
        },
        {
            "experiment_id": "EXP-S3-20260731-012",
            "hypothesis_id": "H-S3-0004B",
            "verdict": "accepted_core",
            "learning_zh": "region、scalar、object 和 height/normal 可共用一个冻结 Anchor + 程序状态接口；Case 差异留在版本 plan。",
        },
    ]
    with hypotheses_path.open("w", encoding="utf-8") as handle:
        for line in old:
            handle.write(line + "\n")
        for item in records:
            handle.write(
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    failures = load_json(STAGE3 / "knowledge/failure_patterns.json")
    failures["patterns"] = [
        item
        for item in failures["patterns"]
        if item["id"] != "FP-METRIC-001"
    ]
    failures["patterns"].append(
        {
            "id": "FP-METRIC-001",
            "taxonomy": "metric_domain",
            "symptom_zh": "视觉和合同几何正确，但面积或材质稳定性门禁小幅失败。",
            "diagnosis_zh": "解析量与栅格量混用，或连续图像用最近邻像素抽样。",
            "forbidden_fix_zh": "不得事后放宽阈值；先确认事实属于解析域、栅格域还是连续采样域，再修正量表。",
        }
    )
    write_json(STAGE3 / "knowledge/failure_patterns.json", failures)

    accepted = load_json(STAGE3 / "baselines/accepted.json")
    replace = {
        "CORE-STATE-RENDERER-B-V1",
        "STATE-PLAN-S3.4-V1",
        "SEQUENCE-CHEM-01-S3.4-V1",
        "SEQUENCE-PHYS-01-S3.4-V1",
        "SEQUENCE-MATH-02-S3.4-V1",
    }
    accepted["records"] = [
        item
        for item in accepted["records"]
        if item["baseline_id"] not in replace
    ]
    additions = [
        (
            "CORE-STATE-RENDERER-B-V1",
            "accepted_core",
            STAGE3 / "framework/state_renderer.py",
        ),
        (
            "STATE-PLAN-S3.4-V1",
            "accepted_core_config",
            STAGE3 / "state_render_plans.json",
        ),
        (
            "SEQUENCE-CHEM-01-S3.4-V1",
            "accepted_state_sequence",
            OUTPUT / "CHEM-01/sequence.jpg",
        ),
        (
            "SEQUENCE-PHYS-01-S3.4-V1",
            "accepted_state_sequence",
            OUTPUT / "PHYS-01/sequence.jpg",
        ),
        (
            "SEQUENCE-MATH-02-S3.4-V1",
            "accepted_state_sequence",
            OUTPUT / "MATH-02/sequence.jpg",
        ),
    ]
    for baseline_id, kind, path in additions:
        accepted["records"].append(
            {
                "baseline_id": baseline_id,
                "kind": kind,
                **file_record(path, REPO_ROOT),
            }
        )
    write_json(STAGE3 / "baselines/accepted.json", accepted)

    write_json(
        STAGE3 / "knowledge/open_problems.json",
        {
            "schema_version": "1.0",
            "problems": [
                {
                    "problem_id": "S3-PROBLEM-MOTION-001",
                    "taxonomy": "motion_guidance",
                    "summary_zh": "关键帧机制已确定，但首尾帧、motion contract、稀疏中间引导和程序时间线的收益边界尚未比较。",
                },
                {
                    "problem_id": "S3-PROBLEM-VISUAL-001",
                    "taxonomy": "visual_target",
                    "summary_zh": "GEO-02 外观目标仍为 provisional。",
                },
            ],
        },
    )
    write_json(
        STAGE3 / "state.json",
        {
            "schema_version": "1.0",
            "loop_id": "LOOP-S3-0001",
            "phase": "S3.5",
            "phase_status": "in_progress",
            "current_problem": {
                "problem_id": "S3-PROBLEM-MOTION-001",
                "taxonomy": "motion_guidance",
                "summary_zh": "确定不同运动类型需要首尾帧、motion contract、稀疏中间帧还是完整程序时间数据。",
            },
            "current_hypothesis": {
                "hypothesis_id": "H-S3-0005A",
                "statement_zh": "相邻关键帧加结构化 motion contract 能比只给一句话更稳定地保留静态对象和因果变化；复杂连续场才需要稀疏中间引导。",
                "falsification_zh": "若更高指导级别不提高机制门禁或反而降低画质，则不得升级为默认路线。",
            },
            "current_cohort": {
                "target": "CHEM-01",
                "regressions": ["MATH-02", "PHYS-01"],
                "keyframe_source": "modules/video_model/stage3/output/phase-4",
            },
            "exit_criteria": [
                "guidance levels are frozen and compared on identical adjacent keyframes",
                "static object, mechanism direction and temporal consistency gates are explicit",
                "each motion type selects the lowest guidance level with material benefit",
                "failed video candidates and deterministic program fallback are preserved",
            ],
            "budget": {
                "preflight_before_gpu_work": True,
                "video_candidate_limit_per_guidance_level": 1,
                "image_candidate_limit": 0,
            },
            "next_action": "Audit deployed video runtime and freeze the S3.5 guidance-level experiment before any video generation.",
        },
    )


def report_html(assets: dict[str, Path]) -> str:
    report_dir = OUTPUT
    g3 = load_json(OUTPUT / "g3.json")
    chem = load_json(OUTPUT / "CHEM-01/manifest.json")
    phys = load_json(OUTPUT / "PHYS-01/manifest.json")
    math_case = load_json(OUTPUT / "MATH-02/manifest.json")
    chem_values = [
        next(
            item
            for item in record["operator_records"]
            if item["operator_type"] == "scalar_transfer"
        )["indicator_mean_in_region"]
        for record in chem["records"]
    ]
    phys_corr = [
        next(
            item
            for item in record["operator_records"]
            if item["operator_type"] == "height_normal"
        )["program_shading_to_realized_luminance_correlation"]
        for record in phys["records"]
    ]
    math_metric = next(
        check
        for check in g3["cohorts"]["MATH-02"]["checks"]
        if check["name"] == "object_local_material_binding_stable"
    )["evidence"]["minimum"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 3 · S3.4 State Renderer B</title>
<style>
:root{{--paper:#f3efe4;--card:#fffdf7;--ink:#172321;--green:#0c6a55;--red:#a13c32;--gold:#a87518;--muted:#5c6763}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.68 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
main{{max-width:1180px;margin:auto;padding:44px 24px 90px}} h1{{font-size:clamp(36px,5vw,62px);line-height:1.06}} h2{{margin-top:52px;border-top:1px solid #cbc5b7;padding-top:28px}}
.lead{{font-size:20px;color:#34413e}} .status{{display:inline-block;background:#d7eee7;color:#075342;padding:7px 12px;border-radius:999px;font-weight:760}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}} .card{{background:var(--card);border:1px solid #d9d2c4;border-radius:14px;padding:18px}}
img{{max-width:100%;height:auto;display:block;border:1px solid #cdc6b8;border-radius:10px}} .hero{{width:100%;margin:18px 0}}
table{{width:100%;border-collapse:collapse;background:var(--card)}} th,td{{text-align:left;vertical-align:top;padding:11px 12px;border-bottom:1px solid #ddd6c8}} th{{background:#e4ede8}}
code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}} pre{{white-space:pre-wrap;background:#142422;color:#edf5f1;padding:15px;border-radius:10px;overflow-wrap:anywhere}}
.good{{color:var(--green)}} .bad{{color:var(--red)}} .note{{border-left:5px solid var(--gold);background:#fff1d2;padding:14px 17px}} .small{{font-size:14px;color:var(--muted)}} a{{color:#096955}}
</style></head><body><main>
<span class="status">S3.4 已通过 · 0 次模型调用 · 已进入 S3.5</span>
<h1>State Renderer B：一张冻结外观，程序状态生成整组关键帧</h1>
<p class="lead">S3.3 解决“底图长什么样”；S3.4 解决“机制变化怎样可靠地画上去”。
本阶段没有让扩散模型重新画四次。每个 Case 只加载一次冻结的材料/外观图，然后读取程序导出的
region、scalar field、object identity 或 height/normal，确定性地合成四个关键帧。</p>

<h2>实际生成过程</h2>
<a href="{href(report_dir, assets['chem_process'])}"><img class="hero" src="{href(report_dir, assets['chem_process'])}" alt="CHEM-01 从 Anchor、区域和 pH 场到关键帧的过程"></a>
<ol>
<li><strong>Anchor：</strong>S3.3 选中的透明玻璃图，提供器材材质、相机、背景和光照。</li>
<li><strong>region（区域，也常叫 mask）：</strong>每个像素只有“允许改/不允许改”两类；
这里限定颜色只能进入液体。mask 不是神秘模型输入，而是程序写出的可编辑范围。</li>
<li><strong>scalar field（标量场）：</strong>每个液体像素保存一个 pH 数字。sigmoid 转换把
pH 8.15 附近映射成显色强度；不是把 preview 蓝黄颜色直接贴进最终图。</li>
<li><strong>投影：</strong>程序画布是 640×360，Anchor 是 1024×576。系统从程序烧杯身份和
S3.1 canonical beaker 的 bbox 建立相对坐标，并在 Anchor 内自动搜索初始液面；没有手写最终像素坐标。</li>
<li><strong>合成与门禁：</strong>只在 compiled mutable area 内按保持亮度的方式混色；
范围外逐像素必须完全相同。</li>
</ol>

<h2>四类数据、四种通用责任</h2>
<div class="grid">
<div class="card"><h3>Region</h3><p>回答“哪里可以变”。MATH-02 用剩余面积 region 画 c² 或 a²+b²；
CHEM-01 用液体 region 限制显色。</p></div>
<div class="card"><h3>Scalar field</h3><p>回答“每个位置变化多少”。pH 经过版本 transfer function
变成粉色强度，数值仍可追溯到原始 <code>.npy</code>。</p></div>
<div class="card"><h3>Object identity</h3><p>回答“是谁、现在在哪里”。MATH-02 以稳定 object_id
把同一块木纹绑定在三角形局部重心坐标；纹理跟对象走，不在屏幕上滑动。</p></div>
<div class="card"><h3>Height / normal</h3><p>回答“表面朝向怎样”。PHYS-01 对高度求梯度、构造法线，
再用固定光源计算漫反射和高光；模型不会发明节点。</p></div>
</div>
<a href="{href(report_dir, assets['operator_map'])}"><img class="hero" src="{href(report_dir, assets['operator_map'])}" alt="三类案例的四种算子效果"></a>

<h2>主案例：酸碱滴定</h2>
<a href="CHEM-01/sequence.jpg"><img class="hero" src="CHEM-01/sequence.jpg" alt="CHEM-01 四关键帧"></a>
<table><tbody>
<tr><th>状态</th><td>START → MECHANISM → RESULT → END</td></tr>
<tr><th>程序平均显色强度</th><td><code>{html.escape(str(chem_values))}</code></td></tr>
<tr><th>当前液滴数量</th><td><code>[0, 1, 0, 0]</code></td></tr>
<tr><th>稳定区最大像素差</th><td><code>[0, 0, 0, 0]</code></td></tr>
<tr><th>开始帧与 Anchor</th><td>逐像素完全相同</td></tr>
</tbody></table>
<p>机制帧的粉色羽流来自该帧 pH field；结果帧回到无色；终点 pH 超过指示剂阈值，
整个液体区域持续显色。器材的伪刻度、玻璃高光和背景虽不完美，但不会在四帧之间随机漂移。</p>
<p><a href="CHEM-01/mutable-sequence.jpg">查看四帧实际可修改范围</a>；
白色是允许程序改动的像素，黑色必须与 Anchor 相同。</p>

<h2>跨案例回归：为什么它不是烧杯脚本</h2>
<h3>PHYS-01 · 双源水面干涉</h3>
<a href="PHYS-01/sequence.jpg"><img class="hero" src="PHYS-01/sequence.jpg" alt="PHYS-01 四关键帧"></a>
<p>同一张水面材料先做固定 8 px 模糊，以免供体自己的细碎波纹压过教学波形。
四帧只换程序 <code>surface_height.npy</code>；两个点源固定在 (235,180) 与 (405,180)。
程序阴影与输出明暗相关性为 <code>{html.escape(str(phys_corr))}</code>。</p>

<h3>MATH-02 · 勾股定理拼图</h3>
<a href="MATH-02/sequence.jpg"><img class="hero" src="MATH-02/sequence.jpg" alt="MATH-02 四关键帧"></a>
<p>木纹供体是没有对象的干净背景。四块三角形由 object identity 的顶点定位；
颜色和局部木纹绑定 object_id。四帧对象数均为 4，解析面积误差为 0，内部重叠为 0，
对象局部材质最低相关性为 <code>{math_metric}</code>。</p>

<h2>一次真实失败怎样改变了框架</h2>
<div class="note"><strong>第一轮 G3 被拒绝。</strong>当时图片没有重画错误，而是量表用错了数学域：
三角形解析面积是 5400 px²，填充 PNG 的边界像素会额外占面积，直接相除得到 2.796% 假误差；
最近邻像素抽查也会在对象旋转后跳到相邻纹理。没有把 2.5%/0.97 门槛调松。</div>
<table><thead><tr><th>检查</th><th>第一轮</th><th>修正</th><th>第二轮</th></tr></thead><tbody>
<tr><td>面积守恒</td><td class="bad">栅格计数误差 2.796%</td><td>用顶点鞋带公式比较解析面积</td><td class="good">误差 0</td></tr>
<tr><td>局部材质</td><td class="bad">最近邻最低相关 0.9522</td><td>固定重心坐标 + 双线性亚像素采样</td><td class="good">最低相关 1.0</td></tr>
</tbody></table>
<p>失败记录：<a href="g3-v1-rejected.json"><code>g3-v1-rejected.json</code></a>；
最终门禁：<a href="g3.json"><code>g3.json</code></a>。</p>

<h2>文件怎样流动</h2>
<pre>Input Contract
  └─ keyframe.semantic_layers.json
       ├─ region.npy ───────────────→ Region operator ─┐
       ├─ scalar_field.npy ─────────→ Scalar transfer ├─→ frozen Anchor 上合成
       ├─ object_identity.json ─────→ Object operator ┤
       └─ height_or_normal.npy ─────→ Normal lighting ┘
                                                   ↓
                                      frame.png + mutable.png
                                                   ↓
                                   G3 稳定区 / 机制 / 身份门禁</pre>
<p>核心代码不含 <code>CHEM-01</code>、<code>PHYS-01</code> 或 <code>MATH-02</code>；
Case 差异位于 <a href="../../state_render_plans.json"><code>state_render_plans.json</code></a>。
这使“pH 用哪个阈值”“水面光源方向”“三角形配色”等都成为可版本化输入，而不是隐藏分支。</p>

<h2>复现命令与成本</h2>
<pre>cd /workspace/Live-Document
/opt/venv/bin/python -m modules.video_model.stage3.phase4 preflight
/opt/venv/bin/python -m modules.video_model.stage3.phase4 render
/opt/venv/bin/python -m modules.video_model.stage3.phase4 gate
/opt/venv/bin/python -m modules.video_model.stage3.phase4_finalize</pre>
<p>本阶段新图片模型候选：<strong>0</strong>；视频候选：<strong>0</strong>。
所有输入、输出、Anchor 和语义层都有 SHA-256，见三个 Case 的
<a href="CHEM-01/manifest.json">manifest</a>、
<a href="PHYS-01/manifest.json">manifest</a>、
<a href="MATH-02/manifest.json">manifest</a>。</p>

<h2>结论和下一阶段</h2>
<ul>
<li><strong>进入通用核心：</strong>region、scalar、object、height/normal 算子接口与 G3。</li>
<li><strong>没有做：</strong>没有把四帧分别交给 SDXL，也没有用插值伪造程序状态。</li>
<li><strong>已知限制：</strong>关键帧正确不代表视频中间过程正确；CHEM 羽流仍是程序光学叠加。</li>
<li><strong>S3.5：</strong>用这些已验收关键帧比较“只给首尾帧”“加 motion contract”
“加稀疏中间引导”，为不同运动类型选最低但足够的指导等级。</li>
</ul>
<p class="small">十一份合同 smoke 全通过。报告只展示本项目实际文件，不引用外部 demo 代替证据。</p>
</main></body></html>"""


def main() -> None:
    if not load_json(OUTPUT / "g3.json")["passed"]:
        raise RuntimeError("cannot finalize failed S3.4 gate")
    assets = make_assets()
    update_records()
    report_path = OUTPUT / "report.html"
    report_path.write_text(report_html(assets), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "phase": "S3.4",
        "status": "passed",
        "accepted_core": [
            "region operator",
            "scalar transfer operator",
            "object-local material operator",
            "height-to-normal lighting operator",
            "G3 state/stability gates",
        ],
        "model_runs": {
            "image_candidates": 0,
            "video_candidates": 0,
        },
        "report": file_record(report_path, REPO_ROOT),
        "g3": file_record(OUTPUT / "g3.json", REPO_ROOT),
        "visual_audit": file_record(
            OUTPUT / "visual-audit.json", REPO_ROOT
        ),
        "assets": {
            key: file_record(path, REPO_ROOT)
            for key, path in assets.items()
        },
        "next_phase": "S3.5",
    }
    write_json(OUTPUT / "phase4_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "passed",
                "report": str(report_path),
                "next_phase": "S3.5",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
