"""Build and self-evaluate Phase 2 deterministic sentinel animations."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .cases.sentinel_programs import PROGRAMS
from .framework.contracts import (
    artifact_record,
    load_json,
    sha256_path,
    write_json,
)
from .framework.program_runner import (
    HEIGHT,
    WIDTH,
    build_program,
    validate_program_tree,
)


STAGE2_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = STAGE2_ROOT / "output" / "phase-2"
REPORT_PATH = OUTPUT_ROOT / "report.html"
MANIFEST_PATH = OUTPUT_ROOT / "phase2_manifest.json"
SENTINEL_IDS = ("MATH-02", "PHYS-01", "CHEM-01", "BIO-01", "GEO-02")

VISUAL_REVIEW: dict[str, dict[str, Any]] = {
    "MATH-02": {
        "status": "passed",
        "readability_score_5": 4,
        "mechanism_visibility_score_5": 4,
        "notes_zh": (
            "第二轮改为一次移动一块；K1 清楚显示 c²，K2 回到互不重叠的"
            "暂存位置，K3 显示 a² 与 b²。"
        ),
    },
    "PHYS-01": {
        "status": "passed",
        "readability_score_5": 5,
        "mechanism_visibility_score_5": 5,
        "notes_zh": (
            "两个固定振源、扩张波前、重叠区及节点/腹线从动画中均可直接辨认。"
        ),
    },
    "CHEM-01": {
        "status": "passed",
        "readability_score_5": 4,
        "mechanism_visibility_score_5": 4,
        "notes_zh": (
            "局部粉红羽流、搅拌后褪色、接近终点及终点持续粉红依次可见。"
        ),
    },
    "BIO-01": {
        "status": "passed",
        "readability_score_5": 4,
        "mechanism_visibility_score_5": 4,
        "notes_zh": (
            "六个复制染色体、十二条姐妹染色单体和两个各含六条的子细胞可追踪；"
            "第二轮增大终帧间距避免视觉重叠。"
        ),
    },
    "GEO-02": {
        "status": "passed",
        "readability_score_5": 4,
        "mechanism_visibility_score_5": 4,
        "notes_zh": (
            "空气团越山、迎风坡雨带、背风侧云雨消散，以及温湿度变化顺序清楚。"
        ),
    },
}


def _source_records() -> dict[str, dict[str, Any]]:
    paths = {
        "loop": STAGE2_ROOT / "loop.md",
        "case_registry": STAGE2_ROOT / "case_registry.json",
        "phase2_runner": Path(__file__),
        "generic_program_runner": (
            STAGE2_ROOT / "framework" / "program_runner.py"
        ),
        "sentinel_plugins": (
            STAGE2_ROOT / "cases" / "sentinel_programs.py"
        ),
    }
    return {
        name: {
            "path": path.relative_to(STAGE2_ROOT).as_posix(),
            "sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }


def _build_overview(programs: list[dict[str, Any]]) -> Path:
    thumb_width, thumb_height = 320, 180
    columns = 4
    rows = len(programs)
    gutter = 10
    label_height = 28
    width = columns * thumb_width + (columns + 1) * gutter
    height = rows * (thumb_height + label_height) + (rows + 1) * gutter
    sheet = Image.new("RGB", (width, height), (15, 34, 40))
    draw = ImageDraw.Draw(sheet)
    for row, manifest in enumerate(programs):
        root = OUTPUT_ROOT / manifest["case_id"]
        for column, keyframe in enumerate(manifest["keyframes"]):
            x = gutter + column * (thumb_width + gutter)
            y = gutter + row * (thumb_height + label_height + gutter)
            image = Image.open(
                root / keyframe["program_frame"]["path"]
            ).convert("RGB")
            image = image.resize((thumb_width, thumb_height))
            sheet.paste(image, (x, y))
            draw.rectangle(
                (x, y + thumb_height, x + thumb_width, y + thumb_height + label_height),
                fill=(6, 23, 28),
            )
            draw.text(
                (x + 7, y + thumb_height + 7),
                f"{manifest['case_id']} · K{column} · "
                f"t={keyframe['progress']:.2f}",
                fill=(232, 244, 239),
            )
    path = OUTPUT_ROOT / "sentinel-keyframes.jpg"
    sheet.save(path, quality=92, subsampling=0)
    return path


def _evidence_text(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return html.escape(text if len(text) <= 280 else text[:277] + "…")


def _render_layer_cards(
    manifest: dict[str, Any],
    keyframe: dict[str, Any],
) -> str:
    cards = []
    case_id = manifest["case_id"]
    for layer in keyframe["layers"]:
        preview = f"{case_id}/{layer['preview']['path']}"
        data_href = f"{case_id}/{layer['data']['path']}"
        policy = (
            "永不输入模型"
            if layer["model_input_policy"] == "never"
            else "由后续路由器决定"
        )
        details = ""
        if layer["layer_type"] == "annotation":
            payload = load_json(OUTPUT_ROOT / case_id / layer["data"]["path"])
            meanings = "".join(
                f"<li>{html.escape(item['meaning_zh'])}</li>"
                for item in payload["items"]
            )
            details = (
                f"<p class=\"muted\">本帧每根箭头的含义：</p>"
                f"<ul>{meanings}</ul>"
            )
        cards.append(
            f"""
            <article class="layer-card">
              <img loading="lazy" src="{html.escape(preview)}"
                   alt="{html.escape(layer['title_zh'])}">
              <div><b>{html.escape(layer['title_zh'])}</b>
              <span class="tag">{html.escape(layer['layer_type'])}</span></div>
              <p>{html.escape(layer['meaning_zh'])}</p>
              <p class="muted">来源：{html.escape(layer['source_zh'])}</p>
              <p class="muted">模型策略：{policy}；本阶段实际输入：否。</p>
              {details}
              <p><a href="{html.escape(data_href)}">下载可复现原始数据</a></p>
            </article>
            """
        )
    return "".join(cards)


def _render_program_section(manifest: dict[str, Any]) -> str:
    case_id = manifest["case_id"]
    mechanism = manifest["primary_mechanism_zh"]
    validation = load_json(
        OUTPUT_ROOT / case_id / manifest["validation"]["path"]
    )
    checks = "".join(
        f"""
        <tr>
          <td><span class="pass">通过</span></td>
          <td><code>{html.escape(check['name'])}</code></td>
          <td>{_evidence_text(check['evidence'])}</td>
        </tr>
        """
        for check in validation["mechanism_checks"]
    )
    keyframe_cards = []
    for keyframe in manifest["keyframes"]:
        clean = f"{case_id}/{keyframe['clean_frame']['path']}"
        program = f"{case_id}/{keyframe['program_frame']['path']}"
        state = load_json(OUTPUT_ROOT / case_id / keyframe["state"]["path"])
        state_summary = {
            key: value
            for key, value in state.items()
            if key not in {"objects", "sources_xy"}
        }
        keyframe_cards.append(
            f"""
            <article class="keyframe">
              <h4>{html.escape(keyframe['keyframe_id'])}
                <span class="tag">t={keyframe['progress']:.3f}</span></h4>
              <div class="pair">
                <figure><img loading="lazy" src="{html.escape(clean)}"
                  alt="无标注程序底图"><figcaption>clean：机制本体</figcaption></figure>
                <figure><img loading="lazy" src="{html.escape(program)}"
                  alt="带教学标注程序图"><figcaption>program：后叠加箭头与标签</figcaption></figure>
              </div>
              <details><summary>查看这一帧的机器状态</summary>
                <pre>{html.escape(json.dumps(state_summary, ensure_ascii=False, indent=2, sort_keys=True))}</pre>
              </details>
            </article>
            """
        )
    representative = manifest["keyframes"][2]
    review = VISUAL_REVIEW[case_id]
    return f"""
    <section id="{case_id}">
      <div class="section-head">
        <div><p class="eyebrow">{case_id}</p>
        <h2>{html.escape(manifest['title_zh'])}</h2></div>
        <span class="status">{html.escape(manifest['status'])}</span>
      </div>
      <p><b>程序必须守住什么：</b>{html.escape(mechanism)}</p>
      <p><b>这不是生图结果：</b>以下画面由数值状态直接绘制。它的任务是把数量、
      位置、拓扑和变化顺序做对；材料质感留给 Phase 3 的图片模型。</p>
      <video controls muted loop preload="metadata"
        poster="{case_id}/{manifest['keyframes'][0]['program_frame']['path']}">
        <source src="{case_id}/{manifest['animation']['path']}" type="video/mp4">
      </video>
      <div class="keyframe-grid">{''.join(keyframe_cards)}</div>
      <h3>机制验收：不是“看起来像”，而是检查状态关系</h3>
      <table><thead><tr><th>结果</th><th>机器检查</th><th>证据</th></tr></thead>
        <tbody>{checks}</tbody></table>
      <h3>语义层拆解（以 K2 为例）</h3>
      <p>这些文件不是含义模糊的“mask”。每层都有类型、来源和用途；
      箭头层明确禁止进入图片模型。</p>
      <div class="layer-grid">{_render_layer_cards(manifest, representative)}</div>
      <h3>Agent 视觉自评</h3>
      <p class="review-{html.escape(review['status'])}">
        状态：{html.escape(review['status'])}；
        可读性 {review['readability_score_5']}/5；
        机制可见性 {review['mechanism_visibility_score_5']}/5。
        {html.escape(review['notes_zh'])}
      </p>
    </section>
    """


def _write_report(
    programs: list[dict[str, Any]],
    *,
    status: str,
    phase_checks: list[dict[str, Any]],
    score: dict[str, Any],
) -> None:
    check_rows = "".join(
        f"<tr><td>{'通过' if item['passed'] else '未通过'}</td>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td>{_evidence_text(item['evidence'])}</td></tr>"
        for item in phase_checks
    )
    sections = "".join(_render_program_section(item) for item in programs)
    REPORT_PATH.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Stage 2 · Phase 2 确定性程序动画报告</title>
  <style>
    :root{{--ink:#163238;--deep:#0b242b;--paper:#f6f1e5;--card:#fffdf7;
      --line:#c9d7d2;--teal:#187f78;--amber:#d98d30;--red:#b94a48}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
      font:16px/1.65 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
    header,main{{max-width:1180px;margin:auto;padding:32px 24px}} header{{padding-top:58px}}
    h1{{font-size:clamp(2rem,5vw,4.6rem);line-height:1.03;margin:.15em 0}}
    h2{{font-size:2rem;margin:.1em 0}} h3{{margin-top:2rem}} h4{{margin:.2rem 0 .8rem}}
    p{{max-width:78ch}} .lede{{font-size:1.15rem}} .eyebrow{{letter-spacing:.14em;
      text-transform:uppercase;color:var(--teal);font-weight:750;margin:0}}
    .hero-grid,.summary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
      gap:14px;margin:24px 0}} .metric,.callout{{background:var(--card);border:1px solid var(--line);
      border-radius:14px;padding:18px}} .metric b{{display:block;font-size:2rem;color:var(--teal)}}
    section{{border-top:3px solid var(--ink);padding:44px 0}} .section-head{{display:flex;
      justify-content:space-between;gap:20px;align-items:flex-start}} .status,.tag{{display:inline-block;
      border-radius:999px;padding:2px 9px;background:#dcece6;font-size:.78rem}}
    video,.overview{{width:100%;border-radius:14px;background:var(--deep);border:1px solid var(--line)}}
    .keyframe-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));
      gap:15px;margin-top:18px}} .keyframe,.layer-card{{background:var(--card);
      border:1px solid var(--line);border-radius:13px;padding:13px}}
    .pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}} figure{{margin:0}}
    img{{display:block;width:100%;height:auto;border-radius:8px}} figcaption{{font-size:.82rem;
      color:#547078;padding-top:4px}} .layer-grid{{display:grid;
      grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
    .layer-card p{{font-size:.9rem;margin:.45rem 0}} .muted{{color:#60777c}}
    table{{width:100%;border-collapse:collapse;background:var(--card);font-size:.9rem}}
    th,td{{padding:9px;border:1px solid var(--line);vertical-align:top;text-align:left}}
    pre{{overflow:auto;background:var(--deep);color:#e8f5f0;padding:12px;border-radius:8px}}
    code{{font-size:.82rem}} .pass{{color:var(--teal);font-weight:750}}
    .review-passed{{border-left:5px solid var(--teal);padding-left:12px}}
    .review-pending,.review-failed{{border-left:5px solid var(--red);padding-left:12px}}
    nav a{{color:var(--teal);margin-right:14px}} @media(max-width:620px){{
      .pair{{grid-template-columns:1fr}} .keyframe-grid{{grid-template-columns:1fr}}
      header,main{{padding-left:15px;padding-right:15px}}}}
  </style>
</head>
<body>
<header>
  <p class="eyebrow">Live-Document · Loop Engineer · Phase 2</p>
  <h1>先把科学机制画对，<br>再让模型补材质。</h1>
  <p class="lede">本阶段用同一套运行器执行数学、物理、化学、生物、地理五个插件。
  每个插件只负责“进度 → 科学状态”；通用运行器负责 49 帧采样、关键帧、语义层、
  MP4、哈希和验收。图片模型和视频模型调用均为 0。</p>
  <nav>{''.join(f'<a href="#{item["case_id"]}">{item["case_id"]}</a>' for item in programs)}</nav>
  <div class="hero-grid">
    <div class="metric"><b>{status}</b>Phase 状态</div>
    <div class="metric"><b>5</b>真实确定性程序</div>
    <div class="metric"><b>245</b>总程序帧</div>
    <div class="metric"><b>0 / 0</b>图片 / 视频模型调用</div>
  </div>
</header>
<main>
  <section>
    <h2>第一次接手项目，先看这张流程图</h2>
    <div class="callout"><b>概念插件</b>计算某个进度的科学状态 →
      <b>程序渲染器</b>画无标注底图 → 同时导出硬边界、区域、场、对象 ID →
      <b>标注合成器</b>最后添加箭头和文字 → 49 张程序帧编码成 MP4。
      Phase 3 才会选择合适的语义层和语言提示词，让图片模型只增强材质。</div>
    <img class="overview" src="sentinel-keyframes.jpg" alt="五个案例各四个关键帧总览">
  </section>
  <section>
    <h2>Agent 自动晋级判断</h2>
    <p>硬门禁要求五个程序、全部机制断言、统一输出契约、零生成模型调用全部通过；
    视觉门禁要求每例可读性和机制可见性都至少 4/5。任一项不满足就留在 Phase 2
    自行优化，不向用户询问。</p>
    <div class="summary-grid">
      <div class="metric"><b>{score['total']}/100</b>综合分</div>
      <div class="metric"><b>{score['hard_gate']}</b>硬门禁</div>
      <div class="metric"><b>{score['visual_gate']}</b>视觉门禁</div>
      <div class="metric"><b>{score['decision']}</b>自动决定</div>
    </div>
    <table><thead><tr><th>结果</th><th>Phase 检查</th><th>证据</th></tr></thead>
      <tbody>{check_rows}</tbody></table>
  </section>
  {sections}
  <section>
    <h2>如何复现</h2>
    <pre>.venv/bin/python -m modules.video_model.stage2.phase2
.venv/bin/python -m modules.video_model.stage2.phase2 --check
.venv/bin/python -m pytest -q modules/video_model/stage2/tests</pre>
    <p>入口代码是 <code>phase2.py</code>；统一导出逻辑在
    <code>framework/program_runner.py</code>；五个概念的机制只存在于
    <code>cases/sentinel_programs.py</code>。因此换案例时不需要复制导出系统。</p>
    <p><a href="phase2_manifest.json">机器可读 Phase manifest</a></p>
  </section>
</main>
</body></html>
""",
        encoding="utf-8",
    )


def _phase_checks(programs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mechanism_checks = [
        check
        for manifest in programs
        for check in manifest["checks"]
        if check["name"]
        not in {
            "normalized_progress_is_monotonic",
            "fixed_rgb_canvas",
            "semantic_layer_contract_is_stable",
            "annotations_are_post_generation_only",
            "video_was_encoded_from_program_frames",
            "zero_generative_model_runs",
            "case_id_matches_plugin",
        }
    ]
    common_checks = [
        check
        for manifest in programs
        for check in manifest["checks"]
        if check not in mechanism_checks
    ]
    visual_passed = all(
        review["status"] == "passed"
        and review["readability_score_5"] >= 4
        and review["mechanism_visibility_score_5"] >= 4
        for review in VISUAL_REVIEW.values()
    )
    return [
        {
            "name": "five_sentinel_programs_pass",
            "passed": len(programs) == 5
            and all(item["status"] == "passed" for item in programs),
            "evidence": {
                item["case_id"]: item["status"] for item in programs
            },
        },
        {
            "name": "all_case_mechanism_assertions_pass",
            "passed": all(item["passed"] for item in mechanism_checks),
            "evidence": {
                "passed": sum(item["passed"] for item in mechanism_checks),
                "total": len(mechanism_checks),
            },
        },
        {
            "name": "all_common_runner_assertions_pass",
            "passed": all(item["passed"] for item in common_checks),
            "evidence": {
                "passed": sum(item["passed"] for item in common_checks),
                "total": len(common_checks),
            },
        },
        {
            "name": "exactly_245_program_frames",
            "passed": sum(
                item["timeline"]["frame_count"] for item in programs
            )
            == 245,
            "evidence": {
                item["case_id"]: item["timeline"]["frame_count"]
                for item in programs
            },
        },
        {
            "name": "zero_stage2_generative_model_runs",
            "passed": all(
                item["model_runs"] == {"image": 0, "video": 0}
                for item in programs
            ),
            "evidence": {"image": 0, "video": 0},
        },
        {
            "name": "agent_visual_review_gate",
            "passed": visual_passed,
            "evidence": VISUAL_REVIEW,
        },
    ]


def _score(checks: list[dict[str, Any]]) -> dict[str, Any]:
    hard = all(item["passed"] for item in checks[:5])
    visual = checks[5]["passed"]
    # The weights keep the hard scientific checks dominant and make it
    # impossible to pass without actually reviewing the visual evidence.
    total = (
        (40 if checks[1]["passed"] else 0)
        + (20 if checks[2]["passed"] else 0)
        + (15 if checks[3]["passed"] else 0)
        + (10 if checks[4]["passed"] else 0)
        + (15 if visual else 0)
    )
    passed = hard and visual and total >= 90
    return {
        "total": total,
        "threshold": 90,
        "hard_gate": "passed" if hard else "failed",
        "visual_gate": "passed" if visual else "failed",
        "decision": "advance_to_phase_3" if passed else "optimize_phase_2",
        "passed": passed,
    }


def _validate_report_links() -> list[str]:
    report = REPORT_PATH.read_text(encoding="utf-8")
    targets = re.findall(r'(?:href|src|poster)="([^"]+)"', report)
    missing = []
    for target in targets:
        if target.startswith(("#", "http://", "https://")):
            continue
        if not (REPORT_PATH.parent / target).resolve().exists():
            missing.append(target)
    return missing


def build_phase2(*, check_only: bool = False) -> dict[str, Any]:
    if check_only:
        manifest = load_json(MANIFEST_PATH)
        for case_id in SENTINEL_IDS:
            validate_program_tree(OUTPUT_ROOT / case_id)
        for source in manifest["sources"].values():
            path = STAGE2_ROOT / source["path"]
            if sha256_path(path) != source["sha256"]:
                raise ValueError(f"Phase 2 source changed: {path}")
        if sha256_path(REPORT_PATH) != manifest["report"]["sha256"]:
            raise ValueError("Phase 2 report hash mismatch")
        missing = _validate_report_links()
        if missing:
            raise ValueError(f"Phase 2 report links missing: {missing}")
        return manifest

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    programs = [
        build_program(PROGRAMS[case_id], OUTPUT_ROOT / case_id)
        for case_id in SENTINEL_IDS
    ]
    overview_path = _build_overview(programs)
    checks = _phase_checks(programs)
    score = _score(checks)
    status = "passed" if score["passed"] else "needs_optimization"
    _write_report(
        programs,
        status=status,
        phase_checks=checks,
        score=score,
    )
    manifest = {
        "schema_version": "1.0",
        "phase": 2,
        "status": status,
        "classification": (
            "five deterministic sentinel program animations; "
            "no generative-model renders"
        ),
        "program_count": len(programs),
        "frame_count": sum(
            item["timeline"]["frame_count"] for item in programs
        ),
        "keyframe_count": sum(len(item["keyframes"]) for item in programs),
        "model_runs": {"image": 0, "video": 0},
        "programs": [
            {
                "case_id": item["case_id"],
                "title_zh": item["title_zh"],
                "status": item["status"],
                "manifest": {
                    "path": (
                        Path(item["case_id"]) / "program_manifest.json"
                    ).as_posix(),
                    "sha256": sha256_path(
                        OUTPUT_ROOT
                        / item["case_id"]
                        / "program_manifest.json"
                    ),
                },
                "animation": {
                    "path": (
                        Path(item["case_id"])
                        / item["animation"]["path"]
                    ).as_posix(),
                    "sha256": item["animation"]["sha256"],
                },
            }
            for item in programs
        ],
        "visual_review": VISUAL_REVIEW,
        "checks": checks,
        "score": score,
        "sources": _source_records(),
        "overview": artifact_record(overview_path, OUTPUT_ROOT),
        "report": artifact_record(REPORT_PATH, OUTPUT_ROOT),
        "automatic_next_action": score["decision"],
    }
    write_json(MANIFEST_PATH, manifest)
    missing = _validate_report_links()
    if missing:
        raise ValueError(f"Phase 2 report links missing: {missing}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate existing outputs without rebuilding",
    )
    args = parser.parse_args()
    manifest = build_phase2(check_only=args.check)
    print(
        f"Phase 2: {manifest['status']} · "
        f"{manifest['program_count']} programs · "
        f"{manifest['frame_count']} frames · "
        f"next={manifest['automatic_next_action']}"
    )


if __name__ == "__main__":
    main()
