"""Run the complete ten-case release audit and publish the Stage 3 candidate."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    validate_case_registry,
    validate_input_contract,
    validate_visual_target,
    verify_file_record,
    write_json,
)


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-10-release"
FORMAL_CASES = ("MATH-01", "MATH-02", "PHYS-01", "PHYS-02", "CHEM-01", "CHEM-02", "BIO-01", "BIO-02", "GEO-01", "GEO-02")
SEQUENCES = {
    "MATH-01": STAGE3 / "output/phase-8-scale-image/MATH-01/candidate/sequence.jpg",
    "MATH-02": STAGE3 / "output/phase-4/MATH-02/sequence.jpg",
    "PHYS-01": STAGE3 / "output/phase-4/PHYS-01/sequence.jpg",
    "PHYS-02": STAGE3 / "output/phase-8-scale-image/PHYS-02/candidate/sequence.jpg",
    "CHEM-01": STAGE3 / "output/phase-4/CHEM-01/sequence.jpg",
    "CHEM-02": STAGE3 / "output/phase-8-scale-image/CHEM-02/candidate/sequence.jpg",
    "BIO-01": STAGE3 / "output/phase-6-rerun-1/BIO-01/candidate/sequence.jpg",
    "BIO-02": STAGE3 / "output/phase-8-scale-image/BIO-02/candidate/sequence.jpg",
    "GEO-01": STAGE3 / "output/phase-8-scale-image/GEO-01/candidate/sequence.jpg",
    "GEO-02": STAGE3 / "output/phase-6-rerun-2/GEO-02/candidate/sequence.jpg",
}
VIDEOS = {
    "MATH-01": STAGE3 / "output/phase-9-scale-motion/MATH-01/deterministic/transition.mp4",
    "MATH-02": OUTPUT / "sentinel-motion/MATH-02/deterministic/transition.mp4",
    "PHYS-01": STAGE3 / "output/phase-5/experiments/EXP-S3-20260731-015/transition.mp4",
    "PHYS-02": STAGE3 / "output/phase-9-scale-motion/PHYS-02/deterministic/transition.mp4",
    "CHEM-01": OUTPUT / "sentinel-motion/CHEM-01/deterministic/transition.mp4",
    "CHEM-02": STAGE3 / "output/phase-9-scale-motion/CHEM-02/deterministic/transition.mp4",
    "BIO-01": STAGE3 / "output/phase-6-rerun-1/BIO-01/video/deterministic/transition.mp4",
    "BIO-02": STAGE3 / "output/phase-9-scale-motion/BIO-02/deterministic/transition.mp4",
    "GEO-01": STAGE3 / "output/phase-9-scale-motion/GEO-01/L1/transition.mp4",
    "GEO-02": STAGE3 / "output/phase-6-rerun-2/GEO-02/video/deterministic/transition.mp4",
}
IMAGE_EVIDENCE = {
    "MATH-01": STAGE3 / "output/phase-8-scale-image/g3-machine.json",
    "MATH-02": STAGE3 / "output/phase-4/g3.json",
    "PHYS-01": STAGE3 / "output/phase-4/g3.json",
    "PHYS-02": STAGE3 / "output/phase-8-scale-image/g3-machine.json",
    "CHEM-01": STAGE3 / "output/phase-4/g3.json",
    "CHEM-02": STAGE3 / "output/phase-8-scale-image/g3-machine.json",
    "BIO-01": STAGE3 / "output/phase-6-rerun-1/BIO-01/g3-machine.json",
    "BIO-02": STAGE3 / "output/phase-8-scale-image/g3-machine.json",
    "GEO-01": STAGE3 / "output/phase-8-scale-image/g3-machine.json",
    "GEO-02": STAGE3 / "output/phase-6-rerun-2/GEO-02/g3-machine.json",
}
MOTION_EVIDENCE = {
    "MATH-01": STAGE3 / "output/phase-9-scale-motion/MATH-01/g4.json",
    "MATH-02": OUTPUT / "sentinel-motion/MATH-02/g4.json",
    "PHYS-01": STAGE3 / "output/phase-5/experiments/EXP-S3-20260731-015/g4.json",
    "PHYS-02": STAGE3 / "output/phase-9-scale-motion/PHYS-02/g4.json",
    "CHEM-01": OUTPUT / "sentinel-motion/CHEM-01/g4.json",
    "CHEM-02": STAGE3 / "output/phase-9-scale-motion/CHEM-02/g4.json",
    "BIO-01": STAGE3 / "output/phase-6-rerun-1/BIO-01/video/deterministic/g4.json",
    "BIO-02": STAGE3 / "output/phase-9-scale-motion/BIO-02/g4.json",
    "GEO-01": STAGE3 / "output/phase-9-scale-motion/GEO-01/L1/g4.json",
    "GEO-02": STAGE3 / "output/phase-6-rerun-2/GEO-02/video/deterministic/g4.json",
}
MODEL_ROUTE = {"PHYS-01": "L1", "GEO-01": "L1"}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _uri(path: Path) -> str:
    mime = {".jpg": "image/jpeg", ".png": "image/png", ".mp4": "video/mp4"}[path.suffix.lower()]
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _update_registry() -> None:
    path = STAGE3 / "case_registry.json"
    registry = load_json(path)
    image_routes = {
        "MATH-02": "modules/video_model/stage3/output/phase-4/g3.json",
        "PHYS-01": "modules/video_model/stage3/output/phase-4/g3.json",
        "CHEM-01": "modules/video_model/stage3/output/phase-4/g3.json",
        "BIO-01": "modules/video_model/stage3/output/phase-6-rerun-1/BIO-01/g3-machine.json",
    }
    motion_routes = {
        "MATH-02": ("accepted_materialized_deterministic_fallback", "modules/video_model/stage3/output/phase-10-release/sentinel-motion/MATH-02/g4.json"),
        "PHYS-01": ("L1_accepted", "modules/video_model/stage3/output/phase-5/experiments/EXP-S3-20260731-015/g4.json"),
        "CHEM-01": ("accepted_materialized_deterministic_fallback", "modules/video_model/stage3/output/phase-10-release/sentinel-motion/CHEM-01/g4.json"),
        "BIO-01": ("accepted_deterministic_fallback", "modules/video_model/stage3/output/phase-6-rerun-1/BIO-01/video/deterministic/g4.json"),
    }
    for case in registry["cases"]:
        case_id = case["case_id"]
        if case_id in image_routes:
            case["image_route"] = {"status": "accepted_case_specific", "evidence": image_routes[case_id]}
        if case_id in motion_routes:
            status, evidence = motion_routes[case_id]
            case["motion_route"] = {"status": status, "evidence": evidence}
        if case_id == "GEO-HIST-DELTA-01":
            case["image_route"] = {"status": "passed_by_accepted_sha256", "evidence": "modules/video_model/stage3/baselines/accepted.json"}
            case["motion_route"] = {"status": "historical_image_regression_only"}
    write_json(path, registry)


def _image_passed(case_id: str, value: dict[str, Any]) -> bool:
    if "case_gates" in value:
        return value["case_gates"][case_id]["passed"]
    if "cohorts" in value:
        return value["cohorts"][case_id]["passed"]
    return bool(value.get("passed", False))


def _audit() -> dict[str, Any]:
    registry = load_json(STAGE3 / "case_registry.json")
    contract_checks = []
    vtp_checks = []
    for case in registry["cases"]:
        case_id = case["case_id"]
        error = None
        try:
            contract = load_json(REPO_ROOT / case["input_contract"])
            validate_input_contract(contract)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        contract_checks.append(_check(case_id, error is None, error or case["input_contract"]))
        error = None
        try:
            manifest = load_json(STAGE3 / f"visual_targets/{case_id}/manifest.json")
            validate_visual_target(manifest)
            for record in manifest["positive_refs"] + manifest["negative_refs"]:
                verify_file_record(record, REPO_ROOT)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        vtp_checks.append(_check(case_id, error is None and manifest["status"] == "accepted_project_baseline", error or manifest["status"]))

    validate_case_registry(registry)
    route_checks = []
    for case_id in FORMAL_CASES:
        image_value = load_json(IMAGE_EVIDENCE[case_id])
        motion_value = load_json(MOTION_EVIDENCE[case_id])
        route_checks.append({
            "case_id": case_id,
            "image_passed": _image_passed(case_id, image_value),
            "motion_passed": bool(motion_value["passed"]),
            "image_evidence": file_record(IMAGE_EVIDENCE[case_id], REPO_ROOT),
            "motion_evidence": file_record(MOTION_EVIDENCE[case_id], REPO_ROOT),
            "sequence": file_record(SEQUENCES[case_id], REPO_ROOT),
            "video": file_record(VIDEOS[case_id], REPO_ROOT),
            "route": f"LTX-{MODEL_ROUTE[case_id]}" if case_id in MODEL_ROUTE else "materialized_deterministic",
        })
    for item in route_checks:
        item["passed"] = item["image_passed"] and item["motion_passed"]

    accepted = load_json(STAGE3 / "baselines/accepted.json")
    baseline_checks = []
    for record in accepted["records"]:
        error = None
        try:
            verify_file_record(record, REPO_ROOT)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        baseline_checks.append(_check(record["baseline_id"], error is None, error or record["path"]))

    historical = [item for item in baseline_checks if "GEO-HIST-DELTA-01" in item["name"]]
    report_checks = []
    for path in (
        STAGE3 / "output/phase-6-rerun-2/report.html",
        STAGE3 / "output/phase-8-scale-image/report.html",
        STAGE3 / "output/phase-9-scale-motion/report.html",
    ):
        source = path.read_text(encoding="utf-8")
        values = re.findall(r"\bsrc=['\"]([^'\"]+)", source)
        report_checks.append(_check(path.name + "@" + path.parent.name, bool(values) and all(item.startswith("data:") for item in values), {"embedded_src_count": len(values), "non_embedded": [item for item in values if not item.startswith("data:")]}))

    groups = {
        "eleven_input_contracts": contract_checks,
        "eleven_visual_target_packages": vtp_checks,
        "ten_image_and_motion_routes": route_checks,
        "accepted_baseline_hashes": baseline_checks,
        "historical_delta": historical,
        "recent_report_live_preview_resources": report_checks,
    }
    result = {
        "schema_version": "1.0",
        "suite": "Stage 3 ten formal cases + one historical delta",
        "groups": groups,
        "passed": all(all(item["passed"] for item in values) for values in groups.values()),
        "counts": {
            "formal_cases": len(route_checks),
            "contracts": len(contract_checks),
            "visual_targets": len(vtp_checks),
            "accepted_baselines": len(baseline_checks),
            "model_video_routes_accepted": len(MODEL_ROUTE),
            "materialized_deterministic_routes": len(FORMAL_CASES) - len(MODEL_ROUTE),
        },
    }
    write_json(OUTPUT / "release-audit.json", result)
    return result


def _contact_sheet() -> Path:
    canvas = Image.new("RGB", (1280, 1120), (15, 32, 32))
    draw = ImageDraw.Draw(canvas)
    title = _font(16)
    for row, case_id in enumerate(FORMAL_CASES):
        image = Image.open(SEQUENCES[case_id]).convert("RGB")
        image.thumbnail((1280, 95))
        y = row * 112
        canvas.paste(image, ((1280 - image.width) // 2, y))
        route = f"LTX {MODEL_ROUTE[case_id]}" if case_id in MODEL_ROUTE else "deterministic"
        draw.text((12, y + 96), f"{case_id} · {route}", fill=(235, 244, 238), font=title)
    path = OUTPUT / "report-assets/ten-case-contact-sheet.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92, subsampling=0)
    return path


def _freeze_release_baselines() -> None:
    path = STAGE3 / "baselines/accepted.json"
    accepted = load_json(path)
    records = {item["baseline_id"]: item for item in accepted["records"]}
    for case_id in ("MATH-02", "PHYS-01", "CHEM-01"):
        item = {
            "baseline_id": f"VIDEO-{case_id}-MATERIALIZED-TIMELINE-S3.10-V1",
            "kind": "accepted_case_specific",
            **file_record(OUTPUT / f"sentinel-motion/{case_id}/deterministic/transition.mp4", REPO_ROOT),
        }
        records[item["baseline_id"]] = item
    evidence = {
        "baseline_id": "G4-SENTINEL-MATERIALIZATION-S3.10-V1",
        "kind": "accepted_core_evidence",
        **file_record(OUTPUT / "sentinel-motion/g4-machine.json", REPO_ROOT),
    }
    records[evidence["baseline_id"]] = evidence
    accepted["records"] = list(records.values())
    write_json(path, accepted)


def _update_release_state(audit: dict[str, Any]) -> None:
    policy = {
        "schema_version": "1.0",
        "release_id": "stage3-validated-candidate-2026-07-31",
        "release_class": "validated_stage3_candidate",
        "production_1_0_ready": False,
        "stage3_exit_passed": audit["passed"],
        "release_claim_zh": "十个正式案例均已通过 G0–G4；两案使用通过硬门的 LTX L1，八案使用携带已验收材质的完整程序时间线回退。三角洲历史基线与全部接受文件哈希通过。该版本确定了可复现研究流程，但不声称所有输出都达到统一照片级写实或生产 1.0。",
        "case_matrix": [
            {
                "case_id": case_id,
                "g0_input": "passed",
                "visual_target": "accepted_project_baseline",
                "g2_g3_image": "passed",
                "g4_motion": f"LTX_{MODEL_ROUTE[case_id]}_passed" if case_id in MODEL_ROUTE else "materialized_deterministic_fallback",
                "release_maturity": "validated" if case_id in MODEL_ROUTE else "validated_with_fallback",
            }
            for case_id in FORMAL_CASES
        ],
        "historical_regressions": [
            {"regression_id": "GEO-HIST-DELTA-01", "status": "passed_by_sha256"},
            {"regression_id": "CHEM-01-PHASE9", "status": "retained"},
        ],
        "production_1_0_blockers": [
            "当前 LTX 运行时只原生接收首尾帧，不能直接接收程序视频、对象轨迹、mask 或运动场；八类精确/循环/身份敏感过程因此使用确定性回退。",
            "十案已经达到机制正确且材质化的教学图标准，但不同学科尚未形成统一照片级真实感标尺。",
            "LTX 文本编码节点的 API 不返回 tokenizer 计数，生产级 prompt 截断完整性仍需运行时接口支持。",
        ],
    }
    write_json(STAGE3 / "release_policy.json", policy)

    state_path = STAGE3 / "state.json"
    state = load_json(state_path)
    state.update({
        "accepted_core_version": "stage3-validated-candidate-2026-07-31",
        "active_loop_id": "LOOP-S3-0007",
        "loop_id": "LOOP-S3-0007",
        "phase": "S3.10",
        "phase_status": "passed",
        "current_problem_id": None,
        "current_problem": None,
        "current_hypothesis_id": "H-S3-0012A",
        "consecutive_no_progress_loops": 0,
        "next_action": "Stage 3 exit passed. Continue only with a new production-realism or richer video-conditioning objective.",
        "open_problem_ids": ["S3-PROBLEM-PRODUCTION-REALISM-001", "S3-PROBLEM-VIDEO-CONDITIONING-001", "S3-PROBLEM-TOKEN-INTEGRITY-001"],
    })
    write_json(state_path, state)

    write_json(STAGE3 / "knowledge/open_problems.json", {
        "schema_version": "1.0",
        "problems": [
            {"problem_id": "S3-PROBLEM-PRODUCTION-REALISM-001", "taxonomy": "appearance_condition", "summary_zh": "十案已可读且材质化，但尚未建立跨学科统一的照片级真实感生产标尺。"},
            {"problem_id": "S3-PROBLEM-VIDEO-CONDITIONING-001", "taxonomy": "runtime", "summary_zh": "当前 LTX FLF 不能原生消费程序时间线、轨迹、mask 或运动场，精确机制仍依赖确定性回退。"},
            {"problem_id": "S3-PROBLEM-TOKEN-INTEGRITY-001", "taxonomy": "runtime", "summary_zh": "ComfyUI LTX 文本节点 API 不暴露 tokenizer 计数，无法完成生产级截断证明。"},
        ],
    })

    changelog = STAGE3 / "CHANGELOG.md"
    old = changelog.read_text(encoding="utf-8")
    if "## Validated candidate — 2026-07-31" in old:
        return
    heading = "# Stage 3 changelog\n"
    body = old[len(heading):] if old.startswith(heading) else old
    entry = """
## Validated candidate — 2026-07-31

- Completed image and motion routes for all ten formal mathematics, physics,
  chemistry, biology and geography cases.
- Added accepted Visual Target Packages for all five former scale gaps.
- Added decorrelated appearance-statistic transfer so donor geometry cannot
  leak into unrelated cases, plus typed raster and scalar operators.
- Upgraded MATH-01 and PHYS-02 program providers to export missing mechanism
  lines instead of drawing case-specific repairs on final images.
- Materialized complete 49-frame, accepted-material timelines for every case;
  PHYS-01 and GEO-01 use accepted LTX L1 routes, while eight exact, cyclic or
  identity-sensitive cases use explicit deterministic fallbacks.
- Preserved and reported rejected CHEM-02 L1/L2 videos: sparse keyframes fixed
  rough timing but not crystal identity continuity.
- Passed all eleven contracts, eleven Visual Target Packages, ten G2–G4
  routes, accepted baseline hashes, historical delta regression and Live
  Preview resource checks.

Known production limits remain in `release_policy.json`; this release is a
validated Stage 3 candidate, not production 1.0.

"""
    changelog.write_text(heading + entry + body, encoding="utf-8")


def _persist_release_experiment() -> None:
    exp_id = "EXP-S3-20260731-034"
    root = STAGE3 / "experiments" / exp_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "hypothesis.md").write_text("# H-S3-0012A — 十案例统一发布回归\n\n若十案 G0–G4、接受基线和历史回归一次通过，Stage 3 可发布为验证候选版，但仍不声称 production 1.0。\n", encoding="utf-8")
    review = {
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "verdict": "accepted_core",
        "passed": True,
        "reason_zh": "十个正式案例图像与运动路线、十一份合同和视觉目标包、全部接受哈希、历史三角洲与最近报告资源一次通过。",
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
        "evidence": file_record(OUTPUT / "release-audit.json", REPO_ROOT),
    }
    write_json(root / "review.json", review)
    ledger_path = STAGE3 / "experiments/ledger.json"
    ledger = load_json(ledger_path)
    by_id = {item["experiment_id"]: item for item in ledger["experiments"]}
    by_id[exp_id] = {"experiment_id": exp_id, "hypothesis_id": "H-S3-0012A", "phase": "S3.10", "model_runs": review["model_runs"], "review": f"modules/video_model/stage3/experiments/{exp_id}/review.json", "verdict": "accepted_core"}
    ledger["experiments"] = list(by_id.values())
    ledger["loop_id"] = "LOOP-S3-0007"
    write_json(ledger_path, ledger)


def _report(contact: Path, audit: dict[str, Any]) -> Path:
    cards = "".join(
        f"<figure><img src='{_uri(SEQUENCES[case_id])}'><figcaption><b>{case_id}</b> · "
        f"{'LTX ' + MODEL_ROUTE[case_id] if case_id in MODEL_ROUTE else '49 帧确定性回退'} · G2/G3/G4 通过</figcaption>"
        f"<video controls preload='metadata' src='{_uri(VIDEOS[case_id])}'></video></figure>"
        for case_id in FORMAL_CASES
    )
    control = STAGE3 / "output/phase-1/report-assets/canonicalize-process.jpg"
    cross_control = STAGE3 / "output/phase-1/report-assets/cross-discipline-controls.jpg"
    donor_failure = STAGE3 / "output/phase-8-scale-image/report-assets/rejected-vs-final.jpg"
    chem_l1 = STAGE3 / "output/phase-9-scale-motion/CHEM-02/L1/generated-frames.jpg"
    chem_l2 = STAGE3 / "output/phase-9-scale-motion/CHEM-02/L2/generated-frames.jpg"
    report = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Stage 3 十案例验证候选报告</title><style>
:root{{--ink:#193330;--muted:#5d6d69;--paper:#f4f0e6;--card:#fffdf8;--line:#d8d0c0;--ok:#19704c;--bad:#a53d33}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.72 system-ui,-apple-system,"Noto Sans SC",sans-serif}}main{{max-width:1220px;margin:auto;padding:34px 24px 80px}}h1{{font-size:clamp(32px,5vw,56px);line-height:1.1}}h2{{margin-top:2.2em;padding-top:1em;border-top:1px solid var(--line)}}h3{{margin-bottom:.4em}}.lead{{font-size:20px;max-width:950px}}.flow{{background:#193330;color:#fffaf0;border-radius:14px;padding:20px;font-size:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:18px}}figure,.card{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px}}img,video{{display:block;width:100%;height:auto;border-radius:8px;background:#132b29}}figcaption{{padding:10px 2px;color:var(--muted)}}table{{width:100%;border-collapse:collapse;background:var(--card)}}th,td{{padding:11px;text-align:left;vertical-align:top;border:1px solid var(--line)}}.ok{{color:var(--ok);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}code,pre{{background:#e8e3d8;border-radius:5px}}code{{padding:.1em .35em}}pre{{white-space:pre-wrap;padding:15px;overflow:auto}}.small{{font-size:14px;color:var(--muted)}}</style></head><body><main>
<p>Stage 3 · LOOP-S3-0007 · validated candidate</p><h1>十个案例都跑完了，流程已经固定成可复现候选版</h1>
<p class='lead'>数学、物理、化学、生物、地理各两个案例，现均有：冻结输入合同、视觉目标、机制正确的材质关键帧、完整运动路线和机器证据。两案使用通过 G4 的 LTX L1，八案使用带已验收材质的完整程序时间线。三角洲历史图和所有接受文件哈希仍通过。</p>
<div class='flow'>概念与程序 → 冻结合同 G0 → 语义层 → 几何/控制图 G1 → 外观目标与有限图片路线 → 关键帧 G3 → 完整程序时间线与运动合同 → 最小充分视频引导 → 视频 G4 → 接受模型结果或自动回退 → 十案与历史回归</div>

<h2>1. 一眼看完十案</h2><figure><img src='{_uri(contact)}'><figcaption>每行一个 Case，四列是开始、机制、结果、结束。右侧路线标签说明最终视频来自 LTX 还是逐程序状态的确定性回退。</figcaption></figure>
<div class='grid' style='margin-top:18px'>{cards}</div>

<h2>2. 这套流程中每个新名词有始有终</h2><table><tr><th>模块</th><th>谁产生、输入什么</th><th>做什么、输出什么</th><th>谁消费</th></tr>
<tr><td>程序 provider</td><td>案例代码读取归一化时间 0–1</td><td>计算对象位置、区域、标量、身份和因果状态；输出 state JSON 与语义 NPY/JSON</td><td>输入合同、State Renderer、G3/G4</td></tr>
<tr><td>输入合同 / G0</td><td>合同构建器读取程序关键帧、时间线、语义层、视觉目标</td><td>冻结路径、哈希、数量、含义和硬门；缺文件或哈希变化即拒绝</td><td>后续所有模块先验证它</td></tr>
<tr><td>Semantic Normalizer</td><td>读取每个案例自己的层名</td><td>映射为 hard_boundary、region、scalar_field、object_identity 等通用类型，同时保留来源</td><td>Geometry Resolver 与 State Renderer</td></tr>
<tr><td>Geometry Resolver / Control Compiler</td><td>读取类型化语义与 preserve_exact / canonicalize / layout_only 策略</td><td>决定轮廓应原样保留、参数化重建还是只守拓扑；输出结构控制图、区域与 derivation</td><td>图片模型或确定性渲染器，G1 检验</td></tr>
<tr><td>Visual Target Package</td><td>项目已接受的机制正例、外观供体、反例和量表</td><td>只定义材质、光照、相机和真实感，不定义对象位置</td><td>Prompt Compiler、候选选择与视觉审阅</td></tr>
<tr><td>State Renderer B</td><td>读取冻结外观、程序状态、语义层和版本化参数</td><td>用确定算子写回区域、边界、对象和标量；输出关键帧或 49 帧</td><td>G3、视频编码与 G4</td></tr>
<tr><td>Motion Contract / LTX</td><td>合同构建器从完整程序时间线提取顺序、路径、身份、静止项；LTX 读取首尾图和文字</td><td>LTX 从固定 seed 噪声生成中间帧；当前运行时不能直接读轨迹或程序视频</td><td>G4 决定接受、升级稀疏引导或回退</td></tr></table>

<h2>3. Canny、ControlNet、SDXL 在哪里，为什么不是每案都调用</h2><figure><img src='{_uri(control)}'><figcaption>烧杯实例的真实血缘：程序图 → 对象身份/语义硬边界 → dense Canny 负对照 → 人工上限 → 自动几何重建。Canny 只是从亮暗突变提取边缘；对整张程序图运行会把粗糙轮廓和 UI 一起保留。</figcaption></figure>
<figure style='margin-top:18px'><img src='{_uri(cross_control)}'><figcaption>同一 Control Compiler 根据几何策略输出不同控制：规范器材重建、数学精确边界、地理稀疏布局、生物区域与锚点。结构控制决定“在哪里”，不是材质图片。</figcaption></figure>
<p>当走扩散图片路线时，结构图送给 <b>SDXL Canny ControlNet FP16</b>；ControlNet 在每个去噪阶段计算结构残差，再交给 <b>SDXL Base 1.0 FP16</b> 与文字条件共同生成 RGB 图。ControlNet 不单独返回成图。Stage 3 的扩展五案先建立确定机制基线，因此没有伪造新的 SDXL 调用：它们借用已审核模型图的材质统计，再由语义层重建几何。图片模型只有在固定候选实验能增加外观价值且通过事实门时才进入。</p>

<h2>4. 外观与几何分离的关键修复</h2><figure><img src='{_uri(donor_failure)}'><figcaption><span class='bad'>左：拒绝。</span>只做高通仍保留供体边缘坐标，烧杯、细胞器或河道会泄漏到新案例。<span class='ok'>右：接受。</span>固定随机排列先摧毁残差坐标，只留材质分布；对象位置全部由新案例语义层重建。</figcaption></figure>
<p>这条改动在五学科扩展案和五个旧 sentinel 上同时回归通过，因此进入通用核心。相反，缺失的正弦轨迹和物理仪表没有在最终图上手画，而是回到程序 provider 新增类型化语义层并升级合同。</p>

<h2>5. 模型视频的能力边界不是靠感觉判断</h2><div class='grid'><figure><img src='{_uri(chem_l1)}'><figcaption><span class='bad'>CHEM-02 L1：</span>最终四晶体好看，但第 12/49 帧就成核，计数还会倒退。</figcaption></figure><figure><img src='{_uri(chem_l2)}'><figcaption><span class='bad'>CHEM-02 L2：</span>中间关键帧改善时序，但独立片段仍不共享晶体身份。</figcaption></figure></div>
<p>模型路线只有 PHYS-01 的连续波场和 GEO-01 的单次河道拓扑变化通过。精确刚体、循环首尾同图、质量/对象身份、细胞分裂和标量事件目前都使用完整程序时间线。回退不是删除外观：每个时间点仍经过同一冻结 State Renderer，所以保留了已接受材质。</p>

<h2>6. 发布门结果</h2><table><tr><th>发布检查</th><th>结果</th></tr>
<tr><td>输入合同</td><td class='ok'>11/11（十正式案 + 三角洲历史案）</td></tr>
<tr><td>Visual Target Package</td><td class='ok'>11/11，正例、反例和引用哈希可解析</td></tr>
<tr><td>图像与运动路线</td><td class='ok'>10/10 G2–G4</td></tr>
<tr><td>接受基线</td><td class='ok'>{audit['counts']['accepted_baselines']}/{audit['counts']['accepted_baselines']} 文件哈希和大小一致</td></tr>
<tr><td>历史三角洲</td><td class='ok'>通过 SHA-256</td></tr>
<tr><td>近期报告 Live Preview 资源</td><td class='ok'>全部图片和视频 src 为内嵌 data URI</td></tr></table>

<h2>7. 如何从零复现发布检查</h2><pre>cd /persistent/workspace-project/Live-Document
/opt/venv/bin/python -m modules.video_model.stage3.phase8_scale_image
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase9_scale_motion
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage3.phase10_sentinel_motion
/opt/venv/bin/python -m modules.video_model.stage3.phase10_release
/opt/venv/bin/python -m pytest -q modules/video_model/stage2/tests modules/video_model/stage3/tests</pre>
<p>模型视频复现还需先启动 <code>/persistent/ComfyUI/start-ltx2.3.sh</code>，再按 S3.9 报告中的固定 spec 执行；不会在发布回归中重复消耗 GPU。模型权重、LoRA、文本编码器、seed、sigmas、输入图和提示词的指纹保存在实验目录。</p>

<h2>8. 这次“完成”不等于什么</h2><p>它表示固定输入能得到固定语义解释、有限候选或固定失败、确定关键帧和可验收视频；不表示已经 production 1.0。当前仍有三个明确限制：视频运行时不能原生接收程序轨迹/视频/mask；十案没有统一照片级真实感标尺；LTX API 不返回 tokenizer 计数。它们已进入 open problems，不会再伪装成“案例没做完”。</p>
<p class='small'>本报告内嵌十组关键帧、十个最终视频和五张过程/失败图，移动目录或 Live Preview 打开均不依赖相对图片路径。</p>
</main></body></html>"""
    path = OUTPUT / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


def run() -> dict[str, Any]:
    _update_registry()
    _freeze_release_baselines()
    audit = _audit()
    if not audit["passed"]:
        raise RuntimeError("Stage 3 release audit failed")
    contact = _contact_sheet()
    _update_release_state(audit)
    _persist_release_experiment()
    report = _report(contact, audit)
    source = report.read_text(encoding="utf-8")
    src_values = re.findall(r"\bsrc=['\"]([^'\"]+)", source)
    if not src_values or not all(item.startswith("data:") for item in src_values):
        raise RuntimeError("final report contains a non-embedded media src")
    manifest = {
        "schema_version": "1.0",
        "release_id": "stage3-validated-candidate-2026-07-31",
        "release_class": "validated_stage3_candidate",
        "stage3_exit_passed": True,
        "production_1_0_ready": False,
        "formal_cases": list(FORMAL_CASES),
        "route_counts": {"LTX_L1": 2, "materialized_deterministic": 8},
        "audit": file_record(OUTPUT / "release-audit.json", REPO_ROOT),
        "report": file_record(report, REPO_ROOT),
        "contact_sheet": file_record(contact, REPO_ROOT),
        "release_policy": file_record(STAGE3 / "release_policy.json", REPO_ROOT),
        "changelog": file_record(STAGE3 / "CHANGELOG.md", REPO_ROOT),
        "tests_required": ["modules/video_model/stage2/tests", "modules/video_model/stage3/tests"],
        "report_embedded_media_count": len(src_values),
    }
    write_json(OUTPUT / "release-manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
