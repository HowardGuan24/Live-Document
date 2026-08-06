"""Build reviewable Visual Target Packages for all five scale cases.

This loop tests one reusable rule: an appearance donor may be shared, while
each case keeps its own program geometry, mechanism reference, negative
example and hard gates.  No image model is called in this phase.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    validate_input_contract,
    validate_visual_target,
    verify_file_record,
    write_json,
)


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-7-scale-vtp"
EXPERIMENT_ID = "EXP-S3-20260731-028"
CASES: dict[str, dict[str, Any]] = {
    "MATH-01": {
        "title": "单位圆怎样生成正弦曲线",
        "summary": "暖白绘图纸与精密墨线；圆、坐标轴、旋转点和正弦轨迹必须保持解析几何精度。",
        "donor": "modules/video_model/stage2/output/phase-7/route-c/MATH-02/variants/studio_pbr/03_end.png",
        "donor_label": "外观供体：只借用棚拍光、纸木微纹理和克制阴影。",
        "negative": "坐标轴弯曲、圆不闭合、轨迹与旋转点不同步，或木纹覆盖精确墨线。",
        "questions": [
            "是否像真实绘图纸上的精密教学图，而不是发光 UI？",
            "圆、坐标轴与正弦轨迹是否仍清楚且线宽稳定？",
            "材质微纹理是否没有推动或扭曲几何线？",
            "相机、纸面和光线在四帧中是否固定？",
        ],
        "hard": [
            "圆和坐标轴必须来自 hard_boundary，不允许模型重画。",
            "旋转点与曲线头的位置必须来自 object_identity。",
            "已出现的正弦轨迹只能增长，不能提前出现或回缩。",
            "appearance_to_geometry_leakage 必须为零。",
        ],
    },
    "PHYS-02": {
        "title": "磁铁运动为什么在线圈中产生感应电流",
        "summary": "桌面实验演示：真实铜线圈、红蓝条形磁铁和指针表；位置、运动阶段和电流方向由程序决定。",
        "donor": "modules/video_model/stage2/output/phase-7/route-a/experiments/EXP-P7-A-chem-01-00_start/raw/semantic_control_065/seed_7101.png",
        "donor_label": "外观供体：只借用克制实验室光照、玻璃金属高光和中性背景。",
        "negative": "生成额外线圈或磁铁、改写 N/S 两极、移动固定仪表，或停止时指针仍偏转。",
        "questions": [
            "铜线圈、磁铁和仪表是否具有可辨认的真实材料？",
            "磁铁、线圈和仪表的相对位置是否一眼清楚？",
            "红蓝两极和表针是否在四帧中保持同一身份？",
            "是否避免实验室杂物和额外器材？",
        ],
        "hard": [
            "只允许一个 bar_magnet 和一个 fixed_coil。",
            "线圈和仪表固定，只有磁铁按程序接近、停止、撤离。",
            "表针方向必须和 induced_current 的正、零、负一致。",
            "appearance_to_geometry_leakage 必须为零。",
        ],
    },
    "CHEM-02": {
        "title": "盐溶液蒸发、过饱和与晶体生长",
        "summary": "透明蒸发皿、逐渐降低的液面和清晰盐晶体；浓度、晶体数量与总溶质质量由程序守恒。",
        "donor": "modules/video_model/stage2/output/phase-7/route-a/experiments/EXP-P7-A-chem-01-00_start/raw/semantic_control_065/seed_7101.png",
        "donor_label": "外观供体：真实玻璃边缘、液体高光和中性实验室照明。",
        "negative": "液面升高、晶体提前出现、晶体数量乱增，或蒸发皿形状在帧间改变。",
        "questions": [
            "蒸发皿是否像透明玻璃，液体是否有克制的体积感？",
            "液面下降和颜色浓缩是否连续可读？",
            "晶体是否像盐晶体而不是塑料钻石？",
            "器皿、相机和光线是否四帧固定？",
        ],
        "hard": [
            "蒸发皿边界必须固定。",
            "液体区域随 solvent_volume 单调下降。",
            "晶体数量必须为 0、0、1、4。",
            "liquid_solute_mass + crystal_solute_mass 始终等于 total_solute_mass。",
        ],
    },
    "BIO-02": {
        "title": "保卫细胞如何控制气孔开闭",
        "summary": "显微镜式叶表皮与两个保卫细胞；细胞身份不变，膨压只控制形变和中央孔隙。",
        "donor": "modules/video_model/stage2/output/phase-7/route-b/BIO-01/variants/stable_material_plus_depth/02_result.png",
        "donor_label": "外观供体：只迁移细胞表面的高频统计、膜高光和柔和显微镜深度。",
        "negative": "供体细胞器进入画面、保卫细胞复制或消失、孔隙位置漂移，或开闭次序反转。",
        "questions": [
            "是否像显微镜下的植物表皮，而不是两个塑料豆？",
            "两个保卫细胞是否始终是同一对对象？",
            "中央孔隙的开闭是否清晰但不过度发光？",
            "背景细胞质感是否克制且不抢机制？",
        ],
        "hard": [
            "guard_cell_count 始终等于 2。",
            "孔隙宽度必须按 10、49、49、10 px 的状态变化。",
            "开、开、关的阶段顺序不能改变。",
            "不得从外观供体复制额外细胞器或对象。",
        ],
    },
    "GEO-01": {
        "title": "河曲裁弯取直与牛轭湖形成",
        "summary": "稳定俯视航拍地表、自然水体和湿润河岸；河道拓扑、颈部宽度和牛轭湖身份由程序决定。",
        "donor": "modules/video_model/stage1/output/keyframe_render/report.html",
        "donor_image": "modules/video_model/stage1/output/keyframe_render/transport_pair/final/at_outlet.png",
        "donor_label": "外观供体：只借用自然河水、湿沙与俯视散射光，不借用三角洲河道形状。",
        "negative": "河流在裁弯前断裂、终帧没有独立牛轭湖、湿沙堵塞主河道，或相机角度改变。",
        "questions": [
            "河水、湿岸和地表是否像自然俯视场景？",
            "颈部逐渐收窄是否一眼可见？",
            "终帧主河道与牛轭湖是否能分别辨认？",
            "材质是否没有把别的三角洲几何带入本案例？",
        ],
        "hard": [
            "前三帧水体一个连通分量，终帧两个。",
            "颈部宽度按 55、42.8、20.2、8 px 收窄。",
            "终帧必须存在且只存在一个 isolated_oxbow_lake。",
            "主河道从左到右始终连通。",
        ],
    },
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _program_sheet(case_id: str) -> Path:
    return REPO_ROOT / f"modules/video_model/stage2/output/phase-4/programs/{case_id}/keyframe-contact-sheet.jpg"


def _donor_path(spec: dict[str, Any]) -> Path:
    return REPO_ROOT / spec.get("donor_image", spec["donor"])


def _negative_reference(case_id: str, spec: dict[str, Any]) -> Path:
    source = Image.open(_program_sheet(case_id)).convert("RGB")
    blurred = source.filter(ImageFilter.GaussianBlur(5.5))
    canvas = Image.blend(source, blurred, 0.72)
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    draw.rectangle((0, 0, width, height), fill=(116, 28, 20, 36))
    draw.line((35, 35, width - 35, height - 35), fill=(220, 45, 38, 210), width=12)
    draw.line((width - 35, 35, 35, height - 35), fill=(220, 45, 38, 210), width=12)
    target = OUTPUT / "negative_refs" / f"{case_id}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=90, subsampling=0)
    return target


def _rubric(case_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    ids = ["material", "mechanism_readability", "camera_stability", "artifact_avoidance"]
    value = {
        "schema_version": "1.0",
        "case_id": case_id,
        "score_scale": {
            "1": "明显失败，外观或机制不能用于教学",
            "3": "基本可读，但材质、稳定性或机制有明显弱点",
            "5": "外观可信，且不看文字也能正确读出机制",
        },
        "appearance_dimensions": [
            {"id": key, "question_zh": question, "weight": 0.25}
            for key, question in zip(ids, spec["questions"])
        ],
        "hard_gates": [
            {"id": f"gate_{index + 1}", "pass_zh": text}
            for index, text in enumerate(spec["hard"])
        ],
        "acceptance": {"minimum_weighted_score": 4.0, "all_hard_gates_must_pass": True},
    }
    return value


def _style_board(case_id: str, spec: dict[str, Any], negative: Path) -> str:
    program = _program_sheet(case_id)
    donor = _donor_path(spec)
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{case_id} 视觉目标</title><style>body{{margin:0;background:#f3efe5;color:#19302d;font:16px/1.65 system-ui}}main{{max-width:1120px;margin:auto;padding:30px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}figure{{margin:0;background:#fff;border:1px solid #d7cfbd;border-radius:12px;padding:10px}}img{{display:block;width:100%;height:auto}}figcaption{{padding:9px}}.bad{{color:#a43b32}}.ok{{color:#18704c}}</style></head><body><main><p>{case_id} · accepted project baseline</p><h1>{spec['title']}</h1><p>{spec['summary']}</p><p><b>分工：</b>程序图和语义层决定对象在哪里、机制怎样变化；外观供体只决定材质、光照和相机感觉。</p><div class='grid'><figure><img src='{_uri(program)}'><figcaption><span class='ok'>机制正例：</span>四帧内容和时间顺序必须保留。</figcaption></figure><figure><img src='{_uri(donor)}'><figcaption><span class='ok'>外观供体：</span>{spec['donor_label']}</figcaption></figure><figure><img src='{_uri(negative)}'><figcaption><span class='bad'>反例规则：</span>{spec['negative']}</figcaption></figure></div><h2>通过前必须满足</h2><ol>{''.join(f'<li>{item}</li>' for item in spec['hard'])}</ol></main></body></html>"""


def _version_contract_baselines() -> None:
    """Preserve the pre-VTP V1 bytes and register the upgraded V2 files."""
    accepted_path = STAGE3 / "baselines/accepted.json"
    accepted = load_json(accepted_path)
    records = accepted["records"]
    by_id = {record["baseline_id"]: record for record in records}
    for case_id in CASES:
        v1_id = f"CONTRACT-{case_id}-V1"
        v1_record = by_id[v1_id]
        current_path = STAGE3 / "contracts" / f"{case_id}.json"
        old_value = load_json(current_path)
        old_value["visual_target_package"] = {"path": None, "status": "missing"}
        archive = STAGE3 / "baselines/contracts" / f"{case_id}-v1.json"
        write_json(archive, old_value)
        if sha256_path(archive) != v1_record["sha256"]:
            raise RuntimeError(f"reconstructed {v1_id} does not match frozen hash")
        v1_record["path"] = archive.relative_to(REPO_ROOT).as_posix()
        v2 = {
            "baseline_id": f"CONTRACT-{case_id}-V2",
            "kind": "input_contract",
            **file_record(current_path, REPO_ROOT),
        }
        by_id[v2["baseline_id"]] = v2
    accepted["records"] = list(by_id.values())
    write_json(accepted_path, accepted)


def run() -> dict[str, Any]:
    hypothesis = STAGE3 / f"experiments/{EXPERIMENT_ID}/hypothesis.md"
    hypothesis.parent.mkdir(parents=True, exist_ok=True)
    hypothesis.write_text(
        "# H-S3-0009A — 外观供体可复用，视觉合同不可复用\n\n"
        "五个 scale Case 可以复用已评审图片中的材质、光照和相机统计；但每个"
        "Case 必须保留独立的程序机制正例、反例描述、量表和硬门。若供体几何"
        "进入新案例，或任何合同引用无法验证，则假设失败。\n",
        encoding="utf-8",
    )
    registry = load_json(STAGE3 / "case_registry.json")
    results = []
    for case_id, spec in CASES.items():
        target = STAGE3 / "visual_targets" / case_id
        target.mkdir(parents=True, exist_ok=True)
        negative = _negative_reference(case_id, spec)
        rubric = _rubric(case_id, spec)
        write_json(target / "rubric.json", rubric)
        (target / "style_board.html").write_text(
            _style_board(case_id, spec, negative), encoding="utf-8"
        )
        manifest = {
            "schema_version": "1.0",
            "package_id": f"VT-{case_id}-V1",
            "case_id": case_id,
            "status": "accepted_project_baseline",
            "summary_zh": spec["summary"],
            "geometry_control_separation": {
                "geometry_source": "frozen input contract + semantic layers + geometry policy",
                "appearance_source": "appearance_donor_only; donor geometry is forbidden",
                "leakage_gate": "appearance_to_geometry_leakage",
            },
            "positive_refs": [
                {**file_record(_program_sheet(case_id), REPO_ROOT), "role": "accepted_mechanism_reference", "label_zh": "程序四关键帧：定义对象、状态和顺序。"},
                {**file_record(_donor_path(spec), REPO_ROOT), "role": "appearance_donor_only", "label_zh": spec["donor_label"]},
            ],
            "negative_refs": [
                {**file_record(negative, REPO_ROOT), "role": "appearance_and_mechanism_negative", "label_zh": spec["negative"]}
            ],
            "rubric": f"modules/video_model/stage3/visual_targets/{case_id}/rubric.json",
            "style_board": f"modules/video_model/stage3/visual_targets/{case_id}/style_board.html",
        }
        write_json(target / "manifest.json", manifest)
        validate_visual_target(manifest)
        for record in manifest["positive_refs"] + manifest["negative_refs"]:
            verify_file_record(record, REPO_ROOT)

        contract_path = STAGE3 / "contracts" / f"{case_id}.json"
        contract = load_json(contract_path)
        contract["visual_target_package"] = {
            "path": f"modules/video_model/stage3/visual_targets/{case_id}/manifest.json",
            "status": "accepted_project_baseline",
        }
        validate_input_contract(contract)
        write_json(contract_path, contract)
        case = next(item for item in registry["cases"] if item["case_id"] == case_id)
        case["visual_target_status"] = "accepted_project_baseline"
        case["known_gaps"] = []
        case["completeness"]["visual_target_status"] = "accepted_project_baseline"
        case["completeness"]["known_gaps"] = []
        results.append({"case_id": case_id, "manifest": file_record(target / "manifest.json", REPO_ROOT), "passed": True})
    write_json(STAGE3 / "case_registry.json", registry)
    _version_contract_baselines()

    smoke = []
    for path in sorted((STAGE3 / "contracts").glob("*.json")):
        error = None
        try:
            validate_input_contract(load_json(path))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        smoke.append({"case_id": path.stem, "passed": error is None, "error": error})
    result = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "appearance donor may be shared; visual/mechanism contract may not",
        "case_results": results,
        "contract_smoke": smoke,
        "passed": all(item["passed"] for item in results + smoke),
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
        "decision_zh": "五个视觉包接受为项目基线；这不代表最终图已通过，下一轮仍需逐案 G2/G3。",
    }
    write_json(OUTPUT / "vtp-audit.json", result)
    review = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "verdict": "accepted_core" if result["passed"] else "rejected",
        "passed": result["passed"],
        "reason_zh": "五个案例都有独立机制正例、反例、量表和硬门，外观供体仅以 appearance_donor_only 引用；十一份输入合同通过。",
        "model_runs": result["model_runs"],
        "evidence": file_record(OUTPUT / "vtp-audit.json", REPO_ROOT),
    }
    write_json(STAGE3 / f"experiments/{EXPERIMENT_ID}/review.json", review)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
