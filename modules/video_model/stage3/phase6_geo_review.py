"""Persist the GEO-02 blind visual review and accepted Visual Target V2."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    write_json,
)


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-6-rerun-2"
VTP = STAGE3 / "visual_targets/GEO-02"
IDS = ("00_start", "01_mechanism", "02_result", "03_end")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _weighted(scores: dict[str, float]) -> float:
    weights = {
        "terrain_and_atmosphere_material": 0.3,
        "lighting_coherence": 0.2,
        "camera_and_scene_stability": 0.2,
        "realism_without_plasticity": 0.2,
        "rain_shadow_readability": 0.1,
    }
    return round(sum(scores[key] * weights[key] for key in weights), 3)


def _blind_sheet() -> Path:
    roots = {
        "A": OUTPUT / "GEO-02/candidate/frames",
        "B": OUTPUT / "GEO-02/negative_control/frames",
    }
    cell = (430, 285)
    canvas = Image.new("RGB", (cell[0] * 4, cell[1] * 2), (13, 29, 32))
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    for row, (option, root) in enumerate(roots.items()):
        for column, frame_id in enumerate(IDS):
            image = Image.open(root / f"{frame_id}.png").convert("RGB")
            image.thumbnail((cell[0] - 10, cell[1] - 42))
            x = column * cell[0]
            y = row * cell[1]
            canvas.paste(image, (x + (cell[0] - image.width) // 2, y + 4))
            draw.text(
                (x + 10, y + cell[1] - 29),
                f"OPTION {option} / {frame_id}",
                fill=(236, 247, 242),
                font=font,
            )
    target = OUTPUT / "report-assets/blind-geo-comparison.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=92, subsampling=0)
    return target


def _rubric() -> dict[str, Any]:
    value = {
        "schema_version": "1.0",
        "score_scale": {
            "1": "明显失败，不能用于教学",
            "3": "基本可读，但地形、大气或降水仍不稳定",
            "5": "材质可信，且迎风降雨—背风变干一眼可读",
        },
        "appearance_dimensions": [
            {
                "id": "terrain_and_atmosphere_material",
                "question_zh": "山体表面、云雾和降水是否像自然地表与大气，而不是塑料模型？",
                "weight": 0.3,
            },
            {
                "id": "lighting_coherence",
                "question_zh": "阴天散射光、山坡明暗与云雾遮挡是否自洽？",
                "weight": 0.2,
            },
            {
                "id": "camera_and_scene_stability",
                "question_zh": "四帧的山体、地面、相机和裁切是否固定？",
                "weight": 0.2,
            },
            {
                "id": "realism_without_plasticity",
                "question_zh": "是否避免玩具山、硬描边、规则雨线网格和扩散伪影？",
                "weight": 0.2,
            },
            {
                "id": "rain_shadow_readability",
                "question_zh": "不看文字时，是否仍能看出湿空气团越山、迎风降雨、背风端变干？",
                "weight": 0.1,
            },
        ],
        "hard_gates": [
            {"id": "appearance_to_geometry_leakage", "pass_zh": "外观底图不改写程序的地形侧别、空气团方向和降雨区。"},
            {"id": "windward_precipitation", "pass_zh": "主降雨质心在山顶左侧，且结果帧为峰值。"},
            {"id": "leeward_drying", "pass_zh": "终帧空气团在背风侧，湿度和降雨明显低于峰值。"},
            {"id": "negative_reference_avoidance", "pass_zh": "云雨没有被解读成第二座山、建筑或硬线条。"},
        ],
        "acceptance": {"minimum_weighted_score": 4.0, "all_hard_gates_must_pass": True},
    }
    write_json(VTP / "rubric.json", value)
    return value


def run() -> dict[str, Any]:
    blind = _blind_sheet()
    scores = {
        "A": {
            "terrain_and_atmosphere_material": 4.5,
            "lighting_coherence": 4.5,
            "camera_and_scene_stability": 5.0,
            "realism_without_plasticity": 4.3,
            "rain_shadow_readability": 4.1,
        },
        "B": {
            "terrain_and_atmosphere_material": 4.5,
            "lighting_coherence": 4.5,
            "camera_and_scene_stability": 5.0,
            "realism_without_plasticity": 4.5,
            "rain_shadow_readability": 1.0,
        },
    }
    machine = load_json(OUTPUT / "GEO-02/g3-machine.json")
    result = {
        "schema_version": "1.0",
        "experiment_id": "EXP-S3-20260731-024",
        "blind_sheet": file_record(blind, REPO_ROOT),
        "blind_map": {"A": "frozen appearance plus program state", "B": "frozen appearance only"},
        "options": {
            "A": {
                "scores_1_to_5": scores["A"],
                "weighted_score": _weighted(scores["A"]),
                "hard_gates_passed": machine["passed"],
                "judgment_zh": "山体材质与光照保持，降雨仅在迎风坡结果帧达峰，空气团稳定越山。",
            },
            "B": {
                "scores_1_to_5": scores["B"],
                "weighted_score": _weighted(scores["B"]),
                "hard_gates_passed": False,
                "judgment_zh": "静态山体外观良好，但四帧完全相同，无法讲解地形雨。",
            },
        },
        "selected_blind_id": "A",
        "passed": machine["passed"] and _weighted(scores["A"]) >= 4.0,
        "decision_scope": "accepted_project_baseline; no user style choice inferred",
    }
    write_json(OUTPUT / "visual-review.json", result)
    _rubric()

    manifest = load_json(VTP / "manifest.json")
    anchor = manifest["positive_refs"][0]
    anchor["label_zh"] = "外观供体：只提供山体材质、阴天光照和空气透视，不提供机制几何。"
    anchor["role"] = "appearance_donor_only"
    sequence = file_record(OUTPUT / "GEO-02/candidate/sequence.jpg", REPO_ROOT)
    manifest.update(
        {
            "package_id": "VT-GEO-02-V2",
            "status": "accepted_project_baseline",
            "summary_zh": "冻结 SDXL 山体外观，由程序标量场和对象身份层添加迎风降雨、越山与背风变干。",
            "positive_refs": [
                anchor,
                {
                    **sequence,
                    "role": "accepted_mechanism_complete_sequence",
                    "label_zh": "已评审四帧：山体固定，空气团左到右，迎风坡降雨达峰后背风端变干。",
                },
            ],
        }
    )
    write_json(VTP / "manifest.json", manifest)

    positive = OUTPUT / "GEO-02/candidate/sequence.jpg"
    negative = Path(REPO_ROOT / manifest["negative_refs"][0]["path"])
    anchor_path = Path(REPO_ROOT / anchor["path"])
    program = REPO_ROOT / "modules/video_model/stage2/output/phase-2/GEO-02/keyframe-contact-sheet.jpg"
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>GEO-02 视觉目标 V2</title>
<style>body{{font:16px/1.65 system-ui;margin:0;padding:28px;background:#f4f1e9;color:#18211d}}main{{max-width:1100px;margin:auto}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}figure{{margin:0;background:white;border:1px solid #d5d0c4;border-radius:12px;padding:10px}}img{{width:100%;height:auto}}figcaption{{padding:8px}}.ok{{color:#16704b;font-weight:700}}</style></head>
<body><main><p>Stage 3 · GEO-02 · accepted project baseline</p><h1>外观与机制分开的地形雨目标</h1>
<p>外观供体只决定山体材质、阴天光照和空气透视；空气团位置、云雨标量场、降雨侧别和时间顺序全部来自程序。</p>
<div class='grid'><figure><img src='{_data_uri(program)}'><figcaption>程序事实源：越山、迎风降雨、背风变干。</figcaption></figure>
<figure><img src='{_data_uri(anchor_path)}'><figcaption>外观供体：不单独证明机制。</figcaption></figure>
<figure><img src='{_data_uri(positive)}'><figcaption><span class='ok'>正例：</span>程序状态回写后的完整四帧。</figcaption></figure>
<figure><img src='{_data_uri(negative)}'><figcaption><b>反例：</b>云雨边缘被 ControlNet 误解为山脊。</figcaption></figure></div>
<p>通过阈值：加权分 4.0，且地形固定、迎风降雨、背风变干、无外观→几何泄漏四个硬门全通过。</p></main></body></html>"""
    (VTP / "style_board.html").write_text(page, encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
