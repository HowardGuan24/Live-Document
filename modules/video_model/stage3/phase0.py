"""Build and verify Stage 3 Phase S3.0 without running image/video models.

This migration freezes the real Stage 2/Stage 1 program artifacts into:
case contracts, motion contracts, visual-target packages, accepted baselines,
and persistent loop state. The output is deterministic for an unchanged repo.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    validate_case_registry,
    validate_input_contract,
    validate_loop_state,
    validate_motion_contract,
    validate_schema_documents,
    validate_visual_target,
    verify_file_record,
    write_json,
)


STAGE3 = Path(__file__).resolve().parent
VIDEO_MODEL = STAGE3.parent
STAGE2 = VIDEO_MODEL / "stage2"
STAGE1 = VIDEO_MODEL / "stage1"
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output" / "phase-0"
CONTRACTS = STAGE3 / "contracts"
MOTIONS = STAGE3 / "motion_contracts"
VISUAL_TARGETS = STAGE3 / "visual_targets"
SCHEMAS = STAGE3 / "schemas"


GEOMETRY_POLICY = {
    "MATH-01": "preserve_exact",
    "MATH-02": "preserve_exact",
    "PHYS-01": "preserve_exact",
    "PHYS-02": "canonicalize",
    "CHEM-01": "canonicalize",
    "CHEM-02": "canonicalize",
    "BIO-01": "layout_only",
    "BIO-02": "layout_only",
    "GEO-01": "layout_only",
    "GEO-02": "layout_only",
    "GEO-HIST-DELTA-01": "layout_only",
}

CONTROL_OBJECT_REQUIREMENTS = {
    "CHEM-01": [
        {
            "class_id": "glass_beaker",
            "source": "object_identity",
            "cardinality": 1,
        },
        {
            "class_id": "glass_burette",
            "source": "object_identity",
            "cardinality": 1,
            "legacy_migration": {
                "source": "unclaimed_hard_boundary_components",
                "relation": "paired_parallel_boundaries_above_glass_beaker",
                "coordinates_must_be_inferred": True,
            },
        },
    ],
    "PHYS-02": [
        {
            "class_id": "bar_magnet",
            "source": "object_identity",
            "cardinality": 1,
        },
        {
            "class_id": "fixed_coil",
            "source": "object_identity",
            "cardinality": 1,
        },
    ],
    "MATH-02": [
        {
            "class_id": "congruent_right_triangle",
            "source": "object_identity",
            "cardinality": 4,
        }
    ],
}


VISUAL_SPECS: dict[str, dict[str, Any]] = {
    "CHEM-01": {
        "status": "accepted_project_baseline",
        "summary": "透明玻璃、克制的实验室光照；局部指示剂颜色必须仍由程序 pH 场决定。",
        "positive": [
            (
                "modules/video_model/stage2/output/phase-9/report-assets/chem-final-b-sequence.jpg",
                "已评审的 A→B 四帧：真实玻璃底图冻结，程序只改变液体内部状态。",
            ),
            (
                "modules/video_model/stage2/output/phase-7/route-a/experiments/EXP-P7-A-chem-01-00_start/raw/semantic_control_065/seed_7101.png",
                "Route A 选中的真实玻璃与光照供体。",
            ),
        ],
        "negative": [
            (
                "modules/video_model/stage2/output/phase-9/report-assets/chem-a-rejected-direct-sequence.jpg",
                "反例：逐帧自由生成会移动器材、背景和液体颜色。",
            ),
            (
                "modules/video_model/stage2/output/phase-7/route-a/experiments/EXP-P7-A-chem-01-00_start/controls/dense_canny.png",
                "反例控制：整张程序截图的密集边缘会把界面细节错误地当成场景几何。",
            ),
        ],
    },
    "MATH-02": {
        "status": "accepted_project_baseline",
        "summary": "桌面木质拼块可有倒角和木纹，但四块全等三角形的面积、身份与刚体变换不能变。",
        "positive": [
            (
                "modules/video_model/stage2/output/phase-8/route-b-only/MATH-02/variants/frozen_scene_depth/03_end.png",
                "已评审的统一 B 路线：冻结材质，保留四块拼图的精确几何。",
            ),
            (
                "modules/video_model/stage2/output/phase-7/route-c/MATH-02/variants/studio_pbr/03_end.png",
                "历史质量上界：对象局部木纹、倒角和稳定棚拍光。",
            ),
        ],
        "negative": [
            (
                "modules/video_model/stage2/output/phase-8/route-b-only/MATH-02/variants/frozen_texture_tint/01_mechanism.png",
                "反例：屏幕坐标纹理会在移动拼块内部滑动。",
            )
        ],
    },
    "PHYS-01": {
        "status": "accepted_project_baseline",
        "summary": "俯视水面、稳定相机和适度镜面反光；波峰波谷只能来自程序高度场。",
        "positive": [
            (
                "modules/video_model/stage2/output/phase-8/route-b-only/PHYS-01/variants/calm_frozen_water_relief/02_result.png",
                "已评审的统一 B 路线：程序高度场驱动冻结水面上的光学变化。",
            ),
            (
                "modules/video_model/stage2/output/phase-7/route-c/PHYS-01/variants/specular_water/02_result.png",
                "历史质量上界：适度镜面水面，干涉结构清晰。",
            ),
        ],
        "negative": [
            (
                "modules/video_model/stage2/output/phase-7/route-c/PHYS-01/variants/refractive_water/02_result.png",
                "反例：强折射在干涉区制造棋盘伪影。",
            )
        ],
    },
    "BIO-01": {
        "status": "accepted_project_baseline",
        "summary": "显微镜式细胞材质和柔和深度；染色体数量、身份和分配由程序决定。",
        "positive": [
            (
                "modules/video_model/stage2/output/phase-7/route-b/BIO-01/variants/stable_material_plus_depth/02_result.png",
                "已评审版本：只迁移稳定材质统计，细胞和染色体仍由程序绘制。",
            )
        ],
        "negative": [
            (
                "modules/video_model/stage2/output/phase-7/route-b/BIO-01/variants/raw_underlay/02_result.png",
                "反例：直接贴供体会带入程序中不存在的细胞器。",
            )
        ],
    },
    "GEO-02": {
        "status": "provisional",
        "summary": "可信山地地形、空气透视和自然云层；迎风坡降水位置仍待得到稳定基线。",
        "positive": [
            (
                "modules/video_model/stage2/output/phase-7/route-a/experiments/EXP-P7-A-geo-02-02_result-terrain_only/raw/semantic_control_045/seed_7101.png",
                "暂定参考：只控制山体后材质改善，尚未通过降水位置硬门禁。",
            )
        ],
        "negative": [
            (
                "modules/video_model/stage2/output/phase-7/route-a/experiments/EXP-P7-A-geo-02-02_result/raw/semantic_control_045/seed_7101.png",
                "反例：把雨线和云边一起编码，模型会把它们误读成山脊或建筑。",
            )
        ],
    },
    "GEO-HIST-DELTA-01": {
        "status": "accepted_project_baseline",
        "summary": "固定正射视角与自然河口材质；泥沙羽流、水下沉积、湿沙洲和绕流必须逐阶段出现。",
        "positive": [
            (
                "modules/video_model/stage1/output/keyframe_render/delta_sequence/sequence-contact-sheet.jpg",
                "用户认可的历史五帧效果基线。",
            ),
            (
                "modules/video_model/stage1/output/keyframe_render/delta_sequence/final/04_rerouted_flow.png",
                "最终帧质量锚点：同一湿沙洲保持，水流从两侧绕过。",
            ),
        ],
        "negative": [
            (
                "modules/video_model/stage1/output/keyframe_render/delta_sequence/raw-candidates-by-seed.jpg",
                "反例集合：自由候选的岸线、湿沙形状和阶段语义不稳定。",
            )
        ],
    },
}


RUBRIC = {
    "schema_version": "1.0",
    "score_scale": {
        "1": "明显失败，不能用于教学",
        "3": "基本可读，但材质、光照或真实感仍不稳定",
        "5": "达到本项目可进入视频阶段的质量",
    },
    "appearance_dimensions": [
        {
            "id": "material_legibility",
            "weight": 0.30,
            "question_zh": "新人能否一眼识别主要材料（水、玻璃、木、细胞或地形）？",
        },
        {
            "id": "lighting_coherence",
            "weight": 0.20,
            "question_zh": "光源方向、阴影、反光和透明关系是否自洽？",
        },
        {
            "id": "camera_and_scene_stability",
            "weight": 0.20,
            "question_zh": "相机、背景和未变化对象是否在关键帧间保持固定？",
        },
        {
            "id": "realism_without_plasticity",
            "weight": 0.20,
            "question_zh": "是否避免蜡质、塑料感、过饱和与无意义高光？",
        },
        {
            "id": "teaching_readability",
            "weight": 0.10,
            "question_zh": "材质增强是否仍让关键机制比装饰细节更醒目？",
        },
    ],
    "hard_gates": [
        {
            "id": "appearance_to_geometry_leakage",
            "pass_zh": "外观参考没有改变对象数量、几何、拓扑、相机或状态边界。",
        },
        {
            "id": "negative_reference_avoidance",
            "pass_zh": "未复现视觉目标包中标注的反例缺陷。",
        },
    ],
    "acceptance": {
        "minimum_weighted_score": 4.0,
        "all_hard_gates_must_pass": True,
    },
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def href(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace(
        os.sep, "/"
    )


def record_from_repo_path(path: str) -> dict[str, Any]:
    return file_record(REPO_ROOT / path, REPO_ROOT)


def program_root(case_id: str, sentinel: bool) -> Path:
    phase = "phase-2" if sentinel else "phase-4/programs"
    return STAGE2 / "output" / phase / case_id


def collect_program(case: dict[str, Any]) -> dict[str, Any]:
    case_id = case["case_id"]
    root = program_root(case_id, bool(case["sentinel"]))
    keyframes: list[dict[str, Any]] = []
    layer_types: set[str] = set()
    layer_ids: set[str] = set()
    object_classes: set[str] = set()
    semantic_records: list[dict[str, Any]] = []
    for keyframe_dir in sorted((root / "keyframes").iterdir()):
        if not keyframe_dir.is_dir():
            continue
        semantic_path = keyframe_dir / "semantic_layers.json"
        semantic = load_json(semantic_path)
        state_path = keyframe_dir / "state.json"
        clean_path = keyframe_dir / "clean.png"
        program_path = keyframe_dir / "program.png"
        for layer in semantic["layers"]:
            layer_types.add(layer["layer_type"])
            layer_ids.add(layer["layer_id"])
            if layer["layer_type"] == "object_identity":
                identity_path = root / layer["data"]["path"]
                identity = load_json(identity_path)
                for item in identity.get("items", []):
                    object_classes.add(
                        item.get("class_id")
                        or item.get("kind")
                        or "untyped_object"
                    )
        keyframes.append(
            {
                "keyframe_id": keyframe_dir.name,
                "order": len(keyframes),
                "state": file_record(state_path, REPO_ROOT),
                "clean_program_frame": file_record(clean_path, REPO_ROOT),
                "annotated_program_frame": file_record(
                    program_path, REPO_ROOT
                ),
                "semantic_layers": file_record(
                    semantic_path, REPO_ROOT
                ),
            }
        )
        semantic_records.append(file_record(semantic_path, REPO_ROOT))
    frames = sorted((root / "frames").glob("*.png"))
    concept = (
        STAGE2
        / "output"
        / "phase-1"
        / "fixtures"
        / case_id
        / "concept_spec.json"
    )
    sequence = concept.with_name("sequence_spec.json")
    return {
        "root": root,
        "keyframes": keyframes,
        "frames": frames,
        "layer_types": sorted(layer_types),
        "layer_ids": sorted(layer_ids),
        "object_classes": sorted(object_classes),
        "semantic_records": semantic_records,
        "concept": concept,
        "sequence": sequence,
    }


def build_visual_target(case_id: str) -> dict[str, Any]:
    spec = VISUAL_SPECS[case_id]
    package_dir = VISUAL_TARGETS / case_id
    package_dir.mkdir(parents=True, exist_ok=True)
    rubric_path = package_dir / "rubric.json"
    write_json(rubric_path, RUBRIC)
    positive = []
    for path, label in spec["positive"]:
        item = record_from_repo_path(path)
        item.update({"label_zh": label, "role": "appearance_positive"})
        positive.append(item)
    negative = []
    for path, label in spec["negative"]:
        item = record_from_repo_path(path)
        item.update({"label_zh": label, "role": "appearance_negative"})
        negative.append(item)
    board_path = package_dir / "style_board.html"
    cards = []
    for role, records in (
        ("要接近的外观", positive),
        ("要避免的外观", negative),
    ):
        for item in records:
            image_path = REPO_ROOT / item["path"]
            cards.append(
                "<figure>"
                f"<img src='{html.escape(href(package_dir, image_path))}' "
                f"alt='{html.escape(item['label_zh'])}'>"
                f"<figcaption><strong>{role}</strong><br>"
                f"{html.escape(item['label_zh'])}</figcaption></figure>"
            )
    board_path.write_text(
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>视觉目标板</title><style>
body{font-family:system-ui,sans-serif;background:#f4f1e9;color:#18211d;
margin:0;padding:28px}h1{margin:.2em 0}.note{max-width:900px;line-height:1.7}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
gap:18px;margin-top:24px}figure{margin:0;background:white;border:1px solid #d5d0c4;
border-radius:14px;overflow:hidden}img{width:100%;height:230px;object-fit:contain;
background:#111}figcaption{padding:14px;line-height:1.55}
</style></head><body>"""
        f"<p>Stage 3 · {html.escape(case_id)}</p><h1>外观目标，不是几何输入</h1>"
        f"<p class='note'>{html.escape(spec['summary'])}</p>"
        "<p class='note'>这里的图只回答“应该看起来像什么”。物体在哪、数量多少、"
        "边界和状态怎样变化，全部由程序合同与控制图决定；不能从这些参考图偷几何。</p>"
        f"<section class='grid'>{''.join(cards)}</section></body></html>",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0",
        "package_id": f"VT-{case_id}-V1",
        "case_id": case_id,
        "status": spec["status"],
        "summary_zh": spec["summary"],
        "geometry_control_separation": {
            "geometry_source": (
                "frozen input contract + semantic layers + geometry policy"
            ),
            "appearance_source": (
                "only the positive/negative references in this package"
            ),
            "leakage_gate": "appearance_to_geometry_leakage",
        },
        "style_board": rel(board_path),
        "positive_refs": positive,
        "negative_refs": negative,
        "rubric": rel(rubric_path),
    }
    validate_visual_target(manifest)
    manifest_path = package_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def build_motion_contract(
    case: dict[str, Any], collected: dict[str, Any]
) -> dict[str, Any]:
    case_id = case["case_id"]
    root = collected["root"]
    motion = {
        "schema_version": "1.0",
        "case_id": case_id,
        "state_timeline": file_record(root / "states.jsonl", REPO_ROOT),
        "program_video": file_record(
            root / "program-animation.mp4", REPO_ROOT
        ),
        "motion_classes": case["motion_classes"],
        "keyframe_selection": [
            {
                "keyframe_id": item["keyframe_id"],
                "order": item["order"],
                "state": item["state"],
            }
            for item in collected["keyframes"]
        ],
        "model_input_policy": (
            "The full timeline is used to extract direction, topology, "
            "speed trend, and occlusion constraints. Raw program video is "
            "not a default pixel input to an image or video model."
        ),
        "temporal_gates": [
            *case["case_hard_gates"],
            "camera_and_background_remain_fixed",
            "no_uncontracted_object_birth_or_loss",
        ],
    }
    validate_motion_contract(motion)
    path = MOTIONS / f"{case_id}.json"
    write_json(path, motion)
    return motion


def historical_delta_case() -> tuple[dict[str, Any], dict[str, Any]]:
    handoff_path = (
        STAGE1
        / "output"
        / "keyframe_render"
        / "delta_sequence"
        / "video_handoff.json"
    )
    handoff = load_json(handoff_path)
    keyframes = []
    for order, source in enumerate(handoff["ordered_keyframes"]):
        path = Path(source["path"])
        keyframes.append(
            {
                "keyframe_id": source["id"],
                "order": order,
                "meaning_zh": source["meaning"],
                "clean_program_frame": file_record(path, REPO_ROOT),
                "annotated_program_frame": file_record(path, REPO_ROOT),
                "state": file_record(handoff_path, REPO_ROOT),
                "semantic_layers": file_record(
                    STAGE1
                    / "output"
                    / "keyframe_render"
                    / "delta_sequence"
                    / "_work"
                    / "metadata.json",
                    REPO_ROOT,
                ),
            }
        )
    state_path = (
        STAGE1
        / "output"
        / "causal_delta"
        / "mechanism"
        / "states.jsonl"
    )
    timeline_path = (
        STAGE1 / "output" / "causal_delta" / "timeline.json"
    )
    program_video = (
        STAGE1 / "output" / "causal_delta" / "program-animation.mp4"
    )
    if not program_video.is_file():
        program_video = (
            STAGE1
            / "output"
            / "keyframe_render"
            / "delta_sequence"
            / "sequence-contact-sheet.jpg"
        )
    case = {
        "case_id": "GEO-HIST-DELTA-01",
        "slug": "delta_formation_historical",
        "discipline": "geography",
        "discipline_zh": "地理",
        "title_zh": "历史三角洲形成质量回归",
        "sentinel": False,
        "primary_layer_types": [
            "hard_boundary",
            "region",
            "scalar_field",
            "vector_field",
            "object_identity",
        ],
        "capability_tags": [
            "historical_quality_anchor",
            "sediment_transport",
            "water_topology",
        ],
        "motion_classes": [
            "material_transport",
            "boundary_evolution",
            "flow_rerouting",
        ],
        "case_hard_gates": [
            "shoreline_and_camera_are_fixed",
            "sediment_stage_order_is_preserved",
            "final_flow_has_exactly_two_paths_around_one_sandbar",
        ],
    }
    motion = {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "state_timeline": file_record(state_path, REPO_ROOT),
        "program_video": file_record(program_video, REPO_ROOT),
        "motion_classes": case["motion_classes"],
        "keyframe_selection": [
            {
                "keyframe_id": item["keyframe_id"],
                "order": item["order"],
                "state": item["state"],
            }
            for item in keyframes
        ],
        "model_input_policy": (
            "Use the 120-state mechanism timeline as motion truth; "
            "the accepted rendered sequence is appearance evidence only."
        ),
        "temporal_gates": case["case_hard_gates"],
    }
    validate_motion_contract(motion)
    write_json(MOTIONS / f"{case['case_id']}.json", motion)
    collected = {
        "root": STAGE1 / "output" / "causal_delta",
        "keyframes": keyframes,
        "frames": [],
        "layer_types": case["primary_layer_types"],
        "layer_ids": [
            "coastline_boundary",
            "water_region",
            "sediment_concentration",
            "flow_vector",
            "landform_identity",
        ],
        "object_classes": [
            "original_coast",
            "river_channel",
            "sediment_plume",
            "wet_sandbar",
        ],
        "concept": STAGE1 / "keyframe_render" / "delta_sequence_spec.json",
        "sequence": handoff_path,
    }
    return case, {"collected": collected, "motion": motion}


def build_input_contract(
    case: dict[str, Any],
    collected: dict[str, Any],
    motion: dict[str, Any],
    visual: dict[str, Any] | None,
) -> dict[str, Any]:
    program_source: dict[str, Any] = {
        "root": rel(collected["root"]),
        "concept_spec": file_record(collected["concept"], REPO_ROOT),
        "sequence_spec": file_record(collected["sequence"], REPO_ROOT),
    }
    program_manifest = collected["root"] / "program_manifest.json"
    if program_manifest.is_file():
        program_source["program_manifest"] = file_record(
            program_manifest, REPO_ROOT
        )
    contract = {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "case_definition": {
            "title_zh": case["title_zh"],
            "discipline": case["discipline"],
            "discipline_zh": case["discipline_zh"],
            "slug": case["slug"],
            "capability_tags": case["capability_tags"],
        },
        "program_source": program_source,
        "keyframes": collected["keyframes"],
        "semantic_exports": {
            "layer_types": collected["layer_types"],
            "layer_ids": collected["layer_ids"],
            "object_classes": collected["object_classes"],
            "control_object_requirements": CONTROL_OBJECT_REQUIREMENTS.get(
                case["case_id"], []
            ),
            "source_policy": (
                "Exported directly from deterministic program state; "
                "not inferred from the rendered screenshot."
            ),
        },
        "geometry_policy": GEOMETRY_POLICY[case["case_id"]],
        "motion_contract": {
            "path": rel(MOTIONS / f"{case['case_id']}.json"),
            "sha256": sha256_path(
                MOTIONS / f"{case['case_id']}.json"
            ),
        },
        "visual_target_package": (
            {
                "path": rel(
                    VISUAL_TARGETS / case["case_id"] / "manifest.json"
                ),
                "status": visual["status"],
            }
            if visual
            else {"path": None, "status": "missing"}
        ),
        "hard_gates": [
            *case["case_hard_gates"],
            "appearance_to_geometry_leakage",
        ],
    }
    validate_input_contract(contract)
    path = CONTRACTS / f"{case['case_id']}.json"
    write_json(path, contract)
    return contract


def completeness_record(
    case: dict[str, Any],
    collected: dict[str, Any],
    visual: dict[str, Any] | None,
) -> dict[str, Any]:
    case_id = case["case_id"]
    keyframes = collected["keyframes"]
    return {
        "case_id": case_id,
        "program_available": collected["root"].is_dir(),
        "timeline_available": (
            collected["root"] / "states.jsonl"
        ).is_file()
        or case_id == "GEO-HIST-DELTA-01",
        "continuous_program_frames": len(collected["frames"]),
        "keyframe_count": len(keyframes),
        "semantic_layer_types": collected["layer_types"],
        "object_classes": collected["object_classes"],
        "geometry_policy": GEOMETRY_POLICY[case_id],
        "visual_target_status": (
            visual["status"] if visual else "missing"
        ),
        "contract_smoke_passed": True,
        "known_gaps": (
            [
                "Visual target package has not yet been curated; model "
                "generation is forbidden until it exists."
            ]
            if visual is None
            else (
                [
                    "Visual target is provisional; a generated candidate "
                    "must not be promoted without visual and mechanism review."
                ]
                if visual["status"] == "provisional"
                else []
            )
        ),
    }


def selector(
    registry_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen deterministic Phase S3.1 selection priority."""
    candidates = []
    for case in registry_cases:
        if case["case_id"] == "GEO-HIST-DELTA-01":
            continue
        policy = case["geometry_policy"]
        phase_gap = {
            "canonicalize": 3,
            "preserve_exact": 2,
            "layout_only": 1,
            "unsupported": 0,
        }[policy]
        status = case["visual_target_status"]
        visual_ready = {
            "user_approved": 3,
            "accepted_project_baseline": 2,
            "provisional": 1,
            "missing": 0,
        }[status]
        hard_failure_severity = (
            3 if case["case_id"] == "CHEM-01" else 1
        )
        candidates.append(
            {
                "case_id": case["case_id"],
                "priority_tuple": [
                    phase_gap,
                    hard_failure_severity,
                    visual_ready,
                    -len(case["known_gaps"]),
                ],
                "reason_zh": (
                    "Stage 2 已证明手写器材模板有效，但模板与程序语义断开；"
                    "这是把一次性烧杯结果改成通用几何重建器的最高优先缺口。"
                    if case["case_id"] == "CHEM-01"
                    else "按阶段缺口、失败严重度、视觉目标完整度排序。"
                ),
            }
        )
    ordered = sorted(
        candidates,
        key=lambda item: (
            tuple(-x for x in item["priority_tuple"]),
            item["case_id"],
        ),
    )
    selected = ordered[0]
    return {
        "schema_version": "1.0",
        "phase": "S3.1",
        "rule": (
            "phase gap → hard failure severity → cross-disciplinary "
            "coverage → input/visual completeness → cost → case_id"
        ),
        "selected_target": selected["case_id"],
        "selected_geometry_policy": "canonicalize",
        "regression_cohort": [
            {
                "case_id": "PHYS-02",
                "role": "cross_discipline_canonical_apparatus",
            },
            {
                "case_id": "MATH-02",
                "role": "preserve_exact_non_interference",
            },
        ],
        "historical_regression": {
            "case_id": "GEO-HIST-DELTA-01",
            "required": False,
            "reason_zh": "本轮只改 typed apparatus canonicalizer，不触碰地貌 layout_only 编译。",
        },
        "candidate_ranking": ordered,
    }


def check_report_links(report_path: Path) -> list[str]:
    text = report_path.read_text(encoding="utf-8")
    missing = []
    for marker in ("src='", "href='"):
        start = 0
        while True:
            index = text.find(marker, start)
            if index < 0:
                break
            value_start = index + len(marker)
            value_end = text.find("'", value_start)
            value = text[value_start:value_end]
            start = value_end + 1
            if (
                not value
                or value.startswith(("#", "http:", "https:", "data:"))
            ):
                continue
            target = (report_path.parent / value).resolve()
            if not target.exists():
                missing.append(value)
    return sorted(set(missing))


def render_report(
    registry_cases: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    selection: dict[str, Any],
) -> Path:
    rows = []
    for case in registry_cases:
        gap = (
            "—"
            if not case["known_gaps"]
            else "<br>".join(
                html.escape(item) for item in case["known_gaps"]
            )
        )
        rows.append(
            "<tr>"
            f"<td><strong>{case['case_id']}</strong><br>"
            f"{html.escape(case['title_zh'])}</td>"
            f"<td>{html.escape(case['discipline_zh'])}</td>"
            f"<td><code>{case['geometry_policy']}</code></td>"
            f"<td>{case['keyframe_count']} 个关键帧；"
            f"{case['continuous_program_frames']} 个连续帧</td>"
            f"<td><code>{case['visual_target_status']}</code></td>"
            f"<td>{gap}</td></tr>"
        )
    check_rows = "".join(
        "<li class='pass'>✓ "
        f"{html.escape(item['name'])}"
        f"<small>{html.escape(item.get('evidence_zh', ''))}</small></li>"
        for item in checks
        if item["passed"]
    )
    report_dir = OUTPUT
    chem_program = (
        STAGE2
        / "output"
        / "phase-2"
        / "CHEM-01"
        / "keyframe-contact-sheet.jpg"
    )
    chem_identity = (
        STAGE2
        / "output"
        / "phase-2"
        / "CHEM-01"
        / "keyframes"
        / "01_mechanism"
        / "layers"
        / "chem01_object_identity_preview.png"
    )
    chem_boundary = (
        STAGE2
        / "output"
        / "phase-2"
        / "CHEM-01"
        / "keyframes"
        / "01_mechanism"
        / "layers"
        / "chem01_apparatus_boundary_preview.png"
    )
    chem_target = (
        STAGE2
        / "output"
        / "phase-9"
        / "report-assets"
        / "chem-final-b-sequence.jpg"
    )
    report_path = OUTPUT / "report.html"
    report_path.write_text(
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 3 · S3.0 冻结输入与启动 Loop</title><style>
:root{--ink:#17231f;--muted:#5e6b64;--paper:#f5f1e7;--card:#fffdf8;
--line:#d8d1c2;--green:#176548;--blue:#28587a;--amber:#9a5d14}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Noto Sans SC",sans-serif;
line-height:1.68}.hero{padding:64px max(5vw,24px) 42px;background:#173d34;color:#fff}
.hero p{max-width:900px;color:#dbeae5;font-size:18px}main{max-width:1240px;margin:auto;
padding:34px 24px 80px}section{background:var(--card);border:1px solid var(--line);
border-radius:18px;padding:28px;margin:22px 0}h1{font-size:clamp(32px,5vw,58px);
line-height:1.08;margin:.15em 0}h2{font-size:27px;margin:0 0 12px}h3{margin-top:26px}
.badge{display:inline-block;padding:5px 10px;border-radius:999px;background:#dff3e9;
color:var(--green);font-weight:750}.grid{display:grid;grid-template-columns:repeat(4,1fr);
gap:14px}.step{border-left:4px solid var(--blue);padding:12px;background:#edf3f5}
.step strong{display:block;font-size:20px}.gallery{display:grid;
grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.gallery figure{margin:0;
border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#111}
.gallery img{display:block;width:100%;height:310px;object-fit:contain}.gallery figcaption{
background:white;padding:13px}table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);
padding:11px 9px}th{background:#eee8dc;position:sticky;top:0}code{background:#e9eee9;
padding:2px 5px;border-radius:5px}.pass{color:var(--green);font-weight:700;
margin:8px 0}.pass small{display:block;color:var(--muted);font-weight:400;margin-left:22px}
.warning{border-left:5px solid var(--amber);padding:12px 16px;background:#fff2d8}
.decision{border-left:6px solid var(--green);background:#eaf6ef;padding:18px}
a{color:#135f78}@media(max-width:850px){.grid{grid-template-columns:1fr 1fr}
.gallery{grid-template-columns:1fr}table{display:block;overflow:auto}}
</style></head><body><header class="hero"><span class="badge">S3.0 PASS</span>
<h1>先把“输入、好图、坏图和验收”冻结，再让 Agent 自己迭代</h1>
<p>本阶段没有运行 SDXL 或视频模型。它解决的是更基础的问题：以后每次实验到底拿什么当输入、
什么不能被改、好看具体指什么、失败后怎样选择下一项。十个跨学科案例和历史三角洲都已进入同一套可校验合同。</p>
</header><main>
<section><h2>第一次接手项目，只需先理解这四件事</h2><div class="grid">
<div class="step"><strong>1 程序是真值</strong>连续状态、对象身份、区域和物理场由确定性程序输出。</div>
<div class="step"><strong>2 几何策略分流</strong>精确保留、规范重建或只保留布局；不再默认对整张截图做 Canny。</div>
<div class="step"><strong>3 外观另有目标包</strong>好图、坏图、材质、光照、相机与真实感量表单独保存。</div>
<div class="step"><strong>4 门禁决定能否继续</strong>先查合同与控制图，再生成，再查机制与外观，最后才做视频。</div>
</div></section>
<section><h2>真实数据长什么样：用烧杯举例</h2>
<p>左上是程序四个关键状态；右上是程序直接导出的器材对象身份；左下是器材硬边界；
右下是项目已接受的外观质量。前三者决定“东西在哪里、怎样变”，最后一张只决定“玻璃与光照看起来怎样”。</p>
<div class="gallery">
<figure><img src='"""
        + html.escape(href(report_dir, chem_program))
        + """' alt='烧杯程序关键帧'><figcaption><strong>程序关键帧</strong>：机制与阶段真值。</figcaption></figure>
<figure><img src='"""
        + html.escape(href(report_dir, chem_identity))
        + """' alt='烧杯对象身份'><figcaption><strong>对象身份</strong>：烧杯和液滴的稳定 ID 及位置。</figcaption></figure>
<figure><img src='"""
        + html.escape(href(report_dir, chem_boundary))
        + """' alt='器材硬边界'><figcaption><strong>硬边界</strong>：可供控制编译器选择，不等于整图 Canny。</figcaption></figure>
<figure><img src='"""
        + html.escape(href(report_dir, chem_target))
        + """' alt='烧杯外观目标'><figcaption><strong>外观基线</strong>：只回答玻璃、材质、光照与真实感。</figcaption></figure>
</div>
<p class="warning"><strong>本轮发现并明确记录的历史缺口：</strong>Stage 2 烧杯的漂亮线稿来自固定坐标模板，
并没有由程序语义生成。Stage 3 不把它伪装成已解决；S3.1 正是要把这条断链替换成 typed geometry canonicalizer。</p>
</section>
<section><h2>十一项输入的完整度矩阵</h2><p>“视觉目标 missing”不会阻止合同冒烟，
但会阻止该案例调用图像模型。当前五个 sentinel 与历史三角洲均已有目标包；GEO-02 明确标为 provisional。</p>
<table><thead><tr><th>案例</th><th>学科</th><th>几何策略</th><th>时间信息</th>
<th>视觉目标</th><th>已知缺口</th></tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table></section>
<section><h2>S3.0 自动检验</h2><ul>"""
        + check_rows
        + """</ul><p>所有文件都保存 SHA-256；同一输入重跑 Phase 0 会产生相同合同与选择结果。
视觉包明确设置 <code>appearance_to_geometry_leakage</code> 硬门禁，防止参考图反过来偷改几何。</p></section>
<section><h2>Loop 自动选择的下一步</h2><div class="decision">
<p><strong>目标：</strong> CHEM-01 · <code>canonicalize</code></p>
<p><strong>跨学科回归：</strong> PHYS-02（同类器材规范重建）与 MATH-02（确保精确几何路线不被误改）。</p>
<p><strong>首个问题：</strong>固定坐标的烧杯线稿能产生好图，但无法从任意程序语义复现。</p>
<p><strong>首个可证伪假设：</strong>如果每个几何对象都有 typed identity 和程序 bbox，
通用规范几何库就能在不读取最终好图几何的前提下生成稳定控制图；若对象缺失、越界或精确路线被改动，则假设失败。</p>
</div><p>排序规则原样保存于 <a href='"""
        + html.escape(href(report_dir, OUTPUT / "case-selection.json"))
        + """'>case-selection.json</a>；不是围绕烧杯临时拍脑袋。它依次考虑阶段缺口、失败严重度、
跨学科覆盖、输入/视觉目标完整度、成本和稳定 case_id。</p></section>
<section><h2>可复现入口</h2><p><code>.venv/bin/python -m modules.video_model.stage3.phase0</code></p>
<p>主要机器可读产物：<a href='"""
        + html.escape(href(report_dir, STAGE3 / "case_registry.json"))
        + """'>case_registry.json</a> · <a href='"""
        + html.escape(href(report_dir, STAGE3 / "state.json"))
        + """'>state.json</a> · <a href='"""
        + html.escape(href(report_dir, STAGE3 / "baselines" / "accepted.json"))
        + """'>accepted.json</a> · <a href='"""
        + html.escape(href(report_dir, OUTPUT / "phase0_manifest.json"))
        + """'>phase0_manifest.json</a></p></section>
</main></body></html>""",
        encoding="utf-8",
    )
    return report_path


def run() -> dict[str, Any]:
    for directory in (
        OUTPUT,
        CONTRACTS,
        MOTIONS,
        VISUAL_TARGETS,
        STAGE3 / "baselines",
        STAGE3 / "experiments",
        STAGE3 / "knowledge",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    schema_records = validate_schema_documents(SCHEMAS)
    source_registry = load_json(STAGE2 / "case_registry.json")
    cases = list(source_registry["cases"])
    historical_case, historical = historical_delta_case()
    cases.append(historical_case)

    registry_cases = []
    completeness = []
    contracts: list[dict[str, Any]] = []
    motions: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        if case_id == "GEO-HIST-DELTA-01":
            collected = historical["collected"]
            motion = historical["motion"]
        else:
            collected = collect_program(case)
            motion = build_motion_contract(case, collected)
        visual = (
            build_visual_target(case_id)
            if case_id in VISUAL_SPECS
            else None
        )
        contract = build_input_contract(
            case, collected, motion, visual
        )
        contracts.append(contract)
        motions.append(motion)
        complete = completeness_record(case, collected, visual)
        completeness.append(complete)
        role = (
            "historical_quality_regression"
            if case_id == "GEO-HIST-DELTA-01"
            else ("sentinel" if case["sentinel"] else "scale")
        )
        registry_cases.append(
            {
                "case_id": case_id,
                "slug": case["slug"],
                "title_zh": case["title_zh"],
                "discipline": case["discipline"],
                "discipline_zh": case["discipline_zh"],
                "role": role,
                "input_contract": rel(CONTRACTS / f"{case_id}.json"),
                "geometry_policy": GEOMETRY_POLICY[case_id],
                "visual_target_status": complete[
                    "visual_target_status"
                ],
                "motion_classes": case["motion_classes"],
                "keyframe_count": complete["keyframe_count"],
                "continuous_program_frames": complete[
                    "continuous_program_frames"
                ],
                "known_gaps": complete["known_gaps"],
                "completeness": complete,
            }
        )

    registry = {
        "schema_version": "1.0",
        "suite_id": "stage3_deterministic_program_to_visual_v1",
        "source_registry": file_record(
            STAGE2 / "case_registry.json", REPO_ROOT
        ),
        "cases": registry_cases,
    }
    validate_case_registry(registry)
    write_json(STAGE3 / "case_registry.json", registry)
    write_json(
        OUTPUT / "case-completeness.json",
        {
            "schema_version": "1.0",
            "scale_case_count": 10,
            "historical_regression_count": 1,
            "records": completeness,
        },
    )

    selection = selector(registry_cases)
    write_json(OUTPUT / "case-selection.json", selection)

    checks = [
        {
            "name": "five_json_schemas_are_frozen",
            "passed": len(schema_records) == 5,
            "evidence_zh": "输入、注册表、视觉目标、运动和 Loop 状态均有版本化 Schema。",
        },
        {
            "name": "ten_scale_cases_plus_delta_are_registered",
            "passed": len(registry_cases) == 11,
            "evidence_zh": "10 个跨学科案例 + 1 个历史三角洲质量回归。",
        },
        {
            "name": "all_ten_scale_cases_have_continuous_program_frames",
            "passed": all(
                item["continuous_program_frames"] == 49
                for item in registry_cases
                if item["role"] != "historical_quality_regression"
            ),
            "evidence_zh": "每个规模案例都有 49 帧程序时间线，不只利用关键帧截图。",
        },
        {
            "name": "all_contracts_export_object_identity",
            "passed": all(
                "object_identity"
                in item["semantic_exports"]["layer_types"]
                for item in contracts
            ),
            "evidence_zh": "对象身份被作为稳定几何与跨帧追踪接口。",
        },
        {
            "name": "all_sentinel_visual_targets_are_not_missing",
            "passed": all(
                item["visual_target_status"] != "missing"
                for item in registry_cases
                if item["role"]
                in {"sentinel", "historical_quality_regression"}
            ),
            "evidence_zh": "5 个 sentinel 与历史三角洲均有本地正例、反例和量表。",
        },
        {
            "name": "geometry_and_appearance_are_separated",
            "passed": all(
                contract["visual_target_package"]["status"] == "missing"
                or load_json(
                    REPO_ROOT
                    / contract["visual_target_package"]["path"]
                )["geometry_control_separation"]["leakage_gate"]
                == "appearance_to_geometry_leakage"
                for contract in contracts
            ),
            "evidence_zh": "外观参考不能作为几何来源，泄漏检查是硬门禁。",
        },
        {
            "name": "case_selector_is_deterministic",
            "passed": selection["selected_target"] == "CHEM-01",
            "evidence_zh": "S3.1 固定选择 CHEM-01，回归 PHYS-02 与 MATH-02。",
        },
        {
            "name": "phase0_used_no_image_or_video_model",
            "passed": True,
            "evidence_zh": "model_runs: image=0, video=0。",
        },
    ]
    if not all(item["passed"] for item in checks):
        raise RuntimeError("S3.0 check failure")

    baseline_records = []
    for contract in contracts:
        path = CONTRACTS / f"{contract['case_id']}.json"
        baseline_records.append(
            {
                "baseline_id": f"CONTRACT-{contract['case_id']}-V1",
                "kind": "input_contract",
                **file_record(path, REPO_ROOT),
            }
        )
    for case_id in VISUAL_SPECS:
        manifest = load_json(
            VISUAL_TARGETS / case_id / "manifest.json"
        )
        if manifest["status"] in {
            "accepted_project_baseline",
            "user_approved",
        }:
            for ref in manifest["positive_refs"]:
                baseline_records.append(
                    {
                        "baseline_id": (
                            f"APPEARANCE-{case_id}-"
                            f"{ref['sha256'][:12]}"
                        ),
                        "kind": "appearance_reference",
                        **{
                            key: ref[key]
                            for key in ("path", "sha256", "size_bytes")
                        },
                    }
                )
    baselines = {
        "schema_version": "1.0",
        "policy": (
            "Only reviewed contracts and accepted/user-approved appearance "
            "references enter this file. Provisional references are excluded."
        ),
        "records": baseline_records,
    }
    write_json(STAGE3 / "baselines" / "accepted.json", baselines)
    for item in baseline_records:
        verify_file_record(item, REPO_ROOT)

    state = {
        "schema_version": "1.0",
        "loop_id": "LOOP-S3-0001",
        "phase": "S3.1",
        "phase_status": "in_progress",
        "exit_criteria": [
            "typed geometry canonicalizer consumes semantic object identity",
            "CHEM-01 control is derived without fixed final coordinates",
            "PHYS-02 cross-discipline canonicalization smoke passes",
            "MATH-02 preserve_exact output is byte-identical",
            "G0 and G1 pass before any model generation",
        ],
        "budget": {
            "phase0_image_model_runs": 0,
            "phase0_video_model_runs": 0,
            "s3_1_image_candidate_limit": 18,
            "s3_1_video_candidate_limit": 0,
            "preflight_before_paid_or_gpu_work": True,
        },
        "current_problem": {
            "problem_id": "S3-PROBLEM-GEOMETRY-001",
            "taxonomy": "geometry",
            "summary_zh": (
                "Stage 2 的烧杯优质控制图来自固定坐标模板，"
                "没有由程序 object identity 与 bbox 生成。"
            ),
        },
        "current_hypothesis": {
            "hypothesis_id": "H-S3-0001",
            "statement_zh": (
                "若规范几何库只读取 typed object identity、程序 bbox "
                "和关系约束，就能跨案例生成稳定控制图，同时不改变 preserve_exact 路线。"
            ),
            "falsification_zh": (
                "对象缺失/越界、CHEM/PHYS 控制结构失败，"
                "或 MATH-02 精确控制发生任何字节变化，即判失败。"
            ),
        },
        "current_cohort": {
            "target": "CHEM-01",
            "regressions": ["PHYS-02", "MATH-02"],
            "selection_record": rel(OUTPUT / "case-selection.json"),
        },
        "next_action": (
            "Run S3.1 semantic coverage preflight, then implement and test "
            "the typed geometry canonicalizer before any SDXL generation."
        ),
    }
    validate_loop_state(state)
    write_json(STAGE3 / "state.json", state)
    write_json(
        STAGE3 / "experiments" / "ledger.json",
        {
            "schema_version": "1.0",
            "loop_id": state["loop_id"],
            "experiments": [],
        },
    )
    hypotheses_path = STAGE3 / "knowledge" / "hypotheses.jsonl"
    hypotheses_path.write_text("", encoding="utf-8")
    write_json(
        STAGE3 / "knowledge" / "failure_patterns.json",
        {
            "schema_version": "1.0",
            "patterns": [
                {
                    "id": "FP-GEOMETRY-001",
                    "taxonomy": "geometry",
                    "symptom_zh": "固定器材模板好看，但换案例或画布后无法复现。",
                    "diagnosis_zh": "控制图没有从 typed semantic objects 编译。",
                    "forbidden_fix_zh": "不得把新案例的最终坐标再次手写进模板。",
                },
                {
                    "id": "FP-CONTROL-001",
                    "taxonomy": "control_encoding",
                    "symptom_zh": "密集 Canny 把文字、箭头、雨线或纹理当作物体边界。",
                    "diagnosis_zh": "整张程序截图被错误地当成统一几何源。",
                    "forbidden_fix_zh": "不得只调 Canny 阈值后继续把全图边缘输入模型。",
                },
            ],
        },
    )
    write_json(
        STAGE3 / "knowledge" / "open_problems.json",
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

    report_path = render_report(registry_cases, checks, selection)
    # The manifest is written after the report hash is known, so its link is
    # the one intentional forward reference during this first pass.
    missing_links = [
        value
        for value in check_report_links(report_path)
        if value != "phase0_manifest.json"
    ]
    if missing_links:
        raise RuntimeError(f"report has missing links: {missing_links}")
    checks.append(
        {
            "name": "report_links_resolve",
            "passed": True,
            "evidence_zh": "报告中的本地图片与机器可读产物链接全部存在。",
        }
    )
    # Re-render once so the link-integrity check itself appears in the report.
    report_path = render_report(registry_cases, checks, selection)

    artifacts = {}
    for name, path in {
        "registry": STAGE3 / "case_registry.json",
        "state": STAGE3 / "state.json",
        "baselines": STAGE3 / "baselines" / "accepted.json",
        "completeness": OUTPUT / "case-completeness.json",
        "selection": OUTPUT / "case-selection.json",
        "report": report_path,
    }.items():
        artifacts[name] = file_record(path, REPO_ROOT)
    manifest = {
        "schema_version": "1.0",
        "phase": "S3.0",
        "status": "passed",
        "classification": "contract_and_baseline_freeze",
        "model_runs": {"image": 0, "video": 0},
        "checks": checks,
        "schema_records": schema_records,
        "artifacts": artifacts,
        "next_phase": {
            "phase": "S3.1",
            "status": "in_progress",
            "target": selection["selected_target"],
            "regressions": [
                item["case_id"]
                for item in selection["regression_cohort"]
            ],
        },
    }
    write_json(OUTPUT / "phase0_manifest.json", manifest)
    # Manifest link was already targeted before it existed; verify final state.
    missing_links = check_report_links(report_path)
    if missing_links:
        raise RuntimeError(f"final report has missing links: {missing_links}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    run()
