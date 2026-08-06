"""Render BIO-01's full program timeline through State Renderer B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage2.cases.sentinel_programs import PROGRAMS
from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    write_json,
)
from modules.video_model.stage3.framework.motion import (
    audit_video,
    encode_video,
)
from modules.video_model.stage3.framework.state_renderer import render_plan


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
ROOT = (
    STAGE3
    / "output/phase-6-rerun-1/BIO-01/video/deterministic"
)
TIMELINE = ROOT / "timeline-input"
RENDER = ROOT / "render"
FPS = 24
FRAME_COUNT = 49


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _layer(sample: Any, layer_id: str) -> Any:
    return next(
        item for item in sample.layers if item.layer_id == layer_id
    )


def export_timeline() -> tuple[dict[str, Any], dict[str, Any]]:
    program = PROGRAMS["BIO-01"]
    root_relative = TIMELINE.relative_to(REPO_ROOT).as_posix()
    keyframes = []
    stage_counts: dict[str, int] = {}
    for index in range(FRAME_COUNT):
        progress = index / (FRAME_COUNT - 1)
        sample = program.sample(progress)
        frame_id = f"frame_{index:03d}"
        frame_root = TIMELINE / frame_id
        frame_root.mkdir(parents=True, exist_ok=True)
        state_path = frame_root / "state.json"
        write_json(state_path, sample.state)
        region = _layer(sample, "bio01_cell_region")
        identity = _layer(sample, "bio01_chromosome_identity")
        region_path = frame_root / "bio01_cell_region.npy"
        np.save(region_path, region.data, allow_pickle=False)
        identity_path = frame_root / "bio01_chromosome_identity.json"
        write_json(identity_path, identity.data)
        semantic = {
            "schema_version": "1.0",
            "case_id": "BIO-01",
            "state_id": frame_id,
            "canvas": {
                "width": 640,
                "height": 360,
                "coordinate_system": "pixel_xy_top_left",
            },
            "layers": [
                {
                    "layer_id": region.layer_id,
                    "layer_type": region.layer_type,
                    "title_zh": region.title_zh,
                    "meaning_zh": region.meaning_zh,
                    "source_zh": (
                        "由 Stage 2 确定性程序在该时间点直接重新计算。"
                    ),
                    "model_input_policy": region.model_input_policy,
                    "final_role_zh": region.final_role_zh,
                    "used_as_model_input": False,
                    "data": {
                        "encoding": "npy",
                        "path": (
                            f"{frame_id}/bio01_cell_region.npy"
                        ),
                        "sha256": sha256_path(region_path),
                        "size_bytes": region_path.stat().st_size,
                    },
                },
                {
                    "layer_id": identity.layer_id,
                    "layer_type": identity.layer_type,
                    "title_zh": identity.title_zh,
                    "meaning_zh": identity.meaning_zh,
                    "source_zh": (
                        "由 Stage 2 确定性程序在该时间点直接重新计算。"
                    ),
                    "model_input_policy": identity.model_input_policy,
                    "final_role_zh": identity.final_role_zh,
                    "used_as_model_input": False,
                    "data": {
                        "encoding": "json",
                        "path": (
                            f"{frame_id}/bio01_chromosome_identity.json"
                        ),
                        "sha256": sha256_path(identity_path),
                        "size_bytes": identity_path.stat().st_size,
                    },
                },
            ],
        }
        semantic_path = frame_root / "semantic_layers.json"
        write_json(semantic_path, semantic)
        keyframes.append(
            {
                "keyframe_id": frame_id,
                "order": index,
                "progress": round(progress, 8),
                "state": file_record(state_path, REPO_ROOT),
                "semantic_layers": file_record(
                    semantic_path, REPO_ROOT
                ),
            }
        )
        stage = sample.state["stage"]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    source_contract = load_json(STAGE3 / "contracts/BIO-01.json")
    motion_contract = load_json(STAGE3 / "motion_contracts/BIO-01.json")
    contract = {
        **source_contract,
        "program_source": {
            **source_contract["program_source"],
            "root": root_relative,
            "timeline_materialization": {
                "provider": (
                    "modules.video_model.stage2.cases."
                    "sentinel_programs.PROGRAMS"
                ),
                "sample_count": FRAME_COUNT,
                "progress_rule": "frame_index / 48",
                "provider_source": file_record(
                    REPO_ROOT
                    / "modules/video_model/stage2/cases/"
                    "sentinel_programs.py",
                    REPO_ROOT,
                ),
            },
        },
        "keyframes": keyframes,
    }
    contract_path = ROOT / "timeline-contract.json"
    write_json(contract_path, contract)
    export = {
        "schema_version": "1.0",
        "case_id": "BIO-01",
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "stage_counts": stage_counts,
        "contract": file_record(contract_path, REPO_ROOT),
        "source_program_video": motion_contract["program_video"],
        "source_state_timeline": motion_contract["state_timeline"],
        "note_zh": (
            "这 49 份语义层不是从程序截图抠色得到；每一帧都重新调用同一个"
            "确定性机制 provider，直接导出 region 和 object_identity。"
        ),
    }
    write_json(ROOT / "timeline-export.json", export)
    return contract, export


def _plan(contract: dict[str, Any]) -> dict[str, Any]:
    candidate = next(
        item
        for item in load_json(STAGE3 / "state_render_plans_v2.json")[
            "plans"
        ]
        if item["role"] == "candidate"
    )
    return {
        **candidate,
        "plan_id": "S3.6R1-BIO-01-FULL-TIMELINE-V1",
        "role": "deterministic_video_fallback",
        "contract": (
            ROOT / "timeline-contract.json"
        ).relative_to(STAGE3).as_posix(),
        "keyframe_ids": [
            item["keyframe_id"] for item in contract["keyframes"]
        ],
    }


def _preview(video: Path, output: Path) -> None:
    from modules.video_model.stage3.framework.motion import decode_video

    info, frames = decode_video(video)
    indices = [
        round(index * (len(frames) - 1) / 8) for index in range(9)
    ]
    cell = (320, 206)
    sheet = Image.new("RGB", (960, 618), (13, 29, 32))
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    for slot, frame_index in enumerate(indices):
        image = Image.fromarray(frames[frame_index]).convert("RGB")
        image.thumbnail((320, 176))
        x = slot % 3 * cell[0]
        y = slot // 3 * cell[1]
        sheet.paste(image, (x + (320 - image.width) // 2, y))
        draw.text(
            (x + 10, y + 182),
            f"frame {frame_index} / {frame_index / info['fps']:.2f}s",
            fill=(235, 247, 242),
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, subsampling=0)


def run() -> dict[str, Any]:
    contract, export = export_timeline()
    plan = _plan(contract)
    write_json(ROOT / "state-render-plan.json", plan)
    manifest = render_plan(
        plan, STAGE3, REPO_ROOT, RENDER
    )
    paths = [
        REPO_ROOT / record["output"]["path"]
        for record in manifest["records"]
    ]
    frames = [
        np.asarray(Image.open(path).convert("RGB")) for path in paths
    ]
    video = ROOT / "transition.mp4"
    encode_video(frames, video, fps=FPS)
    g4_config = load_json(STAGE3 / "bio_motion_v2.json")["g4"]
    audit = audit_video(
        video,
        first_path=paths[0],
        last_path=paths[-1],
        config=g4_config,
        sparse_checkpoints=[
            (0, _frame_path("00_start")),
            (16, _frame_path("01_mechanism")),
            (32, _frame_path("02_result")),
            (48, _frame_path("03_end")),
        ],
    )
    preview = ROOT / "generated-frames.jpg"
    _preview(video, preview)
    result = {
        "schema_version": "1.0",
        "classification": (
            "deterministic full-program-timeline State Renderer B fallback"
        ),
        "passed": audit["passed"],
        "timeline_export": export,
        "render_manifest": file_record(
            RENDER / "manifest.json", REPO_ROOT
        ),
        "video": file_record(video, REPO_ROOT),
        "preview": file_record(preview, REPO_ROOT),
        "g4": audit,
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
        "limitation_zh": (
            "运动机制和写实材质稳定，但运动是程序确定性插值，不是视频模型生成；"
            "因此发布路由记为 fallback，不冒充 LTX 成功。"
        ),
    }
    write_json(ROOT / "g4.json", result)
    return result


def _frame_path(keyframe_id: str) -> Path:
    return (
        STAGE3
        / "output/phase-6-rerun-1/BIO-01/candidate/frames"
        / f"{keyframe_id}.png"
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
