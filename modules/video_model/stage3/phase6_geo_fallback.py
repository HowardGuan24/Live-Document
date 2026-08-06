"""Render all 49 GEO-02 program states through the accepted appearance route."""

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
from modules.video_model.stage3.framework.motion import audit_video, decode_video, encode_video
from modules.video_model.stage3.framework.state_renderer import render_plan


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
ROOT = STAGE3 / "output/phase-6-rerun-2/GEO-02/video/deterministic"
TIMELINE = ROOT / "timeline-input"
RENDER = ROOT / "render"
FPS = 24
FRAME_COUNT = 49
LAYER_IDS = ("geo02_humidity_cloud_rain", "geo02_parcel_identity")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _layer(sample: Any, layer_id: str) -> Any:
    return next(item for item in sample.layers if item.layer_id == layer_id)


def export_timeline() -> tuple[dict[str, Any], dict[str, Any]]:
    program = PROGRAMS["GEO-02"]
    root_relative = TIMELINE.relative_to(REPO_ROOT).as_posix()
    keyframes = []
    states = []
    for index in range(FRAME_COUNT):
        progress = index / (FRAME_COUNT - 1)
        sample = program.sample(progress)
        frame_id = f"frame_{index:03d}"
        frame_root = TIMELINE / frame_id
        frame_root.mkdir(parents=True, exist_ok=True)
        state_path = frame_root / "state.json"
        write_json(state_path, sample.state)
        states.append(sample.state)
        layer_records = []
        for layer_id in LAYER_IDS:
            layer = _layer(sample, layer_id)
            if isinstance(layer.data, np.ndarray):
                filename = f"{layer_id}.npy"
                data_path = frame_root / filename
                np.save(data_path, layer.data, allow_pickle=False)
                encoding = "npy"
            else:
                filename = f"{layer_id}.json"
                data_path = frame_root / filename
                write_json(data_path, layer.data)
                encoding = "json"
            layer_records.append(
                {
                    "layer_id": layer.layer_id,
                    "layer_type": layer.layer_type,
                    "title_zh": layer.title_zh,
                    "meaning_zh": layer.meaning_zh,
                    "source_zh": "由 Stage 2 确定性机制 provider 在该时间点直接重新计算。",
                    "model_input_policy": layer.model_input_policy,
                    "final_role_zh": layer.final_role_zh,
                    "used_as_model_input": False,
                    "data": {
                        "encoding": encoding,
                        "path": f"{frame_id}/{filename}",
                        "sha256": sha256_path(data_path),
                        "size_bytes": data_path.stat().st_size,
                    },
                }
            )
        semantic = {
            "schema_version": "1.0",
            "case_id": "GEO-02",
            "state_id": frame_id,
            "canvas": {"width": 640, "height": 360, "coordinate_system": "pixel_xy_top_left"},
            "layers": layer_records,
        }
        semantic_path = frame_root / "semantic_layers.json"
        write_json(semantic_path, semantic)
        keyframes.append(
            {
                "keyframe_id": frame_id,
                "order": index,
                "progress": round(progress, 8),
                "state": file_record(state_path, REPO_ROOT),
                "semantic_layers": file_record(semantic_path, REPO_ROOT),
            }
        )
    source_contract = load_json(STAGE3 / "contracts/GEO-02.json")
    source_motion = load_json(STAGE3 / "motion_contracts/GEO-02.json")
    contract = {
        **source_contract,
        "program_source": {
            **source_contract["program_source"],
            "root": root_relative,
            "timeline_materialization": {
                "provider": "modules.video_model.stage2.cases.sentinel_programs.PROGRAMS",
                "sample_count": FRAME_COUNT,
                "progress_rule": "frame_index / 48",
                "provider_source": file_record(
                    REPO_ROOT / "modules/video_model/stage2/cases/sentinel_programs.py",
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
        "case_id": "GEO-02",
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "contract": file_record(contract_path, REPO_ROOT),
        "source_program_video": source_motion["program_video"],
        "source_state_timeline": source_motion["state_timeline"],
        "state_summary": {
            "parcel_x_ratio_first_last": [states[0]["parcel_x_ratio"], states[-1]["parcel_x_ratio"]],
            "rain_strength_min_peak_final": [
                min(item["rain_strength"] for item in states),
                max(item["rain_strength"] for item in states),
                states[-1]["rain_strength"],
            ],
        },
        "note_zh": "49 份语义层不是从程序视频抠色；每帧都重新调用确定性 provider，直接导出标量场和对象身份。",
    }
    write_json(ROOT / "timeline-export.json", export)
    return contract, export


def _plan(contract: dict[str, Any]) -> dict[str, Any]:
    source = load_json(STAGE3 / "geo_state_render_plan_v1.json")["plan"]
    return {
        **source,
        "plan_id": "S3.6R2-GEO-02-FULL-TIMELINE-V1",
        "role": "deterministic_video_fallback",
        "contract": (ROOT / "timeline-contract.json").relative_to(STAGE3).as_posix(),
        "keyframe_ids": [item["keyframe_id"] for item in contract["keyframes"]],
    }


def _preview(video: Path, target: Path) -> None:
    info, frames = decode_video(video)
    indices = [round(index * (len(frames) - 1) / 8) for index in range(9)]
    sheet = Image.new("RGB", (960, 618), (13, 29, 32))
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    for slot, frame_index in enumerate(indices):
        image = Image.fromarray(frames[frame_index]).convert("RGB")
        image.thumbnail((320, 176))
        x = slot % 3 * 320
        y = slot // 3 * 206
        sheet.paste(image, (x + (320 - image.width) // 2, y))
        draw.text((x + 10, y + 182), f"frame {frame_index} / {frame_index / info['fps']:.2f}s", fill=(235, 247, 242), font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=92, subsampling=0)


def _accepted_keyframe(keyframe_id: str) -> Path:
    return STAGE3 / f"output/phase-6-rerun-2/GEO-02/candidate/frames/{keyframe_id}.png"


def run() -> dict[str, Any]:
    contract, export = export_timeline()
    plan = _plan(contract)
    write_json(ROOT / "state-render-plan.json", plan)
    manifest = render_plan(plan, STAGE3, REPO_ROOT, RENDER)
    paths = [REPO_ROOT / item["output"]["path"] for item in manifest["records"]]
    frames = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    video = ROOT / "transition.mp4"
    encode_video(frames, video, fps=FPS)
    audit = audit_video(
        video,
        first_path=paths[0],
        last_path=paths[-1],
        config=load_json(STAGE3 / "geo_motion_v1.json")["g4"],
        sparse_checkpoints=[
            (0, _accepted_keyframe("00_start")),
            (16, _accepted_keyframe("01_mechanism")),
            (32, _accepted_keyframe("02_result")),
            (48, _accepted_keyframe("03_end")),
        ],
    )
    preview = ROOT / "generated-frames.jpg"
    _preview(video, preview)
    result = {
        "schema_version": "1.0",
        "classification": "deterministic full-program-timeline State Renderer B fallback",
        "passed": audit["passed"],
        "timeline_export": export,
        "render_manifest": file_record(RENDER / "manifest.json", REPO_ROOT),
        "video": file_record(video, REPO_ROOT),
        "preview": file_record(preview, REPO_ROOT),
        "g4": audit,
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
        "limitation_zh": "运动来自完整程序时间线，外观来自已验收的 SDXL 山体供体；这是写实确定性回退，不冒充 LTX 生成运动。",
    }
    write_json(ROOT / "g4.json", result)
    if not result["passed"]:
        raise RuntimeError("GEO-02 deterministic fallback failed G4")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
