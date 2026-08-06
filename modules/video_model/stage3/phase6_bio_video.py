"""Prepare, generate and audit BIO-01 object-division video candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from modules.video_model.stage2.framework.ltx_flf import (
    MODEL_FILES,
    prepare_video_experiment,
    run_video_experiment,
)
from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    write_json,
)
from modules.video_model.stage3.framework.motion import (
    audit_video,
    compile_motion_prompt,
    concatenate_segment_videos,
    decode_video,
)


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-6-rerun-1/BIO-01/video"
EXPERIMENTS = STAGE3 / "experiments"
CONFIG = STAGE3 / "bio_motion_v2.json"
IMAGE_ROOT = (
    STAGE3 / "output/phase-6-rerun-1/BIO-01/candidate/frames"
)
SETTINGS = {
    "width": 576,
    "height": 320,
    "fps": 24,
    "frame_count": 73,
    "noise_seed": 2026073120,
    "guide_strength": 0.7,
    "image_compression": 25,
    "cfg": 1.0,
    "sigmas": [
        1.0,
        0.99375,
        0.9875,
        0.98125,
        0.975,
        0.909375,
        0.725,
        0.421875,
        0.0,
    ],
}
MODEL = {
    "checkpoint": "ltx-2.3-22b-dev-fp8.safetensors",
    "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
    "distilled_lora": (
        "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
    ),
    "lora_strength": 0.5,
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _frame(keyframe_id: str) -> Path:
    return IMAGE_ROOT / f"{keyframe_id}.png"


def _public_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spec.items() if key != "_spec_path"}


def _spec(
    *,
    experiment_id: str,
    first_id: str,
    last_id: str,
    frame_count: int,
    seed: int,
    path: Path,
) -> dict[str, Any]:
    prompt = compile_motion_prompt(
        load_json(CONFIG), guidance_level="L1"
    )
    settings = dict(SETTINGS)
    settings["frame_count"] = frame_count
    settings["noise_seed"] = seed
    value = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "case_id": "BIO-01",
        "motion_class": "object_division",
        "source": {
            "first_keyframe_id": first_id,
            "last_keyframe_id": last_id,
            "first_frame": str(_frame(first_id).resolve()),
            "last_frame": str(_frame(last_id).resolve()),
        },
        "prompt": prompt,
        "model": MODEL,
        "settings": settings,
        "audit": {
            "thresholds": {
                "maximum_endpoint_mae_0_255": 15,
                "maximum_consecutive_frame_mae_0_255": 12,
            }
        },
        "output_prefix": experiment_id.lower().replace("-", "_"),
        "_spec_path": str(path.resolve()),
    }
    write_json(path, _public_spec(value))
    return value


def _motion_contract() -> dict[str, Any]:
    config = load_json(CONFIG)
    source = load_json(STAGE3 / "motion_contracts/BIO-01.json")
    value = {
        "schema_version": "1.0",
        "contract_id": "S3.6R1-BIO-01-OBJECT-DIVISION-V1",
        "case_id": "BIO-01",
        "motion_class": config["motion_class"],
        "source_keyframes": [
            {
                "keyframe_id": keyframe_id,
                "frame": file_record(_frame(keyframe_id), REPO_ROOT),
            }
            for keyframe_id in config["keyframes"]
        ],
        "entities": config["entities"],
        "ordered_events": config["ordered_events"],
        "state_trends": config["state_trends"],
        "invariants": config["invariants"],
        "forbidden": config["forbidden"],
        "g4": config["g4"],
        "timeline_source": source["state_timeline"],
        "program_video_source": source["program_video"],
        "runtime_capability": {
            "first_last_frame_guidance": True,
            "native_middle_frame_guidance": False,
            "program_video_conditioning": False,
            "L2_implementation": (
                "three adjacent FLF calls with duplicate boundary frames removed"
            ),
        },
    }
    write_json(OUTPUT / "motion-contract.json", value)
    return value


def _specs(level: str) -> list[tuple[dict[str, Any], Path]]:
    config = load_json(CONFIG)
    ids = config["keyframes"]
    if level == "L1":
        experiment_id = "EXP-S3-20260731-021"
        root = EXPERIMENTS / experiment_id
        root.mkdir(parents=True, exist_ok=True)
        spec = _spec(
            experiment_id=experiment_id,
            first_id=ids[0],
            last_id=ids[-1],
            frame_count=73,
            seed=SETTINGS["noise_seed"],
            path=root / "spec.json",
        )
        return [(spec, OUTPUT / "L1")]
    if level != "L2":
        raise ValueError(level)
    parent_id = "EXP-S3-20260731-022"
    root = EXPERIMENTS / parent_id
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for index, (first_id, last_id) in enumerate(
        zip(ids, ids[1:], strict=False)
    ):
        child_id = f"{parent_id}-SEG{index + 1}"
        spec = _spec(
            experiment_id=child_id,
            first_id=first_id,
            last_id=last_id,
            frame_count=25,
            seed=SETTINGS["noise_seed"] + index,
            path=root / f"spec-segment-{index + 1}.json",
        )
        result.append(
            (
                spec,
                OUTPUT
                / "L2/segments"
                / f"{index + 1:02d}_{first_id}__{last_id}",
            )
        )
    write_json(
        root / "spec.json",
        {
            "schema_version": "1.0",
            "experiment_id": parent_id,
            "case_id": "BIO-01",
            "guidance_level": "L2",
            "single_variable_from_L1": (
                "add accepted middle keyframes as segment boundaries; "
                "prompt, model settings and total 73 frames stay fixed"
            ),
            "segment_specs": [
                file_record(Path(spec["_spec_path"]), REPO_ROOT)
                for spec, _ in result
            ],
        },
    )
    return result


def preflight(level: str) -> dict[str, Any]:
    image_gate = load_json(
        STAGE3 / "output/phase-6-rerun-1/BIO-01/g3-machine.json"
    )
    summary = load_json(
        STAGE3 / "output/phase-6-rerun-1/render-summary.json"
    )
    contract = _motion_contract()
    specs = _specs(level)
    prepared = []
    for spec, root in specs:
        prepared.append(prepare_video_experiment(spec, root))
    checks = [
        {
            "name": "image_gate_passed_before_video",
            "passed": image_gate["passed"],
        },
        {
            "name": "cross_case_image_regression_passed",
            "passed": summary["cross_case_regression"],
        },
        {
            "name": "all_model_files_exist",
            "passed": all(path.is_file() for path in MODEL_FILES.values()),
        },
        {
            "name": "program_video_is_evidence_not_pixel_conditioning",
            "passed": not contract["runtime_capability"][
                "program_video_conditioning"
            ],
        },
        {
            "name": "all_specs_prepared",
            "passed": len(prepared) == (1 if level == "L1" else 3),
        },
    ]
    result = {
        "schema_version": "1.0",
        "level": level,
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "prepared_runs": [
            {
                "experiment_id": item["experiment_id"],
                "classification": item["classification"],
                "prompt_preflight": item["prompt_preflight"],
            }
            for item in prepared
        ],
    }
    write_json(OUTPUT / level / "preflight.json", result)
    if not result["passed"]:
        raise RuntimeError(f"{level} preflight failed")
    return result


def generate(level: str, server: str) -> None:
    preflight(level)
    specs = _specs(level)
    for spec, root in specs:
        if (root / "transition.mp4").is_file():
            print(f"{spec['experiment_id']}: existing video preserved")
            continue
        result = run_video_experiment(
            spec,
            root,
            server=server,
            timeout_seconds=7200,
        )
        passed = sum(item["passed"] for item in result["hard_checks"])
        print(
            f"{spec['experiment_id']}: generated "
            f"{passed}/{len(result['hard_checks'])} runtime checks"
        )
    if level == "L2":
        sources = [root / "transition.mp4" for _, root in specs]
        assembly = concatenate_segment_videos(
            sources, OUTPUT / "L2/transition.mp4"
        )
        write_json(OUTPUT / "L2/_work/assembly.json", assembly)


def _preview(video: Path, output: Path) -> None:
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


def audit(level: str) -> dict[str, Any]:
    config = load_json(CONFIG)
    video = OUTPUT / level / "transition.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    checkpoints = None
    if level == "L2":
        checkpoints = [
            (index * 24, _frame(keyframe_id))
            for index, keyframe_id in enumerate(config["keyframes"])
        ]
    result = audit_video(
        video,
        first_path=_frame("00_start"),
        last_path=_frame("03_end"),
        config=config["g4"],
        sparse_checkpoints=checkpoints,
    )
    result.update(
        {
            "schema_version": "1.0",
            "case_id": "BIO-01",
            "guidance_level": level,
            "motion_contract": file_record(
                OUTPUT / "motion-contract.json", REPO_ROOT
            ),
            "video_artifact": file_record(video, REPO_ROOT),
        }
    )
    preview = OUTPUT / level / "generated-frames.jpg"
    _preview(video, preview)
    result["preview"] = file_record(preview, REPO_ROOT)
    write_json(OUTPUT / level / "g4.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "generate", "audit")
    )
    parser.add_argument("--level", choices=("L1", "L2"), required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    if args.action == "prepare":
        value = preflight(args.level)
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif args.action == "generate":
        generate(args.level, args.server)
    else:
        value = audit(args.level)
        print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
