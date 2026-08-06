"""Prepare, run, assemble and audit the frozen S3.5 video experiments."""

from __future__ import annotations

import argparse
import json
import shutil
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
    sha256_path,
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
CONFIG = STAGE3 / "motion_guidance.json"
OUTPUT = STAGE3 / "output" / "phase-5"
EXPERIMENTS = STAGE3 / "experiments"
PHASE4 = STAGE3 / "output" / "phase-4"
KEYFRAME_ORDER = [
    "00_start",
    "01_mechanism",
    "02_result",
    "03_end",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _config() -> dict[str, Any]:
    return load_json(CONFIG)


def _segment(config: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [
        item for item in config["segments"] if item["case_id"] == case_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one segment config for {case_id}")
    return matches[0]


def _frame(case_id: str, keyframe_id: str) -> Path:
    return PHASE4 / case_id / "frames" / f"{keyframe_id}.png"


def _base_spec(
    *,
    experiment_id: str,
    case_id: str,
    motion_class: str,
    first_keyframe: str,
    last_keyframe: str,
    prompt: dict[str, str],
    frame_count: int,
    seed: int,
    spec_path: Path,
) -> dict[str, Any]:
    config = _config()
    settings = dict(config["default_settings"])
    settings["frame_count"] = frame_count
    settings["noise_seed"] = seed
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "case_id": case_id,
        "motion_class": motion_class,
        "source": {
            "first_keyframe_id": first_keyframe,
            "last_keyframe_id": last_keyframe,
            "first_frame": str(_frame(case_id, first_keyframe).resolve()),
            "last_frame": str(_frame(case_id, last_keyframe).resolve()),
        },
        "prompt": prompt,
        "model": config["model"],
        "settings": settings,
        "audit": {
            "thresholds": {
                "maximum_endpoint_mae_0_255": 15,
                "maximum_consecutive_frame_mae_0_255": 12,
            }
        },
        "output_prefix": experiment_id.lower().replace("-", "_"),
        "_spec_path": str(spec_path.resolve()),
    }


def _public_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spec.items() if key != "_spec_path"}


def _motion_contract(
    segment: dict[str, Any],
    keyframes: list[str],
    output_path: Path,
) -> dict[str, Any]:
    case_id = segment["case_id"]
    top_level = load_json(STAGE3 / "motion_contracts" / f"{case_id}.json")
    source_records = []
    for keyframe_id in keyframes:
        path = _frame(case_id, keyframe_id)
        source_records.append(
            {
                "keyframe_id": keyframe_id,
                "frame": file_record(path, REPO_ROOT),
            }
        )
    value = {
        "schema_version": "1.0",
        "contract_id": f"S3.5-{case_id}-{'__'.join(keyframes)}-V1",
        "case_id": case_id,
        "segment_id": segment["segment_id"],
        "motion_class": segment["motion_class"],
        "source_keyframes": source_records,
        "entities": segment["entities"],
        "ordered_events": segment["ordered_events"],
        "state_trends": segment["state_trends"],
        "invariants": segment["invariants"],
        "forbidden": segment["forbidden"],
        "g4": segment["g4"],
        "timeline_source": top_level["state_timeline"],
        "program_video_source": top_level["program_video"],
        "model_capability": {
            "native_guides": [
                "first_frame",
                "last_frame",
            ],
            "native_middle_frame": False,
            "program_video_conditioning": False,
        },
        "downstream_use": (
            "The prompt compiler converts ordered events, trends and "
            "invariants to L1 text. G4 reads the numeric g4 section. "
            "The raw program RGB video is evidence, not a style input."
        ),
    }
    write_json(output_path, value)
    return value


def _write_experiment_spec(
    experiment: dict[str, Any],
    segment: dict[str, Any],
) -> list[tuple[dict[str, Any], Path]]:
    experiment_id = experiment["experiment_id"]
    level = experiment["guidance_level"]
    case_id = experiment["case_id"]
    root = EXPERIMENTS / experiment_id
    root.mkdir(parents=True, exist_ok=True)
    contract_dir = OUTPUT / "contracts" / case_id
    contract_dir.mkdir(parents=True, exist_ok=True)
    keyframes = segment["keyframes"]
    contract_path = contract_dir / f"{segment['segment_id']}.json"
    _motion_contract(segment, keyframes, contract_path)
    prompt_level = "L1" if level == "L2" else level
    prompt = compile_motion_prompt(segment, guidance_level=prompt_level)
    write_json(
        root / "hypothesis.json",
        {
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "case_id": case_id,
            "guidance_level": level,
            "single_variable_zh": (
                "CHEM L0/L1 只改变文字；PHYS L1/L2 只增加已验收中间"
                "关键帧作为分段边界；模型、端点、总帧数和门禁保持不变。"
            ),
            "motion_contract": file_record(contract_path, REPO_ROOT),
        },
    )
    if level != "L2":
        spec_path = root / "spec.json"
        frame_count = 25
        if case_id == "PHYS-01":
            frame_count = 73
        elif case_id == "MATH-02":
            frame_count = 49
        spec = _base_spec(
            experiment_id=experiment_id,
            case_id=case_id,
            motion_class=segment["motion_class"],
            first_keyframe=keyframes[0],
            last_keyframe=keyframes[-1],
            prompt=prompt,
            frame_count=frame_count,
            seed=int(_config()["default_settings"]["noise_seed"]),
            spec_path=spec_path,
        )
        write_json(spec_path, _public_spec(spec))
        return [(spec, OUTPUT / "experiments" / experiment_id)]

    specs = []
    for index, (first_id, last_id) in enumerate(
        zip(keyframes, keyframes[1:])
    ):
        child_id = f"{experiment_id}-SEG{index + 1}"
        spec_path = root / f"spec-segment-{index + 1}.json"
        spec = _base_spec(
            experiment_id=child_id,
            case_id=case_id,
            motion_class=segment["motion_class"],
            first_keyframe=first_id,
            last_keyframe=last_id,
            prompt=prompt,
            frame_count=25,
            seed=int(_config()["default_settings"]["noise_seed"]) + index,
            spec_path=spec_path,
        )
        write_json(spec_path, _public_spec(spec))
        specs.append(
            (
                spec,
                OUTPUT
                / "experiments"
                / experiment_id
                / "segments"
                / f"{index + 1:02d}_{first_id}__{last_id}",
            )
        )
    write_json(
        root / "spec.json",
        {
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "case_id": case_id,
            "guidance_level": "L2",
            "implementation": (
                "three adjacent LTX FLF segments; remove duplicate boundary "
                "frames and concatenate"
            ),
            "segment_specs": [
                file_record(Path(spec["_spec_path"]), REPO_ROOT)
                for spec, _ in specs
            ],
            "assembled_frame_count": 73,
        },
    )
    return specs


def preflight() -> dict[str, Any]:
    config = _config()
    source = (STAGE3 / "framework" / "motion.py").read_text(
        encoding="utf-8"
    )
    checks = [
        {
            "name": "runtime_interface_is_explicit",
            "passed": (
                config["runtime"]["native_image_guide_indices"] == [0, -1]
                and not config["runtime"]["supports_native_middle_frame"]
                and not config["runtime"][
                    "supports_program_video_conditioning"
                ]
            ),
        },
        {
            "name": "motion_core_has_no_case_ids",
            "passed": not any(
                token in source
                for token in ("CHEM-01", "PHYS-01", "MATH-02")
            ),
        },
        {
            "name": "all_model_files_exist",
            "passed": all(path.is_file() for path in MODEL_FILES.values()),
        },
        {
            "name": "five_fixed_experiments",
            "passed": len(config["experiments"]) == 5,
        },
    ]
    prepared = []
    for experiment in config["experiments"]:
        segment = _segment(config, experiment["case_id"])
        if segment["segment_id"] != experiment["segment_id"]:
            raise ValueError("experiment/segment mismatch")
        for keyframe_id in segment["keyframes"]:
            path = _frame(segment["case_id"], keyframe_id)
            if not path.is_file():
                raise FileNotFoundError(path)
        specs = _write_experiment_spec(experiment, segment)
        for spec, root in specs:
            result = prepare_video_experiment(spec, root)
            prepared.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "root": root.relative_to(REPO_ROOT).as_posix(),
                    "workflow": result["workflow"],
                    "prompt": result["prompt"],
                }
            )
    checks.append(
        {
            "name": "all_specs_prepared_without_model_call",
            "passed": len(prepared) == 7,
        }
    )
    result = {
        "schema_version": "1.0",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "prepared_runs": prepared,
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
    }
    write_json(OUTPUT / "preflight.json", result)
    if not result["passed"]:
        raise RuntimeError("S3.5 preflight failed")
    return result


def _preview(video_path: Path, output: Path) -> Path:
    info, frames = decode_video(video_path)
    count = min(9, len(frames))
    indices = [
        round(index * (len(frames) - 1) / max(count - 1, 1))
        for index in range(count)
    ]
    cell = (320, 206)
    sheet = Image.new("RGB", (cell[0] * 3, cell[1] * 3), (13, 29, 32))
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    for slot, frame_index in enumerate(indices):
        image = Image.fromarray(frames[frame_index]).convert("RGB")
        image.thumbnail((cell[0], 176))
        x = slot % 3 * cell[0]
        y = slot // 3 * cell[1]
        sheet.paste(
            image,
            (x + (cell[0] - image.width) // 2, y),
        )
        draw.text(
            (x + 10, y + 182),
            f"frame {frame_index} · {frame_index / info['fps']:.2f}s",
            fill=(235, 247, 242),
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, subsampling=0)
    return output


def generate(experiment_filter: str | None, server: str) -> None:
    preflight()
    config = _config()
    for experiment in config["experiments"]:
        experiment_id = experiment["experiment_id"]
        if experiment_filter and experiment_id != experiment_filter:
            continue
        segment = _segment(config, experiment["case_id"])
        specs = _write_experiment_spec(experiment, segment)
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
                f"{passed}/{len(result['hard_checks'])} smoke checks"
            )
        if experiment["guidance_level"] == "L2":
            root = OUTPUT / "experiments" / experiment_id
            sources = [path / "transition.mp4" for _, path in specs]
            assembly = concatenate_segment_videos(
                sources, root / "transition.mp4"
            )
            write_json(root / "_work" / "assembly.json", assembly)
            _preview(root / "transition.mp4", root / "generated-frames.jpg")


def audit() -> dict[str, Any]:
    config = _config()
    records = []
    for experiment in config["experiments"]:
        experiment_id = experiment["experiment_id"]
        case_id = experiment["case_id"]
        level = experiment["guidance_level"]
        segment = _segment(config, case_id)
        root = OUTPUT / "experiments" / experiment_id
        video_path = root / "transition.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        first_path = _frame(case_id, segment["keyframes"][0])
        last_path = _frame(case_id, segment["keyframes"][-1])
        sparse_checkpoints = None
        if len(segment["keyframes"]) > 2:
            last_index = 72
            step = last_index // (len(segment["keyframes"]) - 1)
            sparse_checkpoints = [
                (index * step, _frame(case_id, keyframe_id))
                for index, keyframe_id in enumerate(
                    segment["keyframes"]
                )
            ]
        result = audit_video(
            video_path,
            first_path=first_path,
            last_path=last_path,
            config=segment["g4"],
            sparse_checkpoints=sparse_checkpoints,
        )
        result.update(
            {
                "schema_version": "1.0",
                "experiment_id": experiment_id,
                "case_id": case_id,
                "guidance_level": level,
                "motion_contract": file_record(
                    OUTPUT
                    / "contracts"
                    / case_id
                    / f"{segment['segment_id']}.json",
                    REPO_ROOT,
                ),
                "video_artifact": file_record(video_path, REPO_ROOT),
            }
        )
        preview = root / "generated-frames.jpg"
        if not preview.is_file():
            _preview(video_path, preview)
        result["preview"] = file_record(preview, REPO_ROOT)
        write_json(root / "g4.json", result)
        records.append(result)
    summary = {
        "schema_version": "1.0",
        "experiments": [
            {
                "experiment_id": item["experiment_id"],
                "case_id": item["case_id"],
                "guidance_level": item["guidance_level"],
                "passed": item["passed"],
                "checks": [
                    {
                        "name": check["name"],
                        "passed": check["passed"],
                    }
                    for check in item["checks"]
                ],
            }
            for item in records
        ],
        "model_runs": {
            "image_candidates": 0,
            "video_candidates": 5,
            "video_model_calls": 7,
            "note": (
                "L2 is one assembled candidate made from three adjacent FLF "
                "model calls."
            ),
        },
    }
    write_json(OUTPUT / "g4-summary.json", summary)
    return summary


def install_program_fallbacks() -> dict[str, Any]:
    """Preserve already audited deterministic program fallbacks."""

    sources = {
        "CHEM-01": (
            REPO_ROOT
            / "modules/video_model/stage2/output/phase-5/experiments/"
            "EXP-P5-20260729-007/transition.mp4"
        ),
        "MATH-02": (
            REPO_ROOT
            / "modules/video_model/stage2/output/phase-5/experiments/"
            "EXP-P5-20260729-004/transition.mp4"
        ),
        "PHYS-01": (
            REPO_ROOT
            / "modules/video_model/stage2/output/phase-2/PHYS-01/"
            "program-animation.mp4"
        ),
    }
    records = {}
    for case_id, source in sources.items():
        target = OUTPUT / "fallbacks" / case_id / "program-motion.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            shutil.copy2(source, target)
        records[case_id] = {
            "classification": "deterministic program motion fallback",
            "source": file_record(source, REPO_ROOT),
            "preserved_copy": file_record(target, REPO_ROOT),
            "appearance_limitation_zh": (
                "保留机制和对象身份，但不是 S3.4 写实外观；它是模型运动"
                "失败时的安全回退，不冒充最终写实成片。"
            ),
        }
    result = {"schema_version": "1.0", "fallbacks": records}
    write_json(OUTPUT / "fallbacks" / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--fallbacks", action="store_true")
    parser.add_argument("--experiment")
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    if not any(
        (args.prepare, args.generate, args.audit, args.fallbacks)
    ):
        parser.error("choose an action")
    if args.prepare:
        result = preflight()
        print(f"S3.5 preflight: {result['passed']}")
    if args.generate:
        generate(args.experiment, args.server)
    if args.audit:
        result = audit()
        passed = sum(item["passed"] for item in result["experiments"])
        print(f"S3.5 G4: {passed}/{len(result['experiments'])}")
    if args.fallbacks:
        result = install_program_fallbacks()
        print(f"S3.5 fallbacks: {len(result['fallbacks'])}")


if __name__ == "__main__":
    main()
