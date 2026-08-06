"""Run the two non-cyclic scale cases through the deployed LTX FLF runtime."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from modules.video_model.stage2.framework.ltx_flf import (
    MODEL_FILES,
    prepare_video_experiment,
    run_video_experiment,
)
from modules.video_model.stage3.framework.contracts import file_record, load_json, write_json
from modules.video_model.stage3.framework.motion import (
    concatenate_segment_videos,
    consecutive_metrics,
    decode_video,
    endpoint_metrics,
)


STAGE3 = Path(__file__).resolve().parent
REPO_ROOT = STAGE3.parents[2]
OUTPUT = STAGE3 / "output/phase-9-scale-motion"
EXPERIMENTS = STAGE3 / "experiments"
MODEL = {
    "checkpoint": "ltx-2.3-22b-dev-fp8.safetensors",
    "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
    "distilled_lora": "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
    "lora_strength": 0.5,
}
SETTINGS = {
    "width": 576,
    "height": 320,
    "fps": 24,
    "frame_count": 49,
    "guide_strength": 0.7,
    "image_compression": 25,
    "cfg": 1.0,
    "sigmas": [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0],
}
CASES = {
    "CHEM-02": {
        "experiment_id": "EXP-S3-20260731-031",
        "motion_class": "material_phase_growth",
        "seed": 2026073131,
        "positive": (
            "Locked top-down scientific view of the exact same shallow transparent glass evaporation dish on a neutral laboratory surface. "
            "The blue salt solution slowly evaporates, its liquid surface steadily lowers, then salt nucleates and grows into exactly four small pale faceted crystals. "
            "The dish never moves or deforms. The liquid never rises. Crystals appear only inside the remaining solution and persist. "
            "End exactly at the supplied final frame. One continuous physically understandable transition, fixed camera, stable light."
        ),
        "negative": (
            "extra dish, broken glass, camera movement, zoom, cut, boiling, splashing, smoke, bubbles, liquid rising, "
            "crystals disappearing, more than four crystals, plastic gemstones, text, labels, hands, flicker, ghosting"
        ),
    },
    "GEO-01": {
        "experiment_id": "EXP-S3-20260731-032",
        "motion_class": "boundary_topology_change",
        "seed": 2026073132,
        "positive": (
            "Locked overhead scientific terrain view of the exact same blue meandering river in green land. "
            "The narrow meander neck erodes gradually, the main river cuts through the neck, and the abandoned loop becomes one separate oxbow lake while the main channel stays connected. "
            "No flood, no camera motion and no new tributary. End exactly at the supplied final frame. "
            "One continuous geographically understandable transition, stable materials and lighting."
        ),
        "negative": (
            "camera movement, zoom, perspective change, scene cut, ocean, waterfall, flood, extra river, extra lake, "
            "main channel breaking, oxbow appearing before cutoff, water turning solid, text, labels, flicker, ghosting"
        ),
    },
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _frame(case_id: str, keyframe_id: str) -> Path:
    return STAGE3 / f"output/phase-8-scale-image/{case_id}/candidate/frames/{keyframe_id}.png"


def _spec(case_id: str) -> dict[str, Any]:
    config = CASES[case_id]
    exp_id = config["experiment_id"]
    exp_root = EXPERIMENTS / exp_id
    exp_root.mkdir(parents=True, exist_ok=True)
    settings = dict(SETTINGS)
    settings["noise_seed"] = config["seed"]
    path = exp_root / "spec.json"
    value = {
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "case_id": case_id,
        "motion_class": config["motion_class"],
        "source": {
            "first_keyframe_id": "00_start",
            "last_keyframe_id": "03_end",
            "first_frame": str(_frame(case_id, "00_start").resolve()),
            "last_frame": str(_frame(case_id, "03_end").resolve()),
        },
        "prompt": {"positive": config["positive"], "negative": config["negative"]},
        "model": MODEL,
        "settings": settings,
        "audit": {"thresholds": {"maximum_endpoint_mae_0_255": 15, "maximum_consecutive_frame_mae_0_255": 12}},
        "output_prefix": exp_id.lower().replace("-", "_"),
        "_spec_path": str(path.resolve()),
    }
    write_json(path, {key: item for key, item in value.items() if key != "_spec_path"})
    return value


def _contract(case_id: str) -> dict[str, Any]:
    config = CASES[case_id]
    source = load_json(STAGE3 / f"motion_contracts/{case_id}.json")
    value = {
        "schema_version": "1.0",
        "contract_id": f"S3.9-{case_id}-L1-V1",
        "case_id": case_id,
        "motion_class": config["motion_class"],
        "source_keyframes": [{"keyframe_id": keyframe_id, "frame": file_record(_frame(case_id, keyframe_id), REPO_ROOT)} for keyframe_id in ("00_start", "01_mechanism", "02_result", "03_end")],
        "full_timeline_evidence": file_record(OUTPUT / f"{case_id}/g4.json", REPO_ROOT),
        "state_timeline_source": source["state_timeline"],
        "program_video_source": source["program_video"],
        "runtime_capability": {
            "first_last_frame_guidance": True,
            "native_middle_frame_guidance": False,
            "program_video_conditioning": False,
        },
        "prompt": {"positive": config["positive"], "negative": config["negative"]},
        "g4": {
            "endpoint_mae_max": 15,
            "consecutive_frame_mae_max": 12,
            "case_mechanism_gate": "typed color-region and connected-component trajectory",
        },
    }
    path = OUTPUT / f"{case_id}/L1/motion-contract.json"
    write_json(path, value)
    return value


def prepare(case_id: str) -> dict[str, Any]:
    deterministic = load_json(OUTPUT / f"{case_id}/g4.json")
    image_gate = load_json(STAGE3 / "output/phase-8-scale-image/g3-machine.json")
    contract = _contract(case_id)
    spec = _spec(case_id)
    prepared = prepare_video_experiment(spec, OUTPUT / f"{case_id}/L1")
    checks = [
        {"name": "deterministic_G4_passed_before_model_trial", "passed": deterministic["passed"]},
        {"name": "image_G3_passed_before_model_trial", "passed": image_gate["case_gates"][case_id]["passed"]},
        {"name": "all_model_files_exist", "passed": all(path.is_file() for path in MODEL_FILES.values())},
        {"name": "runtime_limit_is_recorded", "passed": not contract["runtime_capability"]["native_middle_frame_guidance"] and not contract["runtime_capability"]["program_video_conditioning"]},
    ]
    result = {"schema_version": "1.0", "case_id": case_id, "passed": all(item["passed"] for item in checks), "checks": checks, "prepared": prepared}
    write_json(OUTPUT / f"{case_id}/L1/preflight.json", result)
    if not result["passed"]:
        raise RuntimeError(f"{case_id} L1 preflight failed")
    return result


def generate(case_id: str, server: str) -> None:
    prepare(case_id)
    root = OUTPUT / f"{case_id}/L1"
    if (root / "transition.mp4").is_file():
        return
    run_video_experiment(_spec(case_id), root, server=server, timeout_seconds=7200)


def _l2_specs() -> list[tuple[dict[str, Any], Path]]:
    case_id = "CHEM-02"
    parent_id = "EXP-S3-20260731-033"
    root = EXPERIMENTS / parent_id
    root.mkdir(parents=True, exist_ok=True)
    keyframes = ("00_start", "01_mechanism", "02_result", "03_end")
    events = (
        "The blue solution evaporates smoothly and its surface lowers. Absolutely no crystal appears in this segment.",
        "The blue solution continues to lower. Only near the end, exactly one small pale salt crystal nucleates inside the solution and remains.",
        "The existing crystal persists while three additional crystals nucleate and all grow gently, ending with exactly four separate pale faceted crystals.",
    )
    values = []
    for index, (first_id, last_id) in enumerate(zip(keyframes, keyframes[1:])):
        child_id = f"{parent_id}-SEG{index + 1}"
        child_root = OUTPUT / f"{case_id}/L2/segments/{index + 1:02d}_{first_id}__{last_id}"
        settings = dict(SETTINGS)
        settings.update({"frame_count": 17, "noise_seed": 2026073133 + index})
        path = root / f"spec-segment-{index + 1}.json"
        positive = (
            "Locked top-down scientific view of the exact same shallow transparent glass evaporation dish on a neutral laboratory surface. "
            + events[index]
            + " The dish, background, camera and light remain fixed. End exactly at the supplied last frame."
        )
        spec = {
            "schema_version": "1.0",
            "experiment_id": child_id,
            "case_id": case_id,
            "motion_class": "material_phase_growth",
            "source": {
                "first_keyframe_id": first_id,
                "last_keyframe_id": last_id,
                "first_frame": str(_frame(case_id, first_id).resolve()),
                "last_frame": str(_frame(case_id, last_id).resolve()),
            },
            "prompt": {"positive": positive, "negative": CASES[case_id]["negative"]},
            "model": MODEL,
            "settings": settings,
            "audit": {"thresholds": {"maximum_endpoint_mae_0_255": 15, "maximum_consecutive_frame_mae_0_255": 12}},
            "output_prefix": child_id.lower().replace("-", "_"),
            "_spec_path": str(path.resolve()),
        }
        write_json(path, {key: item for key, item in spec.items() if key != "_spec_path"})
        values.append((spec, child_root))
    write_json(root / "spec.json", {
        "schema_version": "1.0",
        "experiment_id": parent_id,
        "case_id": case_id,
        "guidance_level": "L2",
        "single_variable_from_L1": "add the two accepted middle keyframes as three FLF segment boundaries; model and all other settings stay fixed",
        "segment_specs": [file_record(Path(spec["_spec_path"]), REPO_ROOT) for spec, _ in values],
        "assembled_frame_count": 49,
    })
    return values


def prepare_l2() -> dict[str, Any]:
    _contract("CHEM-02")
    specs = _l2_specs()
    prepared = [prepare_video_experiment(spec, root) for spec, root in specs]
    result = {
        "schema_version": "1.0",
        "case_id": "CHEM-02",
        "guidance_level": "L2",
        "passed": len(prepared) == 3 and all(path.is_file() for path in MODEL_FILES.values()),
        "prepared_runs": prepared,
    }
    write_json(OUTPUT / "CHEM-02/L2/preflight.json", result)
    return result


def generate_l2(server: str) -> None:
    if not prepare_l2()["passed"]:
        raise RuntimeError("CHEM-02 L2 preflight failed")
    specs = _l2_specs()
    for spec, root in specs:
        if not (root / "transition.mp4").is_file():
            run_video_experiment(spec, root, server=server, timeout_seconds=7200)
    assembly = concatenate_segment_videos(
        [root / "transition.mp4" for _, root in specs],
        OUTPUT / "CHEM-02/L2/transition.mp4",
    )
    write_json(OUTPUT / "CHEM-02/L2/assembly.json", assembly)


def _components(mask: np.ndarray, minimum: int, maximum: int | None = None) -> tuple[int, list[int]]:
    seen = np.zeros(mask.shape, dtype=bool)
    sizes = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        size = 0
        while queue:
            cy, cx = queue.popleft()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        if size >= minimum and (maximum is None or size <= maximum):
            sizes.append(size)
    return len(sizes), sorted(sizes, reverse=True)


def _blue_mask(frame: np.ndarray) -> np.ndarray:
    value = frame.astype(np.int16)
    return (value[:, :, 2] - value[:, :, 0] > 20) & (value[:, :, 1] - value[:, :, 0] > 15)


def _chem_checks(frames: list[np.ndarray]) -> list[dict[str, Any]]:
    blue = [int(_blue_mask(frame).sum()) for frame in frames]
    counts = []
    sizes = []
    for frame in frames:
        value = frame.astype(np.int16)
        crop = value[int(.40 * 320):int(.92 * 320), int(.18 * 576):int(.82 * 576)]
        bright = (crop.min(axis=2) > 175) & ((crop.max(axis=2) - crop.min(axis=2)) < 50)
        count, component_sizes = _components(bright, 20, 1500)
        counts.append(count)
        sizes.append(component_sizes)
    growth = [index for index, count in enumerate(counts) if count > 0]
    return [
        {"name": "blue_liquid_area_decreases", "passed": blue[-1] <= .75 * blue[0] and max(blue) <= 1.08 * blue[0] and sum(right <= left for left, right in zip(blue, blue[1:])) >= 30, "evidence": blue},
        {"name": "zero_to_four_crystal_components", "passed": counts[0] == 0 and 3 <= counts[-1] <= 5 and max(counts) <= 8, "evidence": {"counts": counts, "component_sizes": sizes}},
        {"name": "crystal_identity_count_does_not_reverse", "passed": all(right >= left for left, right in zip(counts, counts[1:])), "evidence": counts},
        {"name": "crystal_growth_starts_after_concentration_phase", "passed": bool(growth) and growth[0] >= 20 and all(count > 0 for count in counts[growth[0]:]), "evidence": {"first_growth_frame": growth[0] if growth else None, "counts": counts}},
    ]


def _geo_checks(frames: list[np.ndarray]) -> list[dict[str, Any]]:
    counts, sizes = [], []
    for frame in frames:
        count, component_sizes = _components(_blue_mask(frame), 1000)
        counts.append(count)
        sizes.append(component_sizes)
    cutoff = [index for index, count in enumerate(counts) if count == 2]
    return [
        {"name": "one_channel_becomes_channel_plus_one_oxbow", "passed": counts[0] == 1 and counts[-1] == 2 and set(counts) <= {1, 2}, "evidence": {"counts": counts, "component_sizes": sizes}},
        {"name": "topology_change_occurs_after_neck_narrowing", "passed": bool(cutoff) and cutoff[0] >= 24 and all(count == 2 for count in counts[cutoff[0]:]), "evidence": {"first_two_component_frame": cutoff[0] if cutoff else None, "counts": counts}},
        {"name": "main_channel_remains_largest_connected_water_body", "passed": all(values and values[0] >= 2.0 * values[1] for values in sizes if len(values) > 1), "evidence": sizes},
    ]


def _preview(frames: list[np.ndarray], target: Path) -> Path:
    indices = [round(index * (len(frames) - 1) / 8) for index in range(9)]
    canvas = Image.new("RGB", (960, 618), (13, 29, 32))
    draw = ImageDraw.Draw(canvas)
    font = _font(15)
    for slot, frame_index in enumerate(indices):
        image = Image.fromarray(frames[frame_index]).convert("RGB")
        image.thumbnail((320, 176))
        x, y = slot % 3 * 320, slot // 3 * 206
        canvas.paste(image, (x + (320 - image.width) // 2, y))
        draw.text((x + 10, y + 182), f"frame {frame_index} / {frame_index / 24:.2f}s", fill=(235, 247, 242), font=font)
    canvas.save(target, quality=92, subsampling=0)
    return target


def audit(case_id: str, level: str = "L1") -> dict[str, Any]:
    root = OUTPUT / f"{case_id}/{level}"
    video = root / "transition.mp4"
    info, frames = decode_video(video)
    endpoints = endpoint_metrics(frames, _frame(case_id, "00_start"), _frame(case_id, "03_end"))
    consecutive = consecutive_metrics(frames)
    common = [
        {"name": "all_49_frames_are_decodable", "passed": info["frame_count"] == 49, "evidence": info},
        {"name": "first_and_last_follow_supplied_guides", "passed": all(item["mean_absolute_pixel_error_0_255"] <= 15 for item in endpoints.values()), "evidence": endpoints},
        {"name": "no_abrupt_pixel_jump", "passed": consecutive["maximum"] <= 12, "evidence": consecutive},
    ]
    mechanism = _chem_checks(frames) if case_id == "CHEM-02" else _geo_checks(frames)
    preview = _preview(frames, root / "generated-frames.jpg")
    result = {
        "schema_version": "1.0",
        "case_id": case_id,
        "guidance_level": level,
        "model": MODEL,
        "settings": {**SETTINGS, "noise_seed": CASES[case_id]["seed"]},
        "video": file_record(video, REPO_ROOT),
        "preview": file_record(preview, REPO_ROOT),
        "motion_contract": file_record(OUTPUT / f"{case_id}/L1/motion-contract.json", REPO_ROOT),
        "common_checks": common,
        "mechanism_checks": mechanism,
        "passed": all(item["passed"] for item in common + mechanism),
    }
    write_json(root / "g4.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "generate", "audit"))
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--level", choices=("L1", "L2"), default="L1")
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    if args.level == "L2" and args.case != "CHEM-02":
        raise ValueError("only CHEM-02 has a predeclared L2 confirmation trial")
    if args.action == "prepare":
        value = prepare_l2() if args.level == "L2" else prepare(args.case)
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif args.action == "generate":
        generate_l2(args.server) if args.level == "L2" else generate(args.case, args.server)
    else:
        print(json.dumps(audit(args.case, args.level), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
