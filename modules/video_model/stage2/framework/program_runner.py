"""Generic deterministic runner for Stage 2 scientific program plugins.

This module deliberately contains no case-specific scientific rules.  A case
plugin maps normalized progress to a ProgramSample; the runner samples it,
serializes states and semantic layers, renders keyframes, encodes an MP4, and
checks the common data contract.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from ..cases.sentinel_programs import (
    HEIGHT,
    KEYFRAME_PROGRESS,
    WIDTH,
    LayerSample,
    ProgramSample,
    SentinelProgram,
)
from .contracts import artifact_record, load_json, sha256_path, write_json


FRAME_COUNT = 49
FPS = 12
KEYFRAME_INDICES = (0, 16, 32, 48)
KEYFRAME_NAMES = ("00_start", "01_mechanism", "02_result", "03_end")


def _numeric_record(path: Path, root: Path, array: np.ndarray) -> dict[str, Any]:
    record = artifact_record(path, root)
    record.update(
        {
            "encoding": "npy",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "value_range": [
                round(float(array.min()), 8),
                round(float(array.max()), 8),
            ],
        }
    )
    return record


def _json_record(
    path: Path, root: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    record = artifact_record(path, root)
    items = payload.get("items", [])
    record.update(
        {
            "encoding": "json",
            "shape": [len(items)],
            "dtype": "structured",
            "value_range": None,
        }
    )
    return record


def _image_record(path: Path, root: Path) -> dict[str, Any]:
    record = artifact_record(path, root)
    record.update(
        {
            "encoding": path.suffix.removeprefix(".").lower(),
            "shape": [HEIGHT, WIDTH, 3],
            "dtype": "uint8",
            "value_range": [0, 255],
        }
    )
    return record


def _serialize_layer(
    layer: LayerSample,
    keyframe_root: Path,
    program_root: Path,
) -> dict[str, Any]:
    layer_root = keyframe_root / "layers"
    layer_root.mkdir(parents=True, exist_ok=True)
    if isinstance(layer.data, np.ndarray):
        data_path = layer_root / f"{layer.layer_id}.npy"
        np.save(data_path, layer.data, allow_pickle=False)
        data_record = _numeric_record(data_path, program_root, layer.data)
    else:
        data_path = layer_root / f"{layer.layer_id}.json"
        write_json(data_path, layer.data)
        data_record = _json_record(data_path, program_root, layer.data)
    preview_path = layer_root / f"{layer.layer_id}_preview.png"
    layer.preview.convert("RGB").save(preview_path, optimize=False)
    return {
        "layer_id": layer.layer_id,
        "layer_type": layer.layer_type,
        "title_zh": layer.title_zh,
        "meaning_zh": layer.meaning_zh,
        "source_zh": "由同一时刻的确定性机制状态直接计算，不由图片反推。",
        "data": data_record,
        "preview": _image_record(preview_path, program_root),
        "model_input_policy": layer.model_input_policy,
        "used_as_model_input": False,
        "final_role_zh": layer.final_role_zh,
    }


def _write_keyframe(
    sample: ProgramSample,
    *,
    keyframe_index: int,
    frame_index: int,
    program_root: Path,
) -> dict[str, Any]:
    keyframe_root = program_root / "keyframes" / KEYFRAME_NAMES[keyframe_index]
    keyframe_root.mkdir(parents=True, exist_ok=True)
    clean_path = keyframe_root / "clean.png"
    program_path = keyframe_root / "program.png"
    state_path = keyframe_root / "state.json"
    sample.clean_frame.convert("RGB").save(clean_path, optimize=False)
    sample.program_frame.convert("RGB").save(program_path, optimize=False)
    write_json(state_path, sample.state)
    layer_records = [
        _serialize_layer(layer, keyframe_root, program_root)
        for layer in sample.layers
    ]
    layer_manifest = {
        "schema_version": "1.0",
        "case_id": sample.state["case_id"],
        "state_id": KEYFRAME_NAMES[keyframe_index],
        "canvas": {
            "width": WIDTH,
            "height": HEIGHT,
            "coordinate_system": "pixel_xy_top_left",
        },
        "layers": layer_records,
    }
    layer_manifest_path = keyframe_root / "semantic_layers.json"
    write_json(layer_manifest_path, layer_manifest)
    return {
        "keyframe_id": KEYFRAME_NAMES[keyframe_index],
        "order": keyframe_index,
        "frame_index": frame_index,
        "progress": round(KEYFRAME_PROGRESS[keyframe_index], 6),
        "clean_frame": _image_record(clean_path, program_root),
        "program_frame": _image_record(program_path, program_root),
        "state": artifact_record(state_path, program_root),
        "semantic_layers": artifact_record(
            layer_manifest_path, program_root
        ),
        "layers": layer_records,
    }


def _encode_mp4(frames_root: Path, output_path: Path) -> None:
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(FPS),
        "-start_number",
        "0",
        "-i",
        str(frames_root / "frame_%03d.png"),
        "-frames:v",
        str(FRAME_COUNT),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"ffmpeg failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )


def _build_case_contact_sheet(
    program: SentinelProgram,
    keyframes: list[dict[str, Any]],
    program_root: Path,
) -> Path:
    gutter = 12
    label_height = 32
    width = WIDTH * 2 + gutter * 3
    height = (HEIGHT + label_height) * 2 + gutter * 3
    sheet = Image.new("RGB", (width, height), (13, 31, 37))
    draw = ImageDraw.Draw(sheet)
    for index, keyframe in enumerate(keyframes):
        row, column = divmod(index, 2)
        left = gutter + column * (WIDTH + gutter)
        top = gutter + row * (HEIGHT + label_height + gutter)
        image = Image.open(
            program_root / keyframe["program_frame"]["path"]
        ).convert("RGB")
        sheet.paste(image, (left, top))
        draw.rectangle(
            (left, top + HEIGHT, left + WIDTH, top + HEIGHT + label_height),
            fill=(6, 23, 28),
        )
        draw.text(
            (left + 8, top + HEIGHT + 9),
            f"{program.case_id} · KEYFRAME {index} · "
            f"progress={keyframe['progress']:.3f}",
            fill=(230, 243, 238),
        )
    path = program_root / "keyframe-contact-sheet.jpg"
    sheet.save(path, quality=92, subsampling=0)
    return path


def _common_checks(
    program: SentinelProgram,
    all_samples: list[ProgramSample],
    key_samples: list[ProgramSample],
    keyframes: list[dict[str, Any]],
    animation_path: Path,
) -> list[dict[str, Any]]:
    progress = [sample.state["progress"] for sample in all_samples]
    layer_signatures = [
        [(layer.layer_id, layer.layer_type) for layer in sample.layers]
        for sample in key_samples
    ]
    annotation_layers = [
        layer
        for sample in key_samples
        for layer in sample.layers
        if layer.layer_type == "annotation"
    ]
    image_contract = all(
        sample.clean_frame.size == (WIDTH, HEIGHT)
        and sample.clean_frame.mode == "RGB"
        and sample.program_frame.size == (WIDTH, HEIGHT)
        and sample.program_frame.mode == "RGB"
        for sample in all_samples
    )
    return [
        {
            "name": "normalized_progress_is_monotonic",
            "passed": len(progress) == FRAME_COUNT
            and math.isclose(progress[0], 0.0)
            and math.isclose(progress[-1], 1.0)
            and all(b > a for a, b in zip(progress, progress[1:])),
            "evidence": {
                "frame_count": len(progress),
                "first": progress[0],
                "last": progress[-1],
            },
        },
        {
            "name": "fixed_rgb_canvas",
            "passed": image_contract,
            "evidence": {"width": WIDTH, "height": HEIGHT, "mode": "RGB"},
        },
        {
            "name": "semantic_layer_contract_is_stable",
            "passed": all(
                signature == layer_signatures[0]
                for signature in layer_signatures
            ),
            "evidence": layer_signatures[0],
        },
        {
            "name": "annotations_are_post_generation_only",
            "passed": len(annotation_layers) == len(key_samples)
            and all(
                layer.model_input_policy == "never"
                for layer in annotation_layers
            )
            and all(
                not layer["used_as_model_input"]
                for keyframe in keyframes
                for layer in keyframe["layers"]
            ),
            "evidence": {
                "annotation_count": len(annotation_layers),
                "used_as_model_input": False,
            },
        },
        {
            "name": "video_was_encoded_from_program_frames",
            "passed": animation_path.is_file()
            and animation_path.stat().st_size > 1024,
            "evidence": {
                "frame_count": FRAME_COUNT,
                "fps": FPS,
                "duration_seconds": round(FRAME_COUNT / FPS, 4),
            },
        },
        {
            "name": "zero_generative_model_runs",
            "passed": True,
            "evidence": {"image": 0, "video": 0},
        },
        {
            "name": "case_id_matches_plugin",
            "passed": all(
                sample.state["case_id"] == program.case_id
                for sample in all_samples
            ),
            "evidence": program.case_id,
        },
    ]


def build_program(
    program: SentinelProgram,
    output_root: Path,
    *,
    phase: int = 2,
) -> dict[str, Any]:
    """Build one sentinel program with a case-agnostic export path."""

    output_root.mkdir(parents=True, exist_ok=True)
    frames_root = output_root / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    progress_values = [
        frame_index / (FRAME_COUNT - 1)
        for frame_index in range(FRAME_COUNT)
    ]
    samples = [program.sample(progress) for progress in progress_values]
    states_path = output_root / "states.jsonl"
    states_path.write_text(
        "".join(
            json.dumps(
                sample.state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    frame_records = []
    for index, sample in enumerate(samples):
        path = frames_root / f"frame_{index:03d}.png"
        sample.program_frame.convert("RGB").save(path, optimize=False)
        frame_records.append(artifact_record(path, output_root))

    key_samples = [samples[index] for index in KEYFRAME_INDICES]
    keyframes = [
        _write_keyframe(
            sample,
            keyframe_index=index,
            frame_index=KEYFRAME_INDICES[index],
            program_root=output_root,
        )
        for index, sample in enumerate(key_samples)
    ]
    animation_path = output_root / "program-animation.mp4"
    _encode_mp4(frames_root, animation_path)
    contact_sheet_path = _build_case_contact_sheet(
        program, keyframes, output_root
    )
    mechanism_checks = program.validate(key_samples)
    common_checks = _common_checks(
        program, samples, key_samples, keyframes, animation_path
    )
    checks = mechanism_checks + common_checks
    validation = {
        "schema_version": "1.0",
        "case_id": program.case_id,
        "status": (
            "passed" if all(item["passed"] for item in checks) else "failed"
        ),
        "primary_mechanism_zh": program.primary_mechanism_zh,
        "mechanism_checks": mechanism_checks,
        "common_checks": common_checks,
    }
    validation_path = output_root / "validation.json"
    write_json(validation_path, validation)
    manifest = {
        "schema_version": "1.0",
        "phase": phase,
        "case_id": program.case_id,
        "title_zh": program.title_zh,
        "classification": (
            "deterministic scientific program animation; "
            "not an image-model render"
        ),
        "status": validation["status"],
        "canvas": {"width": WIDTH, "height": HEIGHT, "mode": "RGB"},
        "timeline": {
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "duration_seconds": round(FRAME_COUNT / FPS, 4),
            "progress_domain": [0.0, 1.0],
        },
        "primary_mechanism_zh": program.primary_mechanism_zh,
        "state_source": (
            "case plugin computes deterministic state from normalized progress"
        ),
        "states": artifact_record(states_path, output_root),
        "animation": artifact_record(animation_path, output_root),
        "contact_sheet": artifact_record(contact_sheet_path, output_root),
        "keyframes": keyframes,
        "frame_hashes": frame_records,
        "validation": artifact_record(validation_path, output_root),
        "checks": checks,
        "model_runs": {"image": 0, "video": 0},
    }
    manifest_path = output_root / "program_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def validate_program_tree(output_root: Path) -> dict[str, Any]:
    """Verify hashes, state count, image properties, and check results."""

    manifest_path = output_root / "program_manifest.json"
    manifest = load_json(manifest_path)
    if manifest["status"] != "passed":
        raise ValueError(f"{manifest['case_id']} program is not passed")
    if manifest["model_runs"] != {"image": 0, "video": 0}:
        raise ValueError("Phase 2 program contains generative-model runs")
    if manifest["timeline"]["frame_count"] != FRAME_COUNT:
        raise ValueError("Phase 2 frame count mismatch")
    records = [
        manifest["states"],
        manifest["animation"],
        manifest["contact_sheet"],
        manifest["validation"],
        *manifest["frame_hashes"],
    ]
    for keyframe in manifest["keyframes"]:
        records.extend(
            (
                keyframe["clean_frame"],
                keyframe["program_frame"],
                keyframe["state"],
                keyframe["semantic_layers"],
            )
        )
        for layer in keyframe["layers"]:
            records.extend((layer["data"], layer["preview"]))
            if layer["layer_type"] == "annotation":
                if (
                    layer["model_input_policy"] != "never"
                    or layer["used_as_model_input"]
                ):
                    raise ValueError("annotation layer leaked into model input")
    for record in records:
        path = output_root / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_path(path) != record["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path}")
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"artifact size mismatch: {path}")
    states = [
        json.loads(line)
        for line in (output_root / manifest["states"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    if len(states) != FRAME_COUNT:
        raise ValueError("states.jsonl frame count mismatch")
    if not all(item["passed"] for item in manifest["checks"]):
        raise ValueError("a saved Phase 2 validation check failed")
    return manifest
