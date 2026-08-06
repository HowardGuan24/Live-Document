#!/usr/bin/env python3
"""Deterministic Stage 5 program-baseline runtime.

The runtime has exactly three subcommands:

- generate-sequence: evaluate the frozen prototype into one canonical NPZ.
- render-program: render only the saved NPZ and encode a program preview.
- render-teaching: render the saved NPZ with deterministic teaching overlays.

It invokes no Agent, model, or network service.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator
from PIL import Image, ImageDraw, ImageFont


RUNTIME_PREFIX = "__runtime_"
FPS = 12
FRAME_COUNT = 120
DURATION_SECONDS = 10
NATIVE_WIDTH = 220
NATIVE_HEIGHT = 150
SCALE_FACTOR = 4
OUTPUT_WIDTH = NATIVE_WIDTH * SCALE_FACTOR
OUTPUT_HEIGHT = NATIVE_HEIGHT * SCALE_FACTOR
FFMPEG_PATH = Path("/usr/bin/ffmpeg")
FFPROBE_PATH = Path("/usr/bin/ffprobe")
PRESET = "medium"
CRF = 18
TITLE_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
BODY_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
PROTECTED_TEACHING_FIELDS = (
    "original_fracture_mask",
    "entered_water_mask",
    "newly_dissolved_mask",
    "current_opening_mask",
)
TEACHING_LAYOUT = {
    "preset_id": "teaching_overlay_v1",
    "output_size": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
    "regions": {
        "header": [0, 0, 880, 72],
        "legend": [16, 80, 352, 200],
        "caption": [0, 584, 880, 600],
    },
    "text_areas": {
        "title": [20, 8, 680, 64],
        "stage_count": [700, 8, 860, 64],
        "legend": [24, 86, 344, 192],
        "caption": [16, 584, 864, 600],
    },
    "font_sizes": {"title": 28, "stage_count": 20, "legend": 17, "caption": 14},
    "colors_rgba": {
        "header": [18, 28, 36, 190],
        "legend": [18, 28, 36, 185],
        "caption": [12, 18, 24, 220],
        "primary_text": [250, 250, 248, 255],
        "secondary_text": [225, 232, 236, 255],
        "swatch_outline": [245, 245, 242, 255],
    },
    "legend_row_height": 25,
    "caption_max_lines": 2,
    "semantic_overlap_tolerance_pixels": 0,
}

RUNTIME_PATH = Path(__file__).resolve()
SCHEMA_PATH = RUNTIME_PATH.with_name("schema.json")
STAGE5_ROOT = RUNTIME_PATH.parents[2]
REPOSITORY_ROOT = RUNTIME_PATH.parents[5]
PHASE1_SCHEMA_PATH = STAGE5_ROOT / "workflow" / "phase1" / "schema.json"
PHASE2_SCHEMA_PATH = STAGE5_ROOT / "workflow" / "phase2" / "schema.json"

_BASELINE_SCHEMA: dict[str, Any] | None = None
_PHASE2_SCHEMA: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def artifact_ref(path: Path, *, record_as: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {path}")
    return {
        "path": recorded_path(record_as or path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def resolve_recorded_path(raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def verify_artifact(reference: dict[str, Any], label: str) -> Path:
    path = resolve_recorded_path(reference["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != reference["sha256"]:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {reference['sha256']}, got {actual_hash}"
        )
    if "size_bytes" in reference and path.stat().st_size != reference["size_bytes"]:
        raise ValueError(f"{label} size mismatch")
    return path


def load_baseline_schema() -> dict[str, Any]:
    global _BASELINE_SCHEMA
    if _BASELINE_SCHEMA is None:
        _BASELINE_SCHEMA = load_json(SCHEMA_PATH)
        Draft202012Validator.check_schema(_BASELINE_SCHEMA)
    return _BASELINE_SCHEMA


def load_phase2_schema() -> dict[str, Any]:
    global _PHASE2_SCHEMA
    if _PHASE2_SCHEMA is None:
        _PHASE2_SCHEMA = load_json(PHASE2_SCHEMA_PATH)
        Draft202012Validator.check_schema(_PHASE2_SCHEMA)
    return _PHASE2_SCHEMA


def validate_schema(value: Any, schema: dict[str, Any], label: str) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    location = "$"
    for part in error.absolute_path:
        location += f"[{part}]" if isinstance(part, int) else f".{part}"
    raise ValueError(f"{label} failed schema at {location}: {error.message}")


def validate_baseline_artifact(value: Any, definition: str, label: str) -> None:
    schema = load_baseline_schema()
    validate_schema(
        value,
        {
            "$schema": schema["$schema"],
            "$defs": schema["$defs"],
            "$ref": f"#/$defs/{definition}",
        },
        label,
    )


def validate_phase2_artifact(value: Any, definition: str, label: str) -> None:
    schema = load_phase2_schema()
    validate_schema(
        value,
        {
            "$schema": schema["$schema"],
            "$defs": schema["$defs"],
            "$ref": f"#/$defs/{definition}",
        },
        label,
    )


def package_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def environment_record() -> dict[str, Any]:
    return {
        "interpreter": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd().resolve()),
        "packages": {
            "jsonschema": package_version("jsonschema"),
            "numpy": package_version("numpy"),
            "Pillow": package_version("Pillow"),
        },
    }


def execution_record(started_at: str, started_clock: float) -> dict[str, Any]:
    argv = [str(Path(sys.executable).resolve()), str(RUNTIME_PATH), *sys.argv[1:]]
    return {
        "command": shlex.join(argv),
        "argv": argv,
        "exit_code": 0,
        "elapsed_seconds": time.perf_counter() - started_clock,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "environment": environment_record(),
    }


def check(check_id: str, passed: bool, evidence: str) -> dict[str, Any]:
    if not passed:
        raise ValueError(f"{check_id} failed: {evidence}")
    return {"check_id": check_id, "passed": True, "evidence": evidence}


def ensure_output_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_repository_contained(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository root: {resolved}") from exc
    return resolved


def import_prototype(path: Path, required: tuple[str, ...]):
    name = f"live_document_program_baseline_{sha256_file(path)[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import prototype: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [item for item in required if not callable(getattr(module, item, None))]
    if missing:
        raise AttributeError(f"Prototype is missing required callables: {missing}")
    return module


def merge_selected_config(plan: dict[str, Any], selected_id: str) -> dict[str, Any]:
    selected = next(
        (item for item in plan["candidates"] if item["candidate_id"] == selected_id),
        None,
    )
    if selected is None:
        raise ValueError(f"Selected candidate {selected_id} is absent from plan")
    complete = dict(plan.get("fixed_parameters", {}))
    complete.update(selected.get("parameter_overrides", {}))
    return complete


def resolve_upstream_path(raw: str, label: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return ensure_repository_contained(REPOSITORY_ROOT / path, label)


def resolve_frozen_prototype(
    executable_spec: dict[str, Any], plan_path: Path, plan: dict[str, Any]
) -> tuple[Path, str]:
    expected_hash = executable_spec["prototype_sha256"]
    raw_frozen = Path(executable_spec["prototype_entrypoint"])
    if raw_frozen.is_absolute() and raw_frozen.is_file():
        frozen_path = raw_frozen.resolve()
        if sha256_file(frozen_path) == expected_hash:
            ensure_repository_contained(frozen_path, "frozen prototype")
            return frozen_path, "executable_spec_absolute_verified_v1"

    raw_plan = Path(plan["prototype_entrypoint"])
    plan_candidate = (
        raw_plan.resolve()
        if raw_plan.is_absolute()
        else (plan_path.parent / raw_plan).resolve()
    )
    plan_candidate = ensure_repository_contained(plan_candidate, "plan prototype")
    if not plan_candidate.is_file():
        raise FileNotFoundError(f"Plan prototype does not exist: {plan_candidate}")
    if sha256_file(plan_candidate) != expected_hash:
        raise ValueError("Plan prototype does not match frozen prototype SHA-256")
    return plan_candidate, "plan_repository_relative_verified_v1"


def validate_upstream(
    executable_spec_path: Path, plan_path: Path, schedule_path: Path
) -> dict[str, Any]:
    executable_spec = load_json(executable_spec_path)
    plan = load_json(plan_path)
    schedule = load_json(schedule_path)
    validate_phase2_artifact(executable_spec, "executable_spec", "executable spec")
    validate_phase2_artifact(plan, "plan", "supplemental plan")
    validate_baseline_artifact(schedule, "schedule", "schedule")

    semantic_path = resolve_upstream_path(
        executable_spec["semantic_contract_binding"]["path"], "semantic contract"
    )
    if sha256_file(semantic_path) != executable_spec["semantic_contract_binding"]["sha256"]:
        raise ValueError("Semantic-contract binding SHA-256 mismatch")
    semantic_contract = load_json(semantic_path)
    phase1_schema = load_json(PHASE1_SCHEMA_PATH)
    Draft202012Validator.check_schema(phase1_schema)
    validate_schema(semantic_contract, phase1_schema, "semantic contract")

    probe_path = resolve_upstream_path(
        executable_spec["source_probe_result"]["path"], "selected probe result"
    )
    selection_path = resolve_upstream_path(
        executable_spec["source_selection"]["path"], "selection"
    )
    if sha256_file(probe_path) != executable_spec["source_probe_result"]["sha256"]:
        raise ValueError("Selected probe-result binding SHA-256 mismatch")
    if sha256_file(selection_path) != executable_spec["source_selection"]["sha256"]:
        raise ValueError("Selection binding SHA-256 mismatch")
    probe_result = load_json(probe_path)
    selection = load_json(selection_path)
    validate_phase2_artifact(probe_result, "probe_result", "selected probe result")
    validate_phase2_artifact(selection, "selection", "selection")

    attempt_id = executable_spec["implementation_attempt_id"]
    selected_id = executable_spec["selected_candidate_id"]
    if plan["implementation_attempt_id"] != attempt_id:
        raise ValueError("Plan implementation attempt does not match executable spec")
    if probe_result["implementation_attempt_id"] != attempt_id:
        raise ValueError("Probe implementation attempt does not match executable spec")
    if selection["implementation_attempt_id"] != attempt_id:
        raise ValueError("Selection implementation attempt does not match executable spec")
    if probe_result["candidate_id"] != selected_id:
        raise ValueError("Selected probe candidate does not match executable spec")
    if selection.get("selected_candidate_id") != selected_id:
        raise ValueError("Selection candidate does not match executable spec")
    if probe_result["machine_gate_status"] != "passed":
        raise ValueError("Selected probe result did not pass its machine gate")
    if not all(executable_spec["freeze_checks"].values()):
        raise ValueError("Executable spec contains a failed Freeze check")

    accepted = [
        item
        for item in selection["reviewed_candidates"]
        if item.get("decision") == "accept"
    ]
    if len(accepted) != 1 or accepted[0]["candidate_id"] != selected_id:
        raise ValueError("Selection does not contain one exact accepted candidate")

    complete_config = merge_selected_config(plan, selected_id)
    frozen_config = executable_spec["frozen_config"]
    if set(complete_config) != set(frozen_config) or complete_config != frozen_config:
        raise ValueError("Plan fixed parameters plus selected overrides do not equal frozen config")
    if probe_result["complete_config"] != frozen_config:
        raise ValueError("Selected probe config does not equal frozen config")

    prototype_path, prototype_rule = resolve_frozen_prototype(
        executable_spec, plan_path, plan
    )
    if sha256_file(prototype_path) != probe_result["execution"]["prototype_sha256"]:
        raise ValueError("Resolved prototype does not match selected probe execution")

    declared_invariant_ids = [
        item["check_id"]
        for item in plan["implementation"]["semantic_state_contract"]["invariants"]
    ]
    probe_invariant_ids = [item["check_id"] for item in probe_result["machine_checks"]]
    if declared_invariant_ids != probe_invariant_ids:
        raise ValueError("Plan invariant IDs do not exactly match selected probe-result IDs")

    approved_anchors = [
        {
            "anchor_id": item["sample_id"],
            "progress_values": item["progress_values"],
        }
        for item in plan["implementation"]["probe_samples"]
    ]
    if schedule["anchors"] != approved_anchors:
        raise ValueError("Schedule anchors do not exactly equal approved ordered probe samples")
    if float(schedule["duration_seconds"]) != float(semantic_contract["duration_seconds"]):
        raise ValueError("Schedule duration does not equal semantic-contract duration")

    expected_segments = [
        (approved_anchors[index]["anchor_id"], approved_anchors[index + 1]["anchor_id"])
        for index in range(len(approved_anchors) - 1)
    ]
    actual_segments = [
        (item["from_anchor"], item["to_anchor"]) for item in schedule["segments"]
    ]
    if actual_segments != expected_segments:
        raise ValueError("Schedule does not contain every approved adjacent anchor pair exactly once")
    exact_total = (
        schedule["start_hold_frames"]
        + sum(item["transition_frames"] for item in schedule["segments"])
        + schedule["end_hold_frames"]
    )
    if exact_total != schedule["frame_count"]:
        raise ValueError(f"Schedule emits {exact_total} frames, expected {schedule['frame_count']}")
    if schedule["start_hold_frames"] < 1:
        raise ValueError("Schedule must emit the exact initial anchor at frame zero")

    progress_key_sets = [set(item["progress_values"]) for item in approved_anchors]
    if any(keys != progress_key_sets[0] for keys in progress_key_sets[1:]):
        raise ValueError("Approved anchors do not share an exact progress-key set")
    plan_progress_keys = {
        item["progress_variable"]
        for item in plan["implementation"]["progress_realization"]
    }
    if progress_key_sets[0] != plan_progress_keys:
        raise ValueError("Schedule progress keys do not match approved plan progress variables")

    return {
        "executable_spec": executable_spec,
        "plan": plan,
        "schedule": schedule,
        "semantic_contract": semantic_contract,
        "semantic_path": semantic_path,
        "probe_result": probe_result,
        "probe_path": probe_path,
        "selection": selection,
        "selection_path": selection_path,
        "prototype_path": prototype_path,
        "prototype_rule": prototype_rule,
        "frozen_config": frozen_config,
        "declared_invariant_ids": declared_invariant_ids,
    }


def eased_value(raw_value: float, easing: str) -> float:
    if easing == "linear":
        return raw_value
    if easing == "smoothstep":
        return raw_value * raw_value * (3.0 - 2.0 * raw_value)
    raise ValueError(f"Unsupported easing: {easing}")


def materialize_schedule(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = {item["anchor_id"]: item["progress_values"] for item in schedule["anchors"]}
    first_id = schedule["anchors"][0]["anchor_id"]
    final_id = schedule["anchors"][-1]["anchor_id"]
    frames: list[dict[str, Any]] = []

    def append_frame(
        progress: dict[str, Any], from_anchor: str, to_anchor: str, interpolation: float
    ) -> None:
        index = len(frames)
        frames.append(
            {
                "frame_index": index,
                "frame_id": f"frame-{index:06d}",
                "presentation_time_seconds": index / schedule["fps"],
                "semantic_normalized_time": index / (schedule["frame_count"] - 1),
                "source_anchor_interval": {
                    "from_anchor": from_anchor,
                    "to_anchor": to_anchor,
                },
                "interpolation_value": interpolation,
                "progress_values": {key: float(value) for key, value in progress.items()},
            }
        )

    for _ in range(schedule["start_hold_frames"]):
        append_frame(anchors[first_id], first_id, first_id, 0.0)

    for segment in schedule["segments"]:
        source = anchors[segment["from_anchor"]]
        destination = anchors[segment["to_anchor"]]
        count = segment["transition_frames"]
        for step in range(1, count + 1):
            if step == count:
                interpolation = 1.0
                progress = dict(destination)
            else:
                interpolation = eased_value(step / count, segment["easing"])
                progress = {
                    key: float(source[key])
                    + (float(destination[key]) - float(source[key])) * interpolation
                    for key in source
                }
            append_frame(
                progress,
                segment["from_anchor"],
                segment["to_anchor"],
                interpolation,
            )

    for _ in range(schedule["end_hold_frames"]):
        append_frame(anchors[final_id], final_id, final_id, 1.0)

    if len(frames) != schedule["frame_count"]:
        raise ValueError(f"Materialized {len(frames)} frames")
    return frames


def normalized_state_digest(
    states: list[dict[str, Any]], field_names: list[str]
) -> str:
    digest = hashlib.sha256()
    for state in states:
        for field_name in field_names:
            value = np.asarray(state[field_name])
            digest.update(field_name.encode("utf-8") + b"\0")
            digest.update(value.dtype.str.encode("ascii") + b"\0")
            digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
            digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def evaluate_replay(
    prototype,
    frames: list[dict[str, Any]],
    frozen_config: dict[str, Any],
    field_names: list[str],
    expected_descriptors: dict[str, tuple[str, tuple[int, ...]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, tuple[int, ...]]]]:
    static_scene = prototype.build_static_scene(frozen_config)
    states: list[dict[str, Any]] = []
    descriptors: dict[str, tuple[str, tuple[int, ...]]] = {}
    declared = set(field_names)
    for frame in frames:
        state = prototype.evaluate_state(
            static_scene, frame["progress_values"], frozen_config
        )
        if not isinstance(state, dict):
            raise TypeError("evaluate_state must return a dictionary")
        missing = sorted(declared - set(state))
        if missing:
            raise ValueError(f"{frame['frame_id']} omitted semantic fields: {missing}")
        for field_name in field_names:
            if field_name.startswith(RUNTIME_PREFIX):
                raise ValueError(f"Semantic field collides with reserved prefix: {field_name}")
            array = np.asarray(state[field_name])
            if array.dtype == object:
                raise TypeError(f"{field_name} has object dtype at {frame['frame_id']}")
            descriptor = (array.dtype.str, tuple(array.shape))
            if field_name not in descriptors:
                descriptors[field_name] = descriptor
            elif descriptors[field_name] != descriptor:
                raise ValueError(f"{field_name} changed dtype or shape at {frame['frame_id']}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{field_name} contains a non-finite value at {frame['frame_id']}")
        for progress_key, progress_value in frame["progress_values"].items():
            if progress_key not in state or float(state[progress_key]) != float(progress_value):
                raise ValueError(f"{frame['frame_id']} did not preserve {progress_key} exactly")
        state["_probe_sample_id"] = frame["frame_id"]
        states.append(state)

    if expected_descriptors is not None and descriptors != expected_descriptors:
        raise ValueError("Replay changed semantic field dtype or shape descriptors")
    return states, descriptors


def normalize_invariant_results(
    raw_checks: Any, declared_ids: list[str]
) -> list[dict[str, Any]]:
    if not isinstance(raw_checks, list) or not raw_checks:
        raise ValueError("validate_probe returned no checks")
    allowed = {
        "check_id",
        "claim",
        "passed",
        "evidence",
        "observed_count",
        "denominator",
        "state_fields",
        "rendered",
        "sample_id",
        "worst_sample_id",
        "location",
    }
    normalized: list[dict[str, Any]] = []
    for item in raw_checks:
        if not isinstance(item, dict):
            raise TypeError("validate_probe returned a non-object check")
        for required in ("check_id", "claim", "passed", "evidence"):
            if required not in item:
                raise ValueError(f"Invariant result omitted {required}")
        if item["passed"] is not True:
            raise ValueError(
                f"Full-sequence invariant failed: {item['check_id']}: {item['evidence']}"
            )
        normalized.append({key: item[key] for key in allowed if key in item})
    if [item["check_id"] for item in normalized] != declared_ids:
        raise ValueError("Full-sequence checks do not exactly cover declared invariant IDs")
    return normalized


def save_sequence_archive(
    path: Path,
    states: list[dict[str, Any]],
    field_names: list[str],
    frames: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for field_name in field_names:
        arrays[field_name] = np.stack(
            [np.asarray(state[field_name]) for state in states], axis=0
        )
    arrays["__runtime_frame_index"] = np.arange(len(states), dtype=np.int64)
    arrays["__runtime_pts_seconds"] = np.asarray(
        [frame["presentation_time_seconds"] for frame in frames], dtype=np.float64
    )
    arrays["__runtime_semantic_time"] = np.asarray(
        [frame["semantic_normalized_time"] for frame in frames], dtype=np.float64
    )
    arrays["__runtime_canvas_width"] = np.asarray(
        int(states[0]["canvas_width"]), dtype=np.int64
    )
    arrays["__runtime_canvas_height"] = np.asarray(
        int(states[0]["canvas_height"]), dtype=np.int64
    )
    arrays["__runtime_surface_y"] = np.asarray(
        int(states[0]["surface_y"]), dtype=np.int64
    )
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    return arrays


def verify_archive_round_trip(
    path: Path, expected: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("Sequence archive key set changed during serialization")
        for key in archive.files:
            actual = archive[key]
            wanted = expected[key]
            if actual.dtype != wanted.dtype or actual.shape != wanted.shape:
                raise ValueError(f"Sequence archive dtype/shape mismatch for {key}")
            if not np.array_equal(actual, wanted, equal_nan=True):
                raise ValueError(f"Sequence archive value mismatch for {key}")
            loaded[key] = actual.copy()
    return loaded


def anchor_boundary_indices(schedule: dict[str, Any]) -> dict[str, int]:
    mapping = {schedule["anchors"][0]["anchor_id"]: 0}
    cursor = schedule["start_hold_frames"]
    for segment in schedule["segments"]:
        cursor += segment["transition_frames"]
        mapping[segment["to_anchor"]] = cursor - 1
    return mapping


def generate_sequence(args: argparse.Namespace) -> int:
    started_at = utc_now()
    started_clock = time.perf_counter()
    output_dir = args.output_dir.resolve()
    ensure_output_absent(output_dir)
    upstream = validate_upstream(
        args.executable_spec.resolve(), args.plan.resolve(), args.schedule.resolve()
    )
    plan = upstream["plan"]
    schedule = upstream["schedule"]
    frames = materialize_schedule(schedule)
    field_contracts = plan["implementation"]["semantic_state_contract"]["fields"]
    field_names = [item["name"] for item in field_contracts]
    if len(field_names) != len(set(field_names)):
        raise ValueError("Semantic field names are not unique")

    prototype = import_prototype(
        upstream["prototype_path"],
        ("build_static_scene", "evaluate_state", "validate_probe"),
    )
    states, descriptors = evaluate_replay(
        prototype, frames, upstream["frozen_config"], field_names
    )
    invariant_results = normalize_invariant_results(
        prototype.validate_probe(
            states, upstream["semantic_contract"], upstream["frozen_config"]
        ),
        upstream["declared_invariant_ids"],
    )
    first_digest = normalized_state_digest(states, field_names)
    replay_states, _ = evaluate_replay(
        prototype,
        frames,
        upstream["frozen_config"],
        field_names,
        expected_descriptors=descriptors,
    )
    second_digest = normalized_state_digest(replay_states, field_names)
    if first_digest != second_digest:
        raise ValueError("Second in-memory replay changed normalized state digest")

    expected_boundaries = anchor_boundary_indices(schedule)
    for anchor in schedule["anchors"]:
        index = expected_boundaries[anchor["anchor_id"]]
        if frames[index]["progress_values"] != {
            key: float(value) for key, value in anchor["progress_values"].items()
        }:
            raise ValueError(f"Approved anchor missing at expected boundary: {anchor['anchor_id']}")
    first_anchor = {
        key: float(value)
        for key, value in schedule["anchors"][0]["progress_values"].items()
    }
    final_anchor = {
        key: float(value)
        for key, value in schedule["anchors"][-1]["progress_values"].items()
    }
    timestamps = [item["presentation_time_seconds"] for item in frames]

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        archive_path = staging / "sequence.npz"
        expected_arrays = save_sequence_archive(
            archive_path, states, field_names, frames
        )
        loaded_arrays = verify_archive_round_trip(archive_path, expected_arrays)

        semantic_fields = []
        contract_by_name = {item["name"]: item for item in field_contracts}
        for field_name in field_names:
            dtype, state_shape = descriptors[field_name]
            semantic_fields.append(
                {
                    "name": field_name,
                    "kind": contract_by_name[field_name]["kind"],
                    "dtype": dtype,
                    "state_shape": list(state_shape),
                    "archive_shape": list(loaded_arrays[field_name].shape),
                    "consumers": contract_by_name[field_name]["consumers"],
                }
            )

        runtime_meanings = {
            "__runtime_frame_index": "Zero-based canonical frame index.",
            "__runtime_pts_seconds": "Presentation timestamp in seconds at 12 FPS.",
            "__runtime_semantic_time": "Normalized semantic sequence position from zero to one.",
            "__runtime_canvas_width": "Frozen native renderer width.",
            "__runtime_canvas_height": "Frozen native renderer height.",
            "__runtime_surface_y": "Frozen prototype surface row needed by the public renderer.",
        }
        runtime_metadata_fields = [
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "meaning": runtime_meanings[name],
            }
            for name, array in loaded_arrays.items()
            if name.startswith(RUNTIME_PREFIX)
        ]

        checks = [
            check(
                "UPSTREAM_HASHES_VERIFIED",
                True,
                "Executable spec, semantic contract, plan, selection, probe result, and prototype were hash-checked.",
            ),
            check("EXACT_FRAME_COUNT", len(frames) == FRAME_COUNT, f"frames={len(frames)}"),
            check("EXACT_FIRST_ANCHOR", frames[0]["progress_values"] == first_anchor, "frame-000000 equals initial anchor"),
            check("EXACT_FINAL_ANCHOR", frames[-1]["progress_values"] == final_anchor, "frame-000119 equals final anchor"),
            check("APPROVED_ANCHOR_BOUNDARIES", True, f"boundaries={json.dumps(expected_boundaries, sort_keys=True)}"),
            check(
                "MONOTONE_TIMESTAMPS",
                all(right > left for left, right in zip(timestamps, timestamps[1:])),
                f"first={timestamps[0]:.9f}; last={timestamps[-1]:.9f}",
            ),
            check("FINITE_PROGRESS_AND_STATE", True, "All scheduled progress and declared semantic values are finite."),
            check("FULL_SEQUENCE_INVARIANTS", True, f"passed={len(invariant_results)} of {len(upstream['declared_invariant_ids'])}"),
            check("SERIALIZATION_ROUND_TRIP", True, f"keys={len(loaded_arrays)}; allow_pickle=False"),
            check("DETERMINISTIC_REPLAY", first_digest == second_digest, f"digest={first_digest}"),
        ]

        final_archive_path = output_dir / "sequence.npz"
        manifest = {
            "schema_version": "program-baseline-1",
            "artifact_type": "sequence_manifest",
            "created_at_utc": utc_now(),
            "inputs": {
                "executable_spec": artifact_ref(args.executable_spec.resolve()),
                "semantic_contract": artifact_ref(upstream["semantic_path"]),
                "supplemental_plan": artifact_ref(args.plan.resolve()),
                "source_probe_result": artifact_ref(upstream["probe_path"]),
                "source_selection": artifact_ref(upstream["selection_path"]),
                "resolved_prototype": artifact_ref(upstream["prototype_path"]),
            },
            "compatibility_resolution": {
                "plan_binding_rule": "legacy_cross_checked_sibling_v1",
                "prototype_resolution_rule": upstream["prototype_rule"],
                "resolved_prototype_repository_path": recorded_path(upstream["prototype_path"]),
            },
            "frozen_config": upstream["frozen_config"],
            "schedule": {
                "artifact": artifact_ref(args.schedule.resolve()),
                "resolved": schedule,
            },
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "duration_seconds": DURATION_SECONDS,
            "sequence_archive": artifact_ref(
                archive_path, record_as=final_archive_path
            ),
            "semantic_fields": semantic_fields,
            "runtime_metadata_fields": runtime_metadata_fields,
            "frames": frames,
            "full_sequence_invariants": invariant_results,
            "deterministic_replay": {
                "algorithm": "sha256-field-name-dtype-shape-c-bytes-v1",
                "first_digest": first_digest,
                "second_digest": second_digest,
                "matched": True,
            },
            "checks": checks,
            "failures": [],
            "execution": execution_record(started_at, started_clock),
        }
        validate_baseline_artifact(manifest, "sequence_manifest", "sequence manifest")
        manifest_path = staging / "sequence-manifest.json"
        atomic_write_json(manifest_path, manifest)
        validate_baseline_artifact(
            load_json(manifest_path), "sequence_manifest", "written sequence manifest"
        )
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return 0


def load_sequence_archive(
    archive_path: Path, manifest: dict[str, Any]
) -> dict[str, np.ndarray]:
    expected_names = {item["name"] for item in manifest["semantic_fields"]}
    expected_names.update(item["name"] for item in manifest["runtime_metadata_fields"])
    arrays: dict[str, np.ndarray] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_names:
            raise ValueError("Saved sequence keys do not match sequence manifest")
        for key in archive.files:
            value = archive[key]
            if value.dtype == object:
                raise TypeError(f"Saved sequence field {key} has object dtype")
            arrays[key] = value.copy()
    for descriptor in manifest["semantic_fields"]:
        value = arrays[descriptor["name"]]
        if value.dtype.str != descriptor["dtype"] or list(value.shape) != descriptor["archive_shape"]:
            raise ValueError(f"Saved sequence descriptor mismatch: {descriptor['name']}")
    return arrays


def reconstruct_state(
    arrays: dict[str, np.ndarray], semantic_fields: list[dict[str, Any]], index: int
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for descriptor in semantic_fields:
        value = arrays[descriptor["name"]][index]
        state[descriptor["name"]] = value.item() if value.shape == () else value.copy()
    state["canvas_width"] = int(arrays["__runtime_canvas_width"].item())
    state["canvas_height"] = int(arrays["__runtime_canvas_height"].item())
    state["surface_y"] = int(arrays["__runtime_surface_y"].item())
    return state


def build_contact_sheet(
    frame_paths: list[Path], selected_indices: list[int], output_path: Path
) -> None:
    thumb_width = OUTPUT_WIDTH // 2
    thumb_height = OUTPUT_HEIGHT // 2
    label_height = 24
    columns = 4
    rows = (len(selected_indices) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * thumb_width, rows * (thumb_height + label_height)), (235, 235, 232)
    )
    draw = ImageDraw.Draw(sheet)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 14) if font_path.is_file() else ImageFont.load_default()
    for slot, frame_index in enumerate(selected_indices):
        with Image.open(frame_paths[frame_index]) as source:
            thumb = source.convert("RGB").resize(
                (thumb_width, thumb_height), Image.Resampling.NEAREST
            )
        x = (slot % columns) * thumb_width
        y = (slot // columns) * (thumb_height + label_height)
        sheet.paste(thumb, (x, y))
        draw.text(
            (x + 6, y + thumb_height + 4),
            f"frame-{frame_index:06d} | t={frame_index / FPS:.3f}s",
            fill=(20, 20, 20),
            font=font,
        )
    sheet.save(output_path, format="PNG")


def tool_identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required media tool does not exist: {path}")
    result = subprocess.run(
        [str(path), "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    first_line = (result.stdout or result.stderr).splitlines()[0]
    return {"path": str(path), "sha256": sha256_file(path), "version": first_line}


def inspect_media(path: Path) -> dict[str, Any]:
    argv = [
        str(FFPROBE_PATH),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream, found {len(streams)}")
    stream = streams[0]
    decoded = int(stream.get("nb_read_frames") or stream.get("nb_frames"))
    fps = float(Fraction(stream.get("avg_frame_rate") or stream["r_frame_rate"]))
    duration = float(stream.get("duration") or data["format"]["duration"])
    return {
        "decoded_frame_count": decoded,
        "fps": fps,
        "duration_seconds": duration,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
    }


def render_program(args: argparse.Namespace) -> int:
    started_at = utc_now()
    started_clock = time.perf_counter()
    output_dir = args.output_dir.resolve()
    ensure_output_absent(output_dir)
    manifest_path = args.sequence_manifest.resolve()
    sequence_manifest = load_json(manifest_path)
    validate_baseline_artifact(
        sequence_manifest, "sequence_manifest", "sequence manifest"
    )
    for label, reference in sequence_manifest["inputs"].items():
        verify_artifact(reference, f"sequence input {label}")
    schedule_path = verify_artifact(
        sequence_manifest["schedule"]["artifact"], "schedule"
    )
    if load_json(schedule_path) != sequence_manifest["schedule"]["resolved"]:
        raise ValueError("Schedule file does not equal the resolved schedule in manifest")
    archive_path = verify_artifact(
        sequence_manifest["sequence_archive"], "sequence archive"
    )
    prototype_path = verify_artifact(
        sequence_manifest["inputs"]["resolved_prototype"], "prototype"
    )
    arrays = load_sequence_archive(archive_path, sequence_manifest)
    if int(arrays["__runtime_canvas_width"].item()) != NATIVE_WIDTH:
        raise ValueError("Saved native width is not 220")
    if int(arrays["__runtime_canvas_height"].item()) != NATIVE_HEIGHT:
        raise ValueError("Saved native height is not 150")
    if not np.array_equal(
        arrays["__runtime_frame_index"], np.arange(FRAME_COUNT, dtype=np.int64)
    ):
        raise ValueError("Saved frame indices are not contiguous")

    prototype = import_prototype(prototype_path, ("render_program_probe",))
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    committed = False
    try:
        frames_dir = staging / "frames"
        frames_dir.mkdir(parents=True)
        native_path = staging / ".native-frame.png"
        staging_frame_paths: list[Path] = []
        for index in range(FRAME_COUNT):
            state = reconstruct_state(
                arrays, sequence_manifest["semantic_fields"], index
            )
            prototype.render_program_probe([state], str(native_path))
            if not native_path.is_file():
                raise FileNotFoundError(f"Renderer omitted native frame {index}")
            with Image.open(native_path) as native:
                native.load()
                if native.size != (NATIVE_WIDTH, NATIVE_HEIGHT):
                    raise ValueError(f"Native frame {index} has size {native.size}")
                if native.mode != "RGB":
                    raise ValueError(f"Native frame {index} has mode {native.mode}")
                output = native.resize(
                    (OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.NEAREST
                )
                frame_path = frames_dir / f"frame-{index:06d}.png"
                output.save(frame_path, format="PNG")
            staging_frame_paths.append(frame_path)
        native_path.unlink(missing_ok=True)

        expected_names = [f"frame-{index:06d}.png" for index in range(FRAME_COUNT)]
        actual_names = sorted(path.name for path in frames_dir.glob("frame-*.png"))
        if actual_names != expected_names:
            raise ValueError("Rendered source frame sequence is not contiguous")

        boundary_indices = list(
            anchor_boundary_indices(sequence_manifest["schedule"]["resolved"]).values()
        )
        diagnostic_indices = sorted(set(boundary_indices + [30, 60, 89, 119]))
        contact_sheet_path = staging / "contact-sheet.png"
        build_contact_sheet(staging_frame_paths, diagnostic_indices, contact_sheet_path)
        os.replace(staging, output_dir)
        committed = True

        final_frames_dir = output_dir / "frames"
        final_frame_paths = [
            final_frames_dir / f"frame-{index:06d}.png" for index in range(FRAME_COUNT)
        ]
        mp4_path = output_dir / "program-preview.mp4"
        ffmpeg_argv = [
            str(FFMPEG_PATH),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-framerate",
            str(FPS),
            "-start_number",
            "0",
            "-i",
            str(final_frames_dir / "frame-%06d.png"),
            "-frames:v",
            str(FRAME_COUNT),
            "-c:v",
            "libx264",
            "-preset",
            PRESET,
            "-crf",
            str(CRF),
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            "-an",
            str(mp4_path),
        ]
        subprocess.run(ffmpeg_argv, check=True)
        inspection = inspect_media(mp4_path)
        expected_inspection = {
            "decoded_frame_count": FRAME_COUNT,
            "fps": float(FPS),
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "codec": "h264",
            "pixel_format": "yuv420p",
        }
        for key, expected in expected_inspection.items():
            if inspection[key] != expected:
                raise ValueError(
                    f"Media inspection mismatch for {key}: {inspection[key]} != {expected}"
                )
        if abs(inspection["duration_seconds"] - DURATION_SECONDS) > 0.01:
            raise ValueError(
                f"Media duration {inspection['duration_seconds']} is outside ±0.01 seconds"
            )

        frame_records = []
        for index, frame_path in enumerate(final_frame_paths):
            with Image.open(frame_path) as image:
                image.load()
                if image.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT) or image.mode != "RGB":
                    raise ValueError(f"Production frame {index} has unexpected image properties")
            frame_records.append(
                {
                    "frame_index": index,
                    "frame_id": f"frame-{index:06d}",
                    "artifact": artifact_ref(frame_path),
                    "width": OUTPUT_WIDTH,
                    "height": OUTPUT_HEIGHT,
                    "mode": "RGB",
                }
            )

        ffmpeg_identity = tool_identity(FFMPEG_PATH)
        ffprobe_identity = tool_identity(FFPROBE_PATH)
        checks = [
            check("SAVED_SEQUENCE_ONLY", True, "Renderer loaded sequence.npz and did not call state construction or evaluation."),
            check("SINGLE_STATE_PUBLIC_RENDERER", True, "All 120 native images came from render_program_probe([state], path)."),
            check("NATIVE_FRAME_PROPERTIES", True, "All native frames were RGB 220x150."),
            check("INTEGER_NEAREST_SCALING", True, "All production frames were scaled exactly 4x to RGB 880x600 with nearest-neighbor."),
            check("CONTIGUOUS_SOURCE_FRAMES", actual_names == expected_names, "frame-000000.png through frame-000119.png are present exactly once."),
            check("CONTACT_SHEET_CREATED", (output_dir / "contact-sheet.png").is_file(), f"diagnostic frame indices={diagnostic_indices}"),
            check("ENCODED_FRAME_COUNT", inspection["decoded_frame_count"] == FRAME_COUNT, f"decoded_frames={inspection['decoded_frame_count']}"),
            check("ENCODED_FPS", inspection["fps"] == FPS, f"fps={inspection['fps']}"),
            check("ENCODED_DURATION", abs(inspection["duration_seconds"] - DURATION_SECONDS) <= 0.01, f"duration_seconds={inspection['duration_seconds']}; tolerance=±0.01"),
            check("ENCODED_DIMENSIONS", (inspection["width"], inspection["height"]) == (OUTPUT_WIDTH, OUTPUT_HEIGHT), f"dimensions={inspection['width']}x{inspection['height']}"),
            check("ENCODED_CODEC", inspection["codec"] == "h264", f"codec={inspection['codec']}"),
            check("ENCODED_PIXEL_FORMAT", inspection["pixel_format"] == "yuv420p", f"pixel_format={inspection['pixel_format']}"),
        ]
        program_manifest = {
            "schema_version": "program-baseline-1",
            "artifact_type": "program_manifest",
            "created_at_utc": utc_now(),
            "sequence_manifest": artifact_ref(manifest_path),
            "sequence_archive": artifact_ref(archive_path),
            "prototype": artifact_ref(prototype_path),
            "renderer_adapter": {
                "name": "single_state_render_program_probe_v1",
                "runtime": artifact_ref(RUNTIME_PATH),
                "public_callable": "render_program_probe",
                "invocation": "render_program_probe([state], native_output_path)",
                "state_source": "saved_sequence_only",
            },
            "dimensions": {
                "native_width": NATIVE_WIDTH,
                "native_height": NATIVE_HEIGHT,
                "output_width": OUTPUT_WIDTH,
                "output_height": OUTPUT_HEIGHT,
                "scale_factor": SCALE_FACTOR,
                "scaling_mode": "nearest_neighbor_integer_4x",
            },
            "frames": frame_records,
            "contact_sheet": {
                "artifact": artifact_ref(output_dir / "contact-sheet.png"),
                "frame_indices": diagnostic_indices,
            },
            "media_tools": {
                "ffmpeg": ffmpeg_identity,
                "ffprobe": ffprobe_identity,
            },
            "encoding": {
                "command": shlex.join(ffmpeg_argv),
                "argv": ffmpeg_argv,
                "input_fps": FPS,
                "frame_count": FRAME_COUNT,
                "codec": "libx264",
                "pixel_format": "yuv420p",
                "preset": PRESET,
                "crf": CRF,
                "audio": False,
                "frame_interpolation": False,
                "implicit_duplication": False,
            },
            "program_preview": artifact_ref(mp4_path),
            "media_inspection": inspection,
            "checks": checks,
            "failures": [],
            "execution": execution_record(started_at, started_clock),
        }
        validate_baseline_artifact(
            program_manifest, "program_manifest", "program manifest"
        )
        program_manifest_path = output_dir / "program-manifest.json"
        atomic_write_json(program_manifest_path, program_manifest)
        validate_baseline_artifact(
            load_json(program_manifest_path),
            "program_manifest",
            "written program manifest",
        )
    except Exception:
        if not committed and staging.exists():
            shutil.rmtree(staging)
        raise
    return 0


def semantic_stage_windows(schedule: dict[str, Any]) -> list[dict[str, int]]:
    """Map the retained karst anchor sequence to its four Phase 1 stages."""
    boundaries = anchor_boundary_indices(schedule)
    required = {
        "initial",
        "acidified",
        "entry-early",
        "entry-complete",
        "dissolution-early",
        "dissolution-middle",
        "connected-openings",
    }
    if set(boundaries) != required:
        raise ValueError(
            "Retained karst schedule does not have the approved seven-anchor set"
        )
    return [
        {"start_frame": 0, "end_frame": boundaries["acidified"]},
        {
            "start_frame": boundaries["acidified"] + 1,
            "end_frame": boundaries["entry-complete"],
        },
        {
            "start_frame": boundaries["entry-complete"] + 1,
            "end_frame": boundaries["dissolution-early"],
        },
        {
            "start_frame": boundaries["dissolution-early"] + 1,
            "end_frame": FRAME_COUNT - 1,
        },
    ]


def validate_presentation_for_sequence(
    presentation: dict[str, Any],
    sequence_manifest: dict[str, Any],
    semantic_contract: dict[str, Any],
) -> list[dict[str, int]]:
    validate_baseline_artifact(presentation, "presentation", "presentation")
    field_descriptors = {
        item["name"]: item for item in sequence_manifest["semantic_fields"]
    }
    legend_fields = [item["semantic_field"] for item in presentation["legend"]]
    if len(legend_fields) != len(set(legend_fields)):
        raise ValueError("Presentation legend semantic fields must be unique")
    unknown = sorted(set(legend_fields) - set(field_descriptors))
    if unknown:
        raise ValueError(f"Presentation legend uses undeclared fields: {unknown}")
    for field_name in legend_fields:
        descriptor = field_descriptors[field_name]
        if descriptor["state_shape"] != [NATIVE_HEIGHT, NATIVE_WIDTH]:
            raise ValueError(
                f"Legend field {field_name} is not a 220x150 semantic mask"
            )

    stages = presentation["stages"]
    phase1_stages = semantic_contract["stages"]
    if len(stages) != len(phase1_stages):
        raise ValueError("Presentation stage count does not equal Phase 1 stage count")
    if [item["semantic_stage_index"] for item in stages] != list(
        range(len(phase1_stages))
    ):
        raise ValueError("Presentation stage indices are not exact and consecutive")
    if stages[0]["start_frame"] != 0 or stages[-1]["end_frame"] != FRAME_COUNT - 1:
        raise ValueError("Presentation stages do not cover frames 0 through 119")
    for previous, current in zip(stages, stages[1:]):
        if current["start_frame"] != previous["end_frame"] + 1:
            raise ValueError("Presentation stage ranges are not contiguous")
    for stage in stages:
        if stage["start_frame"] > stage["end_frame"]:
            raise ValueError("Presentation stage has an inverted frame range")

    expected_windows = semantic_stage_windows(
        sequence_manifest["schedule"]["resolved"]
    )
    for stage, window in zip(stages, expected_windows):
        intersects = not (
            stage["end_frame"] < window["start_frame"]
            or stage["start_frame"] > window["end_frame"]
        )
        if not intersects:
            raise ValueError(
                f"Presentation stage {stage['semantic_stage_index']} does not "
                "intersect its corresponding schedule interval"
            )
    return expected_windows


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def protected_layout_evidence(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    native_union = np.zeros((NATIVE_HEIGHT, NATIVE_WIDTH), dtype=bool)
    for field_name in PROTECTED_TEACHING_FIELDS:
        if field_name not in arrays:
            raise ValueError(f"Protected semantic field is absent: {field_name}")
        values = arrays[field_name]
        if values.shape != (FRAME_COUNT, NATIVE_HEIGHT, NATIVE_WIDTH):
            raise ValueError(f"Protected semantic field has unexpected shape: {field_name}")
        native_union |= np.any(values.astype(bool), axis=0)
    protected = np.repeat(
        np.repeat(native_union, SCALE_FACTOR, axis=0), SCALE_FACTOR, axis=1
    )
    global_bbox = mask_bbox(protected)
    if global_bbox is None:
        raise ValueError("Protected semantic union has no support")

    regions = []
    for region_id in ("header", "legend", "caption"):
        x0, y0, x1, y1 = TEACHING_LAYOUT["regions"][region_id]
        region_support = protected[y0:y1, x0:x1]
        overlap = int(np.count_nonzero(region_support))
        local_bbox = mask_bbox(region_support)
        if local_bbox is not None:
            local_bbox = [
                local_bbox[0] + x0,
                local_bbox[1] + y0,
                local_bbox[2] + x0,
                local_bbox[3] + y0,
            ]
        if overlap > TEACHING_LAYOUT["semantic_overlap_tolerance_pixels"]:
            raise ValueError(
                f"teaching_overlay_v1 {region_id} collides with {overlap} protected pixels"
            )
        regions.append(
            {
                "region_id": region_id,
                "bbox": [x0, y0, x1, y1],
                "protected_bbox": local_bbox,
                "semantic_overlap_pixels": overlap,
            }
        )
    return {
        "mask": protected,
        "global_bbox": global_bbox,
        "support_pixels": int(np.count_nonzero(protected)),
        "regions": regions,
    }


def render_clean_semantic_viewport(
    prototype, state: dict[str, Any], native_path: Path
) -> Image.Image:
    prototype.render_semantic_probe([state], str(native_path))
    if not native_path.is_file():
        raise FileNotFoundError("Semantic renderer omitted its native frame")
    with Image.open(native_path) as native:
        native.load()
        if native.size != (NATIVE_WIDTH, NATIVE_HEIGHT):
            raise ValueError(f"Semantic native frame has unexpected size {native.size}")
        if native.mode != "RGB":
            raise ValueError(f"Semantic native frame has unexpected mode {native.mode}")
        return native.resize(
            (OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.NEAREST
        ).copy()


def derive_legend_colors(
    prototype,
    arrays: dict[str, np.ndarray],
    semantic_fields: list[dict[str, Any]],
    presentation: dict[str, Any],
    native_path: Path,
) -> list[dict[str, Any]]:
    legend_fields = [item["semantic_field"] for item in presentation["legend"]]
    evidence: list[dict[str, Any]] = []
    for legend_item in presentation["legend"]:
        field_name = legend_item["semantic_field"]
        values = arrays[field_name].astype(bool)
        support_by_frame = np.any(values.reshape(FRAME_COUNT, -1), axis=1)
        supported_indices = np.flatnonzero(support_by_frame)
        if supported_indices.size == 0:
            raise ValueError(f"Legend field has no support: {field_name}")
        frame_index = int(supported_indices[0])
        state = reconstruct_state(arrays, semantic_fields, frame_index)
        prototype.render_semantic_probe([state], str(native_path))
        with Image.open(native_path) as rendered:
            rendered.load()
            if rendered.size != (NATIVE_WIDTH, NATIVE_HEIGHT) or rendered.mode != "RGB":
                raise ValueError("Legend color source frame has invalid properties")
            pixels = np.asarray(rendered, dtype=np.uint8)

        support = values[frame_index]
        other_union = np.zeros_like(support)
        for other_name in legend_fields:
            if other_name != field_name:
                other_union |= arrays[other_name][frame_index].astype(bool)
        exclusive = support & ~other_union
        exclusive_count = int(np.count_nonzero(exclusive))
        sampling_mask = exclusive if exclusive_count > 0 else support
        sampled = pixels[sampling_mask]
        if sampled.size == 0:
            raise ValueError(f"Legend field has no exposed rendered pixels: {field_name}")
        colors, counts = np.unique(sampled.reshape(-1, 3), axis=0, return_counts=True)
        winner = int(np.argmax(counts))
        dominant_count = int(counts[winner])
        confidence = dominant_count / int(sampled.shape[0])
        if confidence < 0.8:
            raise ValueError(
                f"Legend field {field_name} has ambiguous rendered color confidence {confidence:.6f}"
            )
        evidence.append(
            {
                "semantic_field": field_name,
                "label": legend_item["label"],
                "source_frame_index": frame_index,
                "source_frame_id": f"frame-{frame_index:06d}",
                "support_count": int(np.count_nonzero(support)),
                "exclusive_support_count": exclusive_count,
                "sampling_rule": "earliest_support_exclusive_else_full_modal_rgb_v1",
                "selected_rgb": [int(value) for value in colors[winner]],
                "dominant_count": dominant_count,
                "confidence_ratio": confidence,
            }
        )
    native_path.unlink(missing_ok=True)
    return evidence


def rect_contains(outer: list[int], inner: tuple[int, int, int, int]) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
            continue
        if not current:
            raise ValueError(f"Caption word does not fit fixed layout: {word}")
        lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def composite_teaching_overlay(
    base: Image.Image,
    stage: dict[str, Any],
    stage_count: int,
    legend_evidence: list[dict[str, Any]],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> tuple[Image.Image, dict[str, Any]]:
    if base.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT) or base.mode != "RGB":
        raise ValueError("Teaching compositor received an invalid semantic viewport")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = TEACHING_LAYOUT["colors_rgba"]
    for region_id in ("header", "legend", "caption"):
        x0, y0, x1, y1 = TEACHING_LAYOUT["regions"][region_id]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=tuple(colors[region_id]))

    primary = tuple(colors["primary_text"])
    secondary = tuple(colors["secondary_text"])
    title_position = (20, 14)
    title_bbox = draw.textbbox(title_position, stage["title"], font=fonts["title"])
    if not rect_contains(TEACHING_LAYOUT["text_areas"]["title"], title_bbox):
        raise ValueError(f"Stage title does not fit fixed header: {stage['title']}")
    draw.text(title_position, stage["title"], font=fonts["title"], fill=primary)

    stage_text = f"Stage {stage['semantic_stage_index'] + 1} / {stage_count}"
    stage_zero_bbox = draw.textbbox((0, 0), stage_text, font=fonts["stage_count"])
    stage_width = stage_zero_bbox[2] - stage_zero_bbox[0]
    stage_position = (860 - stage_width, 18)
    stage_bbox = draw.textbbox(stage_position, stage_text, font=fonts["stage_count"])
    if not rect_contains(TEACHING_LAYOUT["text_areas"]["stage_count"], stage_bbox):
        raise ValueError("Stage count does not fit fixed header")
    draw.text(stage_position, stage_text, font=fonts["stage_count"], fill=secondary)

    legend_bboxes = []
    for index, item in enumerate(legend_evidence):
        row_y = 90 + index * TEACHING_LAYOUT["legend_row_height"]
        swatch_bbox = (28, row_y, 47, row_y + 19)
        draw.rectangle(swatch_bbox, fill=tuple(item["selected_rgb"]) + (255,))
        draw.rectangle(swatch_bbox, outline=tuple(colors["swatch_outline"]), width=1)
        label_position = (60, row_y - 2)
        label_bbox = draw.textbbox(label_position, item["label"], font=fonts["legend"])
        if not rect_contains(TEACHING_LAYOUT["text_areas"]["legend"], label_bbox):
            raise ValueError(f"Legend label does not fit fixed card: {item['label']}")
        draw.text(label_position, item["label"], font=fonts["legend"], fill=primary)
        legend_bboxes.append(list(label_bbox))

    caption_area = TEACHING_LAYOUT["text_areas"]["caption"]
    caption_lines = wrap_text(
        draw,
        stage["caption"],
        fonts["caption"],
        caption_area[2] - caption_area[0],
    )
    if len(caption_lines) > TEACHING_LAYOUT["caption_max_lines"]:
        raise ValueError("Caption uses more than two rendered lines")
    caption_bboxes = []
    for line_index, line in enumerate(caption_lines):
        position = (caption_area[0] + 2, 582 + line_index * 14)
        line_bbox = draw.textbbox(position, line, font=fonts["caption"])
        if not rect_contains(caption_area, line_bbox):
            raise ValueError("Caption does not fit the fixed bottom safe area")
        draw.text(position, line, font=fonts["caption"], fill=primary)
        caption_bboxes.append(list(line_bbox))

    composed = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    base_pixels = np.asarray(base)
    composed_pixels = np.asarray(composed)
    overlay_region_mask = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=bool)
    for rect in TEACHING_LAYOUT["regions"].values():
        x0, y0, x1, y1 = rect
        overlay_region_mask[y0:y1, x0:x1] = True
    if not np.array_equal(
        base_pixels[~overlay_region_mask], composed_pixels[~overlay_region_mask]
    ):
        raise ValueError("Teaching compositor changed a viewport pixel outside overlay regions")
    return composed, {
        "title_bbox": list(title_bbox),
        "stage_count_bbox": list(stage_bbox),
        "legend_bboxes": legend_bboxes,
        "caption_bboxes": caption_bboxes,
        "caption_line_count": len(caption_lines),
    }


def update_teaching_digest(
    digest: "hashlib._Hash", frame_index: int, image: Image.Image
) -> None:
    digest.update(frame_index.to_bytes(8, "big", signed=False))
    digest.update(np.ascontiguousarray(np.asarray(image, dtype=np.uint8)).tobytes())


def build_teaching_contact_sheet(
    frame_paths: list[Path], frame_indices: list[int], output_path: Path
) -> None:
    sheet = Image.new("RGB", (OUTPUT_WIDTH * 2, OUTPUT_HEIGHT * 2), (20, 20, 20))
    for slot, frame_index in enumerate(frame_indices):
        with Image.open(frame_paths[frame_index]) as source:
            source.load()
            if source.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT) or source.mode != "RGB":
                raise ValueError("Teaching contact-sheet source has invalid properties")
            sheet.paste(source, ((slot % 2) * OUTPUT_WIDTH, (slot // 2) * OUTPUT_HEIGHT))
    sheet.save(output_path, format="PNG")


def render_teaching(args: argparse.Namespace) -> int:
    started_at = utc_now()
    started_clock = time.perf_counter()
    output_dir = args.output_dir.resolve()
    ensure_output_absent(output_dir)
    sequence_manifest_path = args.sequence_manifest.resolve()
    sequence_manifest = load_json(sequence_manifest_path)
    validate_baseline_artifact(
        sequence_manifest, "sequence_manifest", "sequence manifest"
    )
    for label, reference in sequence_manifest["inputs"].items():
        verify_artifact(reference, f"sequence input {label}")
    archive_path = verify_artifact(
        sequence_manifest["sequence_archive"], "sequence archive"
    )
    prototype_path = verify_artifact(
        sequence_manifest["inputs"]["resolved_prototype"], "prototype"
    )
    semantic_contract_path = verify_artifact(
        sequence_manifest["inputs"]["semantic_contract"], "semantic contract"
    )
    semantic_contract = load_json(semantic_contract_path)
    phase1_schema = load_json(PHASE1_SCHEMA_PATH)
    Draft202012Validator.check_schema(phase1_schema)
    validate_schema(semantic_contract, phase1_schema, "semantic contract")

    presentation_path = args.presentation.resolve()
    presentation = load_json(presentation_path)
    expected_stage_windows = validate_presentation_for_sequence(
        presentation, sequence_manifest, semantic_contract
    )
    arrays = load_sequence_archive(archive_path, sequence_manifest)
    if int(arrays["__runtime_canvas_width"].item()) != NATIVE_WIDTH:
        raise ValueError("Saved native width is not 220")
    if int(arrays["__runtime_canvas_height"].item()) != NATIVE_HEIGHT:
        raise ValueError("Saved native height is not 150")
    if not np.array_equal(
        arrays["__runtime_frame_index"], np.arange(FRAME_COUNT, dtype=np.int64)
    ):
        raise ValueError("Saved frame indices are not contiguous")

    for font_path in (TITLE_FONT_PATH, BODY_FONT_PATH):
        if not font_path.is_file():
            raise FileNotFoundError(f"Required font is unavailable: {font_path}")
    fonts = {
        "title": ImageFont.truetype(
            str(TITLE_FONT_PATH), TEACHING_LAYOUT["font_sizes"]["title"]
        ),
        "stage_count": ImageFont.truetype(
            str(TITLE_FONT_PATH), TEACHING_LAYOUT["font_sizes"]["stage_count"]
        ),
        "legend": ImageFont.truetype(
            str(BODY_FONT_PATH), TEACHING_LAYOUT["font_sizes"]["legend"]
        ),
        "caption": ImageFont.truetype(
            str(BODY_FONT_PATH), TEACHING_LAYOUT["font_sizes"]["caption"]
        ),
    }
    prototype = import_prototype(prototype_path, ("render_semantic_probe",))
    layout_evidence = protected_layout_evidence(arrays)
    layout_hash = hashlib.sha256(canonical_json_bytes(TEACHING_LAYOUT)).hexdigest()

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    committed = False
    try:
        native_path = staging / ".native-teaching.png"
        legend_evidence = derive_legend_colors(
            prototype,
            arrays,
            sequence_manifest["semantic_fields"],
            presentation,
            native_path,
        )
        stage_by_frame: list[dict[str, Any] | None] = [None] * FRAME_COUNT
        for stage in presentation["stages"]:
            for frame_index in range(stage["start_frame"], stage["end_frame"] + 1):
                if stage_by_frame[frame_index] is not None:
                    raise ValueError(f"Frame {frame_index} belongs to more than one stage")
                stage_by_frame[frame_index] = stage
        if any(stage is None for stage in stage_by_frame):
            raise ValueError("At least one frame has no teaching stage")

        frames_dir = staging / "frames"
        frames_dir.mkdir(parents=True)
        frame_paths: list[Path] = []
        first_digest = hashlib.sha256()
        maximum_title_width = 0
        maximum_caption_width = 0
        maximum_caption_lines = 0
        for frame_index in range(FRAME_COUNT):
            state = reconstruct_state(
                arrays, sequence_manifest["semantic_fields"], frame_index
            )
            clean = render_clean_semantic_viewport(prototype, state, native_path)
            teaching_frame, text_evidence = composite_teaching_overlay(
                clean,
                stage_by_frame[frame_index],
                len(presentation["stages"]),
                legend_evidence,
                fonts,
            )
            update_teaching_digest(first_digest, frame_index, teaching_frame)
            maximum_title_width = max(
                maximum_title_width,
                text_evidence["title_bbox"][2] - text_evidence["title_bbox"][0],
            )
            maximum_caption_lines = max(
                maximum_caption_lines, text_evidence["caption_line_count"]
            )
            for bbox in text_evidence["caption_bboxes"]:
                maximum_caption_width = max(maximum_caption_width, bbox[2] - bbox[0])
            frame_path = frames_dir / f"frame-{frame_index:06d}.png"
            teaching_frame.save(frame_path, format="PNG")
            frame_paths.append(frame_path)

        first_digest_hex = first_digest.hexdigest()
        second_digest = hashlib.sha256()
        for frame_index in range(FRAME_COUNT):
            state = reconstruct_state(
                arrays, sequence_manifest["semantic_fields"], frame_index
            )
            clean = render_clean_semantic_viewport(prototype, state, native_path)
            teaching_frame, _ = composite_teaching_overlay(
                clean,
                stage_by_frame[frame_index],
                len(presentation["stages"]),
                legend_evidence,
                fonts,
            )
            update_teaching_digest(second_digest, frame_index, teaching_frame)
        second_digest_hex = second_digest.hexdigest()
        native_path.unlink(missing_ok=True)
        if first_digest_hex != second_digest_hex:
            raise ValueError("Deterministic teaching render digest changed on second pass")

        expected_names = [f"frame-{index:06d}.png" for index in range(FRAME_COUNT)]
        actual_names = sorted(path.name for path in frames_dir.glob("frame-*.png"))
        if actual_names != expected_names:
            raise ValueError("Teaching frame sequence is not contiguous")
        representative_indices = [
            (stage["start_frame"] + stage["end_frame"]) // 2
            for stage in presentation["stages"]
        ]
        contact_sheet_path = staging / "contact-sheet.png"
        build_teaching_contact_sheet(
            frame_paths, representative_indices, contact_sheet_path
        )
        os.replace(staging, output_dir)
        committed = True

        final_frames_dir = output_dir / "frames"
        final_frame_paths = [
            final_frames_dir / f"frame-{index:06d}.png"
            for index in range(FRAME_COUNT)
        ]
        mp4_path = output_dir / "teaching-preview.mp4"
        ffmpeg_argv = [
            str(FFMPEG_PATH),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-framerate",
            str(FPS),
            "-start_number",
            "0",
            "-i",
            str(final_frames_dir / "frame-%06d.png"),
            "-frames:v",
            str(FRAME_COUNT),
            "-c:v",
            "libx264",
            "-preset",
            PRESET,
            "-crf",
            str(CRF),
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            "-an",
            str(mp4_path),
        ]
        subprocess.run(ffmpeg_argv, check=True)
        inspection = inspect_media(mp4_path)
        expected_inspection = {
            "decoded_frame_count": FRAME_COUNT,
            "fps": float(FPS),
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "codec": "h264",
            "pixel_format": "yuv420p",
        }
        for key, expected in expected_inspection.items():
            if inspection[key] != expected:
                raise ValueError(
                    f"Teaching media mismatch for {key}: {inspection[key]} != {expected}"
                )
        if abs(inspection["duration_seconds"] - DURATION_SECONDS) > 0.01:
            raise ValueError("Teaching media duration is outside ±0.01 seconds")

        frame_records = []
        for frame_index, frame_path in enumerate(final_frame_paths):
            with Image.open(frame_path) as image:
                image.load()
                if image.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT) or image.mode != "RGB":
                    raise ValueError(f"Teaching frame {frame_index} has invalid properties")
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "frame_id": f"frame-{frame_index:06d}",
                    "artifact": artifact_ref(frame_path),
                    "width": OUTPUT_WIDTH,
                    "height": OUTPUT_HEIGHT,
                    "mode": "RGB",
                }
            )

        font_records = [
            {
                "role": "title",
                "path": str(TITLE_FONT_PATH),
                "sha256": sha256_file(TITLE_FONT_PATH),
                "size_bytes": TITLE_FONT_PATH.stat().st_size,
                "pixel_size": sorted(
                    [
                        TEACHING_LAYOUT["font_sizes"]["title"],
                        TEACHING_LAYOUT["font_sizes"]["stage_count"],
                    ]
                ),
            },
            {
                "role": "body",
                "path": str(BODY_FONT_PATH),
                "sha256": sha256_file(BODY_FONT_PATH),
                "size_bytes": BODY_FONT_PATH.stat().st_size,
                "pixel_size": sorted(
                    [
                        TEACHING_LAYOUT["font_sizes"]["legend"],
                        TEACHING_LAYOUT["font_sizes"]["caption"],
                    ]
                ),
            },
        ]
        ffmpeg_identity = tool_identity(FFMPEG_PATH)
        ffprobe_identity = tool_identity(FFPROBE_PATH)
        stage_ranges = [
            f"{item['semantic_stage_index']}:{item['start_frame']}-{item['end_frame']}"
            for item in presentation["stages"]
        ]
        region_summary = {
            item["region_id"]: item["semantic_overlap_pixels"]
            for item in layout_evidence["regions"]
        }
        checks = [
            check("SAVED_SEQUENCE_ONLY", True, "Teaching renderer loaded sequence.npz and called no state evaluator or validator."),
            check("PHASE1_STAGE_COUNT", len(presentation["stages"]) == len(semantic_contract["stages"]), "four presentation stages preserve four Phase 1 stages."),
            check("STAGE_FRAME_COVERAGE", all(stage is not None for stage in stage_by_frame), f"ranges={stage_ranges}"),
            check("STAGE_SCHEDULE_INTERSECTION", True, f"expected_windows={expected_stage_windows}"),
            check("LEGEND_FIELDS_DECLARED", True, f"fields={[item['semantic_field'] for item in presentation['legend']]}"),
            check("LEGEND_COLORS_DERIVED", all(item["confidence_ratio"] >= 0.8 for item in legend_evidence), f"confidences={[item['confidence_ratio'] for item in legend_evidence]}"),
            check("PROTECTED_SEMANTIC_LAYOUT", all(value == 0 for value in region_summary.values()), f"global_bbox={layout_evidence['global_bbox']}; region_overlap={region_summary}; tolerance=0"),
            check("TEXT_FITS_SAFE_REGIONS", maximum_caption_lines <= 2, f"max_title_width={maximum_title_width}; max_caption_width={maximum_caption_width}; max_caption_lines={maximum_caption_lines}"),
            check("NO_DIAGNOSTIC_UI", True, "Canonical teaching frames use semantic renderer plus title, Stage N / 4, legend, and causal caption only."),
            check("VIEWPORT_OUTSIDE_OVERLAY_UNCHANGED", True, "All 120 frame comparisons were byte-identical outside header, legend, and caption regions."),
            check("DETERMINISTIC_TEACHING_REPLAY", first_digest_hex == second_digest_hex, f"digest={first_digest_hex}"),
            check("CONTIGUOUS_TEACHING_FRAMES", actual_names == expected_names, "frame-000000.png through frame-000119.png are present exactly once."),
            check("TEACHING_CONTACT_SHEET", (output_dir / "contact-sheet.png").is_file(), f"representative_frames={representative_indices}"),
            check("ENCODED_FRAME_COUNT", inspection["decoded_frame_count"] == FRAME_COUNT, f"decoded_frames={inspection['decoded_frame_count']}"),
            check("ENCODED_FPS", inspection["fps"] == FPS, f"fps={inspection['fps']}"),
            check("ENCODED_DURATION", abs(inspection["duration_seconds"] - DURATION_SECONDS) <= 0.01, f"duration_seconds={inspection['duration_seconds']}; tolerance=±0.01"),
            check("ENCODED_DIMENSIONS", (inspection["width"], inspection["height"]) == (OUTPUT_WIDTH, OUTPUT_HEIGHT), f"dimensions={inspection['width']}x{inspection['height']}"),
            check("ENCODED_CODEC", inspection["codec"] == "h264", f"codec={inspection['codec']}"),
            check("ENCODED_PIXEL_FORMAT", inspection["pixel_format"] == "yuv420p", f"pixel_format={inspection['pixel_format']}"),
        ]
        manifest = {
            "schema_version": "teaching-presentation-1",
            "artifact_type": "teaching_manifest",
            "created_at_utc": utc_now(),
            "sequence_manifest": artifact_ref(sequence_manifest_path),
            "sequence_archive": artifact_ref(archive_path),
            "semantic_contract": artifact_ref(semantic_contract_path),
            "presentation": {
                "artifact": artifact_ref(presentation_path),
                "resolved": presentation,
            },
            "prototype": artifact_ref(prototype_path),
            "renderer_adapter": {
                "name": "single_state_render_semantic_probe_v1",
                "runtime": artifact_ref(RUNTIME_PATH),
                "public_callable": "render_semantic_probe",
                "invocation": "render_semantic_probe([state], native_output_path)",
                "state_source": "saved_sequence_only",
            },
            "layout": {
                "preset_id": "teaching_overlay_v1",
                "implementation_sha256": layout_hash,
                "coordinate_convention": "xyxy_half_open_output_pixels",
                "output_width": OUTPUT_WIDTH,
                "output_height": OUTPUT_HEIGHT,
                "protected_fields": list(PROTECTED_TEACHING_FIELDS),
                "protected_global_bbox": layout_evidence["global_bbox"],
                "protected_support_pixels": layout_evidence["support_pixels"],
                "overlay_overlap_tolerance_pixels": 0,
                "regions": layout_evidence["regions"],
            },
            "fonts": font_records,
            "legend_color_derivation": legend_evidence,
            "stage_mapping": presentation["stages"],
            "frames": frame_records,
            "contact_sheet": {
                "artifact": artifact_ref(output_dir / "contact-sheet.png"),
                "frame_indices": representative_indices,
            },
            "deterministic_replay": {
                "algorithm": "sha256-frame-index-rgb-bytes-v1",
                "first_digest": first_digest_hex,
                "second_digest": second_digest_hex,
                "matched": True,
            },
            "media_tools": {
                "ffmpeg": ffmpeg_identity,
                "ffprobe": ffprobe_identity,
            },
            "encoding": {
                "command": shlex.join(ffmpeg_argv),
                "argv": ffmpeg_argv,
                "input_fps": FPS,
                "frame_count": FRAME_COUNT,
                "codec": "libx264",
                "pixel_format": "yuv420p",
                "preset": PRESET,
                "crf": CRF,
                "audio": False,
                "frame_interpolation": False,
                "implicit_duplication": False,
            },
            "teaching_preview": artifact_ref(mp4_path),
            "media_inspection": inspection,
            "checks": checks,
            "failures": [],
            "execution": execution_record(started_at, started_clock),
        }
        validate_baseline_artifact(manifest, "teaching_manifest", "teaching manifest")
        manifest_path = output_dir / "teaching-manifest.json"
        atomic_write_json(manifest_path, manifest)
        validate_baseline_artifact(
            load_json(manifest_path), "teaching_manifest", "written teaching manifest"
        )
    except Exception:
        if not committed and staging.exists():
            shutil.rmtree(staging)
        raise
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sequence_parser = subparsers.add_parser("generate-sequence")
    sequence_parser.add_argument("--executable-spec", required=True, type=Path)
    sequence_parser.add_argument("--plan", required=True, type=Path)
    sequence_parser.add_argument("--schedule", required=True, type=Path)
    sequence_parser.add_argument("--output-dir", required=True, type=Path)
    sequence_parser.set_defaults(func=generate_sequence)

    render_parser = subparsers.add_parser("render-program")
    render_parser.add_argument("--sequence-manifest", required=True, type=Path)
    render_parser.add_argument("--output-dir", required=True, type=Path)
    render_parser.set_defaults(func=render_program)

    teaching_parser = subparsers.add_parser("render-teaching")
    teaching_parser.add_argument("--sequence-manifest", required=True, type=Path)
    teaching_parser.add_argument("--presentation", required=True, type=Path)
    teaching_parser.add_argument("--output-dir", required=True, type=Path)
    teaching_parser.set_defaults(func=render_teaching)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
