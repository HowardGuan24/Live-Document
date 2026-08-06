#!/usr/bin/env python3
"""Deterministic Stage 5 Phase 3 semantic-sequence runtime.

The runtime validates a frozen Phase 2 lineage, materializes an explicit
schedule, evaluates the selected prototype twice, and writes exactly one
deterministic semantic archive plus its manifest. It performs no rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import shutil
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator


RUNTIME_VERSION = "stage5-phase3-runtime-1"
RUNTIME_PREFIX = "__runtime_"
RUNTIME_PATH = Path(__file__).resolve()
SCHEMA_PATH = RUNTIME_PATH.with_name("schema.json")
STAGE5_ROOT = RUNTIME_PATH.parents[2]
REPOSITORY_ROOT = RUNTIME_PATH.parents[5]
PHASE1_SCHEMA_PATH = STAGE5_ROOT / "workflow" / "phase1" / "schema.json"
PHASE2_SCHEMA_PATH = STAGE5_ROOT / "workflow" / "phase2" / "schema.json"

INPUT_ARTIFACTS = (
    "semantic-contract.json",
    "executable-spec.json",
    "plan.json",
    "prototype.py",
    "schedule.json",
)
OUTPUT_ARTIFACTS = ("sequence.npz", "sequence-manifest.json")

_PHASE3_SCHEMA: dict[str, Any] | None = None
_PHASE2_SCHEMA: dict[str, Any] | None = None


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
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
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


def artifact_ref(path: Path, *, path_value: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"artifact does not exist: {path}")
    return {
        "path": path_value or recorded_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def ensure_repository_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository root: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def resolve_recorded_path(raw: str, label: str) -> Path:
    path = Path(raw)
    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    return ensure_repository_file(candidate, label)


def verify_reference(reference: Mapping[str, Any], label: str) -> Path:
    path = resolve_recorded_path(str(reference["path"]), label)
    actual_hash = sha256_file(path)
    if actual_hash != reference["sha256"]:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {reference['sha256']}, got {actual_hash}"
        )
    if "size_bytes" in reference and path.stat().st_size != reference["size_bytes"]:
        raise ValueError(f"{label} size mismatch")
    return path


def load_phase3_schema() -> dict[str, Any]:
    global _PHASE3_SCHEMA
    if _PHASE3_SCHEMA is None:
        _PHASE3_SCHEMA = load_json(SCHEMA_PATH)
        Draft202012Validator.check_schema(_PHASE3_SCHEMA)
    return _PHASE3_SCHEMA


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


def validate_definition(
    value: Any, schema: dict[str, Any], definition: str, label: str
) -> None:
    validate_schema(
        value,
        {
            "$schema": schema["$schema"],
            "$defs": schema["$defs"],
            "$ref": f"#/$defs/{definition}",
        },
        label,
    )


def validate_phase2(value: Any, definition: str, label: str) -> None:
    validate_definition(value, load_phase2_schema(), definition, label)


def merge_selected_config(plan: Mapping[str, Any], selected_id: str) -> dict[str, Any]:
    selected = next(
        (item for item in plan["candidates"] if item["candidate_id"] == selected_id),
        None,
    )
    if selected is None:
        raise ValueError(f"selected candidate is absent from plan: {selected_id}")
    complete = dict(plan["fixed_parameters"])
    complete.update(selected["parameter_overrides"])
    return complete


def import_prototype(path: Path):
    name = f"live_document_phase3_{sha256_file(path)[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import prototype: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("build_static_scene", "evaluate_state", "validate_probe")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(f"prototype is missing required callables: {missing}")
    return module


def validate_lineage(
    *,
    semantic_contract_path: Path,
    executable_spec_path: Path,
    plan_path: Path,
    prototype_path: Path,
    schedule_path: Path,
) -> dict[str, Any]:
    semantic_path = ensure_repository_file(semantic_contract_path, "semantic contract")
    spec_path = ensure_repository_file(executable_spec_path, "executable spec")
    resolved_plan_path = ensure_repository_file(plan_path, "Phase 2 plan")
    resolved_prototype_path = ensure_repository_file(prototype_path, "prototype")
    resolved_schedule_path = ensure_repository_file(schedule_path, "schedule")

    semantic_contract = load_json(semantic_path)
    executable_spec = load_json(spec_path)
    plan = load_json(resolved_plan_path)
    schedule = load_json(resolved_schedule_path)

    phase1_schema = load_json(PHASE1_SCHEMA_PATH)
    Draft202012Validator.check_schema(phase1_schema)
    validate_schema(semantic_contract, phase1_schema, "semantic contract")
    validate_phase2(executable_spec, "executable_spec", "executable spec")
    validate_phase2(plan, "plan", "Phase 2 plan")
    validate_definition(schedule, load_phase3_schema(), "schedule", "schedule")

    semantic_binding = executable_spec["semantic_contract_binding"]
    bound_semantic_path = resolve_recorded_path(
        semantic_binding["path"], "bound semantic contract"
    )
    if bound_semantic_path != semantic_path:
        raise ValueError("explicit semantic contract is not the frozen bound artifact")
    if sha256_file(semantic_path) != semantic_binding["sha256"]:
        raise ValueError("semantic-contract binding SHA-256 mismatch")

    prototype_hash = sha256_file(resolved_prototype_path)
    if prototype_hash != executable_spec["prototype_sha256"]:
        raise ValueError("prototype SHA-256 mismatch against executable spec")
    raw_plan_prototype = Path(plan["prototype_entrypoint"])
    plan_prototype = (
        raw_plan_prototype.resolve()
        if raw_plan_prototype.is_absolute()
        else (resolved_plan_path.parent / raw_plan_prototype).resolve()
    )
    if ensure_repository_file(plan_prototype, "plan prototype") != resolved_prototype_path:
        raise ValueError("explicit prototype does not match the plan entrypoint")

    probe_path = verify_reference(executable_spec["source_probe_result"], "probe result")
    selection_path = verify_reference(executable_spec["source_selection"], "selection")
    probe_result = load_json(probe_path)
    selection = load_json(selection_path)
    validate_phase2(probe_result, "probe_result", "probe result")
    validate_phase2(selection, "selection", "selection")

    if not all(executable_spec["freeze_checks"].values()):
        raise ValueError("executable spec contains a failed Freeze check")
    attempt_id = executable_spec["implementation_attempt_id"]
    selected_id = executable_spec["selected_candidate_id"]
    if plan["implementation_attempt_id"] != attempt_id:
        raise ValueError("plan implementation attempt does not match executable spec")
    if probe_result["implementation_attempt_id"] != attempt_id:
        raise ValueError("probe implementation attempt does not match executable spec")
    if selection["implementation_attempt_id"] != attempt_id:
        raise ValueError("selection implementation attempt does not match executable spec")
    if probe_result["candidate_id"] != selected_id:
        raise ValueError("probe candidate does not match executable spec")
    if selection.get("selected_candidate_id") != selected_id:
        raise ValueError("selection candidate does not match executable spec")
    if selection["selection_status"] != "selected":
        raise ValueError("Phase 2 selection is not selected")
    if probe_result["machine_gate_status"] != "passed":
        raise ValueError("selected probe did not pass its machine gate")
    accepted = [
        item for item in selection["reviewed_candidates"] if item["decision"] == "accept"
    ]
    if len(accepted) != 1 or accepted[0]["candidate_id"] != selected_id:
        raise ValueError("selection does not contain one exact accepted candidate")

    frozen_config = executable_spec["frozen_config"]
    if merge_selected_config(plan, selected_id) != frozen_config:
        raise ValueError("plan-selected complete config does not equal frozen config")
    if probe_result["complete_config"] != frozen_config:
        raise ValueError("probe complete config does not equal frozen config")
    if probe_result["execution"]["prototype_sha256"] != prototype_hash:
        raise ValueError("probe prototype binding does not match explicit prototype")

    state_contract = plan["implementation"]["semantic_state_contract"]
    field_contracts = state_contract["fields"]
    field_names = [item["name"] for item in field_contracts]
    if len(field_names) != len(set(field_names)):
        raise ValueError("semantic field names are not unique")
    invariant_contracts = state_contract["invariants"]
    invariant_ids = [item["check_id"] for item in invariant_contracts]
    if len(invariant_ids) != len(set(invariant_ids)):
        raise ValueError("invariant IDs are not unique")
    if [item["check_id"] for item in probe_result["machine_checks"]] != invariant_ids:
        raise ValueError("probe checks do not exactly cover declared invariant IDs")
    unknown_invariant_fields = sorted(
        {
            name
            for invariant in invariant_contracts
            for name in invariant["fields"]
            if name not in field_names
        }
    )
    if unknown_invariant_fields:
        raise ValueError(f"invariants reference undeclared fields: {unknown_invariant_fields}")

    approved_anchors = [
        {"anchor_id": item["sample_id"], "progress_values": item["progress_values"]}
        for item in plan["implementation"]["probe_samples"]
    ]
    if schedule["anchors"] != approved_anchors:
        raise ValueError("schedule anchors do not exactly equal approved probe samples")
    if float(schedule["duration_seconds"]) != float(semantic_contract["duration_seconds"]):
        raise ValueError("schedule duration does not equal semantic-contract duration")
    if schedule["frame_count"] != round(schedule["duration_seconds"] * schedule["fps"]):
        raise ValueError("frame count does not exactly equal duration multiplied by FPS")
    if schedule["start_hold_frames"] < 1:
        raise ValueError("schedule must emit the exact initial anchor at frame zero")

    expected_pairs = [
        (approved_anchors[index]["anchor_id"], approved_anchors[index + 1]["anchor_id"])
        for index in range(len(approved_anchors) - 1)
    ]
    actual_pairs = [
        (segment["from_anchor"], segment["to_anchor"])
        for segment in schedule["segments"]
    ]
    if actual_pairs != expected_pairs:
        raise ValueError("schedule segments do not match every adjacent anchor exactly")
    emitted = (
        schedule["start_hold_frames"]
        + sum(segment["transition_frames"] for segment in schedule["segments"])
        + schedule["end_hold_frames"]
    )
    if emitted != schedule["frame_count"]:
        raise ValueError(f"schedule emits {emitted} frames, expected {schedule['frame_count']}")

    progress_key_sets = [set(anchor["progress_values"]) for anchor in approved_anchors]
    if any(keys != progress_key_sets[0] for keys in progress_key_sets[1:]):
        raise ValueError("schedule anchors do not share one progress-key set")
    plan_progress = {
        item["progress_variable"]
        for item in plan["implementation"]["progress_realization"]
    }
    if progress_key_sets[0] != plan_progress:
        raise ValueError("schedule progress keys do not match the Phase 2 plan")
    if not plan_progress.issubset(field_names):
        raise ValueError("progress variables are not declared semantic fields")

    return {
        "semantic_contract": semantic_contract,
        "semantic_path": semantic_path,
        "executable_spec": executable_spec,
        "executable_spec_path": spec_path,
        "plan": plan,
        "plan_path": resolved_plan_path,
        "prototype_path": resolved_prototype_path,
        "schedule": schedule,
        "schedule_path": resolved_schedule_path,
        "probe_result": probe_result,
        "probe_path": probe_path,
        "selection": selection,
        "selection_path": selection_path,
        "frozen_config": frozen_config,
        "field_contracts": field_contracts,
        "field_names": field_names,
        "invariant_ids": invariant_ids,
    }


def eased_value(value: float, easing: str) -> float:
    if easing == "linear":
        return value
    if easing == "smoothstep":
        return value * value * (3.0 - 2.0 * value)
    raise ValueError(f"unsupported easing: {easing}")


def materialize_schedule(schedule: Mapping[str, Any]) -> list[dict[str, Any]]:
    anchors = {item["anchor_id"]: item["progress_values"] for item in schedule["anchors"]}
    first_id = schedule["anchors"][0]["anchor_id"]
    final_id = schedule["anchors"][-1]["anchor_id"]
    frames: list[dict[str, Any]] = []

    def append_frame(
        progress: Mapping[str, Any],
        from_anchor: str,
        to_anchor: str,
        interpolation: float,
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
            interpolation = 1.0 if step == count else eased_value(step / count, segment["easing"])
            progress = (
                dict(destination)
                if step == count
                else {
                    key: float(source[key])
                    + (float(destination[key]) - float(source[key])) * interpolation
                    for key in source
                }
            )
            append_frame(progress, segment["from_anchor"], segment["to_anchor"], interpolation)
    for _ in range(schedule["end_hold_frames"]):
        append_frame(anchors[final_id], final_id, final_id, 1.0)
    if len(frames) != schedule["frame_count"]:
        raise ValueError(f"materialized {len(frames)} frames")
    return frames


def numeric_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object or not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{label} must be a non-object numeric or boolean value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains a non-finite value")
    return array


def evaluate_pass(
    *,
    prototype: Any,
    static_scene: Any,
    frames: list[dict[str, Any]],
    frozen_config: dict[str, Any],
    field_names: list[str],
    expected_descriptors: dict[str, tuple[str, tuple[int, ...]]] | None = None,
    expected_runtime_descriptors: dict[str, tuple[str, tuple[int, ...]]] | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[str, tuple[int, ...]]],
    dict[str, tuple[str, tuple[int, ...]]],
]:
    declared = set(field_names)
    states: list[dict[str, Any]] = []
    descriptors: dict[str, tuple[str, tuple[int, ...]]] = {}
    runtime_descriptors: dict[str, tuple[str, tuple[int, ...]]] = {}
    runtime_keys: set[str] | None = None
    for frame in frames:
        raw_state = prototype.evaluate_state(
            static_scene, frame["progress_values"], frozen_config
        )
        if not isinstance(raw_state, dict):
            raise TypeError("evaluate_state must return a dictionary")
        state = dict(raw_state)
        missing = sorted(declared - set(state))
        if missing:
            raise ValueError(f"{frame['frame_id']} omitted semantic fields: {missing}")
        extras = set(state) - declared
        if runtime_keys is None:
            runtime_keys = extras
        elif extras != runtime_keys:
            raise ValueError(f"{frame['frame_id']} changed undeclared state-field keys")
        for field_name in field_names:
            if field_name.startswith(RUNTIME_PREFIX):
                raise ValueError(f"semantic field uses reserved prefix: {field_name}")
            array = numeric_array(state[field_name], f"{frame['frame_id']} {field_name}")
            state[field_name] = array.copy()
            descriptor = (array.dtype.str, tuple(array.shape))
            if field_name not in descriptors:
                descriptors[field_name] = descriptor
            elif descriptors[field_name] != descriptor:
                raise ValueError(f"{field_name} changed dtype or shape at {frame['frame_id']}")
        for key in sorted(extras):
            array = numeric_array(state[key], f"{frame['frame_id']} auxiliary field {key}")
            state[key] = array.copy()
            descriptor = (array.dtype.str, tuple(array.shape))
            if key not in runtime_descriptors:
                runtime_descriptors[key] = descriptor
            elif runtime_descriptors[key] != descriptor:
                raise ValueError(f"auxiliary field {key} changed dtype or shape")
        for progress_key, progress_value in frame["progress_values"].items():
            if float(np.asarray(state[progress_key])) != float(progress_value):
                raise ValueError(f"{frame['frame_id']} did not preserve {progress_key} exactly")
        state["_probe_sample_id"] = frame["frame_id"]
        states.append(state)

    if expected_descriptors is not None and descriptors != expected_descriptors:
        raise ValueError("replay changed semantic field descriptors")
    if (
        expected_runtime_descriptors is not None
        and runtime_descriptors != expected_runtime_descriptors
    ):
        raise ValueError("replay changed auxiliary field descriptors")
    return states, descriptors, runtime_descriptors


def state_digest(states: list[dict[str, Any]], field_names: list[str]) -> str:
    digest = hashlib.sha256()
    for state in states:
        for field_name in field_names:
            value = np.asarray(state[field_name])
            digest.update(field_name.encode("utf-8") + b"\0")
            digest.update(value.dtype.str.encode("ascii") + b"\0")
            digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
            digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def normalize_invariant_results(raw: Any, declared_ids: list[str]) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("validate_probe returned no invariant results")
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
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("validate_probe returned a non-object invariant result")
        for required in ("check_id", "claim", "passed", "evidence"):
            if required not in item:
                raise ValueError(f"invariant result omitted {required}")
        if item["passed"] is not True:
            raise ValueError(
                f"full-sequence invariant failed: {item['check_id']}: {item['evidence']}"
            )
        normalized.append({key: item[key] for key in allowed if key in item})
    if [item["check_id"] for item in normalized] != declared_ids:
        raise ValueError("full-sequence checks do not exactly cover declared invariant IDs")
    return normalized


def build_archive_arrays(
    states: list[dict[str, Any]],
    field_names: list[str],
    runtime_keys: list[str],
    frames: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    arrays = {
        field_name: np.stack([np.asarray(state[field_name]) for state in states], axis=0)
        for field_name in field_names
    }
    arrays["__runtime_frame_index"] = np.arange(len(states), dtype=np.int64)
    arrays["__runtime_pts_seconds"] = np.asarray(
        [frame["presentation_time_seconds"] for frame in frames], dtype=np.float64
    )
    arrays["__runtime_semantic_time"] = np.asarray(
        [frame["semantic_normalized_time"] for frame in frames], dtype=np.float64
    )
    for key in runtime_keys:
        arrays[f"__runtime_state_{key}"] = np.stack(
            [np.asarray(state[key]) for state in states], axis=0
        )
    return arrays


def deterministic_npz_write(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compresslevel=9)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def verify_archive(
    path: Path,
    expected: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        if expected is not None and set(archive.files) != set(expected):
            raise ValueError("sequence archive key set changed during serialization")
        for name in archive.files:
            value = archive[name]
            if value.dtype == object:
                raise TypeError(f"archive field has object dtype: {name}")
            if expected is not None:
                wanted = expected[name]
                if value.dtype != wanted.dtype or value.shape != wanted.shape:
                    raise ValueError(f"archive descriptor mismatch: {name}")
                if not np.array_equal(value, wanted, equal_nan=True):
                    raise ValueError(f"archive values changed: {name}")
            loaded[name] = value.copy()
    return loaded


def runtime_record() -> dict[str, Any]:
    return {
        "runtime_version": RUNTIME_VERSION,
        "runtime_sha256": sha256_file(RUNTIME_PATH),
        "interpreter": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "packages": {
            "numpy": importlib.metadata.version("numpy"),
            "jsonschema": importlib.metadata.version("jsonschema"),
        },
    }


def factual_check(check_id: str, evidence: str) -> dict[str, Any]:
    return {"check_id": check_id, "passed": True, "evidence": evidence}


def build_manifest(
    *,
    upstream: Mapping[str, Any],
    frames: list[dict[str, Any]],
    arrays: Mapping[str, np.ndarray],
    archive_path: Path,
    descriptors: Mapping[str, tuple[str, tuple[int, ...]]],
    runtime_descriptors: Mapping[str, tuple[str, tuple[int, ...]]],
    invariant_results: list[dict[str, Any]],
    first_digest: str,
    second_digest: str,
) -> dict[str, Any]:
    contract_by_name = {item["name"]: item for item in upstream["field_contracts"]}
    semantic_fields = []
    for name in upstream["field_names"]:
        dtype, state_shape = descriptors[name]
        semantic_fields.append(
            {
                "name": name,
                "kind": contract_by_name[name]["kind"],
                "dtype": dtype,
                "state_shape": list(state_shape),
                "archive_shape": list(arrays[name].shape),
                "consumers": contract_by_name[name]["consumers"],
                "content_sha256": array_digest(arrays[name]),
            }
        )
    runtime_fields = [
        {
            "name": "__runtime_frame_index",
            "source_state_key": None,
            "dtype": arrays["__runtime_frame_index"].dtype.str,
            "archive_shape": list(arrays["__runtime_frame_index"].shape),
            "content_sha256": array_digest(arrays["__runtime_frame_index"]),
        },
        {
            "name": "__runtime_pts_seconds",
            "source_state_key": None,
            "dtype": arrays["__runtime_pts_seconds"].dtype.str,
            "archive_shape": list(arrays["__runtime_pts_seconds"].shape),
            "content_sha256": array_digest(arrays["__runtime_pts_seconds"]),
        },
        {
            "name": "__runtime_semantic_time",
            "source_state_key": None,
            "dtype": arrays["__runtime_semantic_time"].dtype.str,
            "archive_shape": list(arrays["__runtime_semantic_time"].shape),
            "content_sha256": array_digest(arrays["__runtime_semantic_time"]),
        },
    ]
    for key in sorted(runtime_descriptors):
        archive_name = f"__runtime_state_{key}"
        runtime_fields.append(
            {
                "name": archive_name,
                "source_state_key": key,
                "dtype": arrays[archive_name].dtype.str,
                "archive_shape": list(arrays[archive_name].shape),
                "content_sha256": array_digest(arrays[archive_name]),
            }
        )

    schedule = upstream["schedule"]
    return {
        "schema_version": "stage5-phase3-sequence-manifest-1",
        "artifact_type": "semantic_sequence",
        "phase": "phase3",
        "source": {
            "semantic_contract": artifact_ref(upstream["semantic_path"]),
            "executable_spec": artifact_ref(upstream["executable_spec_path"]),
            "supplemental_plan": artifact_ref(upstream["plan_path"]),
            "prototype": artifact_ref(upstream["prototype_path"]),
            "source_probe_result": artifact_ref(upstream["probe_path"]),
            "source_selection": artifact_ref(upstream["selection_path"]),
            "schedule": artifact_ref(upstream["schedule_path"]),
        },
        "lineage": {
            "implementation_attempt_id": upstream["executable_spec"]["implementation_attempt_id"],
            "selected_candidate_id": upstream["executable_spec"]["selected_candidate_id"],
            "plan_binding_rule": "explicit_legacy_plan_cross_check_v1",
            "prototype_resolution_rule": "explicit_path_hash_and_plan_match_v1",
        },
        "frozen_config": upstream["frozen_config"],
        "timeline": {
            "authority": "explicit_schedule_artifact",
            "fps": schedule["fps"],
            "frame_count": schedule["frame_count"],
            "duration_seconds": schedule["duration_seconds"],
            "start_hold_frames": schedule["start_hold_frames"],
            "anchors": schedule["anchors"],
            "segments": schedule["segments"],
            "end_hold_frames": schedule["end_hold_frames"],
        },
        "semantic_fields": semantic_fields,
        "runtime_fields": runtime_fields,
        "frames": frames,
        "sequence_archive": artifact_ref(archive_path, path_value="sequence.npz"),
        "replay": {
            "algorithm": "sha256-field-name-dtype-shape-c-bytes-v1",
            "first_digest": first_digest,
            "second_digest": second_digest,
            "matched": True,
        },
        "invariant_results": invariant_results,
        "checks": [
            factual_check("FROZEN_LINEAGE_VERIFIED", "All explicit and frozen Phase 2 bindings matched."),
            factual_check("TIMELINE_EXPLICIT", f"frames={schedule['frame_count']}; fps={schedule['fps']}; duration={schedule['duration_seconds']}"),
            factual_check("SEMANTIC_FIELDS_COMPLETE", f"fields={len(semantic_fields)}"),
            factual_check("INVARIANTS_PASSED", f"checks={len(invariant_results)}"),
            factual_check("ARCHIVE_ROUND_TRIP", f"keys={len(arrays)}; allow_pickle=False"),
            factual_check("DETERMINISTIC_REPLAY", f"digest={first_digest}"),
            factual_check("PHASE_BOUNDARY_PRESERVED", "No renderer or Phase 4-6 artifact was invoked or emitted."),
        ],
        "runtime": runtime_record(),
        "status": "complete",
    }


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    validate_schema(dict(manifest), load_phase3_schema(), "Phase 3 sequence manifest")


def verify_manifest_artifacts(manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    for name, reference in manifest["source"].items():
        verify_reference(reference, f"manifest source {name}")
    archive_ref = manifest["sequence_archive"]
    raw_archive = Path(archive_ref["path"])
    archive_path = (
        raw_archive.resolve()
        if raw_archive.is_absolute()
        else (manifest_path.parent / raw_archive).resolve()
    )
    if not archive_path.is_file():
        raise FileNotFoundError(f"manifest archive does not exist: {archive_path}")
    if sha256_file(archive_path) != archive_ref["sha256"]:
        raise ValueError("manifest archive SHA-256 mismatch")
    if archive_path.stat().st_size != archive_ref["size_bytes"]:
        raise ValueError("manifest archive size mismatch")
    arrays = verify_archive(archive_path)
    descriptors = [*manifest["semantic_fields"], *manifest["runtime_fields"]]
    if set(arrays) != {item["name"] for item in descriptors}:
        raise ValueError("manifest/archive key sets differ")
    for descriptor in descriptors:
        array = arrays[descriptor["name"]]
        if array.dtype.str != descriptor["dtype"]:
            raise ValueError(f"manifest dtype mismatch: {descriptor['name']}")
        if list(array.shape) != descriptor["archive_shape"]:
            raise ValueError(f"manifest shape mismatch: {descriptor['name']}")
        if array_digest(array) != descriptor["content_sha256"]:
            raise ValueError(f"manifest content digest mismatch: {descriptor['name']}")


def failure_record(exc: BaseException, inputs: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": "stage5-phase3-failure-1",
        "phase": "phase3",
        "status": "failed",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "inputs": {name: str(path) for name, path in inputs.items()},
        "traceback": traceback.format_exc(),
    }


def build_full_semantic_sequence(
    *,
    semantic_contract: Path,
    executable_spec: Path,
    plan: Path,
    prototype: Path,
    schedule: Path,
    output_directory: Path,
) -> None:
    output_dir = output_directory.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output already exists and will not be reused: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    inputs = {
        "semantic_contract": semantic_contract,
        "executable_spec": executable_spec,
        "plan": plan,
        "prototype": prototype,
        "schedule": schedule,
    }
    try:
        upstream = validate_lineage(
            semantic_contract_path=semantic_contract,
            executable_spec_path=executable_spec,
            plan_path=plan,
            prototype_path=prototype,
            schedule_path=schedule,
        )
        frames = materialize_schedule(upstream["schedule"])
        module = import_prototype(upstream["prototype_path"])
        static_scene = module.build_static_scene(upstream["frozen_config"])
        first_states, descriptors, runtime_descriptors = evaluate_pass(
            prototype=module,
            static_scene=static_scene,
            frames=frames,
            frozen_config=upstream["frozen_config"],
            field_names=upstream["field_names"],
        )
        first_invariants = normalize_invariant_results(
            module.validate_probe(
                first_states, upstream["semantic_contract"], upstream["frozen_config"]
            ),
            upstream["invariant_ids"],
        )
        first_digest = state_digest(first_states, upstream["field_names"])

        second_states, _, _ = evaluate_pass(
            prototype=module,
            static_scene=static_scene,
            frames=frames,
            frozen_config=upstream["frozen_config"],
            field_names=upstream["field_names"],
            expected_descriptors=descriptors,
            expected_runtime_descriptors=runtime_descriptors,
        )
        second_invariants = normalize_invariant_results(
            module.validate_probe(
                second_states, upstream["semantic_contract"], upstream["frozen_config"]
            ),
            upstream["invariant_ids"],
        )
        second_digest = state_digest(second_states, upstream["field_names"])
        if first_digest != second_digest:
            raise ValueError("second replay changed the semantic state digest")
        if first_invariants != second_invariants:
            raise ValueError("second replay changed invariant evidence")

        arrays = build_archive_arrays(
            first_states,
            upstream["field_names"],
            sorted(runtime_descriptors),
            frames,
        )
        archive_path = output_dir / "sequence.npz"
        deterministic_npz_write(archive_path, arrays)
        loaded = verify_archive(archive_path, arrays)
        manifest = build_manifest(
            upstream=upstream,
            frames=frames,
            arrays=loaded,
            archive_path=archive_path,
            descriptors=descriptors,
            runtime_descriptors=runtime_descriptors,
            invariant_results=first_invariants,
            first_digest=first_digest,
            second_digest=second_digest,
        )
        validate_manifest(manifest)
        manifest_path = output_dir / "sequence-manifest.json"
        atomic_write_json(manifest_path, manifest)
        written = load_json(manifest_path)
        validate_manifest(written)
        verify_manifest_artifacts(manifest_path, written)
    except Exception as exc:
        (output_dir / "sequence-manifest.json").unlink(missing_ok=True)
        (output_dir / "sequence.npz").unlink(missing_ok=True)
        for temporary in output_dir.glob(".*.tmp"):
            temporary.unlink(missing_ok=True)
        atomic_write_json(output_dir / "failure.json", failure_record(exc, inputs))
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("manifest", type=Path)

    build = subparsers.add_parser("build-sequence")
    build.add_argument("--semantic-contract", type=Path, required=True)
    build.add_argument("--executable-spec", type=Path, required=True)
    build.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Explicit Phase 2 plan required to recover legacy field/invariant lineage.",
    )
    build.add_argument("--prototype", type=Path, required=True)
    build.add_argument("--schedule", type=Path, required=True)
    build.add_argument("--output-directory", "--output-dir", dest="output_directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-manifest":
            manifest_path = args.manifest.resolve()
            manifest = load_json(manifest_path)
            validate_manifest(manifest)
            verify_manifest_artifacts(manifest_path, manifest)
            return 0
        if args.command == "build-sequence":
            build_full_semantic_sequence(
                semantic_contract=args.semantic_contract,
                executable_spec=args.executable_spec,
                plan=args.plan,
                prototype=args.prototype,
                schedule=args.schedule,
                output_directory=args.output_directory,
            )
            return 0
        raise AssertionError(f"unhandled command: {args.command}")
    except Exception as exc:
        print(f"phase3 runtime error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
