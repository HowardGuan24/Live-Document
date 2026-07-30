"""Validation helpers for Stage 2 model-free data contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FRAMEWORK_ROOT = Path(__file__).resolve().parent
SCHEMA_ROOT = FRAMEWORK_ROOT / "schemas"
SEMANTIC_LAYER_TYPES = {
    "hard_boundary",
    "region",
    "scalar_field",
    "vector_field",
    "height_or_normal",
    "object_identity",
    "annotation",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, fixture_root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(fixture_root.resolve()).as_posix(),
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _require_keys(
    value: dict[str, Any], required: set[str], *, label: str
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label} missing fields: {missing}")


def _ordered_ids(
    values: list[dict[str, Any]],
    *,
    id_field: str,
    label: str,
) -> list[str]:
    ids = [str(item[id_field]) for item in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} IDs must be unique")
    orders = [int(item["order"]) for item in values]
    if orders != list(range(len(values))):
        raise ValueError(f"{label} orders must be contiguous from zero")
    return ids


def validate_schema_documents() -> list[dict[str, Any]]:
    required = {
        "concept_spec.schema.json",
        "sequence_spec.schema.json",
        "semantic_layers.schema.json",
        "fixture_manifest.schema.json",
        "score_record.schema.json",
    }
    present = {path.name for path in SCHEMA_ROOT.glob("*.json")}
    if not required <= present:
        raise ValueError(
            f"contract schemas missing: {sorted(required - present)}"
        )
    records = []
    for name in sorted(required):
        path = SCHEMA_ROOT / name
        schema = load_json(path)
        if schema.get("$schema") != (
            "https://json-schema.org/draft/2020-12/schema"
        ):
            raise ValueError(f"{name} does not declare JSON Schema 2020-12")
        records.append(
            {
                "name": name,
                "sha256": sha256_path(path),
                "id": schema.get("$id"),
            }
        )
    return records


def validate_concept_spec(
    spec: dict[str, Any], *, expected_case_id: str
) -> None:
    _require_keys(
        spec,
        {
            "schema_version",
            "case_id",
            "title_zh",
            "discipline",
            "learning_goal_zh",
            "assumptions_zh",
            "segments",
            "forbidden_shortcuts_zh",
        },
        label="concept spec",
    )
    if spec["schema_version"] != "1.0":
        raise ValueError("concept spec schema version must be 1.0")
    if spec["case_id"] != expected_case_id:
        raise ValueError("concept spec case ID mismatch")
    if not spec["learning_goal_zh"].strip():
        raise ValueError("concept spec learning goal is empty")
    if not spec["assumptions_zh"] or not spec["forbidden_shortcuts_zh"]:
        raise ValueError("concept assumptions and forbidden shortcuts required")
    segments = spec["segments"]
    if len(segments) != 4:
        raise ValueError("Phase 1 fixtures require four concept segments")
    for item in segments:
        _require_keys(
            item,
            {
                "segment_id",
                "order",
                "meaning_zh",
                "mechanism_condition_zh",
                "only_major_change_zh",
            },
            label="concept segment",
        )
    _ordered_ids(
        segments, id_field="segment_id", label="concept segments"
    )


def validate_sequence_spec(
    spec: dict[str, Any],
    *,
    expected_case_id: str,
    expected_layer_ids: set[str],
) -> None:
    _require_keys(
        spec,
        {
            "schema_version",
            "sequence_id",
            "case_id",
            "canvas",
            "camera",
            "state_source",
            "keyframes",
            "semantic_layer_ids",
            "fixed_across_sequence_zh",
            "model_policy",
        },
        label="sequence spec",
    )
    if spec["schema_version"] != "1.0":
        raise ValueError("sequence spec schema version must be 1.0")
    if spec["case_id"] != expected_case_id:
        raise ValueError("sequence spec case ID mismatch")
    canvas = spec["canvas"]
    if canvas != {
        "width": 320,
        "height": 180,
        "coordinate_system": "pixel_xy_top_left",
    }:
        raise ValueError("Phase 1 fixture canvas must be 320×180 pixel XY")
    if spec["camera"].get("locked") is not True:
        raise ValueError("sequence camera must be locked")
    keyframes = spec["keyframes"]
    if len(keyframes) != 4:
        raise ValueError("Phase 1 fixtures require four keyframes")
    for item in keyframes:
        _require_keys(
            item,
            {
                "keyframe_id",
                "order",
                "state_id",
                "meaning_zh",
                "selection_condition_zh",
                "only_major_change_zh",
                "forbidden_zh",
            },
            label="keyframe",
        )
    _ordered_ids(keyframes, id_field="keyframe_id", label="keyframes")
    if set(spec["semantic_layer_ids"]) != expected_layer_ids:
        raise ValueError("sequence layer IDs do not match fixture layers")
    policy = spec["model_policy"]
    if policy.get("annotations_after_generation") is not True:
        raise ValueError("annotations must be composited after generation")


def validate_states(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 4:
        raise ValueError("fixture states.jsonl must contain four states")
    ids = []
    for index, record in enumerate(records):
        _require_keys(
            record,
            {
                "state_id",
                "order",
                "progress",
                "meaning_zh",
                "fixture_only",
            },
            label="fixture state",
        )
        if record["order"] != index:
            raise ValueError("fixture state order mismatch")
        if record["fixture_only"] is not True:
            raise ValueError("Phase 1 state must declare fixture_only")
        expected_progress = index / 3
        if not math.isclose(record["progress"], expected_progress):
            raise ValueError("fixture state progress mismatch")
        ids.append(record["state_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("fixture state IDs are not unique")
    return records


def _validate_file_record(
    record: dict[str, Any],
    fixture_root: Path,
    *,
    require_semantic_metadata: bool,
) -> Path:
    required = {"path", "sha256", "size_bytes"}
    if require_semantic_metadata:
        required |= {"encoding", "shape", "dtype", "value_range"}
    _require_keys(record, required, label="artifact record")
    path = (fixture_root / record["path"]).resolve()
    try:
        path.relative_to(fixture_root.resolve())
    except ValueError as error:
        raise ValueError(f"fixture artifact escapes root: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_path(path) != record["sha256"]:
        raise ValueError(f"fixture artifact hash mismatch: {path}")
    if path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"fixture artifact size mismatch: {path}")
    return path


def validate_layer_manifest(
    manifest: dict[str, Any], fixture_root: Path
) -> None:
    _require_keys(
        manifest,
        {
            "schema_version",
            "case_id",
            "state_id",
            "canvas",
            "layers",
        },
        label="semantic layer manifest",
    )
    layers = manifest["layers"]
    layer_ids = [layer["layer_id"] for layer in layers]
    if len(layer_ids) != len(set(layer_ids)):
        raise ValueError("semantic layer IDs must be unique")
    expected_size = (
        manifest["canvas"]["width"],
        manifest["canvas"]["height"],
    )
    for layer in layers:
        _require_keys(
            layer,
            {
                "layer_id",
                "layer_type",
                "title_zh",
                "meaning_zh",
                "source_zh",
                "data",
                "preview",
                "model_input_policy",
                "used_as_model_input",
                "final_role_zh",
            },
            label="semantic layer",
        )
        if layer["layer_type"] not in SEMANTIC_LAYER_TYPES:
            raise ValueError(f"unknown layer type: {layer['layer_type']}")
        if layer["used_as_model_input"] is not False:
            raise ValueError("Phase 1 fixture cannot be a model input")
        data_path = _validate_file_record(
            layer["data"], fixture_root, require_semantic_metadata=True
        )
        preview_path = _validate_file_record(
            layer["preview"],
            fixture_root,
            require_semantic_metadata=True,
        )
        with Image.open(preview_path) as preview:
            if preview.size != expected_size or preview.mode != "RGB":
                raise ValueError(
                    f"invalid layer preview {preview_path}: "
                    f"{preview.size}/{preview.mode}"
                )
        data_record = layer["data"]
        if data_path.suffix == ".npy":
            array = np.load(data_path, allow_pickle=False)
            if list(array.shape) != data_record["shape"]:
                raise ValueError(f"array shape mismatch: {data_path}")
            if str(array.dtype) != data_record["dtype"]:
                raise ValueError(f"array dtype mismatch: {data_path}")
            actual_range = [float(array.min()), float(array.max())]
            for actual, expected in zip(
                actual_range, data_record["value_range"]
            ):
                if not math.isclose(actual, expected, abs_tol=1e-6):
                    raise ValueError(f"array range mismatch: {data_path}")
        elif data_path.suffix == ".json":
            payload = load_json(data_path)
            if not isinstance(payload.get("items"), list):
                raise ValueError(f"JSON layer lacks items: {data_path}")
            if data_record["shape"] != [len(payload["items"])]:
                raise ValueError(f"JSON layer count mismatch: {data_path}")
        else:
            raise ValueError(f"unsupported layer encoding: {data_path}")


def validate_fixture_manifest(
    manifest: dict[str, Any], fixture_root: Path
) -> None:
    _require_keys(
        manifest,
        {
            "schema_version",
            "case_id",
            "classification",
            "representative_state_id",
            "concept_spec",
            "sequence_spec",
            "states",
            "clean_frame",
            "program_frame",
            "semantic_layers",
            "control",
            "model_runs",
        },
        label="fixture manifest",
    )
    if manifest["classification"] != (
        "model-free contract fixture, not a finished scientific animation"
    ):
        raise ValueError("fixture classification is not explicit")
    if manifest["model_runs"] != {"image": 0, "video": 0}:
        raise ValueError("Phase 1 fixture must use zero model runs")
    for field in (
        "concept_spec",
        "sequence_spec",
        "states",
        "clean_frame",
        "program_frame",
        "semantic_layers",
    ):
        _validate_file_record(
            manifest[field], fixture_root, require_semantic_metadata=False
        )
    for field in ("clean_frame", "program_frame"):
        path = fixture_root / manifest[field]["path"]
        with Image.open(path) as image:
            if image.size != (320, 180) or image.mode != "RGB":
                raise ValueError(f"invalid fixture frame: {path}")
    control = manifest["control"]
    if control["used_as_model_input"] is not False:
        raise ValueError("Phase 1 control cannot be sent to a model")
    if control["route"] == "off":
        if control["input_layer_ids"] or control["control_preview"] is not None:
            raise ValueError("off control route must have no input or preview")
    elif control["route"] == "sparse_hard_boundary_candidate":
        if len(control["input_layer_ids"]) != 1:
            raise ValueError("sparse control requires one hard-boundary layer")
        _validate_file_record(
            control["control_preview"],
            fixture_root,
            require_semantic_metadata=False,
        )
    else:
        raise ValueError(f"unknown control route: {control['route']}")

