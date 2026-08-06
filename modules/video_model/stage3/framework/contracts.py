"""Deterministic file records and dependency-free Stage 3 contract checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
VISUAL_TARGET_STATUSES = {
    "user_approved",
    "accepted_project_baseline",
    "provisional",
    "missing",
}
GEOMETRY_POLICIES = {
    "preserve_exact",
    "canonicalize",
    "layout_only",
    "unsupported",
}
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


def file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(repo_root.resolve()).as_posix()
    return {
        "path": relative,
        "sha256": sha256_path(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def verify_file_record(record: dict[str, Any], repo_root: Path) -> None:
    path = (repo_root / record["path"]).resolve()
    path.relative_to(repo_root.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_path(path) != record["sha256"]:
        raise ValueError(f"hash mismatch: {record['path']}")
    if path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"size mismatch: {record['path']}")


def _require(value: dict[str, Any], names: set[str], label: str) -> None:
    missing = sorted(names - set(value))
    if missing:
        raise ValueError(f"{label} missing fields: {missing}")


def validate_schema_documents(schema_root: Path) -> list[dict[str, Any]]:
    expected = {
        "input_contract.schema.json",
        "case_registry.schema.json",
        "visual_target.schema.json",
        "motion_contract.schema.json",
        "state.schema.json",
    }
    present = {path.name for path in schema_root.glob("*.json")}
    if expected != present:
        raise ValueError(
            "schema set mismatch: "
            f"missing={sorted(expected - present)}, "
            f"extra={sorted(present - expected)}"
        )
    records: list[dict[str, Any]] = []
    for name in sorted(expected):
        path = schema_root / name
        schema = load_json(path)
        if schema.get("$schema") != (
            "https://json-schema.org/draft/2020-12/schema"
        ):
            raise ValueError(f"{name} is not JSON Schema 2020-12")
        if schema.get("type") != "object" or not schema.get("required"):
            raise ValueError(f"{name} lacks an object contract")
        records.append(
            {"name": name, "id": schema.get("$id"), "sha256": sha256_path(path)}
        )
    return records


def validate_motion_contract(value: dict[str, Any]) -> None:
    _require(
        value,
        {
            "schema_version",
            "case_id",
            "state_timeline",
            "program_video",
            "motion_classes",
            "keyframe_selection",
            "model_input_policy",
            "temporal_gates",
        },
        "motion contract",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("motion contract schema version mismatch")
    if len(value["keyframe_selection"]) < 2:
        raise ValueError("motion contract requires at least two keyframes")
    if not value["motion_classes"] or not value["temporal_gates"]:
        raise ValueError("motion classes and temporal gates are required")
    for record in (
        value["state_timeline"],
        value["program_video"],
    ):
        _require(record, {"path", "sha256", "size_bytes"}, "motion artifact")


def validate_visual_target(value: dict[str, Any]) -> None:
    _require(
        value,
        {
            "schema_version",
            "package_id",
            "case_id",
            "status",
            "geometry_control_separation",
            "style_board",
            "positive_refs",
            "negative_refs",
            "rubric",
        },
        "visual target",
    )
    if value["status"] not in VISUAL_TARGET_STATUSES:
        raise ValueError("unknown visual target status")
    separation = value["geometry_control_separation"]
    _require(
        separation,
        {"geometry_source", "appearance_source", "leakage_gate"},
        "geometry/appearance separation",
    )
    if separation["leakage_gate"] != "appearance_to_geometry_leakage":
        raise ValueError("appearance leakage gate is not frozen")
    if value["status"] != "missing" and not value["positive_refs"]:
        raise ValueError("non-missing visual target requires a positive reference")


def validate_input_contract(value: dict[str, Any]) -> None:
    _require(
        value,
        {
            "schema_version",
            "case_id",
            "case_definition",
            "program_source",
            "keyframes",
            "semantic_exports",
            "geometry_policy",
            "motion_contract",
            "visual_target_package",
            "hard_gates",
        },
        "input contract",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("input contract schema version mismatch")
    if value["geometry_policy"] not in GEOMETRY_POLICIES:
        raise ValueError("unknown geometry policy")
    expected_count = 5 if value["case_id"] == "GEO-HIST-DELTA-01" else 4
    if len(value["keyframes"]) != expected_count:
        raise ValueError(
            f"{value['case_id']} expects {expected_count} teaching keyframes"
        )
    layer_types = set(value["semantic_exports"]["layer_types"])
    if not layer_types <= SEMANTIC_LAYER_TYPES:
        raise ValueError("unknown semantic layer type")
    if "object_identity" not in layer_types:
        raise ValueError("object identity export is required")
    if not value["hard_gates"]:
        raise ValueError("hard gates are required")


def validate_case_registry(value: dict[str, Any]) -> None:
    _require(value, {"schema_version", "suite_id", "cases"}, "case registry")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("case registry schema version mismatch")
    cases = value["cases"]
    ids = [case["case_id"] for case in cases]
    if len(cases) != 11 or len(set(ids)) != 11:
        raise ValueError("registry must contain exactly 10 scale cases + delta")
    expected = {
        "MATH-01",
        "MATH-02",
        "PHYS-01",
        "PHYS-02",
        "CHEM-01",
        "CHEM-02",
        "BIO-01",
        "BIO-02",
        "GEO-01",
        "GEO-02",
        "GEO-HIST-DELTA-01",
    }
    if set(ids) != expected:
        raise ValueError("case registry IDs do not match frozen suite")
    if not all(case["completeness"]["contract_smoke_passed"] for case in cases):
        raise ValueError("one or more cases failed contract smoke")


def validate_loop_state(value: dict[str, Any]) -> None:
    _require(
        value,
        {
            "schema_version",
            "loop_id",
            "phase",
            "phase_status",
            "exit_criteria",
            "budget",
            "current_problem",
            "current_hypothesis",
            "current_cohort",
            "next_action",
        },
        "loop state",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("state schema version mismatch")
    if value["phase_status"] not in {
        "not_started",
        "in_progress",
        "passed",
        "failed",
        "blocked",
    }:
        raise ValueError("unknown phase status")
    if not value["exit_criteria"] or not value["next_action"]:
        raise ValueError("state must have exit criteria and next action")
