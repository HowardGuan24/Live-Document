#!/usr/bin/env python3
"""Deterministic runtime for Live Document Phase 2.

Subcommands:

- run-candidates: execute every candidate from plan.json using one shared prototype.
- freeze: copy the exact selected candidate config into executable-spec.json.

This file performs no implementation design and no visual review.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator


REQUIRED_PROTOTYPE_FUNCTIONS = (
    "build_static_scene",
    "evaluate_state",
    "validate_probe",
    "render_semantic_probe",
    "render_edge_probe",
    "render_program_probe",
)

PHASE2_SCHEMA_PATH = Path(__file__).resolve().with_name("schema.json")
_PHASE2_SCHEMA: dict[str, Any] | None = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return data


def load_phase2_schema() -> dict[str, Any]:
    global _PHASE2_SCHEMA
    if _PHASE2_SCHEMA is None:
        schema = load_json(PHASE2_SCHEMA_PATH)
        Draft202012Validator.check_schema(schema)
        _PHASE2_SCHEMA = schema
    return _PHASE2_SCHEMA


def validate_phase2_artifact(
    data: dict[str, Any], definition: str, label: str
) -> None:
    schema = load_phase2_schema()
    if definition not in schema.get("$defs", {}):
        raise KeyError(f"Unknown Phase 2 schema definition: {definition}")
    fragment_schema = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": f"#/$defs/{definition}",
    }
    validator = Draft202012Validator(fragment_schema)
    errors = sorted(
        validator.iter_errors(data),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "$"
        for part in error.absolute_path:
            location += f"[{part}]" if isinstance(part, int) else f".{part}"
        raise ValueError(
            f"{label} failed schema #/$defs/{definition} at {location}: "
            f"{error.message}"
        )


def canonical_json_bytes(data: Any) -> bytes:
    return (
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(data))


def write_validated_phase2_json(
    path: Path, data: dict[str, Any], definition: str, label: str
) -> None:
    validate_phase2_artifact(data, definition, label)
    write_json(path, data)
    validate_phase2_artifact(load_json(path), definition, label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_prototype(plan_path: Path, plan: dict[str, Any]) -> Path:
    raw = Path(plan["prototype_entrypoint"])
    path = raw if raw.is_absolute() else plan_path.parent / raw
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prototype does not exist: {path}")
    return path


def import_prototype(path: Path):
    module_name = f"live_document_phase2_{sha256_file(path)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import prototype: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [
        name
        for name in REQUIRED_PROTOTYPE_FUNCTIONS
        if not callable(getattr(module, name, None))
    ]
    if missing:
        raise AttributeError(f"Prototype is missing required callables: {missing}")
    return module


def safe_key(parts: list[str]) -> str:
    raw = "__".join(parts)
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_") or "value"


def serialize_state(path: Path, state: dict[str, Any]) -> None:
    """Store arrays and JSON-compatible metadata in one compressed NPZ."""
    if not isinstance(state, dict):
        raise TypeError("evaluate_state must return a dict")

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}

    def walk(value: Any, parts: list[str]) -> None:
        key = safe_key(parts)
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, np.generic):
            metadata[key] = value.item()
        elif isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, parts + [str(child_key)])
        elif isinstance(value, (list, tuple)):
            array = np.asarray(value)
            if array.dtype != object:
                arrays[key] = array
            else:
                metadata[key] = value
        elif value is None or isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            raise TypeError(
                f"Unsupported state value at {'.'.join(parts)}: "
                f"{type(value).__name__}"
            )

    walk(state, ["state"])
    arrays["__metadata_json__"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def validate_checks(
    checks: Any, required_check_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    if not isinstance(checks, list) or not checks:
        raise ValueError("validate_probe must return a non-empty list")
    normalized: list[dict[str, Any]] = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise TypeError(f"Check {index} is not an object")
        for key in ("check_id", "claim", "passed", "evidence"):
            if key not in check:
                raise ValueError(f"Check {index} is missing {key}")
        if not isinstance(check["check_id"], str):
            raise TypeError(f"Check {index} check_id must be a string")
        if not isinstance(check["claim"], str):
            raise TypeError(f"Check {index} claim must be a string")
        if type(check["passed"]) is not bool:
            raise TypeError(f"Check {index} passed must be a boolean")
        if not isinstance(check["evidence"], str):
            raise TypeError(f"Check {index} evidence must be a string")
        normalized.append(dict(check))

    check_ids = [check["check_id"] for check in normalized]
    if len(set(check_ids)) != len(check_ids):
        raise ValueError("validate_probe returned duplicate check_id values")
    missing = sorted((required_check_ids or set()) - set(check_ids))
    if missing:
        raise ValueError(
            "validate_probe omitted declared state invariant checks: "
            f"{missing}"
        )
    return normalized


def artifact_ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def merge_config(fixed: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(fixed)
    merged.update(overrides)
    return merged


def validate_plan_for_execution(plan: dict[str, Any]) -> None:
    candidates = plan["candidates"]
    budget = plan["candidate_budget"]
    if len(candidates) > budget:
        raise ValueError(
            f"candidate count {len(candidates)} exceeds candidate_budget {budget}"
        )
    ids = [item["candidate_id"] for item in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_id values must be unique")

    declared = {
        item["name"] for item in plan["implementation"]["tunable_parameters"]
    }
    used = set(plan.get("fixed_parameters", {}))
    for candidate in candidates:
        used.update(candidate.get("parameter_overrides", {}))
    undeclared = sorted(used - declared)
    if undeclared:
        raise ValueError(
            f"Config parameters are not declared in tunable_parameters: {undeclared}"
        )

    state_contract = plan["implementation"]["semantic_state_contract"]
    field_names = [item["name"] for item in state_contract["fields"]]
    if len(set(field_names)) != len(field_names):
        raise ValueError("semantic state field names must be unique")
    declared_fields = set(field_names)

    for field in state_contract["fields"]:
        field_name = field["name"]
        unknown_dependencies = sorted(
            set(field["depends_on"]) - declared_fields
        )
        if unknown_dependencies:
            raise ValueError(
                f"Derived field {field_name} has undeclared dependencies: "
                f"{unknown_dependencies}"
            )
        if field_name in field["depends_on"]:
            raise ValueError(f"Semantic state field cannot depend on itself: {field_name}")

    invariant_ids: list[str] = []
    for invariant in state_contract["invariants"]:
        invariant_ids.append(invariant["check_id"])
        unknown_fields = sorted(set(invariant["fields"]) - declared_fields)
        if unknown_fields:
            raise ValueError(
                f"State invariant {invariant['check_id']} references undeclared "
                f"fields: {unknown_fields}"
            )
    if len(set(invariant_ids)) != len(invariant_ids):
        raise ValueError("state invariant check_id values must be unique")


def validate_declared_state_fields(
    state: dict[str, Any], declared_field_names: set[str], sample_id: str
) -> None:
    """Require every declared scalar or array field as an exact top-level key."""
    missing = sorted(declared_field_names - set(state))
    if missing:
        raise ValueError(
            f"evaluate_state omitted declared fields at sample {sample_id}: {missing}"
        )


def summarize_failure_groups(output_root: Path) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for result_path in sorted(output_root.glob("candidate-*/probe-result.json")):
        result = load_json(result_path)
        validate_phase2_artifact(
            result,
            "probe_result",
            f"{result.get('candidate_id', result_path.parent.name)} probe-result.json",
        )
        completed.append(result)

    completed_ids = {result["candidate_id"] for result in completed}
    failures_by_check: dict[str, list[dict[str, Any]]] = {}
    for result in completed:
        for check in result["machine_checks"]:
            if check["passed"]:
                continue
            failures_by_check.setdefault(check["check_id"], []).append(
                {"candidate_id": result["candidate_id"], "check": check}
            )

    groups: list[dict[str, Any]] = []
    for check_id in sorted(failures_by_check):
        failures = failures_by_check[check_id]
        failed_ids = sorted({item["candidate_id"] for item in failures})
        shared = len(completed_ids) >= 2 and set(failed_ids) == completed_ids
        if len(completed_ids) == 1:
            scope = "not_comparable"
        elif shared:
            scope = "shared_across_completed_candidates"
        else:
            scope = "candidate_specific"
        reported_classes = sorted(
            {
                item["check"]["failure_class"]
                for item in failures
                if "failure_class" in item["check"]
            }
        )
        reported_targets = sorted(
            {
                item["check"]["recommended_return_target"]
                for item in failures
                if "recommended_return_target" in item["check"]
            }
        )
        groups.append(
            {
                "check_id": check_id,
                "scope": scope,
                "failed_candidate_ids": failed_ids,
                "reported_failure_classes": reported_classes,
                "reported_return_targets": reported_targets,
            }
        )
    return groups


def run_one_candidate(
    *,
    prototype,
    prototype_path: Path,
    semantic_contract: dict[str, Any],
    plan: dict[str, Any],
    candidate: dict[str, Any],
    output_root: Path,
    command_display: str,
) -> None:
    attempt_id = plan["implementation_attempt_id"]
    candidate_id = candidate["candidate_id"]
    candidate_root = output_root / candidate_id
    config_path = candidate_root / "config.json"
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    started = time.perf_counter()
    complete_config: dict[str, Any] = {}

    try:
        candidate_root.mkdir(parents=True, exist_ok=False)
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            complete_config = merge_config(
                plan.get("fixed_parameters", {}),
                candidate.get("parameter_overrides", {}),
            )
            write_json(config_path, complete_config)
            static_scene = prototype.build_static_scene(complete_config)
            states: list[dict[str, Any]] = []
            sampled_states: list[dict[str, Any]] = []
            declared_field_names = {
                item["name"]
                for item in plan["implementation"]["semantic_state_contract"][
                    "fields"
                ]
            }

            for sample in plan["implementation"]["probe_samples"]:
                sample_id = sample["sample_id"]
                progress_values = sample["progress_values"]
                state = prototype.evaluate_state(
                    static_scene,
                    progress_values,
                    complete_config,
                )
                if not isinstance(state, dict):
                    raise TypeError("evaluate_state must return a dict")
                validate_declared_state_fields(
                    state, declared_field_names, sample_id
                )
                state.setdefault("_probe_sample_id", sample_id)
                state.setdefault("_probe_progress_values", progress_values)
                states.append(state)

                raw_path = candidate_root / "states" / f"{sample_id}.npz"
                serialize_state(raw_path, state)
                sampled_states.append(
                    {
                        "sample_id": sample_id,
                        "progress_values": progress_values,
                        "raw_state": artifact_ref(raw_path),
                    }
                )

            semantic_path = candidate_root / "semantic-probe.png"
            edge_path = candidate_root / "edge-probe.png"
            program_path = candidate_root / "program-probe.png"

            prototype.render_semantic_probe(states, str(semantic_path))
            prototype.render_edge_probe(states, str(edge_path))
            prototype.render_program_probe(states, str(program_path))

            for required_path in (semantic_path, edge_path, program_path):
                if not required_path.is_file():
                    raise FileNotFoundError(
                        f"Prototype did not create required artifact: {required_path}"
                    )

            declared_invariant_ids = {
                item["check_id"]
                for item in plan["implementation"]["semantic_state_contract"][
                    "invariants"
                ]
            }
            checks = validate_checks(
                prototype.validate_probe(
                    states,
                    semantic_contract,
                    complete_config,
                ),
                declared_invariant_ids,
            )

        duration = time.perf_counter() - started
        machine_passed = all(check["passed"] for check in checks)
        failures = [
            f'{check["check_id"]}: {check["evidence"]}'
            for check in checks
            if not check["passed"]
        ]

        result = {
            "implementation_attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "complete_config": complete_config,
            "execution": {
                "command": f"{command_display} [candidate={candidate_id}]",
                "exit_code": 0,
                "duration_seconds": duration,
                "prototype_sha256": sha256_file(prototype_path),
                "config_sha256": sha256_file(config_path),
            },
            "sampled_states": sampled_states,
            "probe_artifacts": {
                "semantic_probe": artifact_ref(semantic_path),
                "edge_probe": artifact_ref(edge_path),
                "program_probe": artifact_ref(program_path),
            },
            "machine_checks": checks,
            "machine_gate_status": "passed" if machine_passed else "failed",
            "failure_summary": failures,
        }
        write_validated_phase2_json(
            candidate_root / "probe-result.json",
            result,
            "probe_result",
            f"{candidate_id} probe-result.json",
        )

    except Exception as exc:
        duration = time.perf_counter() - started
        candidate_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "implementation_attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "complete_config": complete_config,
            "execution": {
                "command": f"{command_display} [candidate={candidate_id}]",
                "exit_code": 1,
                "duration_seconds": duration,
                "prototype_sha256": sha256_file(prototype_path),
                "config_sha256": (
                    sha256_file(config_path) if config_path.is_file() else None
                ),
            },
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(candidate_root / "execution-failure.json", failure)
    finally:
        (candidate_root / "stdout.txt").write_text(
            stdout_buffer.getvalue(), encoding="utf-8"
        )
        (candidate_root / "stderr.txt").write_text(
            stderr_buffer.getvalue(), encoding="utf-8"
        )


def run_candidates(args: argparse.Namespace) -> int:
    semantic_contract = load_json(args.semantic_contract)
    plan = load_json(args.plan)
    validate_phase2_artifact(plan, "plan", "plan.json")
    validate_plan_for_execution(plan)
    prototype_path = resolve_prototype(args.plan, plan)
    prototype = import_prototype(prototype_path)

    args.output_root.mkdir(parents=True, exist_ok=False)
    command_display = " ".join(sys.argv)

    for candidate in plan["candidates"]:
        run_one_candidate(
            prototype=prototype,
            prototype_path=prototype_path,
            semantic_contract=semantic_contract,
            plan=plan,
            candidate=candidate,
            output_root=args.output_root,
            command_display=command_display,
        )

    failure_groups = summarize_failure_groups(args.output_root)
    summary = {
        "implementation_attempt_id": plan["implementation_attempt_id"],
        "candidate_count": len(plan["candidates"]),
        "completed_probe_results": len(
            list(args.output_root.glob("candidate-*/probe-result.json"))
        ),
        "execution_failures": len(
            list(args.output_root.glob("candidate-*/execution-failure.json"))
        ),
        "failure_groups": failure_groups,
    }
    write_validated_phase2_json(
        args.output_root / "executor-summary.json",
        summary,
        "executor_summary",
        "executor-summary.json",
    )
    return 0


def freeze_selected(args: argparse.Namespace) -> int:
    selection = load_json(args.selection)
    probe = load_json(args.probe_result)
    plan = load_json(args.plan)
    validate_phase2_artifact(selection, "selection", "selection.json")
    validate_phase2_artifact(probe, "probe_result", "probe-result.json")
    validate_phase2_artifact(plan, "plan", "plan.json")

    if selection.get("selection_status") != "selected":
        raise ValueError("selection_status must be 'selected'")

    selected_id = selection.get("selected_candidate_id")
    if not selected_id:
        raise ValueError("selected_candidate_id is missing")
    if probe.get("candidate_id") != selected_id:
        raise ValueError(
            f"Selection chose {selected_id}, but probe result is for "
            f"{probe.get('candidate_id')}"
        )
    if probe.get("machine_gate_status") != "passed":
        raise ValueError("Selected candidate did not pass the machine gate")

    attempt_id = selection.get("implementation_attempt_id")
    if probe.get("implementation_attempt_id") != attempt_id:
        raise ValueError("Selection and probe implementation attempt IDs do not match")
    if plan.get("implementation_attempt_id") != attempt_id:
        raise ValueError("Plan and selection implementation attempt IDs do not match")

    reviewed = selection.get("reviewed_candidates", [])
    accepted = [item for item in reviewed if item.get("decision") == "accept"]
    if len(accepted) != 1 or accepted[0].get("candidate_id") != selected_id:
        raise ValueError("Exactly the selected candidate must have decision='accept'")
    if accepted[0].get("machine_gate_passed") is not True:
        raise ValueError("Selected review does not confirm machine-gate passage")

    planned = next(
        (item for item in plan["candidates"] if item["candidate_id"] == selected_id),
        None,
    )
    if planned is None:
        raise ValueError("Selected candidate is absent from plan.json")
    planned_config = merge_config(
        plan.get("fixed_parameters", {}),
        planned.get("parameter_overrides", {}),
    )
    frozen_config = probe.get("complete_config")
    if planned_config != frozen_config:
        raise ValueError("Probe config does not exactly match the selected plan candidate")

    prototype_path = resolve_prototype(args.plan, plan)
    actual_prototype_sha = sha256_file(prototype_path)
    probe_prototype_sha = probe.get("execution", {}).get("prototype_sha256")
    if actual_prototype_sha != probe_prototype_sha:
        raise ValueError("Prototype hash does not match the executed probe")

    executable_spec = {
        "implementation_attempt_id": attempt_id,
        "selected_candidate_id": selected_id,
        "prototype_entrypoint": str(prototype_path),
        "prototype_sha256": actual_prototype_sha,
        "source_probe_result": {
            "path": str(args.probe_result),
            "sha256": sha256_file(args.probe_result),
        },
        "source_selection": {
            "path": str(args.selection),
            "sha256": sha256_file(args.selection),
        },
        "frozen_config": frozen_config,
        "semantic_contract_binding": {
            "path": str(args.semantic_contract),
            "sha256": sha256_file(args.semantic_contract),
        },
        "freeze_checks": {
            "selection_matches_candidate": True,
            "prototype_hash_matches_probe": True,
            "config_exactly_matches_probe": True,
            "machine_gate_passed": True,
        },
    }

    write_validated_phase2_json(
        args.output,
        executable_spec,
        "executable_spec",
        "executable-spec.json",
    )
    written = load_json(args.output)
    if written["frozen_config"] != frozen_config:
        raise RuntimeError("Frozen config changed during serialization")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-candidates")
    run_parser.add_argument("--semantic-contract", required=True, type=Path)
    run_parser.add_argument("--plan", required=True, type=Path)
    run_parser.add_argument("--output-root", required=True, type=Path)
    run_parser.set_defaults(func=run_candidates)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--selection", required=True, type=Path)
    freeze_parser.add_argument("--probe-result", required=True, type=Path)
    freeze_parser.add_argument("--semantic-contract", required=True, type=Path)
    freeze_parser.add_argument("--plan", required=True, type=Path)
    freeze_parser.add_argument("--output", required=True, type=Path)
    freeze_parser.set_defaults(func=freeze_selected)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
