#!/usr/bin/env python3
"""Thin, resumable Stage 5 Phase 0-6 orchestration CLI.

This module coordinates the existing phase prompts, schemas, and Runtime CLIs.
It does not import or reproduce phase implementation logic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STAGE5_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = STAGE5_ROOT.parents[2]
RUNS_ROOT = STAGE5_ROOT / "runs"
WORKFLOW_ROOT = STAGE5_ROOT / "workflow"
STATE_NAME = "run-state.json"
STATE_VERSION = "stage5-pipeline-run-state-1"
RUN_ID_RE = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
STATUSES = {"initialized", "running", "waiting_for_agent", "waiting_for_human", "failed", "completed"}
PHASES = tuple(f"phase{i}" for i in range(7))
STEPS = (
    "phase0.author",
    "phase1.author",
    "phase2.build",
    "phase2.run_candidates",
    "phase2.review",
    "phase2.freeze",
    "phase3.schedule",
    "phase3.build_sequence",
    "phase4.author",
    "phase4.render_styles",
    "phase4.human_gate",
    "phase5.build",
    "phase5.run_appearance",
    "phase5.review",
    "phase5.human_gate",
    "phase5.assemble_pack",
    "phase6.render_delivery",
    "completed",
)
AGENT_STEPS = {
    "phase0.author",
    "phase1.author",
    "phase2.build",
    "phase2.review",
    "phase3.schedule",
    "phase4.author",
    "phase5.build",
    "phase5.review",
}
RUNTIME_STEPS = {
    "phase2.run_candidates",
    "phase2.freeze",
    "phase3.build_sequence",
    "phase4.render_styles",
    "phase5.run_appearance",
    "phase5.assemble_pack",
    "phase6.render_delivery",
}
NEXT_STEP = {
    "phase0.author": "phase1.author",
    "phase1.author": "phase2.build",
    "phase2.build": "phase2.run_candidates",
    "phase2.run_candidates": "phase2.review",
    "phase2.review": "phase2.freeze",
    "phase2.freeze": "phase3.schedule",
    "phase3.schedule": "phase3.build_sequence",
    "phase3.build_sequence": "phase4.author",
    "phase4.author": "phase4.render_styles",
    "phase5.build": "phase5.run_appearance",
    "phase5.run_appearance": "phase5.review",
    "phase5.assemble_pack": "phase6.render_delivery",
    "phase6.render_delivery": "completed",
}


class PipelineError(RuntimeError):
    """A recoverable orchestration or validation failure."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json(value))


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PipelineError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_metadata(path: Path) -> tuple[str, int, int]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise PipelineError(f"artifact tree is empty: {path}")
    digest = hashlib.sha256()
    size = 0
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        content_digest = sha256_file(item).encode("ascii")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(content_digest)
        size += item.stat().st_size
    return digest.hexdigest(), size, len(files)


def repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise PipelineError(f"path must remain inside the repository: {path}") from exc


def resolve_repository_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    elif path.parts and path.parts[0] == "modules":
        resolved = (REPOSITORY_ROOT / path).resolve()
    elif base is not None:
        resolved = (base / path).resolve()
    else:
        resolved = (Path.cwd() / path).resolve()
    repository_relative(resolved)
    return resolved


def resolve_external_or_repository_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "modules":
        return (REPOSITORY_ROOT / path).resolve()
    return (Path.cwd() / path).resolve()


def resolve_run(value: str | Path) -> Path:
    path = resolve_repository_path(value)
    try:
        relative = path.relative_to(RUNS_ROOT.resolve())
    except ValueError as exc:
        raise PipelineError(f"run root must be under {repository_relative(RUNS_ROOT)}") from exc
    if len(relative.parts) != 1 or not RUN_ID_RE.fullmatch(relative.name):
        raise PipelineError("run path must identify one canonical runs/<run-id> root")
    if path.is_symlink():
        raise PipelineError("run root may not be a symbolic link")
    return path


def validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise PipelineError("run ID must match ^[a-z][a-z0-9-]{2,63}$")


def choose_run_id(concept: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", concept.lower()).strip("-")[:40].rstrip("-")
    if not slug or not slug[0].isalpha():
        slug = f"concept-{slug}".rstrip("-")
    slug = slug or "concept"
    for number in range(1, 1000):
        candidate = f"{slug}-{number:03d}"
        if len(candidate) <= 64 and not (RUNS_ROOT / candidate).exists():
            return candidate
    raise PipelineError("could not allocate a run ID")


def artifact_record(path: Path, *, phase: str, attempt_id: str, role: str, external: bool) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        digest = sha256_file(resolved)
        size = resolved.stat().st_size
        kind = "file"
        file_count = 1
    elif resolved.is_dir():
        digest, size, file_count = tree_metadata(resolved)
        kind = "tree"
    else:
        raise PipelineError(f"artifact does not exist: {resolved}")
    return {
        "path": repository_relative(resolved),
        "sha256": digest,
        "size_bytes": size,
        "file_count": file_count,
        "kind": kind,
        "phase": phase,
        "attempt_id": attempt_id,
        "role": role,
        "status": "active",
        "external": bool(external),
    }


def artifact_ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineError(f"artifact reference requires a file: {path}")
    return {
        "path": repository_relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def resolve_artifact_reference(reference: Mapping[str, Any], *, base: Path | None = None) -> Path:
    path = resolve_repository_path(str(reference["path"]), base=base)
    if not path.is_file():
        raise PipelineError(f"referenced artifact is missing: {path}")
    if path.stat().st_size != reference.get("size_bytes", path.stat().st_size):
        raise PipelineError(f"referenced artifact size changed: {path}")
    if sha256_file(path) != reference["sha256"]:
        raise PipelineError(f"referenced artifact hash changed: {path}")
    return path


def schema_path(phase: int) -> Path:
    return WORKFLOW_ROOT / f"phase{phase}" / "schema.json"


def validate_json_schema(path: Path, phase: int, definition: str | None = None) -> dict[str, Any]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise PipelineError("jsonschema is required by the Stage 5 workflow") from exc
    document = load_json(path)
    schema = load_json(schema_path(phase))
    Draft202012Validator.check_schema(schema)
    if definition is None:
        contract = schema
    else:
        if definition not in schema.get("$defs", {}):
            raise PipelineError(f"missing Phase {phase} schema definition: {definition}")
        contract = {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{definition}"}
    errors = sorted(Draft202012Validator(contract).iter_errors(document), key=lambda item: list(item.absolute_path))
    if errors:
        error = errors[0]
        location = "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path)
        raise PipelineError(f"Phase {phase} schema validation failed at {location}: {error.message}")
    return document


def runtime_python() -> str:
    # Preserve the invoked virtual-environment entry point. Resolving its
    # symlink would bypass the environment's site-packages.
    return sys.executable


def runtime_path(phase: int) -> Path:
    return WORKFLOW_ROOT / f"phase{phase}" / "runtime.py"


def validate_with_runtime(phase: int, command: str, path: Path, extra: Sequence[str] = ()) -> None:
    argv = [runtime_python(), str(runtime_path(phase)), command, str(path), *extra]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(argv, cwd=REPOSITORY_ROOT, env=environment, text=True, capture_output=True)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise PipelineError(f"Phase {phase} Runtime validation failed: {message}")


def state_path(run_root: Path) -> Path:
    return run_root / STATE_NAME


def phase_from_step(step: str) -> str | None:
    match = re.match(r"^(phase[0-6])\.", step)
    return match.group(1) if match else None


def phase_number(phase: str) -> int:
    return int(phase.removeprefix("phase"))


def validate_state_structure(state: Mapping[str, Any], run_root: Path) -> None:
    required = {
        "schema_version", "run_id", "concept", "status", "current_phase", "current_step",
        "attempts", "artifact_registry", "human_gates", "history", "last_error", "final_delivery",
    }
    if set(state) != required:
        raise PipelineError(f"run-state keys differ from the v1 contract: {sorted(set(state) ^ required)}")
    if state["schema_version"] != STATE_VERSION:
        raise PipelineError("unsupported run-state schema version")
    validate_run_id(str(state["run_id"]))
    if state["run_id"] != run_root.name:
        raise PipelineError("run-state run_id differs from its directory")
    if not isinstance(state["concept"], str) or not state["concept"].strip():
        raise PipelineError("run-state concept must be a nonempty string")
    if state["status"] not in STATUSES:
        raise PipelineError("invalid run status")
    if state["current_step"] not in STEPS:
        raise PipelineError("invalid current step")
    expected_phase = phase_from_step(state["current_step"])
    if expected_phase is not None and state["current_phase"] != expected_phase:
        raise PipelineError("current phase and step disagree")
    if state["current_step"] == "completed" and state["status"] != "completed":
        raise PipelineError("completed step requires completed status")
    if set(state["attempts"]) != set(PHASES):
        raise PipelineError("attempt registry must contain Phase 0 through Phase 6")
    for phase, attempts in state["attempts"].items():
        if not isinstance(attempts, list) or not attempts:
            raise PipelineError(f"{phase} requires at least one attempt")
        seen: set[str] = set()
        for index, attempt in enumerate(attempts, 1):
            expected = f"attempt-{index:03d}"
            if attempt.get("attempt_id") != expected or expected in seen:
                raise PipelineError(f"noncanonical attempt sequence in {phase}")
            seen.add(expected)
            expected_path = run_root / phase / expected
            if resolve_repository_path(attempt.get("path", "")) != expected_path.resolve():
                raise PipelineError(f"attempt path differs from canonical layout: {phase}/{expected}")
            if attempt.get("status") not in {"pending", "active", "completed", "failed", "superseded", "imported"}:
                raise PipelineError(f"invalid attempt status: {phase}/{expected}")
    if not isinstance(state["artifact_registry"], dict):
        raise PipelineError("artifact_registry must be an object")
    for key, record in state["artifact_registry"].items():
        if not isinstance(key, str) or not isinstance(record, dict):
            raise PipelineError("invalid artifact registry entry")
        if record.get("phase") not in PHASES or record.get("kind") not in {"file", "tree"}:
            raise PipelineError(f"invalid artifact ownership: {key}")
        path_value = record.get("path")
        if not isinstance(path_value, str) or Path(path_value).is_absolute():
            raise PipelineError(f"artifact path must be repository-relative: {key}")
        resolve_repository_path(path_value)
        if not re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256", ""))):
            raise PipelineError(f"invalid artifact hash: {key}")
    if set(state["human_gates"]) != {"phase4", "phase5"}:
        raise PipelineError("human_gates must contain Phase 4 and Phase 5")
    for phase, gate in state["human_gates"].items():
        if gate.get("status") not in {"not_reached", "waiting", "approved", "rejected", "superseded"}:
            raise PipelineError(f"invalid {phase} human gate status")
        if not isinstance(gate.get("allowed_selections"), list):
            raise PipelineError(f"invalid {phase} allowed selections")
    history = state["history"]
    if not isinstance(history, list):
        raise PipelineError("history must be an array")
    for index, entry in enumerate(history, 1):
        if entry.get("sequence") != index or not isinstance(entry.get("event"), str):
            raise PipelineError("history sequence is not contiguous")


def load_state(run_root: Path) -> dict[str, Any]:
    path = state_path(run_root)
    if not path.is_file():
        raise PipelineError(f"run state is missing: {path}")
    state = load_json(path)
    validate_state_structure(state, run_root)
    return state


def write_state(run_root: Path, state: dict[str, Any]) -> None:
    validate_state_structure(state, run_root)
    atomic_write_json(state_path(run_root), state)


def record_history(state: dict[str, Any], event: str, **details: Any) -> None:
    entry = {"sequence": len(state["history"]) + 1, "event": event}
    clean = {key: value for key, value in details.items() if value is not None}
    if clean:
        entry["details"] = clean
    state["history"].append(entry)


def active_attempt(state: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    active = [item for item in state["attempts"][phase] if item["status"] not in {"superseded"}]
    if not active:
        raise PipelineError(f"no active attempt for {phase}")
    return active[-1]


def active_attempt_path(state: Mapping[str, Any], phase: str) -> Path:
    return resolve_repository_path(active_attempt(state, phase)["path"])


def set_attempt_status(state: dict[str, Any], phase: str, status: str) -> None:
    active_attempt(state, phase)["status"] = status


def register_artifact(
    state: dict[str, Any], path: Path, *, phase: str, role: str, external: bool = False,
) -> str:
    attempt_id = str(active_attempt(state, phase)["attempt_id"])
    key = f"{phase}.{attempt_id}.{role}"
    record = artifact_record(path, phase=phase, attempt_id=attempt_id, role=role, external=external)
    existing = state["artifact_registry"].get(key)
    if existing is not None and existing != record:
        raise PipelineError(f"registered artifact would be overwritten: {key}")
    state["artifact_registry"][key] = record
    return key


def latest_artifact_record(state: Mapping[str, Any], role: str, phase: str | None = None) -> Mapping[str, Any]:
    candidates = [
        record for record in state["artifact_registry"].values()
        if record["role"] == role and record["status"] == "active" and (phase is None or record["phase"] == phase)
    ]
    if not candidates:
        raise PipelineError(f"missing registered artifact: {phase + '.' if phase else ''}{role}")
    candidates.sort(key=lambda item: (phase_number(item["phase"]), item["attempt_id"]))
    return candidates[-1]


def latest_artifact_path(state: Mapping[str, Any], role: str, phase: str | None = None) -> Path:
    return resolve_repository_path(latest_artifact_record(state, role, phase)["path"])


def create_run_layout(run_root: Path, concept: str, run_id: str) -> dict[str, Any]:
    if run_root.exists():
        raise PipelineError(f"run root already exists and will not be reused: {run_root}")
    run_root.mkdir(parents=True)
    for name in ("tasks", "reports", "decisions"):
        (run_root / name).mkdir()
    attempts: dict[str, list[dict[str, Any]]] = {}
    for phase in PHASES:
        attempt_root = run_root / phase / "attempt-001"
        attempt_root.mkdir(parents=True)
        attempts[phase] = [{
            "attempt_id": "attempt-001",
            "path": repository_relative(attempt_root),
            "status": "active" if phase == "phase0" else "pending",
            "superseded_by": None,
            "reason": None,
        }]
    atomic_write_text(run_root / "concept.txt", concept)
    state: dict[str, Any] = {
        "schema_version": STATE_VERSION,
        "run_id": run_id,
        "concept": concept,
        "status": "waiting_for_agent",
        "current_phase": "phase0",
        "current_step": "phase0.author",
        "attempts": attempts,
        "artifact_registry": {},
        "human_gates": {
            "phase4": {"status": "not_reached", "attempt_id": None, "allowed_selections": [], "selection": None, "decision_artifact": None, "review_packet": None},
            "phase5": {"status": "not_reached", "attempt_id": None, "allowed_selections": [], "selection": None, "decision_artifact": None, "review_packet": None},
        },
        "history": [],
        "last_error": None,
        "final_delivery": {},
    }
    record_history(state, "run_initialized", run_id=run_id, concept_path=repository_relative(run_root / "concept.txt"))
    return state


def schema_summary(phase: int, definition: str | None) -> str:
    schema = load_json(schema_path(phase))
    contract = schema if definition is None else schema["$defs"][definition]
    required = contract.get("required", [])
    constants = {
        key: value["const"] for key, value in contract.get("properties", {}).items()
        if isinstance(value, dict) and "const" in value
    }
    return json.dumps({"schema": repository_relative(schema_path(phase)), "definition": definition or "root", "required": required, "constants": constants}, indent=2)


def task_spec(state: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    step = state["current_step"]
    phase = phase_from_step(step)
    if phase is None:
        raise PipelineError(f"no Agent task for step {step}")
    attempt = active_attempt(state, phase)
    attempt_root = resolve_repository_path(attempt["path"])
    attempt_id = attempt["attempt_id"]
    reports = run_root / "reports"
    common_inputs = [STAGE5_ROOT / "README.md", WORKFLOW_ROOT / "WORKFLOW.md"]
    if step == "phase0.author":
        prompt = WORKFLOW_ROOT / "phase0" / "prompt.md"
        schema = schema_path(0)
        reads = [run_root / "concept.txt", *common_inputs, prompt, schema]
        outputs = [attempt_root / "scope.json", reports / f"phase0-{attempt_id}.md"]
        definition = None
        validation = "Validate scope.json against workflow/phase0/schema.json."
    elif step == "phase1.author":
        prompt = WORKFLOW_ROOT / "phase1" / "prompt.md"
        schema = schema_path(1)
        reads = [latest_artifact_path(state, "scope", "phase0"), *common_inputs, prompt, schema]
        outputs = [attempt_root / "semantic-contract.json", reports / f"phase1-{attempt_id}.md"]
        definition = None
        validation = "Validate semantic-contract.json against workflow/phase1/schema.json."
    elif step == "phase2.build":
        prompt = WORKFLOW_ROOT / "phase2" / "builder_prompt.md"
        schema = schema_path(2)
        reads = [latest_artifact_path(state, "semantic_contract", "phase1"), *common_inputs, prompt, schema]
        outputs = [attempt_root / "plan.json", attempt_root / "prototype.py", reports / f"phase2-builder-{attempt_id}.md"]
        definition = "plan"
        validation = "Validate plan.json against $defs.plan and compile prototype.py without executing it."
    elif step == "phase2.review":
        prompt = WORKFLOW_ROOT / "phase2" / "reviewer_prompt.md"
        schema = schema_path(2)
        reads = [
            latest_artifact_path(state, "semantic_contract", "phase1"),
            latest_artifact_path(state, "plan", "phase2"),
            latest_artifact_path(state, "candidate_outputs", "phase2"),
            *common_inputs, prompt, schema,
        ]
        outputs = [attempt_root / "selection.json", reports / f"phase2-review-{attempt_id}.md"]
        definition = "selection"
        validation = "Validate selection.json against $defs.selection and select only a real machine-passed candidate."
    elif step == "phase3.schedule":
        prompt = WORKFLOW_ROOT / "phase3" / "scheduler_prompt.md"
        schema = schema_path(3)
        reads = [
            latest_artifact_path(state, "semantic_contract", "phase1"),
            latest_artifact_path(state, "plan", "phase2"),
            latest_artifact_path(state, "executable_spec", "phase2"),
            *common_inputs, prompt, schema,
        ]
        outputs = [attempt_root / "schedule.json", reports / f"phase3-schedule-{attempt_id}.md"]
        definition = "schedule"
        validation = "Validate schedule.json against $defs.schedule and preserve every ordered Phase 2 probe anchor exactly."
    elif step == "phase4.author":
        prompt = WORKFLOW_ROOT / "phase4" / "prompt.md"
        schema = schema_path(4)
        reads = [
            latest_artifact_path(state, "semantic_contract", "phase1"),
            latest_artifact_path(state, "sequence_manifest", "phase3"),
            *common_inputs, prompt, schema,
        ]
        outputs = [attempt_root / "presentation.json", reports / f"phase4-authoring-{attempt_id}.md"]
        definition = "presentation"
        validation = "Run workflow/phase4/runtime.py validate-presentation on presentation.json."
    elif step == "phase5.build":
        prompt = WORKFLOW_ROOT / "phase5" / "builder_prompt.md"
        schema = schema_path(5)
        selected_manifest = latest_artifact_path(state, f"teaching_manifest.{state['human_gates']['phase4']['selection']}", "phase4")
        reads = [
            latest_artifact_path(state, "semantic_contract", "phase1"),
            latest_artifact_path(state, "sequence_archive", "phase3"),
            latest_artifact_path(state, "sequence_manifest", "phase3"),
            latest_artifact_path(state, "presentation", "phase4"),
            selected_manifest,
            *common_inputs, prompt, schema,
        ]
        outputs = [attempt_root / "appearance-plan.json", reports / f"phase5-builder-{attempt_id}.md"]
        definition = "appearance_plan"
        validation = "Run workflow/phase5/runtime.py validate-plan on appearance-plan.json."
    elif step == "phase5.review":
        prompt = WORKFLOW_ROOT / "phase5" / "reviewer_prompt.md"
        schema = schema_path(5)
        reads = [
            latest_artifact_path(state, "appearance_plan", "phase5"),
            latest_artifact_path(state, "appearance_execution", "phase5"),
            latest_artifact_path(state, "appearance_outputs", "phase5"),
            *common_inputs, prompt, schema,
        ]
        outputs = [attempt_root / "appearance-review.json", reports / f"phase5-review-{attempt_id}.md"]
        definition = "appearance_review"
        validation = "Run workflow/phase5/runtime.py validate-review on appearance-review.json; recommendation is not approval."
    else:
        raise PipelineError(f"unsupported Agent step: {step}")
    report_path = outputs[-1]
    task_path = run_root / "tasks" / f"{step.replace('.', '-')}-{attempt_id}.md"
    return {
        "step": step,
        "phase": phase,
        "attempt_id": attempt_id,
        "attempt_root": attempt_root,
        "prompt": prompt,
        "schema": schema,
        "definition": definition,
        "reads": reads,
        "outputs": outputs,
        "report": report_path,
        "validation": validation,
        "task_path": task_path,
    }


def write_agent_task(state: dict[str, Any], run_root: Path) -> Path:
    spec = task_spec(state, run_root)
    lines = [
        f"# Stage 5 Agent Task — {spec['step']}",
        "",
        f"- Run root: `{repository_relative(run_root)}`",
        f"- Attempt root: `{repository_relative(spec['attempt_root'])}`",
        f"- Phase and step: `{spec['phase']}` / `{spec['step']}`",
        f"- Authoritative workflow prompt: `{repository_relative(spec['prompt'])}`",
        f"- Schema: `{repository_relative(spec['schema'])}`",
        "",
        "## Allowed reads",
        "",
        *[f"- `{repository_relative(path)}`" for path in spec["reads"]],
        "",
        "## Allowed writes",
        "",
        *[f"- `{repository_relative(path)}`" for path in spec["outputs"]],
        "",
        "Do not modify any other file. Existing attempts and registered artifacts are immutable.",
        "",
        "## Output contract",
        "",
        "```json",
        schema_summary(phase_number(spec["phase"]), spec["definition"]),
        "```",
    ]
    if spec["step"] == "phase2.build":
        lines.extend(["", "`prototype.py` must expose the exact interface required by the authoritative Builder prompt."])
    registered_lines = [
        f"- `{record['role']}`: `{record['path']}` (`{record['sha256']}`)"
        for record in state["artifact_registry"].values() if record["status"] == "active"
    ] or ["- None beyond the allowed reads above."]
    lines.extend([
        "",
        "## Registered inputs",
        "",
        *registered_lines,
        "",
        "## Required validation",
        "",
        spec["validation"],
        f"Write a concise Markdown phase report to `{repository_relative(spec['report'])}`.",
        "",
        "## Stop condition",
        "",
        "Stop after writing and validating only the allowed outputs. Do not run the next Runtime, select for a human, or continue into another phase. Then call `run_pipeline.py next` for this same run.",
        "",
    ])
    atomic_write_text(spec["task_path"], "\n".join(lines))
    state["status"] = "waiting_for_agent"
    record_history(state, "agent_task_emitted", step=spec["step"], task=repository_relative(spec["task_path"]), attempt_id=spec["attempt_id"])
    return spec["task_path"]


def report_if_missing(path: Path, *, title: str, outputs: Iterable[Path]) -> None:
    if path.exists():
        if not path.is_file():
            raise PipelineError(f"phase report path is not a file: {path}")
        return
    lines = [f"# {title}", "", "Validated outputs:", ""]
    lines.extend(f"- `{repository_relative(item)}`" for item in outputs)
    lines.extend(["", "Status: `validated_by_pipeline`", ""])
    atomic_write_text(path, "\n".join(lines))


def transition(state: dict[str, Any], next_step: str, *, event: str) -> None:
    state["current_step"] = next_step
    state["current_phase"] = phase_from_step(next_step) or "phase6"
    state["status"] = "completed" if next_step == "completed" else "running"
    state["last_error"] = None
    if next_step != "completed":
        set_attempt_status(state, state["current_phase"], "active")
    record_history(state, event, next_step=next_step)


def process_agent_output(state: dict[str, Any], run_root: Path) -> bool:
    spec = task_spec(state, run_root)
    data_outputs = spec["outputs"][:-1]
    if not all(path.is_file() for path in data_outputs):
        write_agent_task(state, run_root)
        return False
    step = spec["step"]
    phase = spec["phase"]
    if step == "phase0.author":
        scope = validate_json_schema(data_outputs[0], 0)
        register_artifact(state, data_outputs[0], phase=phase, role="scope")
        report_if_missing(spec["report"], title=f"Phase 0 {spec['attempt_id']} Report", outputs=data_outputs)
        register_artifact(state, spec["report"], phase=phase, role="phase_report")
        set_attempt_status(state, phase, "completed")
        if scope["scope_status"] == "unsuitable":
            state["status"] = "completed"
            state["current_step"] = "completed"
            state["current_phase"] = "phase0"
            state["final_delivery"] = {"completion_reason": "stopped_unsuitable"}
            record_history(state, "stopped_unsuitable")
            write_run_report(state, run_root)
            return True
        transition(state, NEXT_STEP[step], event="phase0_validated")
    elif step == "phase1.author":
        validate_json_schema(data_outputs[0], 1)
        register_artifact(state, data_outputs[0], phase=phase, role="semantic_contract")
        report_if_missing(spec["report"], title=f"Phase 1 {spec['attempt_id']} Report", outputs=data_outputs)
        register_artifact(state, spec["report"], phase=phase, role="phase_report")
        set_attempt_status(state, phase, "completed")
        transition(state, NEXT_STEP[step], event="phase1_validated")
    elif step == "phase2.build":
        validate_json_schema(data_outputs[0], 2, "plan")
        try:
            compile(data_outputs[1].read_text(encoding="utf-8"), str(data_outputs[1]), "exec")
        except SyntaxError as exc:
            raise PipelineError(f"prototype.py does not compile: {exc}") from exc
        register_artifact(state, data_outputs[0], phase=phase, role="plan")
        register_artifact(state, data_outputs[1], phase=phase, role="prototype")
        report_if_missing(spec["report"], title=f"Phase 2 Builder {spec['attempt_id']} Report", outputs=data_outputs)
        register_artifact(state, spec["report"], phase=phase, role="builder_report")
        transition(state, NEXT_STEP[step], event="phase2_builder_validated")
    elif step == "phase2.review":
        validate_json_schema(data_outputs[0], 2, "selection")
        selection = load_json(data_outputs[0])
        if selection.get("selection_status") != "selected":
            raise PipelineError("Phase 2 Reviewer recorded no selection; use retry with the recorded return route")
        candidate_id = selection["selected_candidate_id"]
        probe = latest_artifact_path(state, "candidate_outputs", "phase2") / candidate_id / "probe-result.json"
        if not probe.is_file():
            raise PipelineError("Phase 2 selection does not identify a real executed candidate")
        probe_doc = validate_json_schema(probe, 2, "probe_result")
        if probe_doc.get("machine_gate_status") != "passed":
            raise PipelineError("Phase 2 selection identifies a candidate that did not pass the machine gate")
        register_artifact(state, data_outputs[0], phase=phase, role="selection")
        report_if_missing(spec["report"], title=f"Phase 2 Reviewer {spec['attempt_id']} Report", outputs=data_outputs)
        register_artifact(state, spec["report"], phase=phase, role="review_report")
        transition(state, NEXT_STEP[step], event="phase2_review_validated")
    elif step == "phase3.schedule":
        schedule = validate_json_schema(data_outputs[0], 3, "schedule")
        plan = load_json(latest_artifact_path(state, "plan", "phase2"))
        expected = [
            {"anchor_id": sample["sample_id"], "progress_values": sample["progress_values"]}
            for sample in plan["implementation"]["probe_samples"]
        ]
        if schedule["anchors"] != expected:
            raise PipelineError("Phase 3 schedule changed the ordered Phase 2 probe anchors")
        if schedule["frame_count"] != round(schedule["duration_seconds"] * schedule["fps"]):
            raise PipelineError("Phase 3 schedule duration, FPS, and frame count disagree")
        if schedule["start_hold_frames"] + sum(item["transition_frames"] for item in schedule["segments"]) + schedule["end_hold_frames"] != schedule["frame_count"]:
            raise PipelineError("Phase 3 schedule frame allocation does not close")
        register_artifact(state, data_outputs[0], phase=phase, role="schedule")
        report_if_missing(spec["report"], title=f"Phase 3 Schedule {spec['attempt_id']} Report", outputs=data_outputs)
        register_artifact(state, spec["report"], phase=phase, role="schedule_report")
        transition(state, NEXT_STEP[step], event="phase3_schedule_validated")
    elif step == "phase4.author":
        validate_with_runtime(4, "validate-presentation", data_outputs[0])
        register_artifact(state, data_outputs[0], phase=phase, role="presentation")
        report_if_missing(spec["report"], title=f"Phase 4 Authoring {spec['attempt_id']} Report", outputs=data_outputs)
        register_artifact(state, spec["report"], phase=phase, role="authoring_report")
        transition(state, NEXT_STEP[step], event="phase4_authoring_validated")
    elif step == "phase5.build":
        validate_with_runtime(5, "validate-plan", data_outputs[0])
        register_artifact(state, data_outputs[0], phase=phase, role="appearance_plan")
        report_if_missing(spec["report"], title=f"Phase 5 Builder {spec['attempt_id']} Report", outputs=data_outputs)
        register_artifact(state, spec["report"], phase=phase, role="builder_report")
        transition(state, NEXT_STEP[step], event="phase5_builder_validated")
    elif step == "phase5.review":
        validate_with_runtime(5, "validate-review", data_outputs[0])
        review = load_json(data_outputs[0])
        execution = load_json(latest_artifact_path(state, "appearance_execution", "phase5"))
        if review["appearance_execution"]["sha256"] != sha256_file(latest_artifact_path(state, "appearance_execution", "phase5")):
            raise PipelineError("Phase 5 review does not bind the registered execution")
        executed = {item["candidate_id"] for item in execution["candidates"] if item["candidate_id"] != "baseline"}
        reviewed = {item["candidate_id"] for item in review["candidate_assessments"] if item["candidate_id"] != "baseline"}
        if not executed.issubset(reviewed):
            raise PipelineError("Phase 5 review omitted an executed candidate")
        register_artifact(state, data_outputs[0], phase=phase, role="appearance_review")
        report_if_missing(spec["report"], title=f"Phase 5 Reviewer {spec['attempt_id']} Report", outputs=data_outputs)
        register_artifact(state, spec["report"], phase=phase, role="review_report")
        packet = build_phase5_review_packet(state, run_root)
        gate = state["human_gates"]["phase5"]
        gate.update({
            "status": "waiting",
            "attempt_id": active_attempt(state, "phase5")["attempt_id"],
            "allowed_selections": sorted(executed),
            "selection": None,
            "decision_artifact": None,
            "review_packet": repository_relative(packet),
        })
        state["current_step"] = "phase5.human_gate"
        state["current_phase"] = "phase5"
        state["status"] = "waiting_for_human"
        record_history(state, "phase5_human_gate_opened", allowed_selections=sorted(executed), review_packet=repository_relative(packet))
    else:
        raise PipelineError(f"unhandled Agent step: {step}")
    return True


def runtime_log_paths(state: Mapping[str, Any], phase: str, slug: str) -> tuple[Path, Path, Path]:
    root = active_attempt_path(state, phase) / "runtime-logs"
    return root / f"{slug}-command.json", root / f"{slug}-stdout.txt", root / f"{slug}-stderr.txt"


def invoke_runtime(state: dict[str, Any], phase: str, slug: str, command: Sequence[str]) -> None:
    command_path, stdout_path, stderr_path = runtime_log_paths(state, phase, slug)
    if any(path.exists() for path in (command_path, stdout_path, stderr_path)):
        raise PipelineError(f"Runtime log root already exists; refusing implicit rerun: {command_path.parent}")
    atomic_write_json(command_path, {"argv": list(command), "cwd": repository_relative(REPOSITORY_ROOT)})
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, text=True, capture_output=True)
    atomic_write_text(stdout_path, result.stdout)
    atomic_write_text(stderr_path, result.stderr)
    register_artifact(state, command_path, phase=phase, role=f"runtime_command.{slug}")
    register_artifact(state, stdout_path, phase=phase, role=f"runtime_stdout.{slug}")
    register_artifact(state, stderr_path, phase=phase, role=f"runtime_stderr.{slug}")
    record_history(state, "runtime_invoked", phase=phase, slug=slug, exit_code=result.returncode)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise PipelineError(f"{phase} Runtime {slug} failed: {message}")


def runtime_command(phase: int, subcommand: str, *args: str | Path) -> list[str]:
    return [runtime_python(), str(runtime_path(phase)), subcommand, *[str(item) for item in args]]


def process_phase2_candidates(state: dict[str, Any]) -> None:
    attempt = active_attempt_path(state, "phase2")
    output = attempt / "execution"
    summary = output / "executor-summary.json"
    if not summary.exists():
        if output.exists():
            raise PipelineError("partial Phase 2 candidate output exists; retry instead of overwriting")
        command = runtime_command(
            2, "run-candidates",
            "--semantic-contract", latest_artifact_path(state, "semantic_contract", "phase1"),
            "--plan", latest_artifact_path(state, "plan", "phase2"),
            "--output-root", output,
        )
        invoke_runtime(state, "phase2", "run-candidates", command)
    validate_json_schema(summary, 2, "executor_summary")
    probe_results = sorted(output.glob("candidate-*/probe-result.json"))
    if not probe_results:
        raise PipelineError("Phase 2 produced no completed candidate probe result")
    for result in probe_results:
        validate_json_schema(result, 2, "probe_result")
        register_artifact(state, result, phase="phase2", role=f"probe_result.{result.parent.name}")
    register_artifact(state, summary, phase="phase2", role="executor_summary")
    register_artifact(state, output, phase="phase2", role="candidate_outputs")
    transition(state, NEXT_STEP["phase2.run_candidates"], event="phase2_candidates_validated")


def process_phase2_freeze(state: dict[str, Any]) -> None:
    attempt = active_attempt_path(state, "phase2")
    output = attempt / "executable-spec.json"
    selection_path = latest_artifact_path(state, "selection", "phase2")
    selection = load_json(selection_path)
    candidate_id = selection["selected_candidate_id"]
    probe = latest_artifact_path(state, "candidate_outputs", "phase2") / candidate_id / "probe-result.json"
    if not output.exists():
        command = runtime_command(
            2, "freeze",
            "--selection", selection_path,
            "--probe-result", probe,
            "--semantic-contract", latest_artifact_path(state, "semantic_contract", "phase1"),
            "--plan", latest_artifact_path(state, "plan", "phase2"),
            "--output", output,
        )
        invoke_runtime(state, "phase2", "freeze", command)
    validate_json_schema(output, 2, "executable_spec")
    executable = load_json(output)
    if executable["selected_candidate_id"] != candidate_id:
        raise PipelineError("frozen executable candidate differs from Reviewer selection")
    register_artifact(state, output, phase="phase2", role="executable_spec")
    set_attempt_status(state, "phase2", "completed")
    transition(state, NEXT_STEP["phase2.freeze"], event="phase2_frozen")


def process_phase3_sequence(state: dict[str, Any]) -> None:
    attempt = active_attempt_path(state, "phase3")
    output = attempt / "sequence"
    manifest = output / "sequence-manifest.json"
    archive = output / "sequence.npz"
    if not (manifest.exists() and archive.exists()):
        if output.exists():
            raise PipelineError("partial Phase 3 output exists; retry instead of overwriting")
        command = runtime_command(
            3, "build-sequence",
            "--semantic-contract", latest_artifact_path(state, "semantic_contract", "phase1"),
            "--executable-spec", latest_artifact_path(state, "executable_spec", "phase2"),
            "--plan", latest_artifact_path(state, "plan", "phase2"),
            "--prototype", latest_artifact_path(state, "prototype", "phase2"),
            "--schedule", latest_artifact_path(state, "schedule", "phase3"),
            "--output-directory", output,
        )
        invoke_runtime(state, "phase3", "build-sequence", command)
    validate_with_runtime(3, "validate-manifest", manifest)
    sequence = load_json(manifest)
    if sequence["source"]["semantic_contract"]["sha256"] != sha256_file(latest_artifact_path(state, "semantic_contract", "phase1")):
        raise PipelineError("Phase 3 sequence does not bind the registered semantic contract")
    register_artifact(state, archive, phase="phase3", role="sequence_archive")
    register_artifact(state, manifest, phase="phase3", role="sequence_manifest")
    set_attempt_status(state, "phase3", "completed")
    transition(state, NEXT_STEP["phase3.build_sequence"], event="phase3_sequence_validated")


def phase4_style_ids(presentation_path: Path) -> list[str]:
    presentation = load_json(presentation_path)
    styles = presentation.get("review_style_ids")
    if not isinstance(styles, list) or not styles or any(not isinstance(item, str) for item in styles):
        raise PipelineError("Phase 4 presentation contains no formal review style IDs")
    return styles


def save_png_atomic(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    image.save(temporary, format="PNG", optimize=False)
    os.replace(temporary, path)


def build_image_grid(rows: list[tuple[str, list[tuple[str, Path]]]], destination: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise PipelineError("Pillow is required to assemble human review packets") from exc
    columns = max((len(items) for _, items in rows), default=0)
    if not rows or columns == 0:
        raise PipelineError("cannot build an empty review contact sheet")
    panel_width, panel_height, label_height = 440, 300, 24
    canvas = Image.new("RGB", (columns * panel_width, len(rows) * (panel_height + label_height)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for row_index, (row_label, items) in enumerate(rows):
        for column_index, (column_label, path) in enumerate(items):
            with Image.open(path) as source:
                panel = source.convert("RGB").resize((panel_width, panel_height), Image.Resampling.LANCZOS)
            x = column_index * panel_width
            y = row_index * (panel_height + label_height)
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y + panel_height, x + panel_width, y + panel_height + label_height), fill=(24, 24, 24))
            draw.text((x + 6, y + panel_height + 5), f"{row_label} · {column_label}", fill=(235, 235, 235))
    save_png_atomic(canvas, destination)


def encode_still_review(source: Path, destination: Path, *, seconds: int = 6) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(source),
        "-t", str(seconds), "-r", "12", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise PipelineError(f"review MP4 encoding failed: {result.stderr.strip()}")


def build_phase4_review_packet(state: dict[str, Any], run_root: Path, styles: Sequence[str]) -> Path:
    attempt = active_attempt_path(state, "phase4")
    review = attempt / "review"
    review.mkdir(exist_ok=True)
    manifests: dict[str, dict[str, Any]] = {}
    rows: list[tuple[str, list[tuple[str, Path]]]] = []
    for style in styles:
        manifest_path = attempt / "candidates" / style / "teaching-manifest.json"
        manifest = load_json(manifest_path)
        manifests[style] = manifest
        count = manifest["timeline"]["frame_count"]
        indices = sorted({0, count // 3, (2 * count) // 3, count - 1})
        frames = {item["frame_index"]: resolve_artifact_reference(item["artifact"], base=manifest_path.parent) for item in manifest["frames"]}
        rows.append((style, [(f"frame {index}", frames[index]) for index in indices]))
    contact = review / "style-contact-sheet.png"
    build_image_grid(rows, contact)
    review_mp4 = review / "style-review.mp4"
    encode_still_review(contact, review_mp4)
    metrics = review / "layout-metrics.json"
    atomic_write_json(metrics, {
        "schema_version": "stage5-pipeline-phase4-review-metrics-1",
        "styles": [{
            "style_id": style,
            "teaching_manifest": artifact_ref(attempt / "candidates" / style / "teaching-manifest.json"),
            "timeline": manifests[style]["timeline"],
            "layout": manifests[style]["layout"],
            "layout_checks": manifests[style]["layout_checks"],
            "deterministic_replay": manifests[style]["deterministic_replay"],
        } for style in styles],
    })
    packet = run_root / "tasks" / f"phase4-human-review-{active_attempt(state, 'phase4')['attempt_id']}.md"
    lines = [
        "# Phase 4 Human Review Gate", "",
        f"Allowed style IDs: {', '.join(f'`{style}`' for style in styles)}", "",
        f"- Representative contact sheet: `{repository_relative(contact)}`",
        f"- Review MP4: `{repository_relative(review_mp4)}`",
        f"- Layout metrics: `{repository_relative(metrics)}`",
        "- Agent recommendation: none is treated as approval; inspect any Phase 4 report separately.", "",
        "Approve exactly one real rendered style with:", "",
        "```bash",
        f"python {repository_relative(STAGE5_ROOT / 'run_pipeline.py')} approve \\",
        f"  --run {repository_relative(run_root)} --phase 4 \\",
        "  --selection <style-id> --notes \"<human review notes>\"",
        "```", "",
    ]
    atomic_write_text(packet, "\n".join(lines))
    for path, role in ((contact, "review_contact_sheet"), (review_mp4, "review_mp4"), (metrics, "review_metrics"), (packet, "human_review_packet")):
        register_artifact(state, path, phase="phase4", role=role)
    return packet


def process_phase4_styles(state: dict[str, Any], run_root: Path) -> None:
    attempt = active_attempt_path(state, "phase4")
    presentation = latest_artifact_path(state, "presentation", "phase4")
    styles = phase4_style_ids(presentation)
    for style in styles:
        output = attempt / "candidates" / style
        manifest = output / "teaching-manifest.json"
        if not manifest.exists():
            if output.exists():
                raise PipelineError(f"partial Phase 4 style output exists: {output}")
            command = runtime_command(
                4, "render-teaching",
                "--semantic-contract", latest_artifact_path(state, "semantic_contract", "phase1"),
                "--sequence-archive", latest_artifact_path(state, "sequence_archive", "phase3"),
                "--sequence-manifest", latest_artifact_path(state, "sequence_manifest", "phase3"),
                "--presentation", presentation,
                "--style-id", style,
                "--output-dir", output,
            )
            invoke_runtime(state, "phase4", f"render-{style}", command)
        validate_with_runtime(4, "validate-teaching-manifest", manifest)
        document = load_json(manifest)
        if document["layout"].get("style_id") != style:
            raise PipelineError(f"Phase 4 manifest style mismatch: {style}")
        register_artifact(state, manifest, phase="phase4", role=f"teaching_manifest.{style}")
        register_artifact(state, output / "frames", phase="phase4", role=f"teaching_frames.{style}")
    packet = build_phase4_review_packet(state, run_root, styles)
    gate = state["human_gates"]["phase4"]
    gate.update({
        "status": "waiting",
        "attempt_id": active_attempt(state, "phase4")["attempt_id"],
        "allowed_selections": list(styles),
        "selection": None,
        "decision_artifact": None,
        "review_packet": repository_relative(packet),
    })
    state["current_step"] = "phase4.human_gate"
    state["current_phase"] = "phase4"
    state["status"] = "waiting_for_human"
    record_history(state, "phase4_human_gate_opened", allowed_selections=list(styles), review_packet=repository_relative(packet))


def plan_model_paths(plan: Mapping[str, Any]) -> tuple[Path, Path]:
    generation = plan["local_generation"]
    root = resolve_external_or_repository_path(generation["model_root"])
    inventory = resolve_external_or_repository_path(generation["model_inventory"]["path"])
    return root, inventory


def process_phase5_appearance(state: dict[str, Any]) -> None:
    attempt = active_attempt_path(state, "phase5")
    output = attempt / "execution"
    execution_path = output / "appearance-execution.json"
    plan_path = latest_artifact_path(state, "appearance_plan", "phase5")
    if not execution_path.exists():
        if output.exists():
            raise PipelineError("partial Phase 5 execution exists; retry instead of overwriting")
        plan = load_json(plan_path)
        model_root, inventory = plan_model_paths(plan)
        selected_style = state["human_gates"]["phase4"]["selection"]
        command = runtime_command(
            5, "run-appearance",
            "--semantic-contract", latest_artifact_path(state, "semantic_contract", "phase1"),
            "--sequence-archive", latest_artifact_path(state, "sequence_archive", "phase3"),
            "--sequence-manifest", latest_artifact_path(state, "sequence_manifest", "phase3"),
            "--presentation", latest_artifact_path(state, "presentation", "phase4"),
            "--teaching-manifest", latest_artifact_path(state, f"teaching_manifest.{selected_style}", "phase4"),
            "--appearance-plan", plan_path,
            "--local-model-root", model_root,
            "--model-inventory", inventory,
            "--output-directory", output,
        )
        invoke_runtime(state, "phase5", "run-appearance", command)
    validate_with_runtime(5, "validate-execution", execution_path, ("--appearance-plan", str(plan_path)))
    execution = load_json(execution_path)
    if execution["status"] != "complete_pending_agent_review":
        raise PipelineError("Phase 5 execution is not ready for Reviewer work")
    register_artifact(state, execution_path, phase="phase5", role="appearance_execution")
    register_artifact(state, output, phase="phase5", role="appearance_outputs")
    transition(state, NEXT_STEP["phase5.run_appearance"], event="phase5_execution_validated")


def build_phase5_review_packet(state: dict[str, Any], run_root: Path) -> Path:
    attempt = active_attempt_path(state, "phase5")
    review_root = attempt / "review"
    review_root.mkdir(exist_ok=True)
    execution_path = latest_artifact_path(state, "appearance_execution", "phase5")
    execution = load_json(execution_path)
    material_items: list[tuple[str, Path]] = []
    for job in execution["jobs"]:
        material_items.append((job["candidate_asset_id"], resolve_artifact_reference(job["asset"])))
    material_sheet = review_root / "material-contact-sheet.png"
    build_image_grid([("generated materials", material_items)], material_sheet)
    comparison_rows: list[tuple[str, list[tuple[str, Path]]]] = []
    for candidate in execution["candidates"]:
        comparison_rows.append((candidate["candidate_id"], [
            (probe["probe_id"], resolve_artifact_reference(probe["artifact"])) for probe in candidate["probes"]
        ]))
    comparison = review_root / "baseline-candidate-comparison.png"
    build_image_grid(comparison_rows, comparison)
    review_mp4 = review_root / "appearance-review.mp4"
    encode_still_review(comparison, review_mp4)
    metrics = review_root / "preservation-metrics.json"
    review_path = latest_artifact_path(state, "appearance_review", "phase5")
    review = load_json(review_path)
    atomic_write_json(metrics, {
        "schema_version": "stage5-pipeline-phase5-review-metrics-1",
        "execution_checks": execution["checks"],
        "deterministic_replay": execution["deterministic_replay"],
        "prominence_metrics": execution["prominence_metrics"],
        "review_recommendation": review["recommendation"],
        "review_warnings": review["warnings"],
    })
    allowed = sorted(item["candidate_id"] for item in execution["candidates"] if item["candidate_id"] != "baseline")
    packet = run_root / "tasks" / f"phase5-human-review-{active_attempt(state, 'phase5')['attempt_id']}.md"
    lines = [
        "# Phase 5 Human Review Gate", "",
        f"Allowed executed candidate IDs: {', '.join(f'`{item}`' for item in allowed)}", "",
        f"- Material contact sheet: `{repository_relative(material_sheet)}`",
        f"- Baseline/candidate comparison: `{repository_relative(comparison)}`",
        f"- Review MP4: `{repository_relative(review_mp4)}`",
        f"- Appearance review: `{repository_relative(review_path)}`",
        f"- Preservation metrics: `{repository_relative(metrics)}`", "",
        "The Agent recommendation is evidence only and is not approval.", "",
        "```bash",
        f"python {repository_relative(STAGE5_ROOT / 'run_pipeline.py')} approve \\",
        f"  --run {repository_relative(run_root)} --phase 5 \\",
        "  --selection <candidate-id> --notes \"<human review notes>\"",
        "```", "",
    ]
    atomic_write_text(packet, "\n".join(lines))
    for path, role in ((material_sheet, "material_contact_sheet"), (comparison, "candidate_comparison"), (review_mp4, "review_mp4"), (metrics, "preservation_metrics"), (packet, "human_review_packet")):
        register_artifact(state, path, phase="phase5", role=role)
    return packet


def process_phase5_pack(state: dict[str, Any]) -> None:
    attempt = active_attempt_path(state, "phase5")
    output = attempt / "appearance-pack.json"
    if not output.exists():
        command = runtime_command(
            5, "assemble-pack",
            "--appearance-plan", latest_artifact_path(state, "appearance_plan", "phase5"),
            "--appearance-execution", latest_artifact_path(state, "appearance_execution", "phase5"),
            "--appearance-review", latest_artifact_path(state, "appearance_review", "phase5"),
            "--human-decision", latest_artifact_path(state, "human_decision", "phase5"),
            "--output-path", output,
        )
        invoke_runtime(state, "phase5", "assemble-pack", command)
    validate_with_runtime(5, "validate-pack", output)
    pack = load_json(output)
    if pack["status"] != "approved_for_phase6":
        raise PipelineError("Phase 5 pack does not authorize Phase 6")
    register_artifact(state, output, phase="phase5", role="appearance_pack")
    set_attempt_status(state, "phase5", "completed")
    transition(state, NEXT_STEP["phase5.assemble_pack"], event="phase5_pack_assembled")


def process_phase6_delivery(state: dict[str, Any], run_root: Path) -> None:
    attempt = active_attempt_path(state, "phase6")
    output = attempt / "delivery"
    manifest = output / "delivery-manifest.json"
    selected_style = state["human_gates"]["phase4"]["selection"]
    if not manifest.exists():
        if output.exists():
            raise PipelineError("partial Phase 6 delivery exists; retry instead of overwriting")
        command = runtime_command(
            6, "render-delivery",
            "--semantic-contract", latest_artifact_path(state, "semantic_contract", "phase1"),
            "--sequence-archive", latest_artifact_path(state, "sequence_archive", "phase3"),
            "--sequence-manifest", latest_artifact_path(state, "sequence_manifest", "phase3"),
            "--presentation", latest_artifact_path(state, "presentation", "phase4"),
            "--teaching-frames", latest_artifact_path(state, f"teaching_frames.{selected_style}", "phase4"),
            "--teaching-manifest", latest_artifact_path(state, f"teaching_manifest.{selected_style}", "phase4"),
            "--phase4-human-decision", latest_artifact_path(state, "human_decision", "phase4"),
            "--appearance-pack", latest_artifact_path(state, "appearance_pack", "phase5"),
            "--phase5-human-decision", latest_artifact_path(state, "human_decision", "phase5"),
            "--output-directory", output,
        )
        invoke_runtime(state, "phase6", "render-delivery", command)
    validate_with_runtime(6, "validate-delivery-manifest", manifest)
    document = load_json(manifest)
    evaluation = resolve_artifact_reference(document["diagnostics"]["final_evaluation"])
    mp4 = resolve_artifact_reference(document["delivery_artifacts"]["mp4"]["artifact"])
    gif = resolve_artifact_reference(document["delivery_artifacts"]["gif"]["artifact"])
    frames = resolve_repository_path(document["final_frames"]["directory"])
    for path, role in (
        (manifest, "delivery_manifest"), (evaluation, "final_evaluation"),
        (mp4, "final_mp4"), (gif, "final_gif"), (frames, "final_frames"),
    ):
        register_artifact(state, path, phase="phase6", role=role)
    state["final_delivery"] = {
        "status": document["status"],
        "mp4": repository_relative(mp4),
        "gif": repository_relative(gif),
        "evaluation": repository_relative(evaluation),
        "manifest": repository_relative(manifest),
        "frames": repository_relative(frames),
    }
    set_attempt_status(state, "phase6", "completed")
    transition(state, "completed", event="pipeline_completed")
    write_run_report(state, run_root)


def process_runtime_step(state: dict[str, Any], run_root: Path) -> None:
    step = state["current_step"]
    if step == "phase2.run_candidates":
        process_phase2_candidates(state)
    elif step == "phase2.freeze":
        process_phase2_freeze(state)
    elif step == "phase3.build_sequence":
        process_phase3_sequence(state)
    elif step == "phase4.render_styles":
        process_phase4_styles(state, run_root)
    elif step == "phase5.run_appearance":
        process_phase5_appearance(state)
    elif step == "phase5.assemble_pack":
        process_phase5_pack(state)
    elif step == "phase6.render_delivery":
        process_phase6_delivery(state, run_root)
    else:
        raise PipelineError(f"unsupported Runtime step: {step}")


def set_failure(state: dict[str, Any], message: str) -> None:
    state["status"] = "failed"
    state["last_error"] = {"phase": state["current_phase"], "step": state["current_step"], "message": message}
    set_attempt_status(state, state["current_phase"], "failed")
    record_history(state, "step_failed", phase=state["current_phase"], step=state["current_step"], message=message)


def decision_path(run_root: Path, phase: int, attempt_id: str) -> Path:
    return run_root / "decisions" / f"phase{phase}-{attempt_id}-human-decision.json"


def validate_preexisting_decision(state: dict[str, Any], run_root: Path, phase: int) -> bool:
    gate = state["human_gates"][f"phase{phase}"]
    path = decision_path(run_root, phase, gate["attempt_id"])
    if not path.exists():
        return False
    apply_decision(state, run_root, phase=phase, selection=None, notes=None, existing_path=path)
    return True


def apply_decision(
    state: dict[str, Any], run_root: Path, *, phase: int, selection: str | None,
    notes: str | None, existing_path: Path | None = None,
) -> Path:
    if phase not in (4, 5):
        raise PipelineError("only Phase 4 and Phase 5 have pipeline approval gates")
    phase_name = f"phase{phase}"
    gate = state["human_gates"][phase_name]
    if state["status"] != "waiting_for_human" or state["current_phase"] != phase_name or gate["status"] != "waiting":
        raise PipelineError(f"run is not waiting at the Phase {phase} human gate")
    if existing_path is not None:
        decision = validate_json_schema(existing_path, phase, "human_decision")
        chosen = decision["selected_style_id"] if phase == 4 else decision["selected_candidate_ids"][0]
        output = existing_path
    else:
        if selection not in gate["allowed_selections"]:
            raise PipelineError(f"invalid selection; allowed IDs: {gate['allowed_selections']}")
        if not notes or not notes.strip():
            raise PipelineError("human approval requires nonempty notes")
        output = decision_path(run_root, phase, gate["attempt_id"])
        if output.exists():
            raise PipelineError(f"human decision already exists and will not be overwritten: {output}")
        if phase == 4:
            presentation = latest_artifact_path(state, "presentation", "phase4")
            manifest = latest_artifact_path(state, f"teaching_manifest.{selection}", "phase4")
            manifest_doc = load_json(manifest)
            binding = (
                f"teaching_manifest_sha256={sha256_file(manifest)}; "
                f"frame_replay_digest={manifest_doc['deterministic_replay']['first_digest']}"
            )
            decision = {
                "schema_version": "stage5-phase4-layout-human-decision-2",
                "phase": "phase4",
                "status": "approved",
                "presentation": artifact_ref(presentation),
                "selected_style_id": selection,
                "notes": f"{notes.strip()}; {binding}",
            }
        else:
            decision = {
                "schema_version": "stage5-phase5-human-decision-1",
                "phase": "phase5",
                "appearance_plan": artifact_ref(latest_artifact_path(state, "appearance_plan", "phase5")),
                "appearance_execution": artifact_ref(latest_artifact_path(state, "appearance_execution", "phase5")),
                "appearance_review": artifact_ref(latest_artifact_path(state, "appearance_review", "phase5")),
                "decision": "approved",
                "selected_candidate_ids": [selection],
                "reviewer": f"pipeline_human; notes: {notes.strip()}",
                "decided_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        atomic_write_json(output, decision)
        validate_json_schema(output, phase, "human_decision")
        chosen = selection
    if chosen not in gate["allowed_selections"]:
        raise PipelineError("pre-existing decision selects an ID absent from real gate evidence")
    if phase == 4 and decision.get("status") != "approved":
        raise PipelineError("Phase 4 pre-existing decision is not approved")
    if phase == 5 and decision.get("decision") != "approved":
        raise PipelineError("Phase 5 pre-existing decision is not approved")
    key = register_artifact(state, output, phase=phase_name, role="human_decision", external=existing_path is not None)
    gate.update({"status": "approved", "selection": chosen, "decision_artifact": state["artifact_registry"][key]["path"]})
    record_history(state, f"phase{phase}_human_approved", selection=chosen, decision=repository_relative(output))
    if phase == 4:
        set_attempt_status(state, "phase4", "completed")
        transition(state, "phase5.build", event="phase4_gate_closed")
    else:
        transition(state, "phase5.assemble_pack", event="phase5_gate_closed")
    return output


def next_once(run_root: Path) -> str:
    state = load_state(run_root)
    if state["status"] == "completed":
        print_completion(state, run_root)
        return "done"
    if state["status"] == "failed":
        raise PipelineError(f"run is failed: {state['last_error']}; use retry after diagnosis")
    if state["status"] == "waiting_for_human":
        phase = phase_number(state["current_phase"])
        if validate_preexisting_decision(state, run_root, phase):
            write_state(run_root, state)
            print(f"Registered pre-existing Phase {phase} human decision; next step: {state['current_step']}")
            return "progressed"
        print_human_gate(state)
        return "blocked"
    try:
        if state["current_step"] in AGENT_STEPS:
            progressed = process_agent_output(state, run_root)
            write_state(run_root, state)
            if not progressed:
                spec = task_spec(state, run_root)
                print(f"waiting_for_agent\ntask: {repository_relative(spec['task_path'])}")
                print("expected outputs:")
                for path in spec["outputs"]:
                    print(f"  {repository_relative(path)}")
                return "blocked"
            print(f"advanced to {state['current_step']}")
            return "progressed"
        if state["current_step"] in RUNTIME_STEPS:
            process_runtime_step(state, run_root)
            write_state(run_root, state)
            if state["status"] == "waiting_for_human":
                print_human_gate(state)
                return "blocked"
            if state["status"] == "completed":
                print_completion(state, run_root)
                return "done"
            print(f"advanced to {state['current_step']}")
            return "progressed"
        raise PipelineError(f"step cannot advance: {state['current_step']}")
    except Exception as exc:
        message = str(exc)
        set_failure(state, message)
        write_state(run_root, state)
        raise


def print_human_gate(state: Mapping[str, Any]) -> None:
    phase = state["current_phase"]
    gate = state["human_gates"][phase]
    print("waiting_for_human")
    print(f"phase: {phase}")
    print(f"review packet: {gate['review_packet']}")
    print(f"allowed selections: {', '.join(gate['allowed_selections'])}")


def print_completion(state: Mapping[str, Any], run_root: Path) -> None:
    print("completed")
    delivery = state["final_delivery"]
    if delivery.get("completion_reason") == "stopped_unsuitable":
        print("completion reason: stopped_unsuitable")
    else:
        for key in ("mp4", "gif", "evaluation", "manifest"):
            print(f"{key}: {delivery.get(key)}")
    print(f"run report: {repository_relative(run_root / 'run-report.md')}")


def missing_for_current_step(state: Mapping[str, Any], run_root: Path) -> list[str]:
    if state["current_step"] in AGENT_STEPS:
        spec = task_spec(state, run_root)
        return [repository_relative(path) for path in spec["outputs"] if not path.exists()]
    return []


def verify_registered_artifact(record: Mapping[str, Any]) -> None:
    path = resolve_repository_path(record["path"])
    if record["kind"] == "file":
        if not path.is_file():
            raise PipelineError(f"registered file is missing: {path}")
        digest, size, count = sha256_file(path), path.stat().st_size, 1
    else:
        if not path.is_dir():
            raise PipelineError(f"registered tree is missing: {path}")
        digest, size, count = tree_metadata(path)
    if digest != record["sha256"] or size != record["size_bytes"] or count != record["file_count"]:
        raise PipelineError(f"registered artifact changed: {record['path']}")


def verify_run(run_root: Path) -> dict[str, Any]:
    state = load_state(run_root)
    if (run_root / "concept.txt").read_text(encoding="utf-8") != state["concept"]:
        raise PipelineError("concept.txt differs from run-state concept")
    for record in state["artifact_registry"].values():
        verify_registered_artifact(record)
    for phase, attempts in state["attempts"].items():
        paths = [item["path"] for item in attempts]
        if len(paths) != len(set(paths)) or any(not resolve_repository_path(path).is_dir() for path in paths):
            raise PipelineError(f"attempt roots are missing or reused: {phase}")
    for phase in (4, 5):
        gate = state["human_gates"][f"phase{phase}"]
        if gate["status"] == "approved":
            if gate["selection"] not in gate["allowed_selections"]:
                raise PipelineError(f"Phase {phase} approval is outside real evidence")
            decision = latest_artifact_path(state, "human_decision", f"phase{phase}")
            validate_json_schema(decision, phase, "human_decision")
    if any(record["role"] == "executable_spec" and record["status"] == "active" for record in state["artifact_registry"].values()):
        validate_json_schema(latest_artifact_path(state, "executable_spec", "phase2"), 2, "executable_spec")
    if any(record["role"] == "sequence_manifest" and record["status"] == "active" for record in state["artifact_registry"].values()):
        validate_with_runtime(3, "validate-manifest", latest_artifact_path(state, "sequence_manifest", "phase3"))
    selected_style = state["human_gates"]["phase4"].get("selection")
    if selected_style:
        validate_with_runtime(4, "validate-teaching-manifest", latest_artifact_path(state, f"teaching_manifest.{selected_style}", "phase4"))
    if any(record["role"] == "appearance_pack" and record["status"] == "active" for record in state["artifact_registry"].values()):
        validate_with_runtime(5, "validate-pack", latest_artifact_path(state, "appearance_pack", "phase5"))
    if state["status"] == "completed" and state["final_delivery"].get("completion_reason") != "stopped_unsuitable":
        required = {"status", "mp4", "gif", "evaluation", "manifest", "frames"}
        if set(state["final_delivery"]) != required:
            raise PipelineError("completed run has incomplete final_delivery state")
        validate_with_runtime(6, "validate-delivery-manifest", resolve_repository_path(state["final_delivery"]["manifest"]))
        if state["human_gates"]["phase4"]["status"] != "approved" or state["human_gates"]["phase5"]["status"] != "approved":
            raise PipelineError("completed delivery lacks both human approvals")
    return {
        "status": "verified",
        "run_id": state["run_id"],
        "run_status": state["status"],
        "artifact_count": len(state["artifact_registry"]),
        "history_entries": len(state["history"]),
    }


def write_run_report(state: Mapping[str, Any], run_root: Path) -> Path:
    scope_path = None
    try:
        scope_path = latest_artifact_path(state, "scope", "phase0")
    except PipelineError:
        pass
    scoped = load_json(scope_path).get("scoped_concept") if scope_path else None
    executable = None
    try:
        executable = load_json(latest_artifact_path(state, "executable_spec", "phase2"))
    except PipelineError:
        pass
    reports = sorted(
        record["path"] for record in state["artifact_registry"].values()
        if record["status"] == "active" and "report" in record["role"]
    )
    warnings: list[str] = []
    if state["final_delivery"].get("evaluation"):
        evaluation = load_json(resolve_repository_path(state["final_delivery"]["evaluation"]))
        warnings = list(evaluation.get("known_visual_warnings", []))
    lines = [
        "# Stage 5 Run Report", "",
        "## Concept and scope", "",
        f"- Run ID: `{state['run_id']}`",
        f"- Original concept: {state['concept']}",
        f"- Scoped concept: {scoped or 'Not available' }", "",
        "## Phase results", "",
    ]
    for phase in PHASES:
        attempts = state["attempts"][phase]
        lines.append(f"- {phase}: " + ", ".join(f"`{item['attempt_id']}` ({item['status']})" for item in attempts))
    lines.extend([
        "",
        f"- Selected program candidate: `{executable.get('selected_candidate_id') if executable else 'not available'}`",
        "",
        "## Human decisions", "",
        f"- Phase 4 style: `{state['human_gates']['phase4'].get('selection') or 'not selected'}`",
        f"- Phase 5 candidate: `{state['human_gates']['phase5'].get('selection') or 'not selected'}`",
        "",
        "## Final delivery", "",
    ])
    if state["final_delivery"].get("completion_reason"):
        lines.append(f"- Completion reason: `{state['final_delivery']['completion_reason']}`")
    else:
        for key in ("mp4", "gif", "evaluation", "manifest", "frames"):
            lines.append(f"- {key}: `{state['final_delivery'].get(key, 'not available')}`")
    lines.extend(["", "## Artifact lineage", ""])
    for record in sorted(state["artifact_registry"].values(), key=lambda item: (phase_number(item["phase"]), item["attempt_id"], item["role"])):
        if record["status"] == "active" and record["role"] in {
            "scope", "semantic_contract", "executable_spec", "sequence_manifest", "presentation",
            "human_decision", "appearance_pack", "delivery_manifest",
        }:
            lines.append(f"- {record['phase']} / {record['role']}: `{record['path']}` (`{record['sha256']}`)")
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in warnings] or ["- None recorded."])
    lines.extend(["", "## Reproducibility", ""])
    lines.append(f"- Run state: `{repository_relative(state_path(run_root))}`")
    lines.append(f"- Controller tasks: `{repository_relative(run_root / 'tasks')}`")
    for path in reports:
        lines.append(f"- Phase report: `{path}`")
    for phase in ("phase4", "phase5"):
        packet = state["human_gates"][phase].get("review_packet")
        if packet:
            lines.append(f"- {phase} review packet: `{packet}`")
    lines.append("")
    path = run_root / "run-report.md"
    atomic_write_text(path, "\n".join(lines))
    return path


def retry_run(state: dict[str, Any], run_root: Path, phase_number_value: int, reason: str) -> None:
    if phase_number_value < 0 or phase_number_value > 6:
        raise PipelineError("retry phase must be between 0 and 6")
    if state["status"] == "completed":
        raise PipelineError("a completed run is immutable; initialize a new run")
    current_number = phase_number(state["current_phase"])
    if phase_number_value > current_number:
        raise PipelineError("cannot retry a phase that the run has not reached")
    target = f"phase{phase_number_value}"
    for number in range(phase_number_value, 7):
        phase = f"phase{number}"
        previous = active_attempt(state, phase)
        previous["status"] = "superseded"
        attempt_id = f"attempt-{len(state['attempts'][phase]) + 1:03d}"
        root = run_root / phase / attempt_id
        if root.exists():
            raise PipelineError(f"retry attempt root already exists: {root}")
        root.mkdir(parents=True)
        previous["superseded_by"] = attempt_id
        state["attempts"][phase].append({
            "attempt_id": attempt_id,
            "path": repository_relative(root),
            "status": "active" if number == phase_number_value else "pending",
            "superseded_by": None,
            "reason": reason if number == phase_number_value else "downstream invalidated by upstream retry",
        })
    for record in state["artifact_registry"].values():
        if phase_number(record["phase"]) >= phase_number_value and record["status"] == "active":
            record["status"] = "superseded"
    for gate_phase in (4, 5):
        if gate_phase >= phase_number_value:
            state["human_gates"][f"phase{gate_phase}"] = {
                "status": "not_reached", "attempt_id": None, "allowed_selections": [],
                "selection": None, "decision_artifact": None, "review_packet": None,
            }
    initial_steps = {
        0: "phase0.author", 1: "phase1.author", 2: "phase2.build", 3: "phase3.schedule",
        4: "phase4.author", 5: "phase5.build", 6: "phase6.render_delivery",
    }
    state["current_phase"] = target
    state["current_step"] = initial_steps[phase_number_value]
    state["status"] = "running"
    state["last_error"] = None
    state["final_delivery"] = {}
    record_history(state, "retry_created", phase=target, reason=reason, attempt_id=active_attempt(state, target)["attempt_id"])
    if state["current_step"] in AGENT_STEPS:
        write_agent_task(state, run_root)


def import_completed(args: argparse.Namespace) -> Path:
    validate_run_id(args.run_id)
    run_root = RUNS_ROOT / args.run_id
    state = create_run_layout(run_root, args.concept, args.run_id)
    inputs = {
        "scope": (Path(args.scope), "phase0", None),
        "semantic_contract": (Path(args.semantic_contract), "phase1", None),
        "plan": (Path(args.plan), "phase2", "plan"),
        "prototype": (Path(args.prototype), "phase2", None),
        "executable_spec": (Path(args.executable_spec), "phase2", "executable_spec"),
        "sequence_archive": (Path(args.sequence_archive), "phase3", None),
        "sequence_manifest": (Path(args.sequence_manifest), "phase3", None),
        "presentation": (Path(args.presentation), "phase4", "presentation"),
        "teaching_frames": (Path(args.teaching_frames), "phase4", None),
        "teaching_manifest": (Path(args.teaching_manifest), "phase4", "teaching_manifest"),
        "phase4_decision": (Path(args.phase4_human_decision), "phase4", "human_decision"),
        "appearance_plan": (Path(args.appearance_plan), "phase5", "appearance_plan"),
        "appearance_execution": (Path(args.appearance_execution), "phase5", "appearance_execution"),
        "appearance_review": (Path(args.appearance_review), "phase5", "appearance_review"),
        "phase5_decision": (Path(args.phase5_human_decision), "phase5", "human_decision"),
        "appearance_pack": (Path(args.appearance_pack), "phase5", "appearance_pack"),
    }
    resolved = {key: resolve_repository_path(path) for key, (path, _, _) in inputs.items()}
    validate_json_schema(resolved["scope"], 0)
    validate_json_schema(resolved["semantic_contract"], 1)
    validate_json_schema(resolved["plan"], 2, "plan")
    compile(resolved["prototype"].read_text(encoding="utf-8"), str(resolved["prototype"]), "exec")
    validate_json_schema(resolved["executable_spec"], 2, "executable_spec")
    validate_with_runtime(3, "validate-manifest", resolved["sequence_manifest"])
    validate_with_runtime(4, "validate-presentation", resolved["presentation"])
    validate_with_runtime(4, "validate-teaching-manifest", resolved["teaching_manifest"])
    validate_with_runtime(5, "validate-plan", resolved["appearance_plan"])
    validate_with_runtime(5, "validate-execution", resolved["appearance_execution"], ("--appearance-plan", str(resolved["appearance_plan"])))
    validate_with_runtime(5, "validate-review", resolved["appearance_review"])
    validate_with_runtime(5, "validate-human-decision", resolved["phase5_decision"])
    validate_with_runtime(5, "validate-pack", resolved["appearance_pack"])
    validate_json_schema(resolved["phase4_decision"], 4, "human_decision")
    delivery_root = resolve_repository_path(args.delivery_root)
    delivery_manifest = delivery_root / "delivery-manifest.json"
    validate_with_runtime(6, "validate-delivery-manifest", delivery_manifest)
    for phase in PHASES:
        active_attempt(state, phase)["status"] = "imported"
    role_map = {
        "phase4_decision": "human_decision", "phase5_decision": "human_decision",
        "teaching_manifest": f"teaching_manifest.{load_json(resolved['phase4_decision'])['selected_style_id']}",
        "teaching_frames": f"teaching_frames.{load_json(resolved['phase4_decision'])['selected_style_id']}",
    }
    for key, (_, phase, _) in inputs.items():
        register_artifact(state, resolved[key], phase=phase, role=role_map.get(key, key), external=True)
    manifest_doc = load_json(delivery_manifest)
    evaluation = resolve_artifact_reference(manifest_doc["diagnostics"]["final_evaluation"])
    mp4 = resolve_artifact_reference(manifest_doc["delivery_artifacts"]["mp4"]["artifact"])
    gif = resolve_artifact_reference(manifest_doc["delivery_artifacts"]["gif"]["artifact"])
    frames = resolve_repository_path(manifest_doc["final_frames"]["directory"])
    for path, role in ((delivery_manifest, "delivery_manifest"), (evaluation, "final_evaluation"), (mp4, "final_mp4"), (gif, "final_gif"), (frames, "final_frames")):
        register_artifact(state, path, phase="phase6", role=role, external=True)
    phase4_decision = load_json(resolved["phase4_decision"])
    phase5_decision = load_json(resolved["phase5_decision"])
    state["human_gates"]["phase4"] = {
        "status": "approved", "attempt_id": "attempt-001",
        "allowed_selections": [phase4_decision["selected_style_id"]],
        "selection": phase4_decision["selected_style_id"],
        "decision_artifact": repository_relative(resolved["phase4_decision"]),
        "review_packet": None,
    }
    state["human_gates"]["phase5"] = {
        "status": "approved", "attempt_id": "attempt-001",
        "allowed_selections": list(phase5_decision["selected_candidate_ids"]),
        "selection": phase5_decision["selected_candidate_ids"][0],
        "decision_artifact": repository_relative(resolved["phase5_decision"]),
        "review_packet": None,
    }
    state["status"] = "completed"
    state["current_phase"] = "phase6"
    state["current_step"] = "completed"
    state["final_delivery"] = {
        "status": manifest_doc["status"], "mp4": repository_relative(mp4), "gif": repository_relative(gif),
        "evaluation": repository_relative(evaluation), "manifest": repository_relative(delivery_manifest),
        "frames": repository_relative(frames),
    }
    state["history"] = []
    record_history(state, "completed_run_imported", source_delivery=repository_relative(delivery_root))
    write_run_report(state, run_root)
    write_state(run_root, state)
    verify_run(run_root)
    return run_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a new immutable orchestrated run and first Agent task.")
    init.add_argument("--concept", required=True)
    init.add_argument("--run-id")
    for name in ("status", "next", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--run", required=True, type=Path)
        if name == "next":
            sub.add_argument("--until-blocked", action="store_true")
    approve = commands.add_parser("approve", help="Record a real human Phase 4 or Phase 5 approval.")
    approve.add_argument("--run", required=True, type=Path)
    approve.add_argument("--phase", required=True, type=int, choices=(4, 5))
    approve.add_argument("--selection", required=True)
    approve.add_argument("--notes", required=True)
    retry = commands.add_parser("retry", help="Create immutable retry attempts from an allowed owning phase.")
    retry.add_argument("--run", required=True, type=Path)
    retry.add_argument("--phase", required=True, type=int, choices=range(0, 7))
    retry.add_argument("--reason", required=True)
    imported = commands.add_parser("import-completed", help="Register a completed formal run as read-only external evidence.")
    imported.add_argument("--run-id", required=True)
    imported.add_argument("--concept", required=True)
    for name in (
        "scope", "semantic-contract", "plan", "prototype", "executable-spec", "sequence-archive", "sequence-manifest",
        "presentation", "teaching-frames", "teaching-manifest", "phase4-human-decision", "appearance-plan",
        "appearance-execution", "appearance-review", "phase5-human-decision", "appearance-pack", "delivery-root",
    ):
        imported.add_argument(f"--{name}", required=True, type=Path, dest=name.replace("-", "_"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            concept = args.concept
            if not concept.strip() or "\x00" in concept or "\n" in concept or "\r" in concept:
                raise PipelineError("concept must be one nonempty line")
            run_id = args.run_id or choose_run_id(concept)
            validate_run_id(run_id)
            run_root = RUNS_ROOT / run_id
            state = create_run_layout(run_root, concept, run_id)
            task = write_agent_task(state, run_root)
            write_state(run_root, state)
            print(f"initialized: {repository_relative(run_root)}")
            print(f"task: {repository_relative(task)}")
            return 0
        if args.command == "status":
            run_root = resolve_run(args.run)
            state = load_state(run_root)
            current_attempt = active_attempt(state, state["current_phase"])["attempt_id"]
            print(f"run_id: {state['run_id']}")
            print(f"status: {state['status']}")
            print(f"current_phase: {state['current_phase']}")
            print(f"current_step: {state['current_step']}")
            print(f"current_attempt: {current_attempt}")
            print(f"validated_artifacts: {sum(record['status'] == 'active' for record in state['artifact_registry'].values())}")
            missing = missing_for_current_step(state, run_root)
            print(f"missing_expected_artifacts: {', '.join(missing) if missing else 'none'}")
            gate = state["human_gates"].get(state["current_phase"])
            print(f"human_gate: {gate['status'] if gate else 'not_applicable'}")
            if state["status"] == "waiting_for_agent":
                print("next_action: complete the emitted Agent task, then call next")
            elif state["status"] == "waiting_for_human":
                print("next_action: inspect the review packet and call approve or retry")
            elif state["status"] == "failed":
                print("next_action: diagnose last_error and call retry at the owning phase")
            elif state["status"] == "completed":
                print("next_action: inspect run-report.md and final delivery")
            else:
                print("next_action: call next")
            return 0
        if args.command == "next":
            run_root = resolve_run(args.run)
            while True:
                outcome = next_once(run_root)
                if not args.until_blocked or outcome in {"blocked", "done"}:
                    break
            return 0
        if args.command == "approve":
            run_root = resolve_run(args.run)
            state = load_state(run_root)
            output = apply_decision(state, run_root, phase=args.phase, selection=args.selection, notes=args.notes)
            write_state(run_root, state)
            print(f"approved Phase {args.phase}: {repository_relative(output)}")
            print(f"next step: {state['current_step']}")
            return 0
        if args.command == "retry":
            run_root = resolve_run(args.run)
            state = load_state(run_root)
            retry_run(state, run_root, args.phase, args.reason)
            write_state(run_root, state)
            print(f"retry created: {state['current_phase']} {active_attempt(state, state['current_phase'])['attempt_id']}")
            if state["status"] == "waiting_for_agent":
                print(f"task: {repository_relative(task_spec(state, run_root)['task_path'])}")
            return 0
        if args.command == "verify":
            run_root = resolve_run(args.run)
            print(json.dumps(verify_run(run_root), indent=2, sort_keys=True))
            return 0
        if args.command == "import-completed":
            run_root = import_completed(args)
            print(f"imported completed run: {repository_relative(run_root)}")
            print(f"run report: {repository_relative(run_root / 'run-report.md')}")
            return 0
        raise AssertionError(args.command)
    except PipelineError as exc:
        print(f"stage5 pipeline: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
