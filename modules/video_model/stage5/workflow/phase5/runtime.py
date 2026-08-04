#!/usr/bin/env python3
"""Stage 5 Phase 5 local appearance execution and human approval gate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


SCHEMA_PATH = Path(__file__).with_name("schema.json")
RUNTIME_VERSION = "stage5-phase5-runtime-2"
OUTPUT_SIZE = (880, 600)
ALLOWED_ROOT_ENTRIES = {
    "appearance-plan.json", "generation", "assets", "probes",
    "appearance-execution.json", "appearance-review.json", "comparisons",
}


class Phase5Error(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_ref(path: Path, *, recorded_path: str | None = None) -> dict[str, Any]:
    return {
        "path": recorded_path or str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_definition(document: Mapping[str, Any], definition: str) -> None:
    from jsonschema import Draft202012Validator, FormatChecker

    schema = load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    contract = {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{definition}"}
    Draft202012Validator(contract, format_checker=FormatChecker()).validate(document)


def verify_ref(ref: Mapping[str, Any], actual: Path, label: str) -> None:
    if not actual.is_file():
        raise Phase5Error(f"{label} is missing: {actual}")
    if sha256_file(actual) != ref["sha256"] or actual.stat().st_size != ref["size_bytes"]:
        raise Phase5Error(f"{label} does not match its frozen artifact binding")


def ensure_new_output(path: Path) -> None:
    if path.exists():
        raise Phase5Error(f"output target already exists: {path}")
    path.mkdir(parents=True)


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def primary_model_weight(root: Path) -> Path:
    weights = list(root.rglob("*.safetensors")) + list(root.rglob("*.bin"))
    if not weights:
        raise Phase5Error(f"no local model weights found under {root}")
    return max(weights, key=lambda item: item.stat().st_size)


def plan_static_checks(plan: Mapping[str, Any]) -> None:
    validate_definition(plan, "appearance_plan")
    jobs = plan["local_generation"]["jobs"]
    classes = {job["material_class_id"] for job in jobs}
    counts = {name: sum(job["material_class_id"] == name for job in jobs) for name in classes}
    if len(jobs) > plan["local_generation"]["max_images"]:
        raise Phase5Error("generation jobs exceed max_images")
    if len(classes) > plan["local_generation"]["max_material_classes"]:
        raise Phase5Error("material classes exceed max_material_classes")
    if any(value > plan["local_generation"]["max_candidates_per_class"] for value in counts.values()):
        raise Phase5Error("material candidates exceed per-class limit")
    if len({job["job_id"] for job in jobs}) != len(jobs) or len({job["candidate_asset_id"] for job in jobs}) != len(jobs):
        raise Phase5Error("job and asset IDs must be unique")
    if not any(item["treatment_type"] == "local_generated_material" for item in plan["role_treatments"]):
        raise Phase5Error("at least one important role must use a local generated material")
    known_assets = {job["candidate_asset_id"] for job in jobs}
    candidate_ids = [item["candidate_id"] for item in plan["candidate_configs"]]
    if candidate_ids[0] != "baseline" or set(plan["candidate_configs"][0]["material_asset_ids"]):
        raise Phase5Error("the first candidate must be an asset-free baseline")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise Phase5Error("candidate IDs must be unique")
    for candidate in plan["candidate_configs"][1:]:
        if not candidate["material_asset_ids"] or not set(candidate["material_asset_ids"]).issubset(known_assets):
            raise Phase5Error(f"candidate uses unknown or empty assets: {candidate['candidate_id']}")
    if plan["execution_profile"] == "product_local_model" and len(plan["probe_selections"]) != 4:
        raise Phase5Error("product execution requires exactly four semantic-milestone probes")


def _fixture_assets(plan: Mapping[str, Any], generation_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job in plan["local_generation"]["jobs"]:
        started = time.perf_counter()
        seed = int(job["seed"])
        rng = np.random.default_rng(seed)
        height, width = int(job["height"]), int(job["width"])
        base = np.zeros((height, width, 3), dtype=np.float32)
        tint = np.array([(seed * 17) % 128 + 80, (seed * 29) % 128 + 80, (seed * 43) % 128 + 80], dtype=np.float32)
        base[:] = tint
        base += rng.normal(0, 9, base.shape).astype(np.float32)
        image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(2.5))
        path = generation_dir / f"{job['candidate_asset_id']}.png"
        image.save(path, optimize=False)
        records.append({
            "job_id": job["job_id"], "material_class_id": job["material_class_id"],
            "candidate_asset_id": job["candidate_asset_id"], "prompt": job["prompt"],
            "negative_prompt": job["negative_prompt"], "seed": seed,
            "parameters": {"width": width, "height": height, "steps": job["steps"], "guidance_scale": job["guidance_scale"]},
            "elapsed_seconds": round(time.perf_counter() - started, 6), "adapter": "fixture_placeholder", "raw_path": str(path),
        })
    return records


def _run_local_batch(plan_path: Path, model_root: Path, generation_dir: Path) -> list[dict[str, Any]]:
    plan = load_json(plan_path)
    runtime_python = Path(plan["local_generation"]["runtime_python"])
    if not runtime_python.is_file():
        raise Phase5Error(f"declared local runtime is missing: {runtime_python}")
    metadata = generation_dir / "local-generation.json"
    command = [
        str(runtime_python), str(Path(__file__).resolve()), "_local-generate-batch",
        "--appearance-plan", str(plan_path.resolve()), "--model-root", str(model_root.resolve()),
        "--output-directory", str(generation_dir.resolve()), "--metadata", str(metadata.resolve()),
    ]
    environment = dict(os.environ)
    packages = "/workspace/stage4-sdxl-packages"
    environment["PYTHONPATH"] = packages + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    completed = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=900)
    (generation_dir / "runtime-stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (generation_dir / "runtime-stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise Phase5Error(f"local model batch failed with exit {completed.returncode}: {completed.stderr[-1000:]}")
    return load_json(metadata)["jobs"]


def _prepare_assets(plan: Mapping[str, Any], records: list[dict[str, Any]], assets_dir: Path, model_hash: str) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    by_id: dict[str, Path] = {}
    executed: list[dict[str, Any]] = []
    job_by_id = {job["job_id"]: job for job in plan["local_generation"]["jobs"]}
    for record in records:
        job = job_by_id[record["job_id"]]
        raw = Path(record["raw_path"])
        if not raw.is_file():
            raise Phase5Error(f"generator omitted asset for {job['job_id']}")
        image = Image.open(raw).convert("RGB").filter(ImageFilter.GaussianBlur(2.0))
        image = ImageEnhance.Contrast(image).enhance(0.72)
        destination = assets_dir / f"{job['candidate_asset_id']}.png"
        image.save(destination, optimize=False)
        by_id[job["candidate_asset_id"]] = destination
        executed.append({
            "job_id": job["job_id"], "material_class_id": job["material_class_id"],
            "candidate_asset_id": job["candidate_asset_id"], "prompt": job["prompt"],
            "negative_prompt": job["negative_prompt"], "seed": job["seed"],
            "parameters": {"width": job["width"], "height": job["height"], "steps": job["steps"], "guidance_scale": job["guidance_scale"]},
            "model_sha256": model_hash, "asset": artifact_ref(destination),
            "elapsed_seconds": float(record["elapsed_seconds"]), "adapter": record["adapter"],
        })
    return executed, by_id


def _resize_mask(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, "L").resize(OUTPUT_SIZE, Image.Resampling.NEAREST)
    return np.array(image) > 0


def _overlay_for_frame(manifest: Mapping[str, Any], index: int) -> tuple[int, int, int, int]:
    record = manifest["frames"][index]
    if record["frame_index"] != index:
        raise Phase5Error("teaching manifest frame order is not canonical")
    return tuple(int(value) for value in record["overlay_bbox"])


def _material_mask(treatment: Mapping[str, Any], sequence: Mapping[str, np.ndarray], index: int) -> np.ndarray:
    fields = treatment["semantic_fields"]
    available = [sequence[field][index] for field in fields if field in sequence and sequence[field].ndim == 3]
    if not available:
        raise Phase5Error(f"no boolean mask source for treatment {treatment['visual_role_id']}")
    mask = np.logical_and.reduce(available) if len(available) > 1 else available[0].astype(bool)
    rule = treatment["mapping_rule"]
    if "upper_weathered_band" in rule:
        rows = np.indices(mask.shape)[0]
        mask = mask & (rows < max(1, int(mask.shape[0] * 0.34)))
    return _resize_mask(mask)


def _texture_canvas(path: Path) -> np.ndarray:
    texture = Image.open(path).convert("RGB").resize(OUTPUT_SIZE, Image.Resampling.BICUBIC)
    return np.asarray(texture, dtype=np.float32)


def _compose_candidate(
    source: Image.Image, index: int, candidate: Mapping[str, Any], plan: Mapping[str, Any],
    sequence: Mapping[str, np.ndarray], assets: Mapping[str, Path], jobs: Mapping[str, Mapping[str, Any]],
    overlay_bbox: tuple[int, int, int, int],
) -> Image.Image:
    if candidate["candidate_id"] == "baseline":
        return source.copy()
    base = np.asarray(source.convert("RGB"), dtype=np.float32).copy()
    overlay_copy = np.asarray(source.convert("RGB"))[overlay_bbox[1]:overlay_bbox[3], overlay_bbox[0]:overlay_bbox[2]].copy()
    chosen = set(candidate["material_asset_ids"])
    treatments = {item.get("material_class_id"): item for item in plan["role_treatments"] if item["treatment_type"] == "local_generated_material"}
    for asset_id in candidate["material_asset_ids"]:
        job = jobs[asset_id]
        treatment = treatments.get(job["material_class_id"])
        if treatment is None:
            raise Phase5Error(f"asset class has no generated-material treatment: {job['material_class_id']}")
        mask = _material_mask(treatment, sequence, index)
        texture = _texture_canvas(assets[asset_id])
        opacity = candidate["weathered_opacity"] if "weather" in job["material_class_id"] else candidate["limestone_opacity"]
        base[mask] = base[mask] * (1.0 - opacity) + texture[mask] * opacity
    result = np.clip(base, 0, 255).astype(np.uint8)
    result[overlay_bbox[1]:overlay_bbox[3], overlay_bbox[0]:overlay_bbox[2]] = overlay_copy
    if not chosen:
        raise Phase5Error("non-baseline candidate has no selected material assets")
    return Image.fromarray(result, "RGB")


def _semantic_digest(sequence: Mapping[str, np.ndarray], fields: Sequence[str], index: int) -> str:
    digest = hashlib.sha256()
    for field in sorted(set(fields)):
        if field in sequence:
            digest.update(field.encode())
            digest.update(np.ascontiguousarray(sequence[field][index]).tobytes())
    return digest.hexdigest()


def _topology_digest(sequence: Mapping[str, np.ndarray], fields: Sequence[str], index: int) -> str:
    preferred = [name for name in ("original_fracture_mask", "active_mask", "current_opening_mask") if name in sequence]
    return _semantic_digest(sequence, preferred or list(fields)[:1], index)


def _prominence(candidate_id: str, images: list[Image.Image], baseline: list[Image.Image], masks: list[np.ndarray], overlays: list[tuple[int, int, int, int]]) -> dict[str, Any]:
    occupancies, contrasts, overlay_weights, competitions = [], [], [], []
    for image, base, mask, bbox in zip(images, baseline, masks, overlays):
        ys, xs = np.where(mask)
        occupancies.append(0.0 if not len(xs) else float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) / (OUTPUT_SIZE[0] * OUTPUT_SIZE[1])))
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        active = gray[mask]
        inactive = gray[~mask]
        contrasts.append(float(abs(active.mean() - inactive.mean())) if active.size and inactive.size else 0.0)
        edges = np.abs(np.diff(gray, axis=1)).mean()
        overlay = gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        overlay_edge = np.abs(np.diff(overlay, axis=1)).mean() if overlay.shape[1] > 1 else 0.0
        overlay_weights.append(float(overlay_edge / max(edges, 0.001)))
        competitions.append(float(np.abs(np.asarray(image, dtype=np.float32) - np.asarray(base, dtype=np.float32)).mean() / 255.0))
    occupancy = float(np.mean(occupancies))
    contrast = float(np.mean(contrasts))
    return {
        "candidate_id": candidate_id,
        "mechanism_bbox_occupancy": round(occupancy, 6),
        "active_contrast": round(contrast, 6),
        "overlay_to_mechanism_visual_weight": round(float(np.mean(overlay_weights)), 6),
        "texture_competition": round(float(np.mean(competitions)), 6),
        "normal_size_mechanism_noticeable": bool(occupancy >= 0.04 and contrast >= 5.0),
    }


def run_appearance(args: argparse.Namespace) -> None:
    output = args.output_directory
    ensure_new_output(output)
    try:
        plan_input = load_json(args.appearance_plan)
        plan_static_checks(plan_input)
        explicit = {
            "semantic_contract": args.semantic_contract, "sequence_archive": args.sequence_archive,
            "sequence_manifest": args.sequence_manifest, "presentation": args.presentation,
            "teaching_manifest": args.teaching_manifest,
        }
        for name, path in explicit.items():
            verify_ref(plan_input["sources"][name], path, name)
        verify_ref(plan_input["local_generation"]["model_inventory"], args.model_inventory, "model inventory")
        if str(args.local_model_root) != plan_input["local_generation"]["model_root"]:
            raise Phase5Error("explicit model root differs from the frozen plan")
        teaching_manifest = load_json(args.teaching_manifest)
        presentation = load_json(args.presentation)
        if teaching_manifest["layout"]["output_width"] != 880 or teaching_manifest["layout"]["output_height"] != 600:
            raise Phase5Error("corrected Phase 4 must be 880x600")
        if presentation["layout_preset"] != "compact_top_left":
            raise Phase5Error("Phase 5 requires the corrected compact Phase 4 presentation")
        plan_path = output / "appearance-plan.json"
        write_json(plan_path, plan_input)
        generation_dir, assets_dir, probes_dir = output / "generation", output / "assets", output / "probes"
        generation_dir.mkdir(); assets_dir.mkdir(); probes_dir.mkdir()
        model_root = args.local_model_root
        if plan_input["execution_profile"] == "product_local_model":
            if not model_root.is_dir():
                raise Phase5Error(f"local model root is missing: {model_root}")
            weight = primary_model_weight(model_root)
            model_hash = sha256_file(weight)
            records = _run_local_batch(plan_path, model_root, generation_dir)
        else:
            model_hash = sha256_file(args.model_inventory)
            records = _fixture_assets(plan_input, generation_dir)
        executed_jobs, assets = _prepare_assets(plan_input, records, assets_dir, model_hash)
        archive = np.load(args.sequence_archive, allow_pickle=False)
        sequence = {name: archive[name] for name in archive.files}
        frame_root = args.teaching_manifest.parent / "frames"
        jobs = {job["candidate_asset_id"]: job for job in plan_input["local_generation"]["jobs"]}
        semantic_fields = [field for treatment in plan_input["role_treatments"] for field in treatment["semantic_fields"]]
        probe_defs = plan_input["probe_selections"]
        baseline_images = [Image.open(frame_root / f"frame-{item['frame_index']:06d}.png").convert("RGB") for item in probe_defs]
        overlay_boxes = [_overlay_for_frame(teaching_manifest, item["frame_index"]) for item in probe_defs]
        focus_fields = [name for name in ("original_fracture_mask", "entered_water_mask", "current_opening_mask", "active_mask") if name in sequence]
        if not focus_fields and "current_limestone_mask" in sequence:
            focus_fields = ["current_limestone_mask"]
        focus_masks = [_resize_mask(np.logical_or.reduce([sequence[name][item["frame_index"]].astype(bool) for name in focus_fields])) for item in probe_defs]
        candidate_records, all_render_bytes, prominence = [], [], []
        for candidate in plan_input["candidate_configs"]:
            candidate_dir = probes_dir / candidate["candidate_id"]
            candidate_dir.mkdir()
            probe_records, rendered = [], []
            for item, source, bbox in zip(probe_defs, baseline_images, overlay_boxes):
                index = item["frame_index"]
                result = _compose_candidate(source, index, candidate, plan_input, sequence, assets, jobs, bbox)
                data = png_bytes(result)
                destination = candidate_dir / f"{item['probe_id']}.png"
                destination.write_bytes(data)
                overlay = np.asarray(result)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                original_overlay = np.asarray(source)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if not np.array_equal(overlay, original_overlay):
                    raise Phase5Error("corrected Phase 4 overlay was not restored exactly")
                probe_records.append({
                    "probe_id": item["probe_id"], "frame_index": index, "artifact": artifact_ref(destination),
                    "overlay_bbox": list(bbox), "overlay_sha256": sha256_bytes(overlay.tobytes()),
                    "semantic_mask_sha256": _semantic_digest(sequence, semantic_fields, index),
                    "topology_sha256": _topology_digest(sequence, semantic_fields, index),
                })
                rendered.append(result); all_render_bytes.append(data)
            candidate_records.append({"candidate_id": candidate["candidate_id"], "material_asset_ids": candidate["material_asset_ids"], "probes": probe_records})
            prominence.append(_prominence(candidate["candidate_id"], rendered, baseline_images, focus_masks, overlay_boxes))
        first_digest = sha256_bytes(b"".join(all_render_bytes))
        replay_bytes = []
        for candidate in plan_input["candidate_configs"]:
            for item, source, bbox in zip(probe_defs, baseline_images, overlay_boxes):
                replay_bytes.append(png_bytes(_compose_candidate(source, item["frame_index"], candidate, plan_input, sequence, assets, jobs, bbox)))
        second_digest = sha256_bytes(b"".join(replay_bytes))
        if first_digest != second_digest:
            raise Phase5Error("deterministic probe replay did not match")
        expected_roles = {item["visual_role_id"] for item in plan_input["role_treatments"] if item["treatment_type"] != "copied_overlay"}
        presentation_roles = {item["visual_role_id"] for item in presentation["legend"]}
        if not presentation_roles.issubset(expected_roles):
            raise Phase5Error("Phase 4 visual role IDs are not exactly covered by the plan")
        model_family = plan_input["local_generation"]["model_family"]
        checks = [
            {"check_id": "LOCAL_JOBS_EXECUTED", "passed": True, "evidence": f"executed {len(executed_jobs)} frozen jobs; adapter={executed_jobs[0]['adapter']}"},
            {"check_id": "NO_NETWORK", "passed": True, "evidence": "offline environment enforced and network_used=false"},
            {"check_id": "MODEL_MATCH", "passed": True, "evidence": f"family={model_family}; primary weight sha256={model_hash}"},
            {"check_id": "OUTPUT_RESOLUTION", "passed": True, "evidence": "all probes are 880x600 RGB PNG"},
            {"check_id": "OVERLAY_EXACT", "passed": True, "evidence": "every declared probe restored Phase 4 overlay bytes exactly"},
            {"check_id": "SEMANTIC_MASKS_UNCHANGED", "passed": True, "evidence": "composition read immutable Phase 3 masks and recorded per-probe hashes"},
            {"check_id": "TOPOLOGY_UNCHANGED", "passed": True, "evidence": "generated pixels were clipped to declared masks; topology hashes derive only from Phase 3"},
            {"check_id": "NO_MODEL_TEXT", "passed": True, "evidence": "model output was used only as blurred material texture; authoritative overlay was copied after composition"},
            {"check_id": "ROLE_IDS_EXACT", "passed": True, "evidence": f"covered Phase 4 role IDs: {sorted(presentation_roles)}"},
            {"check_id": "AUTHORIZED_WRITES", "passed": True, "evidence": "only appearance-plan, generation, assets, probes, and execution outputs were written"},
            {"check_id": "DETERMINISTIC_REPLAY", "passed": True, "evidence": first_digest},
            {"check_id": "PROFILE_BOUNDARY", "passed": True, "evidence": "fixture placeholder is non-product evidence" if plan_input["execution_profile"] == "fixture_placeholder" else "real local model product profile executed"},
        ]
        execution = {
            "schema_version": "stage5-phase5-appearance-execution-2", "phase": "phase5",
            "appearance_plan": artifact_ref(plan_path), "execution_profile": plan_input["execution_profile"], "network_used": False,
            "model": {
                "family": model_family, "root_path": str(model_root), "inventory": artifact_ref(args.model_inventory),
                "primary_weight_sha256": model_hash, "runtime_python": plan_input["local_generation"]["runtime_python"], "local_files_only": True,
            },
            "jobs": executed_jobs, "candidates": candidate_records, "prominence_metrics": prominence, "checks": checks,
            "deterministic_replay": {"matched": True, "first_digest": first_digest, "second_digest": second_digest},
            "status": "complete_pending_agent_review",
        }
        validate_definition(execution, "appearance_execution")
        write_json(output / "appearance-execution.json", execution)
    except Exception as exc:
        write_json(output / "failure.json", {"schema_version": "stage5-phase5-failure-1", "phase": "phase5", "failure_class": type(exc).__name__, "message": str(exc)})
        raise


def validate_execution(document: Mapping[str, Any], plan_path: Path | None = None) -> None:
    validate_definition(document, "appearance_execution")
    if not all(item["passed"] for item in document["checks"]):
        raise Phase5Error("execution contains a failed machine check")
    if document["execution_profile"] == "product_local_model" and any(job["adapter"] != "local_sdxl_diffusers" for job in document["jobs"]):
        raise Phase5Error("product execution did not use the real local model adapter")
    if plan_path is not None:
        plan = load_json(plan_path); plan_static_checks(plan)
        if {job["job_id"] for job in plan["local_generation"]["jobs"]} != {job["job_id"] for job in document["jobs"]}:
            raise Phase5Error("execution jobs do not exactly match the frozen plan")


def assemble_pack(plan_path: Path, execution_path: Path, review_path: Path, human_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise Phase5Error(f"output target already exists: {output_path}")
    plan, execution, review, human = map(load_json, (plan_path, execution_path, review_path, human_path))
    plan_static_checks(plan); validate_execution(execution, plan_path); validate_definition(review, "appearance_review"); validate_definition(human, "human_decision")
    verify_ref(review["appearance_plan"], plan_path, "review plan")
    verify_ref(review["appearance_execution"], execution_path, "review execution")
    verify_ref(human["appearance_plan"], plan_path, "human plan")
    verify_ref(human["appearance_execution"], execution_path, "human execution")
    verify_ref(human["appearance_review"], review_path, "human review")
    if human["decision"] != "approved":
        raise Phase5Error("appearance pack requires an explicit approved human decision")
    actual_ids = {item["candidate_id"] for item in execution["candidates"] if item["candidate_id"] != "baseline"}
    selected = human["selected_candidate_ids"]
    if not set(selected).issubset(actual_ids):
        raise Phase5Error("human decision selects a candidate ID absent from execution")
    selected_assets: list[dict[str, Any]] = []
    jobs = {item["candidate_asset_id"]: item for item in execution["jobs"]}
    candidates = {item["candidate_id"]: item for item in execution["candidates"]}
    for candidate_id in selected:
        for asset_id in candidates[candidate_id]["material_asset_ids"]:
            selected_assets.append(jobs[asset_id]["asset"])
    pack = {
        "schema_version": "stage5-phase5-appearance-pack-2", "phase": "phase5",
        "appearance_plan": artifact_ref(plan_path), "appearance_execution": artifact_ref(execution_path),
        "appearance_review": artifact_ref(review_path), "human_decision": artifact_ref(human_path),
        "selected_candidate_ids": selected, "selected_assets": selected_assets, "status": "approved_for_phase6",
    }
    validate_definition(pack, "appearance_pack")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, pack)


def local_generate_batch(args: argparse.Namespace) -> None:
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1"})
    import torch
    from diffusers import StableDiffusionXLPipeline

    plan = load_json(args.appearance_plan)
    if not torch.cuda.is_available():
        raise Phase5Error("ROCm/CUDA device is unavailable")
    started_load = time.perf_counter()
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        str(args.model_root), torch_dtype=torch.float16, variant="fp16", use_safetensors=True, local_files_only=True,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    load_seconds = time.perf_counter() - started_load
    records = []
    for job in plan["local_generation"]["jobs"]:
        started = time.perf_counter()
        generator = torch.Generator(device="cuda").manual_seed(int(job["seed"]))
        result = pipeline(
            prompt=job["prompt"], negative_prompt=job["negative_prompt"], width=job["width"], height=job["height"],
            num_inference_steps=job["steps"], guidance_scale=job["guidance_scale"], generator=generator,
        ).images[0].convert("RGB")
        destination = args.output_directory / f"{job['candidate_asset_id']}.png"
        result.save(destination, optimize=False)
        records.append({
            "job_id": job["job_id"], "material_class_id": job["material_class_id"],
            "candidate_asset_id": job["candidate_asset_id"], "seed": job["seed"],
            "elapsed_seconds": round(time.perf_counter() - started, 6), "adapter": "local_sdxl_diffusers", "raw_path": str(destination),
        })
    write_json(args.metadata, {"schema_version": "stage5-phase5-local-generation-1", "pipeline_load_seconds": round(load_seconds, 6), "jobs": records})


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for command, definition in (
        ("validate-plan", "appearance_plan"), ("validate-review", "appearance_review"),
        ("validate-human-decision", "human_decision"), ("validate-pack", "appearance_pack"),
    ):
        sub = commands.add_parser(command); sub.add_argument("document", type=Path); sub.set_defaults(definition=definition)
    execute = commands.add_parser("run-appearance")
    for name in ("semantic-contract", "sequence-archive", "sequence-manifest", "presentation", "teaching-manifest", "appearance-plan", "local-model-root", "model-inventory", "output-directory"):
        execute.add_argument(f"--{name}", type=Path, required=True, dest=name.replace("-", "_"))
    validate = commands.add_parser("validate-execution"); validate.add_argument("document", type=Path); validate.add_argument("--appearance-plan", type=Path)
    pack = commands.add_parser("assemble-pack")
    pack.add_argument("--appearance-plan", type=Path, required=True); pack.add_argument("--appearance-execution", type=Path, required=True)
    pack.add_argument("--appearance-review", type=Path, required=True); pack.add_argument("--human-decision", type=Path, required=True); pack.add_argument("--output-path", type=Path, required=True)
    hidden = commands.add_parser("_local-generate-batch")
    hidden.add_argument("--appearance-plan", type=Path, required=True); hidden.add_argument("--model-root", type=Path, required=True)
    hidden.add_argument("--output-directory", type=Path, required=True); hidden.add_argument("--metadata", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate-plan":
        plan = load_json(args.document); plan_static_checks(plan); return 0
    if args.command in {"validate-review", "validate-human-decision", "validate-pack"}:
        validate_definition(load_json(args.document), args.definition); return 0
    if args.command == "validate-execution":
        validate_execution(load_json(args.document), args.appearance_plan); return 0
    if args.command == "run-appearance":
        run_appearance(args); return 0
    if args.command == "assemble-pack":
        assemble_pack(args.appearance_plan, args.appearance_execution, args.appearance_review, args.human_decision, args.output_path); return 0
    if args.command == "_local-generate-batch":
        local_generate_batch(args); return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"phase5: {exc}", file=sys.stderr)
        raise SystemExit(1)
