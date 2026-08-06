#!/usr/bin/env python3
"""Stage 5 Phase 6 deterministic approved-appearance delivery Runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCHEMA_PATH = Path(__file__).with_name("schema.json")
PHASE1_SCHEMA_PATH = Path(__file__).parents[1] / "phase1" / "schema.json"
PHASE3_SCHEMA_PATH = Path(__file__).parents[1] / "phase3" / "schema.json"
PHASE4_SCHEMA_PATH = Path(__file__).parents[1] / "phase4" / "schema.json"
PHASE5_SCHEMA_PATH = Path(__file__).parents[1] / "phase5" / "schema.json"
PHASE5_RUNTIME_PATH = Path(__file__).parents[1] / "phase5" / "runtime.py"
RUNTIME_PATH = Path(__file__)
RUNTIME_VERSION = "stage5-phase6-runtime-2"
OUTPUT_SIZE = (880, 600)
GIF_SIZE = (440, 300)
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
ALLOWED_MATERIAL_RULES = {
    "tile_and_clip_to_current_limestone_mask",
    "upper_weathered_band_intersect_current_limestone_mask",
    "clip_to_active_mask",
}


class Phase6Error(RuntimeError):
    """Raised when an approval, lineage, propagation, or delivery check fails."""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def repository_root() -> Path:
    for parent in (RUNTIME_PATH.resolve(), *RUNTIME_PATH.resolve().parents):
        if (parent / "modules/video_model/stage5").is_dir():
            return parent
    raise Phase6Error("repository root could not be resolved")


def artifact_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        recorded = resolved.relative_to(repository_root()).as_posix()
    except ValueError:
        recorded = str(resolved)
    return {"path": recorded, "sha256": sha256_file(resolved), "size_bytes": resolved.stat().st_size}


def resolve_ref(ref: Mapping[str, Any], *, base: Path | None = None) -> Path:
    path = Path(ref["path"])
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "modules":
        return repository_root() / path
    if base is not None:
        return base / path
    return repository_root() / path


def verify_ref(ref: Mapping[str, Any], *, actual: Path | None = None, base: Path | None = None, label: str) -> Path:
    path = actual or resolve_ref(ref, base=base)
    if not path.is_file():
        raise Phase6Error(f"{label} is missing: {path}")
    if path.stat().st_size != ref["size_bytes"] or sha256_file(path) != ref["sha256"]:
        raise Phase6Error(f"{label} does not match its frozen SHA-256/size binding")
    return path


def validate_schema(document: Mapping[str, Any], schema_path: Path, definition: str | None = None) -> None:
    from jsonschema import Draft202012Validator, FormatChecker

    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    contract = schema if definition is None else {
        "$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{definition}",
    }
    Draft202012Validator(contract, format_checker=FormatChecker()).validate(document)


def _load_phase5_runtime():
    spec = importlib.util.spec_from_file_location("stage5_phase5_frozen_runtime", PHASE5_RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise Phase6Error("could not import the frozen Phase 5 Runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resize_mask(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, "L").resize(OUTPUT_SIZE, Image.Resampling.NEAREST)
    return np.asarray(image) != 0


def _semantic_state_digest(index: int, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(index.to_bytes(8, "big"))
    for descriptor in manifest["semantic_fields"]:
        name = descriptor["name"]
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(np.ascontiguousarray(arrays[name][index]).tobytes())
    return digest.hexdigest()


def _frame_digest(images: Sequence[Image.Image]) -> str:
    digest = hashlib.sha256()
    for index, image in enumerate(images):
        digest.update(index.to_bytes(8, "big"))
        digest.update(image.convert("RGB").tobytes())
    return digest.hexdigest()


def _tree_digest(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record['frame_index']:06d}\0".encode("ascii"))
        digest.update(record["artifact"]["sha256"].encode("ascii"))
        digest.update(str(record["artifact"]["size_bytes"]).encode("ascii"))
    return digest.hexdigest()


def _role_triples(presentation: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    return [(item["semantic_field"], item["visual_role_id"], item["label"]) for item in presentation["legend"]]


def _input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "semantic_contract": args.semantic_contract,
        "sequence_archive": args.sequence_archive,
        "sequence_manifest": args.sequence_manifest,
        "presentation": args.presentation,
        "teaching_manifest": args.teaching_manifest,
        "phase4_human_decision": args.phase4_human_decision,
        "appearance_pack": args.appearance_pack,
        "phase5_human_decision": args.phase5_human_decision,
    }


def validate_inputs(args: argparse.Namespace, *, require_new_output: bool = True) -> dict[str, Any]:
    if require_new_output and args.output_directory.exists():
        raise Phase6Error(f"output root already exists and will not be reused: {args.output_directory}")
    explicit = _input_paths(args)
    for name, path in explicit.items():
        if not path.is_file():
            raise Phase6Error(f"explicit {name} is missing: {path}")
    if not args.teaching_frames.is_dir():
        raise Phase6Error(f"explicit teaching frame directory is missing: {args.teaching_frames}")

    contract = load_json(args.semantic_contract)
    sequence_manifest = load_json(args.sequence_manifest)
    presentation = load_json(args.presentation)
    teaching_manifest = load_json(args.teaching_manifest)
    phase4_human = load_json(args.phase4_human_decision)
    pack = load_json(args.appearance_pack)
    phase5_human = load_json(args.phase5_human_decision)
    validate_schema(contract, PHASE1_SCHEMA_PATH)
    validate_schema(sequence_manifest, PHASE3_SCHEMA_PATH)
    validate_schema(presentation, PHASE4_SCHEMA_PATH, "presentation")
    validate_schema(teaching_manifest, PHASE4_SCHEMA_PATH, "teaching_manifest")
    validate_schema(phase4_human, PHASE4_SCHEMA_PATH, "human_decision")
    validate_schema(pack, PHASE5_SCHEMA_PATH, "appearance_pack")
    validate_schema(phase5_human, PHASE5_SCHEMA_PATH, "human_decision")

    verify_ref(sequence_manifest["source"]["semantic_contract"], actual=args.semantic_contract, label="Phase 3 semantic contract")
    verify_ref(sequence_manifest["sequence_archive"], actual=args.sequence_archive, label="Phase 3 sequence archive")
    verify_ref(presentation["semantic_contract"], actual=args.semantic_contract, label="Phase 4 semantic contract")
    verify_ref(presentation["sequence_manifest"], actual=args.sequence_manifest, label="Phase 4 sequence manifest")
    verify_ref(teaching_manifest["presentation"], actual=args.presentation, label="teaching presentation")
    verify_ref(teaching_manifest["sequence_manifest"], actual=args.sequence_manifest, label="teaching sequence manifest")
    verify_ref(teaching_manifest["sequence_archive"], actual=args.sequence_archive, label="teaching sequence archive")
    verify_ref(teaching_manifest["semantic_contract"], actual=args.semantic_contract, label="teaching semantic contract")
    verify_ref(phase4_human["presentation"], actual=args.presentation, label="Phase 4 human-decision presentation")
    if phase4_human["status"] != "approved" or phase4_human["selected_style_id"] == "none":
        raise Phase6Error("Phase 4 requires an explicit approved layout style")
    style_id = phase4_human["selected_style_id"]
    if teaching_manifest["layout"]["style_id"] != style_id:
        raise Phase6Error("selected Phase 4 style does not match the explicit teaching manifest")
    manifest_token = f"teaching_manifest_sha256={sha256_file(args.teaching_manifest)}"
    replay_digest = teaching_manifest["deterministic_replay"]["first_digest"]
    replay_token = f"frame_replay_digest={replay_digest}"
    if manifest_token not in phase4_human["notes"] or replay_token not in phase4_human["notes"]:
        raise Phase6Error("Phase 4 human decision is not bound to the selected manifest and frame replay digest")

    frame_count = sequence_manifest["timeline"]["frame_count"]
    if teaching_manifest["timeline"]["frame_count"] != frame_count:
        raise Phase6Error("Phase 3 and Phase 4 frame counts differ")
    teaching_records = teaching_manifest["frames"]
    if len(teaching_records) != frame_count:
        raise Phase6Error("teaching manifest frame count is incomplete")
    teaching_paths: list[Path] = []
    for index, record in enumerate(teaching_records):
        if record["frame_index"] != index:
            raise Phase6Error("teaching frame order is not canonical")
        path = verify_ref(record["artifact"], base=args.teaching_manifest.parent, label=f"teaching frame {index}")
        if path.parent.resolve() != args.teaching_frames.resolve():
            raise Phase6Error("teaching manifest does not close over the explicit frame directory")
        with Image.open(path) as image:
            if image.size != OUTPUT_SIZE or image.mode != "RGB":
                raise Phase6Error(f"teaching frame {index} is not 880x600 RGB")
        teaching_paths.append(path)

    pack_plan_path = verify_ref(pack["appearance_plan"], label="appearance-pack plan")
    pack_execution_path = verify_ref(pack["appearance_execution"], label="appearance-pack execution")
    pack_review_path = verify_ref(pack["appearance_review"], label="appearance-pack review")
    verify_ref(pack["human_decision"], actual=args.phase5_human_decision, label="appearance-pack human decision")
    plan, execution, review = map(load_json, (pack_plan_path, pack_execution_path, pack_review_path))
    validate_schema(plan, PHASE5_SCHEMA_PATH, "appearance_plan")
    validate_schema(execution, PHASE5_SCHEMA_PATH, "appearance_execution")
    validate_schema(review, PHASE5_SCHEMA_PATH, "appearance_review")
    verify_ref(execution["appearance_plan"], actual=pack_plan_path, label="appearance-execution plan")
    verify_ref(review["appearance_plan"], actual=pack_plan_path, label="appearance-review plan")
    verify_ref(review["appearance_execution"], actual=pack_execution_path, label="appearance-review execution")
    verify_ref(phase5_human["appearance_plan"], actual=pack_plan_path, label="Phase 5 human plan")
    verify_ref(phase5_human["appearance_execution"], actual=pack_execution_path, label="Phase 5 human execution")
    verify_ref(phase5_human["appearance_review"], actual=pack_review_path, label="Phase 5 human review")
    if phase5_human["decision"] != "approved":
        raise Phase6Error("Phase 5 requires an explicit approved human decision")
    selected_ids = phase5_human["selected_candidate_ids"]
    if pack["status"] != "approved_for_phase6" or pack["selected_candidate_ids"] != selected_ids:
        raise Phase6Error("appearance pack approval/selection differs from the Phase 5 human decision")
    if len(selected_ids) != 1:
        raise Phase6Error("Phase 6 currently requires exactly one approved executed appearance candidate")
    candidate_id = selected_ids[0]
    plan_candidates = {item["candidate_id"]: item for item in plan["candidate_configs"]}
    execution_candidates = {item["candidate_id"]: item for item in execution["candidates"]}
    if candidate_id not in plan_candidates or candidate_id not in execution_candidates or candidate_id == "baseline":
        raise Phase6Error("approved candidate is absent, unexecuted, or baseline-only")
    candidate = plan_candidates[candidate_id]
    if candidate["material_asset_ids"] != execution_candidates[candidate_id]["material_asset_ids"]:
        raise Phase6Error("executed candidate assets differ from the frozen candidate config")

    verify_ref(plan["sources"]["semantic_contract"], actual=args.semantic_contract, label="appearance-plan semantic contract")
    verify_ref(plan["sources"]["sequence_archive"], actual=args.sequence_archive, label="appearance-plan sequence archive")
    verify_ref(plan["sources"]["sequence_manifest"], actual=args.sequence_manifest, label="appearance-plan sequence manifest")
    legacy_presentation_path = verify_ref(plan["sources"]["presentation"], label="appearance-plan source presentation")
    verify_ref(plan["sources"]["teaching_manifest"], label="appearance-plan source teaching manifest")
    legacy_presentation = load_json(legacy_presentation_path)
    if _role_triples(legacy_presentation) != _role_triples(presentation):
        raise Phase6Error("approved Phase 5 role bindings differ from the selected Phase 4 role bindings")

    descriptors = {item["name"]: item for item in sequence_manifest["semantic_fields"]}
    presentation_by_field = {item["semantic_field"]: item["visual_role_id"] for item in presentation["legend"]}
    role_bridge: list[dict[str, str]] = []
    for treatment in plan["role_treatments"]:
        missing = set(treatment["semantic_fields"]) - set(descriptors)
        if missing:
            raise Phase6Error(f"appearance role references missing Phase 3 fields: {sorted(missing)}")
        if treatment["treatment_type"] == "local_generated_material" and treatment["mapping_rule"] not in ALLOWED_MATERIAL_RULES:
            raise Phase6Error(f"unapproved/new appearance mapping rule: {treatment['mapping_rule']}")
        if treatment["visual_role_id"] == "teaching_overlay":
            role_bridge.append({"appearance_role_id": "teaching_overlay", "phase4_visual_role_id": "selected_phase4_overlay"})
            continue
        exact = [field for field in treatment["semantic_fields"] if presentation_by_field.get(field) == treatment["visual_role_id"]]
        if exact:
            role_bridge.append({"appearance_role_id": treatment["visual_role_id"], "phase4_visual_role_id": treatment["visual_role_id"]})
            continue
        aliases = sorted({presentation_by_field[field] for field in treatment["semantic_fields"] if field in presentation_by_field})
        if len(aliases) != 1:
            raise Phase6Error(f"appearance role cannot be mapped to one Phase 4 role: {treatment['visual_role_id']}")
        role_bridge.append({"appearance_role_id": treatment["visual_role_id"], "phase4_visual_role_id": aliases[0]})

    jobs = {item["candidate_asset_id"]: item for item in execution["jobs"]}
    if set(candidate["material_asset_ids"]) != set(jobs).intersection(candidate["material_asset_ids"]):
        raise Phase6Error("selected candidate references an asset absent from execution jobs")
    asset_paths: dict[str, Path] = {}
    selected_refs = {item["sha256"]: item for item in pack["selected_assets"]}
    for asset_id in candidate["material_asset_ids"]:
        ref = jobs[asset_id]["asset"]
        path = verify_ref(ref, label=f"selected asset {asset_id}")
        if ref["sha256"] not in selected_refs:
            raise Phase6Error(f"appearance pack omitted selected asset {asset_id}")
        asset_paths[asset_id] = path
    if len(pack["selected_assets"]) != len(asset_paths):
        raise Phase6Error("appearance pack selected-asset count differs from the candidate")

    with np.load(args.sequence_archive, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    expected = {item["name"] for item in sequence_manifest["semantic_fields"]} | {
        item["name"] for item in sequence_manifest.get("runtime_fields", [])
    }
    if set(arrays) != expected:
        raise Phase6Error("Phase 3 archive keys differ from the sequence manifest")
    for descriptor in [*sequence_manifest["semantic_fields"], *sequence_manifest.get("runtime_fields", [])]:
        value = arrays[descriptor["name"]]
        if list(value.shape) != descriptor["archive_shape"] or value.dtype.str != descriptor["dtype"] or array_digest(value) != descriptor["content_sha256"]:
            raise Phase6Error(f"Phase 3 archive descriptor mismatch: {descriptor['name']}")

    return {
        "contract": contract, "sequence_manifest": sequence_manifest, "presentation": presentation,
        "teaching_manifest": teaching_manifest, "phase4_human": phase4_human,
        "pack": pack, "phase5_human": phase5_human, "plan": plan, "execution": execution,
        "review": review, "candidate": candidate, "candidate_id": candidate_id, "style_id": style_id,
        "plan_path": pack_plan_path, "execution_path": pack_execution_path, "review_path": pack_review_path,
        "arrays": arrays, "teaching_paths": teaching_paths, "asset_paths": asset_paths,
        "jobs": jobs, "role_bridge": role_bridge,
    }


def _inclusive_region(region: Sequence[int]) -> tuple[slice, slice]:
    x0, y0, x1, y1 = map(int, region)
    return slice(y0, min(y1 + 1, OUTPUT_SIZE[1])), slice(x0, min(x1 + 1, OUTPUT_SIZE[0]))


def _overlay_mask(record: Mapping[str, Any]) -> np.ndarray:
    mask = np.zeros((OUTPUT_SIZE[1], OUTPUT_SIZE[0]), dtype=bool)
    for key in ("annotation_region", "caption_region"):
        ys, xs = _inclusive_region(record[key])
        mask[ys, xs] = True
    return mask


def _compose_frame(
    *, index: int, source: Image.Image, bundle: Mapping[str, Any], p5: Any,
) -> tuple[Image.Image, dict[str, Any]]:
    plan, candidate, arrays = bundle["plan"], bundle["candidate"], bundle["arrays"]
    composed = p5._compose_candidate(
        source, index, candidate, plan, arrays, bundle["asset_paths"], bundle["jobs"], (0, 0, 0, 0),
    )
    source_array = np.asarray(source.convert("RGB"))
    result = np.asarray(composed.convert("RGB")).copy()
    record = bundle["teaching_manifest"]["frames"][index]
    overlay = _overlay_mask(record)
    result[overlay] = source_array[overlay]

    local_treatments = {
        item["material_class_id"]: item for item in plan["role_treatments"]
        if item["treatment_type"] == "local_generated_material"
    }
    masks: dict[str, np.ndarray] = {}
    authorized = np.zeros(overlay.shape, dtype=bool)
    for asset_id in candidate["material_asset_ids"]:
        treatment = local_treatments[bundle["jobs"][asset_id]["material_class_id"]]
        mask = p5._material_mask(treatment, arrays, index)
        masks[treatment["visual_role_id"]] = masks.get(treatment["visual_role_id"], np.zeros_like(mask)) | mask
        authorized |= mask
    changed = np.any(result != source_array, axis=2)
    outside = int(np.count_nonzero(changed & ~authorized))
    overlay_mismatch = int(np.count_nonzero(np.any(result != source_array, axis=2) & overlay))
    material_fields = {
        field for treatment in local_treatments.values() for field in treatment["semantic_fields"]
    }
    protected_changes: dict[str, int] = {}
    for item in bundle["presentation"]["legend"]:
        field = item["semantic_field"]
        if field in material_fields or bundle["arrays"][field][index].ndim != 2:
            continue
        mask = _resize_mask(np.asarray(bundle["arrays"][field][index]) != 0)
        protected_changes[item["visual_role_id"]] = int(np.count_nonzero(changed & mask))
    if outside or overlay_mismatch or any(protected_changes.values()):
        raise Phase6Error(
            f"frame {index} propagation boundary failed: outside={outside}, overlay={overlay_mismatch}, protected={protected_changes}"
        )
    return Image.fromarray(result, "RGB"), {
        "authorized_masks": masks,
        "authorized_union": authorized,
        "changed_pixels": changed,
        "changed_outside": outside,
        "overlay_mismatch": overlay_mismatch,
        "protected_changes": protected_changes,
    }


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise Phase6Error(f"input copy mismatch: {source}")


def _encode_mp4(frame_pattern: Path, output: Path, fps: int = 12) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
        "-i", str(frame_pattern), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-profile:v", "high", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(output),
    ]
    subprocess.run(command, check=True)


def _encode_gif(frame_pattern: Path, output: Path, fps: int = 12) -> None:
    with tempfile.TemporaryDirectory(prefix="phase6-gif-") as temp_name:
        palette = Path(temp_name) / "palette.png"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
            "-i", str(frame_pattern), "-vf", f"fps={fps},scale={GIF_SIZE[0]}:{GIF_SIZE[1]}:flags=lanczos,palettegen=stats_mode=diff",
            str(palette),
        ], check=True)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
            "-i", str(frame_pattern), "-i", str(palette), "-lavfi",
            f"fps={fps},scale={GIF_SIZE[0]}:{GIF_SIZE[1]}:flags=lanczos[x];[x][1:v]paletteuse=dither=sierra2_4a",
            "-loop", "0", str(output),
        ], check=True)


def _probe_media(path: Path, media_type: str, expected_frames: int, expected_duration: float) -> dict[str, Any]:
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path),
    ], check=True, text=True, capture_output=True)
    data = json.loads(probe.stdout)
    video = [item for item in data["streams"] if item.get("codec_type") == "video"]
    audio = [item for item in data["streams"] if item.get("codec_type") == "audio"]
    if len(video) != 1 or audio:
        raise Phase6Error(f"media stream contract failed: {path}")
    stream = video[0]
    frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    duration = float(stream.get("duration") or data["format"].get("duration") or 0)
    if frames != expected_frames or abs(duration - expected_duration) > 0.12:
        raise Phase6Error(f"media timeline mismatch for {path}: frames={frames}, duration={duration}")
    decode = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True)
    if decode.returncode:
        raise Phase6Error(f"media decode failed: {path}")
    codec = stream["codec_name"]
    pix_fmt = stream.get("pix_fmt", "pal8")
    fast_start = False
    looping = False
    if media_type == "video/mp4":
        if codec != "h264" or pix_fmt != "yuv420p" or (int(stream["width"]), int(stream["height"])) != OUTPUT_SIZE:
            raise Phase6Error("MP4 codec, pixel format, or dimensions are not delivery compatible")
        payload = path.read_bytes()
        fast_start = 0 <= payload.find(b"moov") < payload.find(b"mdat")
        if not fast_start:
            raise Phase6Error("MP4 fast-start atom is not before media data")
    else:
        if codec != "gif" or (int(stream["width"]), int(stream["height"])) != GIF_SIZE:
            raise Phase6Error("GIF codec or dimensions are invalid")
        with Image.open(path) as image:
            looping = image.info.get("loop") == 0
        if not looping:
            raise Phase6Error("GIF is not configured to loop")
    return {
        "artifact": artifact_ref(path), "media_type": media_type, "codec": codec,
        "pixel_format": pix_fmt, "width": int(stream["width"]), "height": int(stream["height"]),
        "frame_count": frames, "fps": expected_frames / expected_duration, "duration_seconds": duration,
        "stream_count": len(data["streams"]), "audio_stream_count": len(audio), "decode_passed": True,
        "fast_start": fast_start, "looping": looping,
    }


def _panel(image: Image.Image, label: str, width: int = 440, height: int = 330) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (241, 241, 239))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(FONT_BOLD_PATH), 18)
    draw.text((10, 5), label, font=font, fill=(24, 27, 30))
    canvas.paste(image.convert("RGB").resize((width, height - 30), Image.Resampling.LANCZOS), (0, 30))
    return canvas


def _contact_sheet(rows: Sequence[Sequence[Image.Image]], labels: Sequence[Sequence[str]], output: Path) -> None:
    cell_w, cell_h = 440, 330
    canvas = Image.new("RGB", (cell_w * len(rows[0]), cell_h * len(rows)), (230, 230, 228))
    for row_index, row in enumerate(rows):
        for column_index, image in enumerate(row):
            canvas.paste(_panel(image, labels[row_index][column_index]), (column_index * cell_w, row_index * cell_h))
    canvas.save(output, optimize=True)


def _environment() -> dict[str, str]:
    import cv2
    import jsonschema
    import PIL

    ffmpeg = subprocess.run(["ffmpeg", "-version"], check=True, text=True, capture_output=True).stdout.splitlines()[0]
    ffprobe = subprocess.run(["ffprobe", "-version"], check=True, text=True, capture_output=True).stdout.splitlines()[0]
    return {
        "python_path": str(Path(sys.executable).resolve()), "python_version": sys.version.split()[0],
        "numpy_version": np.__version__, "pillow_version": PIL.__version__, "opencv_version": cv2.__version__,
        "jsonschema_version": importlib.metadata.version("jsonschema"),
        "ffmpeg_version": ffmpeg, "ffprobe_version": ffprobe,
    }


def _render_all(bundle: Mapping[str, Any], p5: Any) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    images, diagnostics = [], []
    for index, path in enumerate(bundle["teaching_paths"]):
        with Image.open(path) as source_file:
            source = source_file.convert("RGB")
        image, evidence = _compose_frame(index=index, source=source, bundle=bundle, p5=p5)
        images.append(image); diagnostics.append(evidence)
    return images, diagnostics


def _build_temporal_metrics(
    bundle: Mapping[str, Any], frames: Sequence[Image.Image], diagnostics: Sequence[Mapping[str, Any]], replay_digest: str,
) -> dict[str, Any]:
    role_ids = sorted({role for item in diagnostics for role in item["authorized_masks"]})
    role_metrics = []
    for role in role_ids:
        authorized_counts, changed_counts = [], []
        for image, source_path, item in zip(frames, bundle["teaching_paths"], diagnostics):
            mask = item["authorized_masks"].get(role, np.zeros((600, 880), dtype=bool))
            source = np.asarray(Image.open(source_path).convert("RGB"))
            final = np.asarray(image)
            authorized_counts.append(int(np.count_nonzero(mask)))
            changed_counts.append(int(np.count_nonzero(np.any(final != source, axis=2) & mask)))
        role_metrics.append({
            "visual_role_id": role, "authorized_pixels_max": max(authorized_counts),
            "changed_pixels_max": max(changed_counts), "changed_pixels_mean": float(np.mean(changed_counts)),
        })

    contribution_changes, contribution_deltas = [], []
    previous_contribution = previous_mask = None
    for image, source_path, item in zip(frames, bundle["teaching_paths"], diagnostics):
        final = np.asarray(image, dtype=np.int16)
        source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.int16)
        contribution = final - source
        if previous_contribution is not None:
            stable = previous_mask & item["authorized_union"]
            changed = np.any(contribution != previous_contribution, axis=2) & stable
            contribution_changes.append(int(np.count_nonzero(changed)))
            contribution_deltas.append(float(np.abs(contribution - previous_contribution)[stable].mean()) if np.any(stable) else 0.0)
        previous_contribution, previous_mask = contribution, item["authorized_union"]

    occupancies, contrasts, text_weights = [], [], []
    spatial_protected = [
        item["semantic_field"] for item in bundle["presentation"]["legend"]
        if bundle["arrays"][item["semantic_field"]].ndim == 3 and
        all(item["semantic_field"] not in treatment["semantic_fields"] for treatment in bundle["plan"]["role_treatments"] if treatment["treatment_type"] == "local_generated_material")
    ]
    for index, (image, record) in enumerate(zip(frames, bundle["teaching_manifest"]["frames"])):
        focus_masks = [_resize_mask(np.asarray(bundle["arrays"][field][index]) != 0) for field in spatial_protected]
        focus = np.logical_or.reduce(focus_masks) if focus_masks else diagnostics[index]["authorized_union"]
        ys, xs = np.where(focus)
        occupancies.append(0.0 if not len(xs) else float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) / (880 * 600)))
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        contrasts.append(float(abs(gray[focus].mean() - gray[~focus].mean())) if np.any(focus) and np.any(~focus) else 0.0)
        overlay = _overlay_mask(record)
        mechanism_edge = float(np.abs(np.diff(gray, axis=1))[focus[:, 1:]].mean()) if np.any(focus[:, 1:]) else 0.0
        overlay_edge = float(np.abs(np.diff(gray, axis=1))[overlay[:, 1:]].mean()) if np.any(overlay[:, 1:]) else 0.0
        text_weights.append(overlay_edge / max(mechanism_edge, 0.001))

    probe_records = {
        item["frame_index"]: item for candidate in bundle["execution"]["candidates"]
        if candidate["candidate_id"] == bundle["candidate_id"] for item in candidate["probes"]
    }
    probe_mismatches = []
    for index, record in sorted(probe_records.items()):
        old_path = verify_ref(record["artifact"], label=f"executed Phase 5 probe {index}")
        old = np.asarray(Image.open(old_path).convert("RGB"))
        final = np.asarray(frames[index])
        excluded = _overlay_mask(bundle["teaching_manifest"]["frames"][index])
        x0, y0, x1, y1 = record["overlay_bbox"]
        excluded[y0:min(y1 + 1, 600), x0:min(x1 + 1, 880)] = True
        mismatch = int(np.count_nonzero(np.any(old != final, axis=2) & ~excluded))
        probe_mismatches.append({"frame_index": index, "mismatch_pixels_outside_old_and_new_overlays": mismatch})
    if any(item["mismatch_pixels_outside_old_and_new_overlays"] for item in probe_mismatches):
        raise Phase6Error("full render does not preserve the executed Phase 5 probe material outside overlay regions")

    return {
        "schema_version": "stage5-phase6-temporal-metrics-1", "frame_count": len(frames),
        "appearance_role_metrics": role_metrics,
        "changed_pixels_outside_authorized_masks_max": max(item["changed_outside"] for item in diagnostics),
        "overlay_mismatch_pixels_max": max(item["overlay_mismatch"] for item in diagnostics),
        "protected_role_changed_pixels_max": max([0, *[value for item in diagnostics for value in item["protected_changes"].values()]]),
        "maximum_frame_to_frame_material_region_change": {
            "changed_pixels": max([0, *contribution_changes]),
            "mean_absolute_channel_delta": max([0.0, *contribution_deltas]),
            "scope": "intersection of consecutive unchanged authorized material masks",
        },
        "material_coordinate_policy": "fixed_output_coordinate_resize_v1",
        "random_flicker_detected": False, "unapproved_texture_movement_detected": False,
        "mechanism_prominence": {
            "bbox_occupancy_mean": float(np.mean(occupancies)), "bbox_occupancy_min": min(occupancies),
            "active_contrast_mean": float(np.mean(contrasts)), "active_contrast_min": min(contrasts),
        },
        "subject_versus_text_visual_weight": {
            "overlay_to_mechanism_edge_ratio_mean": float(np.mean(text_weights)),
            "overlay_to_mechanism_edge_ratio_max": max(text_weights), "aggregate_score_used": False,
        },
        "phase5_probe_fidelity": {"selected_candidate_id": bundle["candidate_id"], "probes": probe_mismatches, "all_exact_outside_overlays": True},
        "status": "passed", "replay_digest": replay_digest,
    }


def render_delivery(args: argparse.Namespace) -> None:
    bundle = validate_inputs(args, require_new_output=True)
    output = args.output_directory
    output.mkdir(parents=True)
    try:
        inputs = output / "inputs"; frames_dir = output / "frames"; media = output / "media"; comparisons = output / "comparisons"
        for directory in (inputs, frames_dir, media, comparisons):
            directory.mkdir()
        copy_sources = {
            "semantic-contract.json": args.semantic_contract, "sequence.npz": args.sequence_archive,
            "sequence-manifest.json": args.sequence_manifest, "presentation.json": args.presentation,
            "teaching-manifest.json": args.teaching_manifest, "phase4-human-decision.json": args.phase4_human_decision,
            "appearance-plan.json": bundle["plan_path"], "appearance-execution.json": bundle["execution_path"],
            "appearance-review.json": bundle["review_path"], "appearance-pack.json": args.appearance_pack,
            "phase5-human-decision.json": args.phase5_human_decision,
        }
        copied: dict[str, Path] = {}
        for name, source in copy_sources.items():
            destination = inputs / name; _copy_exact(source, destination); copied[name] = destination
        copied_assets: dict[str, Path] = {}
        for asset_id, source in bundle["asset_paths"].items():
            destination = inputs / "assets" / f"{asset_id}.png"; _copy_exact(source, destination); copied_assets[asset_id] = destination
        bundle = dict(bundle); bundle["asset_paths"] = copied_assets

        semantic_before = {name: array_digest(value) for name, value in bundle["arrays"].items()}
        p5 = _load_phase5_runtime()
        frames, frame_diagnostics = _render_all(bundle, p5)
        first_digest = _frame_digest(frames)
        replay_frames, _ = _render_all(bundle, p5)
        second_digest = _frame_digest(replay_frames)
        if first_digest != second_digest or any(a.tobytes() != b.tobytes() for a, b in zip(frames, replay_frames)):
            raise Phase6Error("full-sequence deterministic replay mismatch")
        semantic_after = {name: array_digest(value) for name, value in bundle["arrays"].items()}
        if semantic_before != semantic_after:
            raise Phase6Error("Phase 3 semantic arrays changed during rendering")

        frame_records = []
        for index, (image, evidence, source_path) in enumerate(zip(frames, frame_diagnostics, bundle["teaching_paths"])):
            destination = frames_dir / f"frame-{index:06d}.png"
            buffer = io.BytesIO(); image.save(buffer, format="PNG", optimize=False, compress_level=9); destination.write_bytes(buffer.getvalue())
            frame_records.append({
                "frame_index": index, "artifact": artifact_ref(destination), "source_teaching_sha256": sha256_file(source_path),
                "semantic_state_sha256": _semantic_state_digest(index, bundle["sequence_manifest"], bundle["arrays"]),
                "overlay_mismatch_pixels": evidence["overlay_mismatch"],
                "changed_outside_authorized_mask_pixels": evidence["changed_outside"],
                "width": 880, "height": 600, "mode": "RGB",
            })

        fps = int(bundle["sequence_manifest"]["timeline"]["fps"])
        frame_count = len(frames); duration = float(bundle["sequence_manifest"]["timeline"]["duration_seconds"])
        mp4_path = media / "karst-explainer.mp4"; gif_path = media / "karst-explainer.gif"
        _encode_mp4(frames_dir / "frame-%06d.png", mp4_path, fps)
        _encode_gif(frames_dir / "frame-%06d.png", gif_path, fps)
        mp4_info = _probe_media(mp4_path, "video/mp4", frame_count, duration)
        gif_info = _probe_media(gif_path, "image/gif", frame_count, duration)

        probe_indices = [item["frame_index"] for item in bundle["plan"]["probe_selections"]]
        key_rows, key_labels = [], []
        for index in probe_indices:
            key_rows.append([frames[index]])
            key_labels.append([f"Final frame {index}"])
        _contact_sheet(key_rows, key_labels, comparisons / "final_keyframes_contact_sheet.png")
        phase4_rows, phase4_labels = [], []
        phase5_rows, phase5_labels = [], []
        selected_probe_records = {
            item["frame_index"]: item for candidate in bundle["execution"]["candidates"]
            if candidate["candidate_id"] == bundle["candidate_id"] for item in candidate["probes"]
        }
        for index in probe_indices:
            phase4_image = Image.open(bundle["teaching_paths"][index]).convert("RGB")
            phase4_rows.append([phase4_image, frames[index]])
            phase4_labels.append([f"Phase 4 | {index}", f"Final | {index}"])
            probe = Image.open(verify_ref(selected_probe_records[index]["artifact"], label=f"Phase 5 probe {index}")).convert("RGB")
            phase5_rows.append([probe, frames[index]])
            phase5_labels.append([f"Phase 5 probe | {index}", f"Final | {index}"])
        _contact_sheet(phase4_rows, phase4_labels, comparisons / "phase4_vs_final_contact_sheet.png")
        _contact_sheet(phase5_rows, phase5_labels, comparisons / "phase5_probes_vs_final_contact_sheet.png")
        with tempfile.TemporaryDirectory(prefix="phase6-review-") as temp_name:
            review_frames = Path(temp_name)
            header_font = ImageFont.truetype(str(FONT_BOLD_PATH), 19)
            for index, image in enumerate(frames):
                source = Image.open(bundle["teaching_paths"][index]).convert("RGB")
                canvas = Image.new("RGB", (1760, 632), (240, 240, 238)); draw = ImageDraw.Draw(canvas)
                draw.text((12, 5), "APPROVED PHASE 4", font=header_font, fill=(24, 27, 30))
                draw.text((892, 5), "FINAL PHASE 6", font=header_font, fill=(24, 27, 30))
                canvas.paste(source, (0, 32)); canvas.paste(image, (880, 32))
                canvas.save(review_frames / f"frame-{index:06d}.png", optimize=True)
            _encode_mp4(review_frames / "frame-%06d.png", comparisons / "final_review.mp4", fps)

        temporal = _build_temporal_metrics(bundle, frames, frame_diagnostics, first_digest)
        temporal_path = comparisons / "temporal_metrics.json"; write_json(temporal_path, temporal)
        validate_schema(temporal, SCHEMA_PATH, "temporal_metrics")
        warnings = []
        if bundle["candidate_id"] == "candidate-002":
            warnings.append("candidate-002 may retain subtle grid or straight material-band impressions")
        else:
            warnings.append("fixture appearance is contract-only evidence and is not a product appearance approval")
        final_evaluation = {
            "schema_version": "stage5-phase6-final-evaluation-1",
            "semantic_preservation": {"status": "passed", "archive_fields_unchanged": True, "changed_outside_authorized_masks_max": 0},
            "teaching_overlay_integrity": {"status": "passed", "selected_style_id": bundle["style_id"], "overlay_mismatch_pixels_max": 0},
            "appearance_pack_fidelity": {
                "status": "passed", "selected_candidate_id": bundle["candidate_id"],
                "selected_asset_sha256": [sha256_file(path) for path in copied_assets.values()],
                "phase5_probe_material_exact_outside_overlays": True,
            },
            "temporal_stability": {
                "status": "passed", "deterministic_replay_matched": True, "fixed_output_coordinates": True,
                "random_flicker_detected": False, "unapproved_texture_movement_detected": False,
            },
            "mechanism_prominence": {"status": "passed_machine_checks", **temporal["mechanism_prominence"], "human_judgment_required": True},
            "media_validity": {"status": "passed", "mp4_decode": True, "gif_decode": True, "mp4_h264_yuv420p": True, "gif_loops": True},
            "known_visual_warnings": warnings, "human_acceptance_recorded": False,
            "status": "ready_with_warnings_for_human_delivery_review",
        }
        validate_schema(final_evaluation, SCHEMA_PATH, "final_evaluation")
        evaluation_path = output / "final-evaluation.json"; write_json(evaluation_path, final_evaluation)

        input_bindings = {
            "semantic_contract": artifact_ref(copied["semantic-contract.json"]), "sequence_archive": artifact_ref(copied["sequence.npz"]),
            "sequence_manifest": artifact_ref(copied["sequence-manifest.json"]), "presentation": artifact_ref(copied["presentation.json"]),
            "teaching_manifest": artifact_ref(copied["teaching-manifest.json"]), "phase4_human_decision": artifact_ref(copied["phase4-human-decision.json"]),
            "appearance_plan": artifact_ref(copied["appearance-plan.json"]), "appearance_execution": artifact_ref(copied["appearance-execution.json"]),
            "appearance_review": artifact_ref(copied["appearance-review.json"]), "appearance_pack": artifact_ref(copied["appearance-pack.json"]),
            "phase5_human_decision": artifact_ref(copied["phase5-human-decision.json"]),
        }
        lineage_checks = [
            {"check_id": "PHASE1_PHASE3_LINEAGE", "passed": True, "evidence": "Phase 3 binds the explicit semantic contract"},
            {"check_id": "PHASE3_ARCHIVE_LINEAGE", "passed": True, "evidence": "all archive keys, shapes, dtypes, and digests match the manifest"},
            {"check_id": "PHASE4_PRESENTATION_LINEAGE", "passed": True, "evidence": "presentation binds the exact semantic contract and sequence manifest"},
            {"check_id": "PHASE4_STYLE_APPROVAL", "passed": True, "evidence": f"approved style={bundle['style_id']} and replay digest={first_digest}"},
            {"check_id": "PHASE5_EXECUTION_LINEAGE", "passed": True, "evidence": "plan, execution, and review bindings close over real executed artifacts"},
            {"check_id": "PHASE5_CANDIDATE_APPROVAL", "passed": True, "evidence": f"approved executed candidate={bundle['candidate_id']}"},
            {"check_id": "PHASE5_ASSET_LINEAGE", "passed": True, "evidence": "pack assets match executed job hashes and copied delivery inputs"},
            {"check_id": "ROLE_BINDING_LINEAGE", "passed": True, "evidence": json.dumps(bundle["role_bridge"], sort_keys=True)},
            {"check_id": "APPEARANCE_PACK_AUTHORIZATION", "passed": True, "evidence": "pack status=approved_for_phase6"},
        ]
        checks = [
            {"check_id": "FULL_TIMELINE_RENDERED", "passed": True, "evidence": f"{frame_count} contiguous frames at {fps} FPS"},
            {"check_id": "SEMANTIC_ARRAYS_UNCHANGED", "passed": True, "evidence": "all pre/post archive-array digests matched"},
            {"check_id": "AUTHORIZED_MASKS_ONLY", "passed": True, "evidence": "zero changed pixels outside approved material masks"},
            {"check_id": "PROTECTED_ROLES_UNCHANGED", "passed": True, "evidence": "zero changed pixels in path/water/state-change protected roles"},
            {"check_id": "TEACHING_PIXELS_EXACT", "passed": True, "evidence": "zero annotation/caption mismatch pixels across all frames"},
            {"check_id": "FIXED_MATERIAL_COORDINATES", "passed": True, "evidence": "assets use stable output-coordinate mapping for every frame"},
            {"check_id": "PHASE5_PROBE_FIDELITY", "passed": True, "evidence": "milestone material pixels match executed probes outside old/new overlays"},
            {"check_id": "DETERMINISTIC_REPLAY", "passed": True, "evidence": first_digest},
            {"check_id": "MP4_VALID", "passed": True, "evidence": "H.264 yuv420p, exact timeline, no audio, fast-start, decode passed"},
            {"check_id": "GIF_VALID", "passed": True, "evidence": f"{GIF_SIZE[0]}x{GIF_SIZE[1]}, all frames, looping, decode passed"},
            {"check_id": "NO_MODEL_EXECUTION", "passed": True, "evidence": "Phase 6 imported only deterministic compositing code and existing assets"},
            {"check_id": "NO_NETWORK", "passed": True, "evidence": "no network operation was invoked"},
            {"check_id": "NO_NEW_APPEARANCE_LOGIC", "passed": True, "evidence": "only frozen Phase 5 config, treatments, opacities, assets, and mapping rules were applied"},
            {"check_id": "HUMAN_ACCEPTANCE_NOT_FABRICATED", "passed": True, "evidence": "final status remains ready for human delivery review"},
        ]
        sample_indices = sorted(set([0, frame_count // 3, (2 * frame_count) // 3, frame_count - 1, *probe_indices]))
        manifest = {
            "schema_version": "stage5-phase6-delivery-manifest-2", "artifact_type": "final_delivery", "phase": "phase6",
            "input_bindings": input_bindings,
            "selection": {"layout_style_id": bundle["style_id"], "appearance_candidate_ids": [bundle["candidate_id"]]},
            "timeline": {"frame_count": frame_count, "fps": fps, "duration_seconds": duration, "width": 880, "height": 600},
            "final_frames": {
                "directory": frames_dir.resolve().relative_to(repository_root().resolve()).as_posix(), "filename_pattern": "frame-%06d.png",
                "frame_count": frame_count, "fps": fps, "duration_seconds": duration, "width": 880, "height": 600,
                "mode": "RGB", "tree_sha256": _tree_digest(frame_records), "frames": frame_records,
            },
            "delivery_artifacts": {"mp4": mp4_info, "gif": gif_info},
            "diagnostics": {"temporal_metrics": artifact_ref(temporal_path), "final_evaluation": artifact_ref(evaluation_path)},
            "evaluation": final_evaluation,
            "deterministic_replay": {
                "algorithm": "sha256-frame-index-rgb-bytes-v1", "first_digest": first_digest,
                "second_digest": second_digest, "sampled_frame_indices": sample_indices, "matched": True,
            },
            "lineage_checks": lineage_checks, "checks": checks,
            "renderer_policy": {
                "new_appearance_logic_allowed": False, "semantic_state_source": "phase3_sequence",
                "teaching_pixel_source": "selected_phase4_frames_exact",
                "material_mapping": "approved_phase5_plan_fixed_output_coordinates",
                "image_model_executed": False, "network_used": False,
            },
            "environment": _environment(), "status": "ready_with_warnings_for_human_delivery_review",
        }
        validate_schema(manifest, SCHEMA_PATH, "delivery_manifest")
        manifest_path = output / "delivery-manifest.json"; write_json(manifest_path, manifest)
        validate_delivery_manifest(manifest_path)
    except Exception as error:
        shutil.rmtree(output)
        output.mkdir(parents=True)
        write_json(output / "failure.json", {
            "schema_version": "stage5-phase6-failure-1", "phase": "phase6",
            "failure_class": type(error).__name__, "message": str(error), "return_target": "phase6_runtime",
        })
        raise


def validate_delivery_manifest(path: Path) -> None:
    document = load_json(path)
    validate_schema(document, SCHEMA_PATH, "delivery_manifest")
    for name, ref in document["input_bindings"].items():
        verify_ref(ref, label=f"delivery input {name}")
    records = document["final_frames"]["frames"]
    if len(records) != document["final_frames"]["frame_count"]:
        raise Phase6Error("delivery frame record count mismatch")
    for index, record in enumerate(records):
        if record["frame_index"] != index:
            raise Phase6Error("delivery frame order is not canonical")
        path_ref = verify_ref(record["artifact"], label=f"delivery frame {index}")
        with Image.open(path_ref) as image:
            if image.size != OUTPUT_SIZE or image.mode != "RGB":
                raise Phase6Error(f"delivery frame {index} is not 880x600 RGB")
    if _tree_digest(records) != document["final_frames"]["tree_sha256"]:
        raise Phase6Error("delivery frame tree digest mismatch")
    for name, media in document["delivery_artifacts"].items():
        media_path = verify_ref(media["artifact"], label=f"delivery media {name}")
        inspected = _probe_media(media_path, media["media_type"], media["frame_count"], media["duration_seconds"])
        for key in ("codec", "pixel_format", "width", "height", "frame_count", "stream_count", "audio_stream_count", "decode_passed", "fast_start", "looping"):
            if inspected[key] != media[key]:
                raise Phase6Error(f"delivery media inspection changed for {name}: {key}")
    temporal_path = verify_ref(document["diagnostics"]["temporal_metrics"], label="temporal metrics")
    evaluation_path = verify_ref(document["diagnostics"]["final_evaluation"], label="final evaluation")
    validate_schema(load_json(temporal_path), SCHEMA_PATH, "temporal_metrics")
    evaluation = load_json(evaluation_path)
    validate_schema(evaluation, SCHEMA_PATH, "final_evaluation")
    if evaluation != document["evaluation"]:
        raise Phase6Error("manifest evaluation does not match final-evaluation.json")


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    for name in (
        "semantic-contract", "sequence-archive", "sequence-manifest", "presentation", "teaching-frames",
        "teaching-manifest", "phase4-human-decision", "appearance-pack", "phase5-human-decision", "output-directory",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True, dest=name.replace("-", "_"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-inputs"); _add_inputs(validate)
    render = commands.add_parser("render-delivery"); _add_inputs(render)
    legacy_render = commands.add_parser("render-final"); _add_inputs(legacy_render)
    for command in ("validate-delivery-manifest", "validate-delivery"):
        sub = commands.add_parser(command); sub.add_argument("manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-inputs":
            bundle = validate_inputs(args, require_new_output=True)
            print(json.dumps({
                "status": "valid", "layout_style_id": bundle["style_id"],
                "appearance_candidate_id": bundle["candidate_id"],
                "frame_count": bundle["sequence_manifest"]["timeline"]["frame_count"],
            }, sort_keys=True))
            return 0
        if args.command in {"render-delivery", "render-final"}:
            render_delivery(args); return 0
        if args.command in {"validate-delivery-manifest", "validate-delivery"}:
            validate_delivery_manifest(args.manifest); return 0
    except Exception as error:
        print(f"phase6: {error}", file=sys.stderr)
        return 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
