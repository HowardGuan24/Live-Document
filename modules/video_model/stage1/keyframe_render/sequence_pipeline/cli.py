"""Run the config-driven mechanism keyframe sequence pipeline."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

from ..first_frame_test import _contact_sheet
from .adapters import get_state_adapter
from .candidates import generate_candidates
from .composite import compose_sequence
from .controls import build_control
from .evaluate import evaluate_sequence
from .projection import Projection
from .prompts import build_prompt_record
from .report import write_prepare_audit_report, write_report
from .schema import default_output_root, load_spec, resolve_stage_path
from .semantic_layers import get_semantic_builder
from .utils import image_record, sha256, stable_hash, write_json


def _recorded_paths_exist(value: Any) -> bool:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str):
            if not Path(value["path"]).is_file():
                return False
        return all(_recorded_paths_exist(item) for item in value.values())
    if isinstance(value, list):
        return all(_recorded_paths_exist(item) for item in value)
    return True


def prepare(
    spec: dict[str, Any],
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    adapter = get_state_adapter(spec["state_adapter"])
    semantic_builder = get_semantic_builder(spec["semantic_builder"])
    records, sources = adapter(spec)
    signature_files = [
        Path(spec["_spec_path"]),
        *[
            resolve_stage_path(value)
            for value in spec["paths"].values()
            if resolve_stage_path(value).is_file()
        ],
        *[Path(record["program_frame"]) for record in records.values()],
        Path(__file__),
        Path(inspect.getsourcefile(adapter) or __file__),
        Path(semantic_builder.__file__ or __file__),
        Path(__file__).with_name("projection.py"),
        Path(__file__).with_name("semantic_layers.py"),
        Path(__file__).with_name("controls.py"),
        Path(__file__).with_name("prompts.py"),
    ]
    signature = stable_hash(
        {
            "files": {
                str(path.resolve()): sha256(path)
                for path in signature_files
            },
            "sequence_id": spec["sequence_id"],
        }
    )
    manifest_path = (
        output_root / "_work" / "manifests" / "prepare.json"
    )
    if manifest_path.is_file() and not force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("input_signature") == signature
            and _recorded_paths_exist(previous)
        ):
            previous["cache"] = {
                "reused": True,
                "policy": (
                    "spec, source data, selected program frames, anchor, "
                    "control source and prepare code hashes all match"
                ),
            }
            write_json(manifest_path, previous)
            write_prepare_audit_report(spec, output_root, previous)
            return previous
    size = (int(spec["canvas"]["width"]), int(spec["canvas"]["height"]))
    projection = Projection(spec["projection"], size)
    semantic_context = semantic_builder.prepare_context(
        records, spec, projection
    )
    source_root = output_root / "_work" / "source"
    audit_root = source_root / "mechanism_frames"
    summary_root = source_root / "state_summaries"
    audit_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    ordered_specs = [spec["anchor"], *spec["keyframes"]]
    manifest: dict[str, Any] = {
        "status": "prepared",
        "sequence_id": spec["sequence_id"],
        "spec_path": spec["_spec_path"],
        "spec_sha256": sha256(Path(spec["_spec_path"])),
        "input_signature": signature,
        "cache": {"reused": False},
        "output_root": str(output_root.resolve()),
        "sources": sources,
        "projection": spec["projection"],
        "semantic_builder": spec["semantic_builder"],
        "semantic_context": {
            key: value
            for key, value in semantic_context.items()
            if not key.startswith("_")
        },
        "anchor": {},
        "keyframes": {},
    }
    program_panels: list[tuple[str, Path]] = []
    for item in ordered_specs:
        record = records[item["id"]]
        program_source = Path(record["program_frame"])
        program_target = audit_root / f"{record['id']}.png"
        shutil.copyfile(program_source, program_target)
        program_panels.append(
            (
                f"{record['id']} | display {record['display_frame']} "
                f"| state {record['state_frame']}",
                program_target,
            )
        )
        summary = {
            "id": record["id"],
            "display_frame": record["display_frame"],
            "state_frame": record["state_frame"],
            "beat_id": record["beat_id"],
            "caption": record["caption"],
            "meaning": record["meaning"],
            "stats": record["stats"],
            "program_frame": image_record(
                program_target, model_input=False
            ),
        }
        write_json(summary_root / f"{record['id']}.json", summary)
        layers, arrays = semantic_builder.build_layers(
            record, projection, output_root, semantic_context
        )
        entry: dict[str, Any] = {
            **summary,
            "semantic_layers": layers,
        }
        if item["id"] == spec["anchor"]["id"]:
            manifest["anchor"] = entry
            continue
        entry["control"] = build_control(
            spec,
            record,
            projection,
            arrays["geometry_source"],
            arrays["hard_boundary"],
            output_root,
        )
        entry["prompt"] = build_prompt_record(
            spec, item, output_root
        )
        entry["output_filename"] = item["output_filename"]
        manifest["keyframes"][item["id"]] = entry

    _contact_sheet(
        program_panels,
        output_root / "source-comparison.jpg",
        columns=min(3, len(program_panels)),
    )
    final_root = output_root / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    anchor_source = resolve_stage_path(spec["paths"]["visual_anchor"])
    anchor_target = final_root / spec["anchor"]["output_filename"]
    shutil.copyfile(anchor_source, anchor_target)
    manifest["anchor"]["final"] = image_record(
        anchor_target,
        classification="existing selected Stage 1.2 visual anchor",
    )
    write_json(manifest_path, manifest)
    write_prepare_audit_report(spec, output_root, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not any(
        (
            args.prepare,
            args.generate,
            args.compose,
            args.evaluate,
            args.report,
        )
    ):
        parser.error("choose at least one pipeline stage")
    spec = load_spec(args.spec)
    output_root = (
        args.output.resolve()
        if args.output
        else default_output_root(spec)
    )
    result: dict[str, Any] = {}
    if args.prepare:
        result["prepare"] = prepare(
            spec, output_root, force=args.force
        )["status"]
    if args.generate:
        result["generate"] = generate_candidates(
            spec, output_root, force=args.force
        )["status"]
    if args.compose:
        result["compose"] = compose_sequence(
            spec, output_root, force=args.force
        )["status"]
    if args.evaluate:
        result["evaluate"] = evaluate_sequence(
            spec, output_root, force=args.force
        )["status"]
    if args.report:
        result["report"] = str(write_report(spec, output_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
