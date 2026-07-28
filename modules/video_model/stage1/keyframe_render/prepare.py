"""Prepare clean semantic keyframes from the audited mechanism timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ..causal_delta.config import MECHANISM_ROOT, OUTPUT_ROOT, SimulationConfig, original_land
from ..causal_delta.render import render_clean_base
from ..causal_delta.validate import load_states


KEYFRAME_OUTPUT_ROOT = OUTPUT_ROOT.parent / "keyframe_render"
WORK_DIR_NAME = "_work"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_keyframe(
    timeline: list[dict[str, Any]],
    beat_id: str,
) -> dict[str, Any]:
    """Resolve the end of a teaching beat without hard-coding display indices."""

    candidates = [entry for entry in timeline if entry["beat_id"] == beat_id]
    if not candidates:
        raise ValueError(f"beat not present in timeline: {beat_id}")
    return max(
        candidates,
        key=lambda entry: (
            entry["state_frame"],
            entry["display_frame"],
        ),
    )


def semantic_masks(
    state: dict[str, Any],
    config: SimulationConfig,
) -> dict[str, np.ndarray]:
    base_land = original_land(config)
    land = np.asarray(state["land"], dtype=bool)
    new_land = np.asarray(state["new_land"], dtype=bool)
    thickness = np.asarray(state["thick"], dtype=float)
    yy, xx = np.indices(base_land.shape)
    river = (~land) & (xx < config.coastline_x)
    underwater = (thickness > 0.001) & ~land & (xx >= config.coastline_x)
    ocean = (~land) & ~river & ~underwater
    masks = {
        "original_land": base_land,
        "river": river,
        "ocean": ocean,
        "underwater_deposit": underwater,
        "new_land": new_land,
    }
    coverage = np.zeros_like(base_land, dtype=np.uint8)
    for mask in masks.values():
        coverage += mask.astype(np.uint8)
    if not np.all(coverage == 1):
        values, counts = np.unique(coverage, return_counts=True)
        raise ValueError(f"semantic masks must be exhaustive and exclusive: {dict(zip(values, counts))}")
    return masks


def _save_mask(mask: np.ndarray, path: Path, config: SimulationConfig) -> None:
    image = Image.fromarray(np.uint8(mask) * 255, mode="L").resize(
        (config.canvas_width, config.canvas_height),
        Image.Resampling.NEAREST,
    )
    image.save(path)


def prepare(
    mechanism_root: Path = MECHANISM_ROOT,
    causal_output_root: Path = OUTPUT_ROOT,
    output_root: Path = KEYFRAME_OUTPUT_ROOT,
) -> dict[str, Any]:
    states = load_states(mechanism_root / "states.jsonl")
    timeline = json.loads((causal_output_root / "timeline.json").read_text(encoding="utf-8"))
    config_data = json.loads((mechanism_root / "simulation_config.json").read_text(encoding="utf-8"))
    config = SimulationConfig(**config_data)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = output_root / WORK_DIR_NAME
    base_root = work_root / "base_images"
    mask_root = work_root / "semantic_masks"
    edge_root = work_root / "control_edges"
    base_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    edge_root.mkdir(parents=True, exist_ok=True)
    selections = {
        "first": semantic_keyframe(timeline, "accumulate"),
        "last": semantic_keyframe(timeline, "threshold_change"),
    }
    manifest: dict[str, Any] = {
        "status": "prepared",
        "selection_method": "semantic beat lookup in timeline.json",
        "source_timeline": str((causal_output_root / "timeline.json").resolve()),
        "keyframes": {},
    }
    for name, selection in selections.items():
        state = states[selection["state_frame"]]
        base = render_clean_base(
            state,
            config,
            antialias=True,
            show_new_land_edge=False,
        )
        base_path = base_root / f"{name}.png"
        base.save(base_path)
        try:
            import cv2

            gray = np.asarray(base.convert("L"))
            edge = Image.fromarray(cv2.Canny(gray, 100, 200), mode="L")
            conditioning_method = "OpenCV Canny thresholds 100/200"
        except ImportError:
            edge = ImageOps.autocontrast(
                base.convert("L").filter(ImageFilter.FIND_EDGES)
            ).filter(ImageFilter.GaussianBlur(0.55))
            conditioning_method = "Pillow FIND_EDGES fallback"
        edge_path = edge_root / f"{name}_canny.png"
        edge.save(edge_path)
        masks = semantic_masks(state, config)
        mask_records = {}
        for category, mask in masks.items():
            mask_path = mask_root / f"{name}_{category}.png"
            _save_mask(mask, mask_path, config)
            mask_records[category] = {
                "path": str(mask_path.resolve()),
                "grid_cells": int(mask.sum()),
                "sha256": _sha256(mask_path),
            }
        manifest["keyframes"][name] = {
            "beat_id": selection["beat_id"],
            "display_frame": selection["display_frame"],
            "state_frame": selection["state_frame"],
            "is_hold": selection["is_hold"],
            "clean_base": str(base_path.resolve()),
            "conditioning_image": str(edge_path.resolve()),
            "conditioning_method": conditioning_method,
            "masks": mask_records,
            "removed_overlays": [
                "flow arrows",
                "sediment particles",
                "captions",
                "legend",
                "progress dots",
                "bottom panel",
            ],
        }
    (work_root / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-root", type=Path, default=MECHANISM_ROOT)
    parser.add_argument("--causal-output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=KEYFRAME_OUTPUT_ROOT)
    args = parser.parse_args()
    result = prepare(args.mechanism_root, args.causal_output, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
