"""Build sparse geometry controls and their explanatory artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .projection import Projection
from .schema import resolve_stage_path
from .utils import image_record


def _scaled_mechanism_geometry(
    geometry_source: np.ndarray,
    projection: Projection,
) -> Image.Image:
    image = Image.fromarray(
        np.uint8(geometry_source > 0) * 255, mode="L"
    )
    return image.resize(
        (projection.mechanism_width * 8, projection.mechanism_height * 8),
        Image.Resampling.NEAREST,
    )


def build_control(
    spec: dict[str, Any],
    record: dict[str, Any],
    projection: Projection,
    geometry_source: np.ndarray,
    hard_boundary: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    control_root = output_root / "_work" / "controls"
    geometry_path = (
        control_root / "geometry_source" / f"{record['id']}.png"
    )
    projected_path = (
        control_root / "projected_boundaries" / f"{record['id']}.png"
    )
    canny_path = control_root / "canny" / f"{record['id']}.png"
    overlay_path = (
        control_root / "anchor_overlay" / f"{record['id']}.png"
    )
    for path in (
        geometry_path,
        projected_path,
        canny_path,
        overlay_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    _scaled_mechanism_geometry(geometry_source, projection).save(
        geometry_path
    )

    base_path = resolve_stage_path(spec["paths"]["base_control"])
    base = Image.open(base_path).convert("L")
    if base.size != (projection.width, projection.height):
        base = base.resize(
            (projection.width, projection.height),
            Image.Resampling.NEAREST,
        )
    base_array = np.asarray(base, dtype=np.uint8)
    dynamic_edges = cv2.morphologyEx(
        np.uint8(hard_boundary > 0) * 255,
        cv2.MORPH_GRADIENT,
        np.ones((5, 5), dtype=np.uint8),
    )
    projected = np.maximum(
        base_array,
        cv2.GaussianBlur(dynamic_edges, (0, 0), 0.8),
    )
    Image.fromarray(projected, mode="L").save(projected_path)
    canny = np.uint8(projected >= 42) * 255
    Image.fromarray(canny, mode="L").save(canny_path)

    anchor = np.asarray(
        Image.open(resolve_stage_path(spec["paths"]["visual_anchor"]))
        .convert("RGB")
        .resize((projection.width, projection.height)),
        dtype=np.float32,
    )
    overlay = anchor.copy()
    edge = canny > 0
    overlay[edge] = overlay[edge] * 0.25 + np.array(
        [255.0, 61.0, 42.0], dtype=np.float32
    ) * 0.75
    Image.fromarray(np.uint8(np.clip(overlay, 0, 255))).save(overlay_path)

    unique = sorted(int(value) for value in np.unique(canny))
    if unique != [0, 255]:
        raise ValueError(f"ControlNet input is not binary: {unique}")
    manifest = {
        "geometry_source": image_record(
            geometry_path,
            meaning=(
                "机制网格中的总陆地；用于理解数据来源，不输入模型"
            ),
            model_input=False,
        ),
        "projected_boundaries": image_record(
            projected_path,
            meaning="机制新陆地边界投影后与固定河岸海岸合并的中间图",
            model_input=False,
        ),
        "canny": image_record(
            canny_path,
            meaning="实际输入 SDXL Canny ControlNet 的黑底白线图",
            model_input=True,
            edge_pixels=int(edge.sum()),
            edge_fraction=round(float(edge.mean()), 6),
            unique_values=unique,
        ),
        "anchor_overlay": image_record(
            overlay_path,
            meaning="红线显示 Canny 在视觉锚点上的实际位置",
            model_input=False,
        ),
        "base_control": {
            "source": str(base_path.resolve()),
            "copied_from_stage_1_2": True,
        },
        "derivation": [
            "load the fixed Stage 1.2 coast and river-bank control",
            "project the current state's new_land region",
            "extract only its outer boundary",
            "merge and binarize at threshold 42",
        ],
    }
    return manifest
