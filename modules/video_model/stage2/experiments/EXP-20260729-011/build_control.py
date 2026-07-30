"""Normalize semantic line-art occupancy without changing its topology."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
STAGE2_ROOT = ROOT.parents[1]
SOURCE = (
    STAGE2_ROOT
    / "output/phase-2/MATH-02/keyframes/03_end/layers"
    / "math02_hard_boundary.npy"
)
OUTPUT = ROOT / "centered_hard_boundary.png"
METADATA = ROOT / "centered_hard_boundary.json"
WIDTH = 1024
HEIGHT = 576
TARGET_SHORT_EDGE_FRACTION = 0.90


def build() -> None:
    source = np.load(SOURCE)
    ys, xs = np.nonzero(source > 0)
    if not len(xs):
        raise ValueError("source boundary is empty")
    source_bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )
    crop = Image.fromarray(source, mode="L").crop(source_bbox)
    target_size = int(round(HEIGHT * TARGET_SHORT_EDGE_FRACTION))
    resized = crop.resize((target_size, target_size), Image.Resampling.NEAREST)
    canvas = Image.new("L", (WIDTH, HEIGHT), 0)
    paste_xy = ((WIDTH - target_size) // 2, (HEIGHT - target_size) // 2)
    canvas.paste(resized, paste_xy)
    canvas.convert("RGB").save(OUTPUT, optimize=False)

    METADATA.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "classification": (
                    "deterministic occupancy-normalized semantic line art"
                ),
                "source": str(SOURCE.relative_to(STAGE2_ROOT)),
                "source_bbox_xyxy": source_bbox,
                "canvas": {"width": WIDTH, "height": HEIGHT},
                "target_size_px": target_size,
                "target_short_edge_fraction": TARGET_SHORT_EDGE_FRACTION,
                "paste_xy": paste_xy,
                "topology_change": False,
                "excluded": [
                    "new semantic lines",
                    "labels",
                    "arrows",
                    "material shading",
                    "model-generated pixels"
                ],
                "output": OUTPUT.name
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True
        )
        + "\n",
        encoding="utf-8"
    )


if __name__ == "__main__":
    build()

