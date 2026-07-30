"""Expose the program water boundary as the projection protection control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
STAGE2_ROOT = ROOT.parents[1]
SOURCE = (
    STAGE2_ROOT
    / "output/phase-4/programs/GEO-01/keyframes/03_end/layers"
    / "geo01_water_boundary.npy"
)
OUTPUT = ROOT / "protected_water_boundary.png"
METADATA = ROOT / "protected_water_boundary.json"


def build() -> None:
    boundary = np.load(SOURCE, allow_pickle=False)
    Image.fromarray(boundary, mode="L").convert("RGB").save(
        OUTPUT, optimize=False
    )
    METADATA.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "classification": "deterministic program hard boundary",
                "source": str(SOURCE.relative_to(STAGE2_ROOT)),
                "meaning_zh": (
                    "主河、牛轭湖、捷径和两个封堵的全部水陆边界；"
                    "作为材质投影不可修改带。"
                ),
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

