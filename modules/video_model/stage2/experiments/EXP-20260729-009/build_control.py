"""Build a semantic, text-free titration apparatus line-art control."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "semantic_apparatus_line_art.png"
METADATA = ROOT / "semantic_apparatus_line_art.json"
WIDTH = 1024
HEIGHT = 576


def build() -> None:
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    white = 255
    line = 4

    # Long graduated burette, stopcock, and narrow outlet tip.
    draw.rounded_rectangle(
        (492, 18, 532, 176), radius=11, outline=white, width=line
    )
    for y in range(42, 152, 18):
        tick = 12 if (y // 18) % 2 else 18
        draw.line((492, y, 492 + tick, y), fill=white, width=2)
    draw.ellipse((494, 148, 530, 184), outline=white, width=line)
    draw.line((468, 166, 556, 166), fill=white, width=line)
    draw.ellipse((548, 158, 564, 174), outline=white, width=3)
    draw.line((504, 182, 504, 222), fill=white, width=3)
    draw.line((520, 182, 520, 222), fill=white, width=3)
    draw.line((504, 222, 511, 238), fill=white, width=3)
    draw.line((520, 222, 513, 238), fill=white, width=3)

    # Open cylindrical beaker with an oval rim, slightly tapered sides,
    # curved base, liquid surface, and a few short graduations.
    draw.ellipse((320, 228, 704, 282), outline=white, width=line)
    draw.line((322, 255, 350, 492), fill=white, width=line)
    draw.line((702, 255, 674, 492), fill=white, width=line)
    draw.arc((350, 468, 674, 516), 0, 180, fill=white, width=line)
    draw.arc((350, 468, 674, 516), 180, 360, fill=white, width=line)
    draw.ellipse((346, 398, 678, 434), outline=white, width=3)
    for y, length in ((326, 24), (354, 16), (382, 24), (446, 18)):
        draw.line((674 - length, y, 674, y), fill=white, width=2)

    image.convert("RGB").save(OUTPUT, optimize=False)
    METADATA.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "canvas": {"width": WIDTH, "height": HEIGHT},
                "classification": (
                    "deterministic text-free semantic line-art control"
                ),
                "components": [
                    "one open cylindrical beaker with oval rim",
                    "one liquid surface ellipse",
                    "one graduated burette",
                    "one stopcock and handle",
                    "one narrow outlet tip ending above the beaker",
                ],
                "excluded": [
                    "labels",
                    "arrows",
                    "UI",
                    "material shading",
                    "model-generated pixels",
                ],
                "output": OUTPUT.name,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()

