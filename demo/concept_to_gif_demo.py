from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 960, 540
BACKGROUND = (10, 18, 35)
NODE_COLORS = [
    (37, 99, 235),
    (124, 58, 237),
    (219, 39, 119),
    (245, 158, 11),
]
LABELS = ["Document", "Key concepts", "Scenes", "Animated GIF"]
CENTERS = [(145, 270), (370, 270), (595, 270), (820, 270)]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def render_frame(frame_index: int, total_frames: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font = font(42, bold=True)
    label_font = font(24, bold=True)
    small_font = font(19)
    draw.text(
        (WIDTH // 2, 72),
        "Concept-to-animation pipeline",
        fill=(235, 242, 255),
        font=title_font,
        anchor="mm",
    )
    draw.text(
        (WIDTH // 2, 118),
        "Extract meaning, plan scenes, then render motion",
        fill=(142, 164, 196),
        font=small_font,
        anchor="mm",
    )

    progress = frame_index / max(total_frames - 1, 1)
    for index, ((cx, cy), label, color) in enumerate(zip(CENTERS, LABELS, NODE_COLORS)):
        start = index * 0.19
        visibility = ease((progress - start) / 0.18)

        if index > 0:
            previous_x, previous_y = CENTERS[index - 1]
            line_progress = ease((progress - (start - 0.09)) / 0.12)
            end_x = previous_x + 80 + (cx - 80 - (previous_x + 80)) * line_progress
            draw.line(
                (previous_x + 80, previous_y, end_x, cy),
                fill=(86, 111, 151),
                width=7,
            )
            if line_progress > 0.94:
                draw.polygon(
                    [(cx - 80, cy), (cx - 98, cy - 11), (cx - 98, cy + 11)],
                    fill=(86, 111, 151),
                )

        if visibility <= 0:
            continue

        pulse = 1.0 + 0.035 * math.sin(frame_index * 0.35 + index)
        half_width = int(82 * visibility * pulse)
        half_height = int(62 * visibility * pulse)
        glow = tuple(min(255, channel + 30) for channel in color)
        draw.rounded_rectangle(
            (cx - half_width - 8, cy - half_height - 8, cx + half_width + 8, cy + half_height + 8),
            radius=28,
            outline=glow,
            width=4,
        )
        draw.rounded_rectangle(
            (cx - half_width, cy - half_height, cx + half_width, cy + half_height),
            radius=24,
            fill=color,
        )
        draw.text((cx, cy), label, fill="white", font=label_font, anchor="mm")

    if progress > 0.82:
        fade = int(255 * ease((progress - 0.82) / 0.12))
        draw.text(
            (WIDTH // 2, 440),
            "A local LLM can produce the scene plan; Python renders the result.",
            fill=(fade, fade, fade),
            font=small_font,
            anchor="mm",
        )

    return image


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "concept_pipeline.gif"
    total_frames = 52
    frames = [render_frame(index, total_frames) for index in range(total_frames)]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=75,
        loop=0,
        optimize=True,
    )
    print(output_path)


if __name__ == "__main__":
    main()
