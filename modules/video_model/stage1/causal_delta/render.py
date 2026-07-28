"""Render mechanism states without modifying or inferring mechanism data."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import SimulationConfig, original_land, water_depth


COLORS = {
    "sea_deep": np.array([36, 101, 126], dtype=np.float64),
    "sea_shallow": np.array([91, 159, 169], dtype=np.float64),
    "original_land": np.array([116, 128, 91], dtype=np.float64),
    "original_land_light": (148, 154, 112),
    "deposit": np.array([168, 113, 59], dtype=np.float64),
    "new_land": np.array([151, 169, 91], dtype=np.float64),
    "new_land_edge": (244, 204, 102),
    "flow": (110, 213, 243),
    "flow_slow": (123, 190, 210),
    "particle": (199, 121, 55),
    "particle_edge": (109, 61, 30),
}


ENGLISH_CAPTIONS = {
    "transport": "Suspended sediment is carried downstream",
    "decelerate": "The river slows as it enters open water",
    "accumulate": "Sediment builds up below the surface",
    "threshold_change": "Deposits exceed water depth and emerge",
    "reroute": "New land reroutes the flow into branches",
}


def find_font() -> tuple[Path, bool]:
    candidates = [
        os.environ.get("DELTA_FONT"),
        "/tmp/noto-cjk/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
        "/tmp/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for value in candidates:
        if value and Path(value).is_file():
            path = Path(value)
            has_cjk = "Noto" in path.name or "wqy" in path.name.lower()
            return path, has_cjk
    raise RuntimeError("no usable TrueType/OpenType font found")


def _font(size: int) -> tuple[ImageFont.FreeTypeFont, bool, str]:
    path, has_cjk = find_font()
    return ImageFont.truetype(str(path), size), has_cjk, str(path)


def _edge(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="edge")
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~eroded


def _state_arrays(state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thickness = np.asarray(state["thick"], dtype=np.float64)
    land = np.asarray(state["land"], dtype=bool)
    new_land = np.asarray(state["new_land"], dtype=bool)
    return thickness, land, new_land


def render_clean_base(
    state: dict[str, Any],
    config: SimulationConfig | None = None,
    *,
    antialias: bool = False,
    show_new_land_edge: bool = True,
) -> Image.Image:
    """Render only semantic geography; no arrows, particles, text or panels."""

    config = config or SimulationConfig()
    base_land = original_land(config)
    depth = water_depth(config)
    thickness, _, new_land = _state_arrays(state)
    height, width = thickness.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    offshore = np.maximum(0.0, xx - config.coastline_x)
    shallow_factor = np.exp(-offshore / 35.0)[:, :] * 0.52
    shallow_factor += 0.10 * np.cos((yy - config.river_center_y) / 10.0)
    shallow_factor = np.clip(shallow_factor, 0.06, 0.72)[..., None]
    rgb = COLORS["sea_deep"] * (1.0 - shallow_factor) + COLORS["sea_shallow"] * shallow_factor

    rgb[base_land] = COLORS["original_land"]
    finite_depth = np.where(np.isfinite(depth), depth, 1.0)
    deposit_strength = np.clip(thickness / np.maximum(finite_depth, 0.08), 0.0, 1.0)
    deposit_mask = (thickness > 0.001) & ~base_land & ~new_land
    alpha = (0.20 + 0.52 * deposit_strength)[..., None]
    blended = rgb * (1.0 - alpha) + COLORS["deposit"] * alpha
    rgb[deposit_mask] = blended[deposit_mask]
    rgb[new_land] = COLORS["new_land"]

    image = Image.fromarray(np.uint8(np.clip(rgb, 0, 255)), mode="RGB").resize(
        (config.canvas_width, config.canvas_height),
        Image.Resampling.LANCZOS if antialias else Image.Resampling.NEAREST,
    )
    draw = ImageDraw.Draw(image, "RGBA")
    cell_x = config.canvas_width / config.grid_width
    cell_y = config.canvas_height / config.grid_height

    # Restrained terrain hatching clipped to original land.
    hatch = Image.new("RGBA", image.size, (0, 0, 0, 0))
    hatch_draw = ImageDraw.Draw(hatch)
    for diagonal in range(-config.canvas_height, config.canvas_width, 22):
        hatch_draw.line(
            (diagonal, 0, diagonal + config.canvas_height, config.canvas_height),
            fill=(*COLORS["original_land_light"], 55),
            width=1,
        )
    land_mask = Image.fromarray(np.uint8(base_land) * 255).resize(
        image.size, Image.Resampling.NEAREST
    )
    image.paste(hatch, (0, 0), Image.composite(hatch.getchannel("A"), Image.new("L", image.size), land_mask))
    draw = ImageDraw.Draw(image, "RGBA")

    if show_new_land_edge:
        new_edge = _edge(new_land)
        for y, x in np.argwhere(new_edge):
            draw.rectangle(
                (
                    x * cell_x,
                    y * cell_y,
                    (x + 1) * cell_x,
                    (y + 1) * cell_y,
                ),
                outline=(*COLORS["new_land_edge"], 215),
                width=1,
            )
    return image


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int, int],
    width: int = 2,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = 4.2
    left = (
        end[0] - head * math.cos(angle - 0.60),
        end[1] - head * math.sin(angle - 0.60),
    )
    right = (
        end[0] - head * math.cos(angle + 0.60),
        end[1] - head * math.sin(angle + 0.60),
    )
    draw.polygon((end, left, right), fill=color)


def render_state(
    state: dict[str, Any],
    *,
    beat_index: int,
    config: SimulationConfig | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    config = config or SimulationConfig()
    image = render_clean_base(state, config)
    draw = ImageDraw.Draw(image, "RGBA")
    cell_x = config.canvas_width / config.grid_width
    cell_y = config.canvas_height / config.grid_height

    # Suspended particles are shown using a fixed audit-friendly sample.
    for particle in state["particles"]:
        if particle["id"] % 7:
            continue
        px = (particle["x"] + 0.5) * cell_x
        py = (particle["y"] + 0.5) * cell_y
        radius = 2.25
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=(*COLORS["particle"], 225),
            outline=(*COLORS["particle_edge"], 180),
            width=1,
        )

    for x, y, flow_x, flow_y, speed in state["flow_samples"]:
        if speed <= 0:
            continue
        start = ((x + 0.5) * cell_x, (y + 0.5) * cell_y)
        length = 4.0 + 14.0 * min(1.0, speed / config.river_speed)
        end = (
            start[0] + flow_x / speed * length,
            start[1] + flow_y / speed * length,
        )
        slow_mix = min(1.0, speed / config.river_speed)
        color = tuple(
            int(COLORS["flow_slow"][index] * (1 - slow_mix) + COLORS["flow"][index] * slow_mix)
            for index in range(3)
        )
        _arrow(draw, start, end, (*color, 205), width=2)

    title_font, has_cjk, font_path = _font(22)
    caption_font, _, _ = _font(25)
    small_font, _, _ = _font(14)
    tiny_font, _, _ = _font(12)
    title = "三角洲形成机制" if has_cjk else "HOW A RIVER DELTA FORMS"
    caption = (
        state["caption"] if has_cjk else ENGLISH_CAPTIONS[state["beat_id"]]
    )

    draw.rounded_rectangle((18, 16, 258, 51), radius=9, fill=(7, 29, 38, 205))
    draw.text((31, 21), title, font=title_font, fill=(242, 246, 231, 255))

    # Fixed legend.
    legend_box = (553, 16, 750, 82)
    draw.rounded_rectangle(legend_box, radius=9, fill=(7, 29, 38, 198))
    _arrow(draw, (568, 36), (590, 36), (*COLORS["flow"], 255), width=2)
    draw.text((599, 27), "水流" if has_cjk else "flow", font=small_font, fill="white")
    draw.ellipse((568, 55, 575, 62), fill=(*COLORS["particle"], 255))
    draw.text((582, 50), "悬沙" if has_cjk else "sediment", font=small_font, fill="white")
    new_land_color = tuple(int(channel) for channel in COLORS["new_land"])
    draw.rectangle((657, 54, 666, 63), fill=(*new_land_color, 255))
    draw.text((672, 50), "新陆" if has_cjk else "new land", font=small_font, fill="white")

    panel_top = 432
    draw.rectangle((0, panel_top, config.canvas_width, config.canvas_height), fill=(5, 24, 31, 225))
    caption_box = draw.textbbox((0, 0), caption, font=caption_font)
    caption_width = caption_box[2] - caption_box[0]
    draw.text(
        ((config.canvas_width - caption_width) / 2, panel_top + 8),
        caption,
        font=caption_font,
        fill=(248, 242, 213, 255),
    )
    stage_names_cn = ["输送", "减速", "累积", "出水", "分流"]
    stage_names_en = ["carry", "slow", "build", "emerge", "branch"]
    stage_names = stage_names_cn if has_cjk else stage_names_en
    centers = np.linspace(172, 596, 5)
    for index, (center, label) in enumerate(zip(centers, stage_names)):
        active = index <= beat_index
        fill = (231, 178, 78, 255) if active else (95, 121, 129, 255)
        draw.ellipse((center - 5, 477, center + 5, 487), fill=fill)
        bbox = draw.textbbox((0, 0), label, font=tiny_font)
        draw.text(
            (center - (bbox[2] - bbox[0]) / 2, 491),
            label,
            font=tiny_font,
            fill=(229, 235, 229, 240),
        )
        if index < 4:
            draw.line((center + 8, 482, centers[index + 1] - 8, 482), fill=(106, 135, 143, 150), width=2)

    return image, {"font_path": font_path, "cjk_font": has_cjk}
