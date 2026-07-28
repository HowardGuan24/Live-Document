"""Configuration and fixed geometry for the causal delta simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
STAGE_ROOT = PACKAGE_ROOT.parent
OUTPUT_ROOT = STAGE_ROOT / "output" / "causal_delta"
MECHANISM_ROOT = OUTPUT_ROOT / "mechanism"
FRAMES_ROOT = OUTPUT_ROOT / "frames"


@dataclass(frozen=True)
class SimulationConfig:
    """The accepted historical configuration plus deterministic geometry."""

    grid_width: int = 96
    grid_height: int = 64
    coastline_x: int = 38
    state_count: int = 120
    fps: int = 12
    random_seed: int = 1909
    sediment_per_frame: int = 14
    river_speed: float = 1.34
    expansion_rate: float = 0.35
    sea_min_speed: float = 0.16
    base_settling_rate: float = 0.02
    deposit_mass: float = 0.12
    dispersion_radius: int = 2
    lateral_perturbation: float = 0.055
    mouth_protection_columns: int = 2
    stable_channel_frames: int = 5
    flow_sample_spacing: int = 4
    river_center_y: float = 31.5
    river_half_width: int = 4
    canvas_width: int = 768
    canvas_height: int = 512
    bathymetry_version: str = "mouth-platform-v2"
    base_depth: float = 0.62
    offshore_depth_slope: float = 0.020
    mouth_bar_center_x: float = 47.0
    mouth_bar_sigma_x: float = 8.5
    mouth_bar_sigma_y: float = 5.2
    mouth_bar_amplitude: float = 0.46
    depth_scale: float = 0.78
    north_shoal_center_x: float = 46.0
    north_shoal_center_y: float = 26.0
    north_shoal_sigma_x: float = 3.2
    north_shoal_sigma_y: float = 2.8
    north_shoal_amplitude: float = 0.10
    offshore_shoal_center_x: float = 52.0
    offshore_shoal_center_y: float = 25.5
    offshore_shoal_sigma_x: float = 2.0
    offshore_shoal_sigma_y: float = 1.0
    offshore_shoal_amplitude: float = 0.33
    connector_shoal_center_x: float = 48.5
    connector_shoal_center_y: float = 26.0
    connector_shoal_sigma_x: float = 0.8
    connector_shoal_sigma_y: float = 0.65
    connector_shoal_amplitude: float = 0.025
    south_scour_center_x: float = 46.0
    south_scour_center_y: float = 36.0
    south_scour_sigma_x: float = 4.0
    south_scour_sigma_y: float = 3.2
    south_scour_amplitude: float = 0.14
    minimum_water_depth: float = 0.11

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


BEATS = (
    ("transport", 0, 27, "河流输送悬浮泥沙"),
    ("decelerate", 28, 49, "河水入海后减速"),
    ("accumulate", 50, 100, "泥沙在水下逐层累积"),
    ("threshold_change", 101, 107, "沉积超过水深，沙洲出水"),
    ("reroute", 108, 119, "新生陆地使水流绕行分流"),
)


def beat_for_frame(frame: int) -> tuple[str, str]:
    for beat_id, start, end, caption in BEATS:
        if start <= frame <= end:
            return beat_id, caption
    raise ValueError(f"frame outside configured timeline: {frame}")


def original_land(config: SimulationConfig) -> np.ndarray:
    """Return immutable pre-existing land, with a river cut through it."""

    height, width = config.grid_height, config.grid_width
    yy, xx = np.indices((height, width))
    land = xx < config.coastline_x
    channel = np.abs(yy - config.river_center_y) <= config.river_half_width
    land[channel & (xx < config.coastline_x)] = False
    return land


def water_depth(config: SimulationConfig) -> np.ndarray:
    """Create a shallow mouth platform without encoding a target delta outline."""

    height, width = config.grid_height, config.grid_width
    yy, xx = np.indices((height, width), dtype=np.float64)
    offshore = np.maximum(0.0, xx - config.coastline_x)
    depth = config.base_depth + config.offshore_depth_slope * offshore

    # A smooth, geologically plausible mouth bar: geometry is a scalar depth
    # field, never a target land mask. Land still emerges only through the
    # global thickness > depth rule.
    mouth_bar = np.exp(
        -((xx - config.mouth_bar_center_x) ** 2) / (2.0 * config.mouth_bar_sigma_x**2)
        - ((yy - config.river_center_y) ** 2) / (2.0 * config.mouth_bar_sigma_y**2)
    )
    depth -= config.mouth_bar_amplitude * mouth_bar
    # Historical attempt 08 used a broad shallow platform plus a smaller
    # natural shoal north of the river axis. The asymmetry gives the first bar
    # somewhere to nucleate without prescribing its final outline.
    depth *= config.depth_scale
    north_shoal = np.exp(
        -((xx - config.north_shoal_center_x) ** 2)
        / (2.0 * config.north_shoal_sigma_x**2)
        - ((yy - config.north_shoal_center_y) ** 2)
        / (2.0 * config.north_shoal_sigma_y**2)
    )
    depth -= config.north_shoal_amplitude * north_shoal
    offshore_shoal = np.exp(
        -((xx - config.offshore_shoal_center_x) ** 2)
        / (2.0 * config.offshore_shoal_sigma_x**2)
        - ((yy - config.offshore_shoal_center_y) ** 2)
        / (2.0 * config.offshore_shoal_sigma_y**2)
    )
    depth -= config.offshore_shoal_amplitude * offshore_shoal
    connector_shoal = np.exp(
        -((xx - config.connector_shoal_center_x) ** 2)
        / (2.0 * config.connector_shoal_sigma_x**2)
        - ((yy - config.connector_shoal_center_y) ** 2)
        / (2.0 * config.connector_shoal_sigma_y**2)
    )
    depth -= config.connector_shoal_amplitude * connector_shoal
    # The faster southern branch keeps a modest scour trough. This prevents a
    # disconnected shoal from being classified as land while leaving its
    # underwater sediment visible.
    south_scour = np.exp(
        -((xx - config.south_scour_center_x) ** 2)
        / (2.0 * config.south_scour_sigma_x**2)
        - ((yy - config.south_scour_center_y) ** 2)
        / (2.0 * config.south_scour_sigma_y**2)
    )
    depth += config.south_scour_amplitude * south_scour
    depth = np.clip(depth, config.minimum_water_depth, 2.2)
    depth[original_land(config)] = np.inf
    return depth
