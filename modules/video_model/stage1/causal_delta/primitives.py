"""Reusable causal primitives for flow, transport, deposition and land change."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import SimulationConfig


@dataclass
class Particle:
    id: int
    x: float
    y: float
    age: int = 0
    state: str = "suspended"

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "id": self.id,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "age": self.age,
            "state": self.state,
        }


def decelerate(offshore_distance: np.ndarray | float, config: SimulationConfig) -> np.ndarray:
    """Map expansion into lower speed while retaining a minimum sea current."""

    distance = np.asarray(offshore_distance, dtype=np.float64)
    speed = config.river_speed / (1.0 + config.expansion_rate * np.maximum(distance, 0.0))
    return np.maximum(config.sea_min_speed, speed)


def _shift(array: np.ndarray, dy: int, dx: int, fill: float = 0.0) -> np.ndarray:
    result = np.full_like(array, fill)
    src_y = slice(max(0, -dy), min(array.shape[0], array.shape[0] - dy))
    src_x = slice(max(0, -dx), min(array.shape[1], array.shape[1] - dx))
    dst_y = slice(max(0, dy), min(array.shape[0], array.shape[0] + dy))
    dst_x = slice(max(0, dx), min(array.shape[1], array.shape[1] + dx))
    result[dst_y, dst_x] = array[src_y, src_x]
    return result


def flow_field(
    land: np.ndarray,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute a deterministic downstream field around current obstacles."""

    height, width = land.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    offshore = np.maximum(0.0, xx - config.coastline_x)
    speed = np.where(
        xx < config.coastline_x,
        config.river_speed,
        decelerate(offshore, config),
    )

    # Flow fans gently into open water. This is a direction field, not a target
    # channel map.
    spread = np.clip((yy - config.river_center_y) / (offshore + 14.0), -0.46, 0.46)
    vx = speed / np.sqrt(1.0 + spread**2)
    vy = vx * spread

    # Obstacle feedback: the gradient of a locally blurred land field pushes
    # water away from newly emerged cells. Original coastline is excluded so it
    # does not overwhelm the river mouth.
    original_coast = xx < config.coastline_x
    obstacle = (land & ~original_coast).astype(np.float64)
    if obstacle.any():
        influence = obstacle.copy()
        for radius, weight in ((1, 0.80), (2, 0.52), (3, 0.32), (5, 0.16)):
            influence += weight * (
                _shift(obstacle, radius, 0)
                + _shift(obstacle, -radius, 0)
                + _shift(obstacle, 0, radius)
                + _shift(obstacle, 0, -radius)
            )
        grad_y, grad_x = np.gradient(influence)
        vx -= 0.72 * speed * grad_x
        vy -= 1.08 * speed * grad_y
        # Keep the teaching model downstream-directed.
        vx = np.maximum(vx, config.sea_min_speed * 0.35)

    vx[land] = 0.0
    vy[land] = 0.0
    magnitude = np.hypot(vx, vy)
    return vx, vy, magnitude


def reroute(
    particle: Particle,
    proposed_x: float,
    proposed_y: float,
    land: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> tuple[float, float]:
    """Choose a nearby downstream water cell when a particle hits new land."""

    height, width = land.shape
    px = int(np.clip(round(proposed_x), 0, width - 1))
    py = int(np.clip(round(proposed_y), 0, height - 1))
    if not land[py, px]:
        return proposed_x, proposed_y

    candidates: list[tuple[float, float, float]] = []
    current_x = int(np.clip(round(particle.x), 0, width - 1))
    current_y = int(np.clip(round(particle.y), 0, height - 1))
    for dy_offset in (-3, -2, -1, 1, 2, 3):
        cy = int(np.clip(current_y + dy_offset, 0, height - 1))
        cx = int(np.clip(current_x + 1, 0, width - 1))
        if land[cy, cx]:
            continue
        score = float(vx[cy, cx] - 0.20 * abs(dy_offset) + 0.08 * abs(vy[cy, cx]))
        candidates.append((score, float(cx), float(cy)))
    if candidates:
        _, candidate_x, candidate_y = max(candidates)
        return candidate_x, candidate_y
    return particle.x, particle.y


def transport(
    particles: Iterable[Particle],
    land: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> list[Particle]:
    """Move suspended particles along the current flow field."""

    height, width = land.shape
    transported: list[Particle] = []
    for particle in particles:
        ix = int(np.clip(round(particle.x), 0, width - 1))
        iy = int(np.clip(round(particle.y), 0, height - 1))
        next_x = particle.x + float(vx[iy, ix])
        next_y = (
            particle.y
            + float(vy[iy, ix])
            + float(rng.normal(0.0, config.lateral_perturbation))
        )
        next_x, next_y = reroute(particle, next_x, next_y, land, vx, vy)
        particle.x = float(np.clip(next_x, 0.0, width - 1.001))
        particle.y = float(np.clip(next_y, 0.0, height - 1.001))
        particle.age += 1
        transported.append(particle)
    return transported


def accumulation_kernel(radius: int) -> np.ndarray:
    yy, xx = np.indices((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    distance_sq = (xx - radius) ** 2 + (yy - radius) ** 2
    sigma = max(0.8, radius * 0.82)
    kernel = np.exp(-distance_sq / (2.0 * sigma**2))
    return kernel / kernel.sum()


def accumulate(
    thickness: np.ndarray,
    settled_positions: Iterable[tuple[float, float]],
    config: SimulationConfig,
) -> np.ndarray:
    """Add settled mass to a continuous, non-decreasing thickness field."""

    updated = thickness.copy()
    radius = config.dispersion_radius
    kernel = accumulation_kernel(radius)
    height, width = thickness.shape
    for x, y in settled_positions:
        cx, cy = int(round(x)), int(round(y))
        for ky in range(-radius, radius + 1):
            py = cy + ky
            if py < 0 or py >= height:
                continue
            for kx in range(-radius, radius + 1):
                px = cx + kx
                if px < 0 or px >= width:
                    continue
                updated[py, px] += config.deposit_mass * kernel[ky + radius, kx + radius]
    return updated


def threshold_change(
    original_land: np.ndarray,
    thickness: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the one allowed land transition rule."""

    new_land = (~original_land) & (thickness > depth)
    land = original_land | new_land
    return land, new_land


def connected_component_count(mask: np.ndarray) -> int:
    remaining = mask.copy()
    components = 0
    height, width = mask.shape
    while remaining.any():
        components += 1
        start_y, start_x = np.argwhere(remaining)[0]
        stack = [(int(start_y), int(start_x))]
        remaining[start_y, start_x] = False
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and remaining[ny, nx]:
                    remaining[ny, nx] = False
                    stack.append((ny, nx))
    return components


def channel_count(
    land: np.ndarray,
    magnitude: np.ndarray,
    new_land: np.ndarray,
    config: SimulationConfig,
) -> int:
    """Measure separated flowing segments just downstream of the delta front."""

    if not new_land.any():
        return 1
    front_x = int(np.argwhere(new_land)[:, 1].max())
    lower = int(config.river_center_y - 13)
    upper = int(config.river_center_y + 14)
    active = (~land[lower:upper, front_x]) & (
        magnitude[lower:upper, front_x] > config.sea_min_speed * 0.92
    )
    # Ignore one-cell flicker and count contiguous runs at least two cells wide.
    count = 0
    run = 0
    for value in np.r_[active, False]:
        if value:
            run += 1
        else:
            if run >= 2:
                count += 1
            run = 0
    return count
