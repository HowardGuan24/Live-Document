"""Run the causal model and serialize every mechanism state as JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    MECHANISM_ROOT,
    SimulationConfig,
    beat_for_frame,
    original_land,
    water_depth,
)
from .primitives import (
    Particle,
    accumulate,
    channel_count,
    flow_field,
    threshold_change,
    transport,
)


def _round_matrix(array: np.ndarray, digits: int = 5) -> list[list[float]]:
    return np.round(array, digits).tolist()


def _flow_samples(
    vx: np.ndarray,
    vy: np.ndarray,
    magnitude: np.ndarray,
    land: np.ndarray,
    spacing: int,
) -> list[list[float]]:
    rows: list[list[float]] = []
    for y in range(spacing // 2, land.shape[0], spacing):
        for x in range(spacing // 2, land.shape[1], spacing):
            if land[y, x]:
                continue
            rows.append(
                [
                    x,
                    y,
                    round(float(vx[y, x]), 4),
                    round(float(vy[y, x]), 4),
                    round(float(magnitude[y, x]), 4),
                ]
            )
    return rows


def run_simulation(
    config: SimulationConfig | None = None,
    output_root: Path = MECHANISM_ROOT,
) -> dict[str, Any]:
    config = config or SimulationConfig()
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.random_seed)
    base_land = original_land(config)
    depth = water_depth(config)
    thickness = np.zeros_like(depth, dtype=np.float64)
    land = base_land.copy()
    new_land = np.zeros_like(base_land)
    particles: list[Particle] = []
    next_particle_id = 0
    settled_total = 0
    first_coast_arrival: int | None = None
    first_settling: int | None = None
    first_emergence: int | None = None
    states_path = output_root / "states.jsonl"

    with states_path.open("w", encoding="utf-8") as state_file:
        for frame in range(config.state_count):
            vx, vy, magnitude = flow_field(land, config)

            for _ in range(config.sediment_per_frame):
                particles.append(
                    Particle(
                        id=next_particle_id,
                        x=0.5,
                        y=float(
                            config.river_center_y
                            + rng.uniform(
                                -config.river_half_width + 0.35,
                                config.river_half_width - 0.35,
                            )
                        ),
                    )
                )
                next_particle_id += 1

            particles = transport(particles, land, vx, vy, config, rng)
            if first_coast_arrival is None and any(
                particle.x >= config.coastline_x for particle in particles
            ):
                first_coast_arrival = frame

            remaining: list[Particle] = []
            settled_positions: list[tuple[float, float]] = []
            for particle in particles:
                ix = int(np.clip(round(particle.x), 0, config.grid_width - 1))
                iy = int(np.clip(round(particle.y), 0, config.grid_height - 1))
                eligible_x = config.coastline_x + config.mouth_protection_columns
                if particle.x < eligible_x:
                    remaining.append(particle)
                    continue
                slowdown = 1.0 - min(1.0, float(magnitude[iy, ix]) / config.river_speed)
                probability = config.base_settling_rate * slowdown**2
                if rng.random() < probability:
                    particle.state = "settled"
                    settled_positions.append((particle.x, particle.y))
                else:
                    remaining.append(particle)
            particles = remaining
            if settled_positions:
                settled_total += len(settled_positions)
                if first_settling is None:
                    first_settling = frame
                thickness = accumulate(thickness, settled_positions, config)

            land, new_land = threshold_change(base_land, thickness, depth)
            if first_emergence is None and new_land.any():
                first_emergence = frame

            # Store the field that results from this frame's land transition so
            # each saved state is internally traceable.
            vx, vy, magnitude = flow_field(land, config)
            beat_id, caption = beat_for_frame(frame)
            stats = {
                "injected_particles": next_particle_id,
                "suspended_particles": len(particles),
                "settled_particles": settled_total,
                "settled_this_frame": len(settled_positions),
                "max_thickness": round(float(thickness.max()), 6),
                "underwater_deposit_cells": int(
                    ((thickness > 0.002) & ~land & np.isfinite(depth)).sum()
                ),
                "new_land_cells": int(new_land.sum()),
                "new_land_front_x": (
                    int(np.argwhere(new_land)[:, 1].max()) if new_land.any() else None
                ),
                "deposit_front_x": (
                    int(np.argwhere(thickness > 0.002)[:, 1].max())
                    if (thickness > 0.002).any()
                    else None
                ),
                "raw_channel_count": channel_count(
                    land, magnitude, new_land, config
                ),
            }
            state = {
                "frame": frame,
                "time_seconds": round(frame / config.fps, 4),
                "beat_id": beat_id,
                "caption": caption,
                "particles": [particle.to_dict() for particle in particles],
                "thick": _round_matrix(thickness),
                "land": land.astype(np.uint8).tolist(),
                "new_land": new_land.astype(np.uint8).tolist(),
                "flow_samples": _flow_samples(
                    vx, vy, magnitude, land, config.flow_sample_spacing
                ),
                "stats": stats,
            }
            state_file.write(json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n")

    config_path = output_root / "simulation_config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    final_stats = state["stats"]
    channel_history = []
    with states_path.open(encoding="utf-8") as handle:
        for line in handle:
            channel_history.append(json.loads(line)["stats"]["raw_channel_count"])
    stable_tail = 0
    for count in reversed(channel_history):
        if 2 <= count <= 3:
            stable_tail += 1
        else:
            break
    summary = {
        "status": "simulated",
        "state_count": config.state_count,
        "first_coast_arrival_frame": first_coast_arrival,
        "first_settling_frame": first_settling,
        "first_emergence_frame": first_emergence,
        "final_new_land_cells": final_stats["new_land_cells"],
        "final_new_land_front_x": final_stats["new_land_front_x"],
        "final_deposit_front_x": final_stats["deposit_front_x"],
        "final_channel_count": final_stats["raw_channel_count"],
        "stable_channel_tail_frames": stable_tail,
        "states_path": str(states_path.resolve()),
    }
    (output_root / "simulation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MECHANISM_ROOT)
    args = parser.parse_args()
    summary = run_simulation(output_root=args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
