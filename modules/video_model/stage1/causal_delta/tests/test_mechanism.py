from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from modules.video_model.stage1.causal_delta.config import (
    MECHANISM_ROOT,
    SimulationConfig,
    original_land,
    water_depth,
)
from modules.video_model.stage1.causal_delta.primitives import (
    accumulate,
    decelerate,
    threshold_change,
)
from modules.video_model.stage1.causal_delta.simulate import run_simulation
from modules.video_model.stage1.causal_delta.validate import validate


def test_deceleration_is_bounded_and_monotonic() -> None:
    config = SimulationConfig()
    speeds = decelerate(np.arange(20), config)
    assert np.all(speeds[1:] <= speeds[:-1])
    assert np.all(speeds >= config.sea_min_speed)


def test_accumulation_never_removes_mass() -> None:
    config = SimulationConfig()
    before = np.zeros((config.grid_height, config.grid_width))
    after = accumulate(before, [(45.0, 31.0)], config)
    assert np.all(after >= before)
    assert np.isclose(after.sum(), config.deposit_mass)


def test_threshold_is_the_only_land_transition() -> None:
    config = SimulationConfig()
    base = original_land(config)
    depth = water_depth(config)
    thickness = np.zeros_like(depth)
    thickness[30, 45] = depth[30, 45] + 0.01
    land, emerged = threshold_change(base, thickness, depth)
    assert land[30, 45]
    assert emerged[30, 45]
    assert np.array_equal(land, base | (thickness > depth))


def test_full_simulation_passes_gate(tmp_path: Path) -> None:
    mechanism = tmp_path / "mechanism"
    summary = run_simulation(output_root=mechanism)
    result = validate(mechanism / "states.jsonl", mechanism)
    assert summary["first_coast_arrival_frame"] == 28
    assert summary["first_settling_frame"] >= 28
    assert summary["first_emergence_frame"] == 101
    assert result["passed"], json.dumps(result, ensure_ascii=False, indent=2)
