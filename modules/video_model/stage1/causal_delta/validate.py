"""Independently validate serialized mechanism states before rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .config import MECHANISM_ROOT, SimulationConfig, original_land, water_depth
from .primitives import connected_component_count


def load_states(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        states = [json.loads(line) for line in handle if line.strip()]
    if not states:
        raise ValueError(f"no states found in {path}")
    return states


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _mean_sample_speed(
    state: dict[str, Any],
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> float:
    values = [
        sample[4]
        for sample in state["flow_samples"]
        if x_min <= sample[0] <= x_max and y_min <= sample[1] <= y_max
    ]
    return float(np.mean(values)) if values else 0.0


def _mask_sheet(states: list[dict[str, Any]], output: Path) -> None:
    indices = [0, 32, 50, 80, 100, 101, 107, 119]
    scale = 3
    tile_w, tile_h = 96 * scale, 64 * scale
    sheet = Image.new("RGB", (tile_w * 4, tile_h * 2), "#10242c")
    for tile_index, state_index in enumerate(indices):
        state = states[state_index]
        new_land = np.asarray(state["new_land"], dtype=bool)
        thick = np.asarray(state["thick"], dtype=float)
        rgb = np.zeros((64, 96, 3), dtype=np.uint8)
        rgb[:] = (45, 105, 127)
        rgb[thick > 0.002] = (158, 115, 70)
        rgb[new_land] = (132, 157, 88)
        image = Image.fromarray(rgb).resize((tile_w, tile_h), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 95, 17), fill=(8, 21, 27))
        draw.text((5, 3), f"state {state_index}", fill="white")
        sheet.paste(image, ((tile_index % 4) * tile_w, (tile_index // 4) * tile_h))
    sheet.save(output)


def _flow_sheet(states: list[dict[str, Any]], output: Path) -> None:
    indices = [28, 100, 101, 119]
    scale = 4
    tile_w, tile_h = 96 * scale, 64 * scale
    sheet = Image.new("RGB", (tile_w * 2, tile_h * 2), "#163b4b")
    for tile_index, state_index in enumerate(indices):
        state = states[state_index]
        image = Image.new("RGB", (tile_w, tile_h), "#2f778e")
        draw = ImageDraw.Draw(image)
        land = np.asarray(state["land"], dtype=bool)
        for y, x in np.argwhere(land):
            color = "#7c8c67" if x < 38 else "#91a95f"
            draw.rectangle(
                (x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1),
                fill=color,
            )
        for x, y, flow_x, flow_y, speed in state["flow_samples"]:
            start = (x * scale, y * scale)
            length = 2.2 + 5.0 * speed / 1.34
            end = (
                start[0] + flow_x / max(speed, 1e-8) * length,
                start[1] + flow_y / max(speed, 1e-8) * length,
            )
            draw.line((start, end), fill="#86d6ef", width=1)
        draw.rectangle((0, 0, 90, 18), fill="#081b22")
        draw.text((5, 3), f"state {state_index}", fill="white")
        sheet.paste(image, ((tile_index % 2) * tile_w, (tile_index // 2) * tile_h))
    sheet.save(output)


def _ascii_evidence(states: list[dict[str, Any]], output: Path) -> None:
    sections = []
    for state_index in (100, 101, 107, 119):
        state = states[state_index]
        new_land = np.asarray(state["new_land"], dtype=bool)
        thick = np.asarray(state["thick"], dtype=float)
        lines = [f"state {state_index}"]
        for y in range(18, 46):
            row = []
            for x in range(34, 68):
                if new_land[y, x]:
                    row.append("#")
                elif thick[y, x] > 0.002:
                    row.append("+")
                else:
                    row.append(".")
            lines.append("".join(row))
        sections.append("\n".join(lines))
    output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def validate(
    states_path: Path = MECHANISM_ROOT / "states.jsonl",
    output_root: Path = MECHANISM_ROOT,
) -> dict[str, Any]:
    states = load_states(states_path)
    config_data = json.loads((output_root / "simulation_config.json").read_text(encoding="utf-8"))
    config = SimulationConfig(**config_data)
    base_land = original_land(config)
    depth = water_depth(config)
    thicknesses = [np.asarray(state["thick"], dtype=float) for state in states]
    new_lands = [np.asarray(state["new_land"], dtype=bool) for state in states]
    lands = [np.asarray(state["land"], dtype=bool) for state in states]

    monotonic_thickness = all(
        np.all(current + 1e-5 >= previous)
        for previous, current in zip(thicknesses, thicknesses[1:])
    )
    monotonic_land = all(
        np.all(current | ~previous)
        for previous, current in zip(new_lands, new_lands[1:])
    )
    threshold_exact = all(
        np.array_equal(land, base_land | (thickness > depth))
        for land, thickness in zip(lands, thicknesses)
    )

    first_arrival = next(
        (
            state["frame"]
            for state in states
            if any(particle["x"] >= config.coastline_x for particle in state["particles"])
        ),
        None,
    )
    first_settling = next(
        (state["frame"] for state in states if state["stats"]["settled_particles"] > 0),
        None,
    )
    first_emergence = next(
        (state["frame"] for state in states if state["stats"]["new_land_cells"] > 0),
        None,
    )
    first_underwater = next(
        (
            state["frame"]
            for state in states
            if state["stats"]["underwater_deposit_cells"] > 0
        ),
        None,
    )
    underwater_lead = (
        first_emergence - first_underwater
        if first_underwater is not None and first_emergence is not None
        else 0
    )

    speed_state = states[40]
    upstream_speed = _mean_sample_speed(speed_state, 4, 32, 27, 36)
    mouth_speed = _mean_sample_speed(speed_state, 40, 48, 24, 40)
    speed_ratio = mouth_speed / upstream_speed if upstream_speed else 1.0

    channel_history = [state["stats"]["raw_channel_count"] for state in states]
    stable_tail = 0
    for count in reversed(channel_history):
        if 2 <= count <= 3:
            stable_tail += 1
        else:
            break
    final_new_land = new_lands[-1]
    component_count = connected_component_count(final_new_land)
    final_front = states[-1]["stats"]["new_land_front_x"]
    deposit_front = states[-1]["stats"]["deposit_front_x"]

    checks = [
        _check(
            "state_count",
            len(states) == config.state_count,
            {"actual": len(states), "expected": config.state_count},
        ),
        _check("thickness_monotonic", monotonic_thickness, "all cells non-decreasing"),
        _check("new_land_monotonic", monotonic_land, "all emerged cells persist"),
        _check(
            "land_threshold_exact",
            threshold_exact,
            "land == original_land OR (thickness > depth)",
        ),
        _check(
            "arrival_before_settling",
            first_arrival is not None
            and first_settling is not None
            and first_arrival <= first_settling,
            {"first_arrival": first_arrival, "first_settling": first_settling},
        ),
        _check(
            "visible_underwater_stage",
            underwater_lead >= 7,
            {
                "first_underwater_deposit": first_underwater,
                "first_emergence": first_emergence,
                "lead_frames": underwater_lead,
            },
        ),
        _check(
            "emergence_in_threshold_stage",
            first_emergence is not None and 101 <= first_emergence <= 107,
            {"first_emergence": first_emergence, "required_range": [101, 107]},
        ),
        _check(
            "mouth_deceleration",
            speed_ratio < 0.60,
            {
                "upstream_mean_speed": round(upstream_speed, 5),
                "mouth_mean_speed": round(mouth_speed, 5),
                "ratio": round(speed_ratio, 5),
            },
        ),
        _check(
            "final_channel_count",
            2 <= channel_history[-1] <= 3,
            channel_history[-1],
        ),
        _check(
            "stable_channels",
            stable_tail >= config.stable_channel_frames,
            {"tail_frames": stable_tail, "required": config.stable_channel_frames},
        ),
        _check(
            "new_land_connected",
            bool(final_new_land.any()) and component_count == 1,
            {"components": component_count, "cells": int(final_new_land.sum())},
        ),
        _check(
            "land_and_deposit_extent",
            final_front is not None
            and final_front >= 50
            and deposit_front is not None
            and deposit_front >= 56,
            {"land_front_x": final_front, "deposit_front_x": deposit_front},
        ),
        _check(
            "state_traceability",
            all(
                state["frame"] == index
                and state["beat_id"]
                and state["flow_samples"]
                and "stats" in state
                for index, state in enumerate(states)
            ),
            "frame, beat, flow samples and stats present in every state",
        ),
    ]
    result = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "model_usage": [],
        "gpu_usage": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _mask_sheet(states, output_root / "land-mask-contact-sheet.png")
    _flow_sheet(states, output_root / "flow-contact-sheet.png")
    _ascii_evidence(states, output_root / "land-mask-ascii.txt")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-root", type=Path, default=MECHANISM_ROOT)
    args = parser.parse_args()
    result = validate(args.mechanism_root / "states.jsonl", args.mechanism_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
