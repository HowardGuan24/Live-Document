"""Adapt the delta mechanism trace to generic keyframe state records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ....causal_delta.validate import load_states
from ..schema import resolve_stage_path


def load_selected_states(
    spec: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    paths = spec["paths"]
    states_path = resolve_stage_path(paths["states"])
    timeline_path = resolve_stage_path(paths["timeline"])
    config_path = resolve_stage_path(paths["simulation_config"])
    states = load_states(states_path)
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selections = [spec["anchor"], *spec["keyframes"]]
    records: dict[str, dict[str, Any]] = {}
    for item in selections:
        display_frame = int(item["display_frame"])
        state_frame = int(item["state_frame"])
        timeline_entry = next(
            (
                entry
                for entry in timeline
                if entry["display_frame"] == display_frame
                and entry["state_frame"] == state_frame
            ),
            None,
        )
        if timeline_entry is None:
            raise ValueError(
                f"display/state pair not found: {display_frame}/{state_frame}"
            )
        state = states[state_frame]
        stats = state["stats"]
        records[item["id"]] = {
            "id": item["id"],
            "display_frame": display_frame,
            "state_frame": state_frame,
            "beat_id": state["beat_id"],
            "caption": state["caption"],
            "meaning": item["meaning"],
            "program_frame": str(
                (
                    resolve_stage_path(paths["program_frames"])
                    / f"{display_frame:04d}.png"
                ).resolve()
            ),
            "particles": state["particles"],
            "thickness": state["thick"],
            "land": state["land"],
            "new_land": state["new_land"],
            "flow_samples": state["flow_samples"],
            "stats": stats,
        }
    return records, {
        "states": str(states_path.resolve()),
        "timeline": str(timeline_path.resolve()),
        "simulation_config": str(config_path.resolve()),
        "config": config,
    }
