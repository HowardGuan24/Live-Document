"""Evaluate delta-specific statistics without coupling them to the core."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def _component_count(path: str) -> int:
    array = np.asarray(Image.open(path).convert("L")) > 127
    count, _ = cv2.connectedComponents(np.uint8(array))
    return count - 1


def evaluate_case(
    spec: dict[str, Any],
    prepared: dict[str, Any],
    composed: dict[str, Any],
) -> list[dict[str, Any]]:
    del composed
    rules = spec["case_evaluation"]
    ordered_ids = [item["id"] for item in spec["keyframes"]]
    stats = {
        item_id: prepared["keyframes"][item_id]["stats"]
        for item_id in ordered_ids
    }
    checks: list[dict[str, Any]] = []

    actual_new_land = [
        stats[item_id]["new_land_cells"] for item_id in ordered_ids
    ]
    expected_new_land = rules["new_land_cells"]
    checks.append(
        {
            "name": "delta_new_land_state_counts",
            "scope": "delta_causal case",
            "passed": actual_new_land == expected_new_land,
            "evidence": {
                "actual": actual_new_land,
                "expected": expected_new_land,
            },
        }
    )

    underwater = [
        stats[item_id]["underwater_deposit_cells"]
        for item_id in ordered_ids
    ]
    checks.append(
        {
            "name": "underwater_deposit_monotonic",
            "scope": "delta_causal case",
            "passed": all(
                right >= left
                for left, right in zip(underwater, underwater[1:])
            ),
            "evidence": underwater,
        }
    )

    components = {}
    component_passed = True
    for item_id, expected in rules["new_land_components"].items():
        path = prepared["keyframes"][item_id]["semantic_layers"][
            "new_land_binary"
        ]["path"]
        actual = _component_count(path)
        components[item_id] = {"actual": actual, "expected": expected}
        component_passed &= actual == expected
    checks.append(
        {
            "name": "new_land_connectivity",
            "scope": "delta_causal case",
            "passed": component_passed,
            "evidence": components,
        }
    )

    channels = [
        stats[item_id]["raw_channel_count"] for item_id in ordered_ids
    ]
    expected_final = int(rules["final_channel_count"])
    checks.append(
        {
            "name": "final_channel_count",
            "scope": "delta_causal case",
            "passed": channels[-1] == expected_final,
            "evidence": {
                "sequence": channels,
                "expected_final": expected_final,
            },
        }
    )
    return checks
