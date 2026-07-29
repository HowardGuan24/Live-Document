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

    contrast_evidence = {}
    contrast_passed = True
    for item_id, minimum in rules.get(
        "minimum_new_land_rgb_contrast", {}
    ).items():
        actual = float(
            composed["keyframes"][item_id]["new_land_rgb_contrast"]
        )
        contrast_evidence[item_id] = {
            "actual": actual,
            "minimum": float(minimum),
        }
        contrast_passed &= actual >= float(minimum)
    checks.append(
        {
            "name": "emergent_wet_sand_is_visually_distinct",
            "scope": "delta_causal case",
            "passed": contrast_passed,
            "evidence": contrast_evidence,
        }
    )

    final_id = ordered_ids[-1]
    actual_paths = int(
        composed["keyframes"][final_id]["visible_flow_path_count"]
    )
    expected_paths = int(rules["final_visible_flow_paths"])
    actual_flow_difference = float(
        composed["keyframes"][final_id][
            "mean_flow_path_difference_0_255"
        ]
    )
    minimum_flow_difference = float(
        rules["minimum_final_flow_path_difference"]
    )
    checks.append(
        {
            "name": "final_two_flow_paths_are_visible",
            "scope": "delta_causal case",
            "passed": (
                actual_paths == expected_paths
                and actual_flow_difference >= minimum_flow_difference
            ),
            "evidence": {
                "actual_path_count": actual_paths,
                "expected_path_count": expected_paths,
                "mean_pixel_difference": actual_flow_difference,
                "minimum_mean_pixel_difference": minimum_flow_difference,
            },
        }
    )
    return checks
