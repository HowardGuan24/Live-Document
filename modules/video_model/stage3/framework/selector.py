"""Deterministic S3.2 candidate gates and tie-break selection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    write_json,
)


def _image_features(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 45, 110, L2gradient=True) > 0
    return {
        "mean_saturation_0_255": round(
            float(hsv[:, :, 1].mean()), 6
        ),
        "mean_luminance_0_255": round(float(gray.mean()), 6),
        "luminance_std_0_255": round(float(gray.std()), 6),
        "edge_fraction": round(float(edges.mean()), 8),
        "dark_fraction": round(float((gray < 28).mean()), 8),
        "highlight_fraction": round(
            float((gray > 242).mean()), 8
        ),
    }


def _control_coverage(
    candidate: Image.Image,
    control: Image.Image,
    object_boxes: dict[str, list[int]],
) -> dict[str, Any]:
    rgb = np.asarray(candidate.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 110, L2gradient=True)
    dilated = cv2.dilate(
        edges, np.ones((9, 9), np.uint8), iterations=1
    ) > 0
    binary = np.asarray(control.convert("L")) > 0

    def coverage(mask: np.ndarray) -> float:
        return float(dilated[mask].mean()) if mask.any() else 0.0

    per_object = {}
    for class_id, box in object_boxes.items():
        x0, y0, x1, y1 = box
        local = np.zeros_like(binary)
        local[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1] = True
        local &= binary
        per_object[class_id] = round(coverage(local), 8)
    apparatus = np.zeros_like(binary)
    for box in object_boxes.values():
        x0, y0, x1, y1 = box
        pad = 22
        apparatus[
            max(0, y0 - pad) : min(binary.shape[0], y1 + pad + 1),
            max(0, x0 - pad) : min(binary.shape[1], x1 + pad + 1),
        ] = True
    outside_edge_fraction = (
        float((edges > 0)[~apparatus].mean())
        if (~apparatus).any()
        else 0.0
    )
    near_control = cv2.dilate(
        np.uint8(binary) * 255,
        np.ones((25, 25), np.uint8),
        iterations=1,
    ) > 0
    local_edges = (edges > 0) & apparatus
    edge_precision = (
        float(near_control[local_edges].mean())
        if local_edges.any()
        else 0.0
    )
    ordered = sorted(
        object_boxes.items(), key=lambda item: item[1][1]
    )
    disconnected = True
    lower_center_edge_fraction = 0.0
    if len(ordered) >= 2:
        upper_box = ordered[0][1]
        lower_box = ordered[-1][1]
        connected_input = cv2.dilate(
            edges, np.ones((3, 3), np.uint8), iterations=1
        )
        _, labels = cv2.connectedComponents(
            np.uint8(connected_input > 0), connectivity=8
        )
        ux0, uy0, ux1, uy1 = upper_box
        lx0, ly0, lx1, ly1 = lower_box
        upper_components = set(
            np.unique(labels[max(0, uy0) : uy1 + 1, ux0 : ux1 + 1])
        ) - {0}
        lower_components = set(
            np.unique(labels[max(0, ly0) : ly1 + 1, lx0 : lx1 + 1])
        ) - {0}
        disconnected = not bool(
            upper_components & lower_components
        )
        overlap_left = max(upper_box[0], lower_box[0])
        overlap_right = min(upper_box[2], lower_box[2])
        center_x = int(round((overlap_left + overlap_right) / 2))
        half_width = max(
            12, int(round((overlap_right - overlap_left) * 0.2))
        )
        interior_top = min(lower_box[3], lower_box[1] + 12)
        interior_bottom = int(
            round(
                lower_box[1]
                + (lower_box[3] - lower_box[1]) * 0.65
            )
        )
        roi = (edges > 0)[
            interior_top:interior_bottom,
            max(0, center_x - half_width) : min(
                binary.shape[1], center_x + half_width + 1
            ),
        ]
        lower_center_edge_fraction = (
            float(roi.mean()) if roi.size else 0.0
        )
    return {
        "total": round(coverage(binary), 8),
        "per_object": per_object,
        "outside_apparatus_edge_fraction": round(
            outside_edge_fraction, 8
        ),
        "candidate_edge_precision_within_12px": round(
            edge_precision, 8
        ),
        "separate_objects_remain_disconnected": disconnected,
        "lower_object_center_edge_fraction": round(
            lower_center_edge_fraction, 8
        ),
    }


def _internal_landmarks(
    candidate: Image.Image,
    requirements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate primitive-declared internal identity landmarks.

    The implementation is intentionally data driven.  It knows how to count
    a declared landmark shape, but it does not contain Case IDs or assume
    which discipline supplied the object.
    """

    rgb = np.asarray(candidate.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 110, L2gradient=True) > 0
    records: list[dict[str, Any]] = []
    for requirement in requirements:
        kind = requirement["kind"]
        if kind != "repeated_horizontal_segments":
            records.append(
                {
                    "landmark_id": requirement["landmark_id"],
                    "kind": kind,
                    "passed": False,
                    "reason": "unsupported landmark evaluator",
                }
            )
            continue
        x0, y0, x1, y1 = requirement["roi_xyxy"]
        roi = edges[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1]
        minimum_pixels = int(
            requirement["minimum_edge_pixels_per_row"]
        )
        active_rows = np.flatnonzero(
            roi.sum(axis=1) >= minimum_pixels
        )
        maximum_gap = int(requirement["maximum_row_gap_px"])
        groups: list[list[int]] = []
        for row in active_rows:
            row = int(row)
            if not groups or row > groups[-1][-1] + maximum_gap:
                groups.append([row])
            else:
                groups[-1].append(row)
        group_centers = [
            int(round(y0 + sum(group) / len(group)))
            for group in groups
        ]
        minimum_groups = int(
            requirement["minimum_distinct_groups"]
        )
        expected_centers = requirement.get(
            "expected_group_center_y", []
        )
        tolerance = int(
            requirement.get(
                "maximum_expected_position_error_px", 0
            )
        )
        matched_expected_centers = [
            expected
            for expected in expected_centers
            if any(
                abs(actual - expected) <= tolerance
                for actual in group_centers
            )
        ]
        detected_value = (
            len(matched_expected_centers)
            if expected_centers
            else len(groups)
        )
        records.append(
            {
                "object_id": requirement["object_id"],
                "class_id": requirement["class_id"],
                "landmark_id": requirement["landmark_id"],
                "kind": kind,
                "roi_xyxy": requirement["roi_xyxy"],
                "detected_distinct_groups": len(groups),
                "detected_group_center_y": group_centers,
                "expected_group_center_y": expected_centers,
                "matched_expected_group_center_y": (
                    matched_expected_centers
                ),
                "matched_expected_group_count": len(
                    matched_expected_centers
                ),
                "maximum_expected_position_error_px": tolerance,
                "minimum_distinct_groups": minimum_groups,
                "passed": detected_value >= minimum_groups,
            }
        )
    return records


def _similarity(
    features: dict[str, float], target: dict[str, float]
) -> float:
    scales = {
        "mean_saturation_0_255": 70.0,
        "mean_luminance_0_255": 70.0,
        "luminance_std_0_255": 45.0,
        "edge_fraction": 0.12,
        "dark_fraction": 0.30,
        "highlight_fraction": 0.30,
    }
    values = [
        max(
            0.0,
            1.0
            - abs(features[name] - target[name]) / scales[name],
        )
        for name in scales
    ]
    return float(sum(values) / len(values))


def evaluate_and_select(
    matrix_path: Path,
    metadata_path: Path,
    geometry_gate_path: Path,
    output_dir: Path,
    repo_root: Path,
    selector_policy_path: Path | None = None,
) -> dict[str, Any]:
    matrix = load_json(matrix_path)
    selector_policy = (
        load_json(selector_policy_path)
        if selector_policy_path is not None
        else matrix["selector"]
    )
    metadata = load_json(metadata_path)
    geometry_gate = load_json(geometry_gate_path)
    control_path = repo_root / matrix["geometry_control"]["path"]
    control = Image.open(control_path).convert("RGB")
    target_path = (
        repo_root / matrix["appearance_target"]["positive_reference"]
    )
    target_features = _image_features(
        Image.open(target_path).convert("RGB")
    )
    object_boxes = {
        item["class_id"]: item["output_bbox_xyxy"]
        for item in geometry_gate["rendered_objects"]
        if item["class_id"] in {"glass_beaker", "glass_burette"}
    }
    thresholds = selector_policy["hard_gates"]
    weights = selector_policy["score_weights"]
    experiment_root = metadata_path.parents[1]
    records = []
    for candidate_record in metadata["candidates"]:
        candidate_path = experiment_root / candidate_record["path"]
        candidate = Image.open(candidate_path).convert("RGB")
        features = _image_features(candidate)
        coverage = _control_coverage(
            candidate, control, object_boxes
        )
        landmark_records = _internal_landmarks(
            candidate,
            geometry_gate.get("required_internal_landmarks", []),
        )
        per_object = coverage["per_object"]
        hard_checks = [
            {
                "name": "total_control_coverage",
                "passed": coverage["total"]
                >= thresholds["minimum_total_control_coverage"],
                "value": coverage["total"],
                "threshold": thresholds[
                    "minimum_total_control_coverage"
                ],
            },
            {
                "name": "beaker_control_coverage",
                "passed": per_object.get("glass_beaker", 0)
                >= thresholds["minimum_beaker_control_coverage"],
                "value": per_object.get("glass_beaker", 0),
                "threshold": thresholds[
                    "minimum_beaker_control_coverage"
                ],
            },
            {
                "name": "burette_control_coverage",
                "passed": per_object.get("glass_burette", 0)
                >= thresholds["minimum_burette_control_coverage"],
                "value": per_object.get("glass_burette", 0),
                "threshold": thresholds[
                    "minimum_burette_control_coverage"
                ],
            },
            {
                "name": "mean_saturation",
                "passed": features["mean_saturation_0_255"]
                <= thresholds["maximum_mean_saturation_0_255"],
                "value": features["mean_saturation_0_255"],
                "threshold": thresholds[
                    "maximum_mean_saturation_0_255"
                ],
            },
            {
                "name": "luminance_std",
                "passed": thresholds[
                    "minimum_luminance_std_0_255"
                ]
                <= features["luminance_std_0_255"]
                <= thresholds["maximum_luminance_std_0_255"],
                "value": features["luminance_std_0_255"],
                "threshold": [
                    thresholds["minimum_luminance_std_0_255"],
                    thresholds["maximum_luminance_std_0_255"],
                ],
            },
            {
                "name": "dark_fraction",
                "passed": features["dark_fraction"]
                <= thresholds["maximum_dark_fraction"],
                "value": features["dark_fraction"],
                "threshold": thresholds["maximum_dark_fraction"],
            },
            {
                "name": "highlight_fraction",
                "passed": features["highlight_fraction"]
                <= thresholds["maximum_highlight_fraction"],
                "value": features["highlight_fraction"],
                "threshold": thresholds["maximum_highlight_fraction"],
            },
        ]
        if "minimum_candidate_edge_precision_within_12px" in thresholds:
            hard_checks.extend(
                [
                    {
                        "name": "candidate_edge_precision_within_12px",
                        "passed": coverage[
                            "candidate_edge_precision_within_12px"
                        ]
                        >= thresholds[
                            "minimum_candidate_edge_precision_within_12px"
                        ],
                        "value": coverage[
                            "candidate_edge_precision_within_12px"
                        ],
                        "threshold": thresholds[
                            "minimum_candidate_edge_precision_within_12px"
                        ],
                    },
                    {
                        "name": "outside_apparatus_edge_fraction",
                        "passed": coverage[
                            "outside_apparatus_edge_fraction"
                        ]
                        <= thresholds[
                            "maximum_outside_apparatus_edge_fraction"
                        ],
                        "value": coverage[
                            "outside_apparatus_edge_fraction"
                        ],
                        "threshold": thresholds[
                            "maximum_outside_apparatus_edge_fraction"
                        ],
                    },
                    {
                        "name": "lower_object_center_edge_fraction",
                        "passed": coverage[
                            "lower_object_center_edge_fraction"
                        ]
                        <= thresholds[
                            "maximum_lower_object_center_edge_fraction"
                        ],
                        "value": coverage[
                            "lower_object_center_edge_fraction"
                        ],
                        "threshold": thresholds[
                            "maximum_lower_object_center_edge_fraction"
                        ],
                    },
                    {
                        "name": "separate_objects_remain_disconnected",
                        "passed": coverage[
                            "separate_objects_remain_disconnected"
                        ],
                        "value": coverage[
                            "separate_objects_remain_disconnected"
                        ],
                        "threshold": True,
                    },
                ]
            )
        if thresholds.get(
            "required_internal_landmarks_must_pass", False
        ):
            hard_checks.extend(
                {
                    "name": (
                        "internal_landmark:"
                        f"{item['object_id']}:{item['landmark_id']}"
                    ),
                    "passed": item["passed"],
                    "value": item.get(
                        "matched_expected_group_count",
                        item.get("detected_distinct_groups", 0),
                    ),
                    "threshold": item.get(
                        "minimum_distinct_groups", 1
                    ),
                }
                for item in landmark_records
            )
        hard_pass = all(item["passed"] for item in hard_checks)
        min_object = min(per_object.values()) if per_object else 0.0
        control_score = 0.65 * coverage["total"] + 0.35 * min_object
        appearance_score = _similarity(features, target_features)
        neutral_score = max(
            0.0,
            1.0
            - features["mean_saturation_0_255"]
            / thresholds["maximum_mean_saturation_0_255"],
        )
        exposure_score = max(
            0.0,
            1.0
            - features["dark_fraction"]
            - features["highlight_fraction"],
        )
        total = (
            weights["control_fidelity"] * control_score
            + weights["accepted_appearance_similarity"]
            * appearance_score
            + weights["neutral_material"] * neutral_score
            + weights["exposure_balance"] * exposure_score
        )
        records.append(
            {
                "candidate_id": (
                    f"{candidate_record['configuration_id']}-"
                    f"s{candidate_record['seed']}"
                ),
                "configuration_id": candidate_record[
                    "configuration_id"
                ],
                "seed": int(candidate_record["seed"]),
                "path": candidate_record["path"],
                "sha256": candidate_record["sha256"],
                "hard_gate_passed": hard_pass,
                "hard_checks": hard_checks,
                "control_coverage": coverage,
                "internal_landmarks": landmark_records,
                "appearance_features": features,
                "scores": {
                    "control_fidelity": round(control_score, 8),
                    "accepted_appearance_similarity": round(
                        appearance_score, 8
                    ),
                    "neutral_material": round(neutral_score, 8),
                    "exposure_balance": round(exposure_score, 8),
                    "total": round(total, 8),
                },
            }
        )
    priority = {
        "auto_control_065": 0,
        "auto_control_080": 1,
        "auto_control_050": 2,
    }
    eligible = [item for item in records if item["hard_gate_passed"]]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -item["scores"]["total"],
            -min(item["control_coverage"]["per_object"].values()),
            priority[item["configuration_id"]],
            item["seed"],
        ),
    )
    selected = ranked[0] if ranked else None
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_artifact = None
    if selected:
        source = experiment_root / selected["path"]
        selected_path = output_dir / "selected.png"
        shutil.copyfile(source, selected_path)
        if sha256_path(selected_path) != selected["sha256"]:
            raise ValueError("selected candidate copy changed bytes")
        selected_artifact = file_record(selected_path, repo_root)
    result = {
        "schema_version": "1.0",
        "selector_version": selector_policy.get(
            "selector_id",
            matrix["selector"].get("version", "external_selector"),
        ),
        "matrix_id": matrix["matrix_id"],
        "status": "selected" if selected else "no_candidate_passed",
        "input_signatures": {
            "matrix_sha256": sha256_path(matrix_path),
            "generation_metadata_sha256": sha256_path(metadata_path),
            "geometry_gate_sha256": sha256_path(geometry_gate_path),
            "control_sha256": sha256_path(control_path),
            "appearance_target_sha256": sha256_path(target_path),
            "selector_policy_sha256": (
                sha256_path(selector_policy_path)
                if selector_policy_path is not None
                else sha256_path(matrix_path)
            ),
        },
        "appearance_target_features": target_features,
        "records": records,
        "eligible_count": len(eligible),
        "selected_candidate_id": (
            selected["candidate_id"] if selected else None
        ),
        "selected_candidate": selected,
        "selected_artifact": selected_artifact,
        "tie_break": selector_policy["tie_break"],
        "limitations_zh": [
            "自动量表能检查控制覆盖、曝光、饱和度、外观统计和 primitive 声明的内部几何标志。",
            "它仍不能仅凭传统图像特征可靠评价高级材质审美或未在 primitive provider 中声明的语义部件；阶段报告仍需独立视觉审计。",
        ],
    }
    write_json(output_dir / "selection.json", result)
    return result
