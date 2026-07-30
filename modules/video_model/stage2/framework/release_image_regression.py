"""Deterministic cross-frame material projection for release regression."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .contracts import artifact_record, load_json, sha256_path, write_json


KEYFRAME_IDS = (
    "00_start",
    "01_mechanism",
    "02_result",
    "03_end",
)


def _float_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _blur(image: Image.Image, radius: float) -> np.ndarray:
    return _float_rgb(
        image.convert("RGB").filter(
            ImageFilter.GaussianBlur(radius=radius)
        )
    )


def _layer_path(
    semantic_path: Path,
    layer_id: str,
) -> tuple[dict[str, Any], Path]:
    manifest = load_json(semantic_path)
    matches = [
        layer
        for layer in manifest["layers"]
        if layer["layer_id"] == layer_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"layer must resolve exactly once: {layer_id}"
        )
    layer = matches[0]
    return (
        layer,
        semantic_path.parents[2] / layer["data"]["path"],
    )


def _object_mask(
    payload: dict[str, Any],
    size: tuple[int, int],
    *,
    include_class_ids: list[str] | None = None,
) -> np.ndarray:
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for item in payload["items"]:
        if (
            include_class_ids is not None
            and item.get("class_id") not in include_class_ids
        ):
            continue
        geometry = item["geometry"]
        kind = geometry["kind"]
        if kind == "polygon":
            draw.polygon(
                [tuple(point) for point in geometry["points"]],
                fill=255,
            )
        elif kind == "polyline":
            draw.line(
                [tuple(point) for point in geometry["points"]],
                fill=255,
                width=7,
                joint="curve",
            )
        elif kind == "ellipse":
            draw.ellipse(geometry["bbox_xyxy"], fill=255)
        else:
            raise ValueError(
                f"unsupported object geometry for protection: {kind}"
            )
    return np.asarray(image, dtype=np.uint8) > 0


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    size = pixels * 2 + 1
    image = Image.fromarray(np.uint8(mask) * 255, mode="L")
    return (
        np.asarray(
            image.filter(ImageFilter.MaxFilter(size)),
            dtype=np.uint8,
        )
        > 0
    )


def _layer_mask(
    semantic_path: Path,
    config: dict[str, Any],
    size: tuple[int, int],
) -> np.ndarray:
    layer, path = _layer_path(semantic_path, config["layer_id"])
    kind = config["kind"]
    if kind == "object_identity":
        mask = _object_mask(
            load_json(path),
            size,
            include_class_ids=config.get("include_class_ids"),
        )
    else:
        data = np.load(path, allow_pickle=False)
        if kind == "binary":
            mask = data > 0
        elif kind == "scalar_field":
            mask = data >= float(config["threshold"])
        else:
            raise ValueError(f"unsupported protected layer kind: {kind}")
    if mask.shape != (size[1], size[0]):
        mask = (
            np.asarray(
                Image.fromarray(np.uint8(mask) * 255).resize(
                    size, Image.Resampling.NEAREST
                )
            )
            > 0
        )
    return _dilate(mask, int(config.get("dilation_px", 0)))


def _allowed_mask(
    semantic_path: Path,
    layer_id: str,
    size: tuple[int, int],
) -> np.ndarray:
    layer, path = _layer_path(semantic_path, layer_id)
    if layer["layer_type"] != "region":
        raise ValueError("allowed material layer must be region")
    data = np.load(path, allow_pickle=False) > 0
    if data.shape != (size[1], size[0]):
        data = (
            np.asarray(
                Image.fromarray(np.uint8(data) * 255).resize(
                    size, Image.Resampling.NEAREST
                )
            )
            > 0
        )
    return data


def _residual_stack(
    donors: list[Path],
    size: tuple[int, int],
    *,
    blur_radius: float,
    residual_clip: float,
) -> np.ndarray:
    records = []
    for path in donors:
        donor = Image.open(path).convert("RGB").resize(
            size, Image.Resampling.LANCZOS
        )
        residual = _float_rgb(donor) - _blur(
            donor, blur_radius
        )
        records.append(
            np.clip(residual, -residual_clip, residual_clip)
        )
    return np.stack(records, axis=0)


def _composite(
    base: Image.Image,
    residual: np.ndarray,
    mutable: np.ndarray,
    gain: float,
) -> tuple[Image.Image, dict[str, float]]:
    base_array = _float_rgb(base)
    output = base_array.copy()
    output[mutable] = np.clip(
        output[mutable] + residual[mutable] * gain,
        0,
        255,
    )
    image = Image.fromarray(np.uint8(np.rint(output)), mode="RGB")
    delta = np.abs(output - base_array)
    low_base = _blur(base, 12.0)
    low_output = _blur(image, 12.0)
    return image, {
        "non_mutable_max_abs_difference_0_255": round(
            float(delta[~mutable].max(initial=0.0)), 6
        ),
        "mutable_mean_abs_detail_change_0_255": round(
            float(delta[mutable].mean()) if mutable.any() else 0.0,
            6,
        ),
        "low_frequency_mse_vs_program": round(
            float(np.square(low_output - low_base).mean()), 6
        ),
    }


def _comparison_sheet(
    rows: list[tuple[str, Path, Path]],
    output: Path,
) -> None:
    thumb = (320, 180)
    label = 28
    sheet = Image.new(
        "RGB",
        (thumb[0] * 2, len(rows) * (thumb[1] + label)),
        (13, 35, 41),
    )
    draw = ImageDraw.Draw(sheet)
    for row, (name, before, after) in enumerate(rows):
        y = row * (thumb[1] + label)
        for column, path in enumerate((before, after)):
            image = Image.open(path).convert("RGB").resize(thumb)
            sheet.paste(image, (column * thumb[0], y))
        draw.text(
            (8, y + thumb[1] + 7),
            f"{name} · program",
            fill=(235, 244, 240),
        )
        draw.text(
            (thumb[0] + 8, y + thumb[1] + 7),
            f"{name} · material enhanced",
            fill=(235, 244, 240),
        )
    sheet.save(output, quality=92, subsampling=0)


def build_release_image_regression(
    *,
    stage2_root: Path,
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    program_root = stage2_root / config["program_root"]
    donors = [stage2_root / path for path in config["donors"]]
    output_root.mkdir(parents=True, exist_ok=True)
    final_root = output_root / "final"
    final_root.mkdir(exist_ok=True)
    first_base = Image.open(
        program_root / "keyframes/00_start/clean.png"
    ).convert("RGB")
    size = first_base.size
    settings = config["settings"]
    stack = _residual_stack(
        donors,
        size,
        blur_radius=float(settings["blur_radius"]),
        residual_clip=float(settings["residual_clip"]),
    )
    median = np.median(stack, axis=0)
    keyframes = []
    sheet_rows = []
    applied_residuals = []
    for keyframe_id in KEYFRAME_IDS:
        keyframe_root = program_root / "keyframes" / keyframe_id
        base_path = keyframe_root / "clean.png"
        semantic_path = keyframe_root / "semantic_layers.json"
        base = Image.open(base_path).convert("RGB")
        allowed = _allowed_mask(
            semantic_path,
            config["allowed_region_layer_id"],
            size,
        )
        protected = np.zeros(
            (size[1], size[0]), dtype=bool
        )
        for protected_config in config["protected_layers"]:
            protected |= _layer_mask(
                semantic_path, protected_config, size
            )
        mutable = allowed & ~protected
        image, metrics = _composite(
            base,
            median,
            mutable,
            float(settings["residual_gain"]),
        )
        output_path = final_root / f"{keyframe_id}.png"
        image.save(output_path, optimize=False)
        applied = (
            np.asarray(image, dtype=np.int16)
            - np.asarray(base, dtype=np.int16)
        )
        applied_residuals.append((applied, mutable))
        keyframes.append(
            {
                "keyframe_id": keyframe_id,
                "program": artifact_record(base_path, stage2_root),
                "final": artifact_record(
                    output_path, output_root
                ),
                "semantic_layers": artifact_record(
                    semantic_path, stage2_root
                ),
                "allowed_pixel_fraction": round(
                    float(allowed.mean()), 8
                ),
                "mutable_pixel_fraction": round(
                    float(mutable.mean()), 8
                ),
                "metrics": metrics,
            }
        )
        sheet_rows.append((keyframe_id, base_path, output_path))

    leave_root = output_root / "leave-one-out"
    leave_root.mkdir(exist_ok=True)
    last_root = program_root / "keyframes/03_end"
    last_base = Image.open(last_root / "clean.png").convert("RGB")
    semantic_path = last_root / "semantic_layers.json"
    allowed = _allowed_mask(
        semantic_path,
        config["allowed_region_layer_id"],
        size,
    )
    protected = np.zeros((size[1], size[0]), dtype=bool)
    for protected_config in config["protected_layers"]:
        protected |= _layer_mask(
            semantic_path, protected_config, size
        )
    mutable = allowed & ~protected
    leave_arrays = []
    leave_records = []
    for index, donor in enumerate(donors):
        residual = np.median(np.delete(stack, index, axis=0), axis=0)
        image, metrics = _composite(
            last_base,
            residual,
            mutable,
            float(settings["residual_gain"]),
        )
        path = leave_root / f"exclude_{index + 1}.png"
        image.save(path, optimize=False)
        leave_arrays.append(_float_rgb(image))
        leave_records.append(
            {
                "excluded_donor": artifact_record(
                    donor, stage2_root
                ),
                "composite": artifact_record(path, output_root),
                "metrics": metrics,
            }
        )
    pairwise = [
        float(np.abs(left - right).mean())
        for left, right in itertools.combinations(leave_arrays, 2)
    ]
    residual_consistency = []
    for (left, left_mutable), (right, right_mutable) in zip(
        applied_residuals, applied_residuals[1:]
    ):
        shared = left_mutable & right_mutable
        residual_consistency.append(
            float(
                np.abs(
                    left[shared].astype(np.float32)
                    - right[shared].astype(np.float32)
                ).max(initial=0.0)
            )
        )
    comparison_path = output_root / "sequence-comparison.jpg"
    _comparison_sheet(sheet_rows, comparison_path)
    maximum_non_mutable = max(
        item["metrics"][
            "non_mutable_max_abs_difference_0_255"
        ]
        for item in keyframes
    )
    maximum_low_frequency_mse = max(
        item["metrics"]["low_frequency_mse_vs_program"]
        for item in keyframes
    )
    maximum_leave_one_out = max(pairwise)
    maximum_residual_drift = max(
        residual_consistency, default=0.0
    )
    mean_visible_detail_change = float(
        np.mean(
            [
                item["metrics"][
                    "mutable_mean_abs_detail_change_0_255"
                ]
                for item in keyframes
            ]
        )
    )
    minimum_visible_detail_change = float(
        settings.get("minimum_visible_detail_change_0_255", 0.2)
    )
    hard_checks = [
        {
            "name": "all_non_allowed_and_protected_pixels_are_exact",
            "passed": maximum_non_mutable == 0,
            "evidence": maximum_non_mutable,
        },
        {
            "name": "low_frequency_program_structure_is_preserved",
            "passed": maximum_low_frequency_mse < 1.0,
            "evidence": maximum_low_frequency_mse,
        },
        {
            "name": "leave_one_out_material_projection_is_stable",
            "passed": maximum_leave_one_out < 2.0,
            "evidence": round(maximum_leave_one_out, 6),
        },
        {
            "name": "same_material_residual_is_stable_across_keyframes",
            "passed": maximum_residual_drift <= 1.0,
            "evidence": round(maximum_residual_drift, 6),
        },
        {
            "name": "material_enhancement_is_not_an_unchanged_no_op",
            "passed": (
                mean_visible_detail_change
                >= minimum_visible_detail_change
            ),
            "evidence": {
                "mean_mutable_detail_change_0_255": round(
                    mean_visible_detail_change, 6
                ),
                "minimum": minimum_visible_detail_change,
            },
        },
    ]
    manifest = {
        "schema_version": "1.0",
        "status": (
            "passed"
            if all(item["passed"] for item in hard_checks)
            else "failed"
        ),
        "case_id": config["case_id"],
        "discipline": config["discipline"],
        "visual_review": config["visual_review"],
        "classification": (
            "deterministic program sequence plus robust reused "
            "model-material residual"
        ),
        "model_runs": {"image": 0, "video": 0},
        "reused_model_donors": len(donors),
        "settings": settings,
        "allowed_region_layer_id": config[
            "allowed_region_layer_id"
        ],
        "protected_layers": config["protected_layers"],
        "donors": [
            artifact_record(path, stage2_root) for path in donors
        ],
        "donor_evidence": artifact_record(
            stage2_root / config["donor_evidence"], stage2_root
        ),
        "keyframes": keyframes,
        "leave_one_out": leave_records,
        "leave_one_out_pairwise_mae_0_255": {
            "maximum": round(maximum_leave_one_out, 6),
            "values": [round(value, 6) for value in pairwise],
        },
        "cross_frame_applied_residual_max_difference_0_255": [
            round(value, 6) for value in residual_consistency
        ],
        "hard_checks": hard_checks,
        "comparison": artifact_record(
            comparison_path, output_root
        ),
        "source_hashes": {
            "config_sha256": sha256_path(
                stage2_root / "phase6_image_routes.json"
            ),
            "runner_sha256": sha256_path(Path(__file__)),
        },
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest
