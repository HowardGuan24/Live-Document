"""Project model-generated material detail without moving program geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .contracts import artifact_record, load_json, write_json


def _float_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _blur_array(image: Image.Image, radius: float) -> np.ndarray:
    return _float_rgb(image.filter(ImageFilter.GaussianBlur(radius)))


def project_material_residual(
    base: Image.Image,
    donor: Image.Image,
    protected_boundary: Image.Image,
    *,
    blur_radius: float,
    residual_gain: float,
    protect_dilation_px: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Add only donor high frequencies in areas not protected by program data."""

    base = base.convert("RGB")
    donor = donor.convert("RGB").resize(base.size, Image.Resampling.LANCZOS)
    protected = protected_boundary.convert("L").resize(
        base.size, Image.Resampling.NEAREST
    )
    kernel = protect_dilation_px * 2 + 1
    if kernel % 2 == 0:
        kernel += 1
    protected = protected.filter(ImageFilter.MaxFilter(kernel))
    protected_array = np.asarray(protected, dtype=np.uint8) > 0
    allowed = (~protected_array).astype(np.float32)

    base_array = _float_rgb(base)
    donor_array = _float_rgb(donor)
    donor_low = _blur_array(donor, blur_radius)
    residual = donor_array - donor_low
    projected = np.clip(
        base_array + residual * residual_gain * allowed[:, :, None],
        0,
        255,
    )
    result = Image.fromarray(np.uint8(np.round(projected)), mode="RGB")
    difference = np.abs(projected - base_array)
    low_base = _blur_array(base, blur_radius)
    low_result = _blur_array(result, blur_radius)
    low_donor = _blur_array(donor, blur_radius)
    metrics = {
        "protected_pixel_fraction": round(
            float(protected_array.mean()), 8
        ),
        "protected_max_abs_difference_0_255": round(
            float(
                difference[protected_array].max()
                if protected_array.any()
                else 0
            ),
            6,
        ),
        "allowed_mean_abs_detail_change_0_255": round(
            float(
                difference[~protected_array].mean()
                if (~protected_array).any()
                else 0
            ),
            6,
        ),
        "composite_low_frequency_mse_vs_program": round(
            float(np.square(low_result - low_base).mean()), 6
        ),
        "donor_low_frequency_mse_vs_program": round(
            float(np.square(low_donor - low_base).mean()), 6
        ),
    }
    return result, metrics


def _comparison_sheet(
    base_path: Path,
    donor_and_composites: list[tuple[str, Path, Path]],
    output_path: Path,
) -> None:
    thumb_width, thumb_height = 384, 216
    label_height = 32
    gutter = 10
    columns = 3
    rows = len(donor_and_composites)
    width = columns * thumb_width + (columns + 1) * gutter
    height = rows * (thumb_height + label_height) + (rows + 1) * gutter
    sheet = Image.new("RGB", (width, height), (13, 31, 37))
    draw = ImageDraw.Draw(sheet)
    base = Image.open(base_path).convert("RGB").resize(
        (thumb_width, thumb_height)
    )
    for row, (label, donor_path, composite_path) in enumerate(
        donor_and_composites
    ):
        images = (
            ("程序结构底图", base),
            (
                "模型 raw 材质供体",
                Image.open(donor_path).convert("RGB").resize(
                    (thumb_width, thumb_height)
                ),
            ),
            (
                "高频残差 composite",
                Image.open(composite_path).convert("RGB").resize(
                    (thumb_width, thumb_height)
                ),
            ),
        )
        for column, (title, image) in enumerate(images):
            left = gutter + column * (thumb_width + gutter)
            top = gutter + row * (thumb_height + label_height + gutter)
            sheet.paste(image, (left, top))
            draw.rectangle(
                (
                    left,
                    top + thumb_height,
                    left + thumb_width,
                    top + thumb_height + label_height,
                ),
                fill=(5, 23, 28),
            )
            draw.text(
                (left + 8, top + thumb_height + 9),
                f"{label} · {title}",
                fill=(232, 244, 239),
            )
    sheet.save(output_path, quality=92, subsampling=0)


def build_material_projection(
    experiment_root: Path,
    *,
    blur_radius: float = 10.0,
    residual_gain: float = 0.8,
    protect_dilation_px: int = 10,
) -> dict[str, Any]:
    """Build deterministic composites from all raw candidates."""

    prepared = load_json(experiment_root / "_work" / "prepare.json")
    generated = load_json(experiment_root / "_work" / "generate.json")
    base_path = experiment_root / prepared["source"]["clean_keyframe"]["path"]
    boundary_path = (
        experiment_root
        / prepared["controls"]["sparse_identity_boundary"]["path"]
    )
    base = Image.open(base_path).convert("RGB")
    boundary = Image.open(boundary_path).convert("RGB")
    records = []
    sheet_rows = []
    for candidate in generated["candidates"]:
        donor_path = experiment_root / candidate["path"]
        composite_path = (
            experiment_root
            / "composite"
            / "material_residual"
            / f"seed_{candidate['seed']}.png"
        )
        composite_path.parent.mkdir(parents=True, exist_ok=True)
        composite, metrics = project_material_residual(
            base,
            Image.open(donor_path),
            boundary,
            blur_radius=blur_radius,
            residual_gain=residual_gain,
            protect_dilation_px=protect_dilation_px,
        )
        composite.save(composite_path, optimize=False)
        record = {
            "seed": candidate["seed"],
            "base": artifact_record(base_path, experiment_root),
            "raw_material_donor": artifact_record(
                donor_path, experiment_root
            ),
            "protected_boundary": artifact_record(
                boundary_path, experiment_root
            ),
            "composite": artifact_record(
                composite_path, experiment_root
            ),
            "classification": (
                "deterministic composite: program base plus model-derived "
                "high-frequency material residual"
            ),
            "settings": {
                "blur_radius": blur_radius,
                "residual_gain": residual_gain,
                "protect_dilation_px": protect_dilation_px,
            },
            "metrics": metrics,
        }
        records.append(record)
        sheet_rows.append(
            (f"seed {candidate['seed']}", donor_path, composite_path)
        )
    sheet_path = experiment_root / "material-projection-comparison.jpg"
    _comparison_sheet(base_path, sheet_rows, sheet_path)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": generated["experiment_id"],
        "case_id": generated["case_id"],
        "method_zh": (
            "对 raw 材质供体做高斯低通，再把供体减去低通得到的中高频残差，"
            "只加到程序未保护区域；不采用供体的大尺度构图、对象或波形。"
        ),
        "settings": {
            "blur_radius": blur_radius,
            "residual_gain": residual_gain,
            "protect_dilation_px": protect_dilation_px,
        },
        "records": records,
        "comparison_sheet": artifact_record(sheet_path, experiment_root),
        "hard_checks": [
            {
                "name": "all_protected_pixels_are_bit_exact",
                "passed": all(
                    item["metrics"][
                        "protected_max_abs_difference_0_255"
                    ]
                    == 0
                    for item in records
                ),
                "evidence": [
                    item["metrics"][
                        "protected_max_abs_difference_0_255"
                    ]
                    for item in records
                ],
            },
            {
                "name": "composite_discards_donor_low_frequency_layout",
                "passed": all(
                    item["metrics"][
                        "composite_low_frequency_mse_vs_program"
                    ]
                    < item["metrics"][
                        "donor_low_frequency_mse_vs_program"
                    ]
                    for item in records
                ),
                "evidence": [
                    {
                        "composite": item["metrics"][
                            "composite_low_frequency_mse_vs_program"
                        ],
                        "raw_donor": item["metrics"][
                            "donor_low_frequency_mse_vs_program"
                        ],
                    }
                    for item in records
                ],
            },
        ],
    }
    manifest_path = experiment_root / "_work" / "projection.json"
    write_json(manifest_path, manifest)
    return manifest


def _ensemble_image(
    base: Image.Image,
    residuals: np.ndarray,
    immutable_array: np.ndarray,
    *,
    residual_gain: float,
) -> tuple[Image.Image, dict[str, float]]:
    median_residual = np.median(residuals, axis=0)
    allowed = (~immutable_array).astype(np.float32)
    base_array = _float_rgb(base)
    projected = np.clip(
        base_array
        + median_residual * residual_gain * allowed[:, :, None],
        0,
        255,
    )
    result = Image.fromarray(np.uint8(np.round(projected)), mode="RGB")
    difference = np.abs(projected - base_array)
    return result, {
        "protected_max_abs_difference_0_255": round(
            float(
                difference[immutable_array].max()
                if immutable_array.any()
                else 0
            ),
            6,
        ),
        "allowed_mean_abs_detail_change_0_255": round(
            float(
                difference[~immutable_array].mean()
                if (~immutable_array).any()
                else 0
            ),
            6,
        ),
    }


def build_ensemble_material_projection(
    experiment_root: Path,
    *,
    blur_radius: float = 10.0,
    residual_gain: float = 1.0,
    residual_clip: float = 10.0,
    protect_dilation_px: int = 10,
    variant_id: str = "default",
    allowed_region_layer_id: str | None = None,
) -> dict[str, Any]:
    """Use robust cross-seed median residuals to suppress donor objects."""

    prepared = load_json(experiment_root / "_work" / "prepare.json")
    generated = load_json(experiment_root / "_work" / "generate.json")
    if len(generated["candidates"]) < 3:
        raise ValueError("ensemble projection requires at least three donors")
    base_path = experiment_root / prepared["source"]["clean_keyframe"]["path"]
    boundary_path = (
        experiment_root
        / prepared["controls"]["sparse_identity_boundary"]["path"]
    )
    base = Image.open(base_path).convert("RGB")
    boundary = Image.open(boundary_path).convert("L").resize(
        base.size, Image.Resampling.NEAREST
    )
    kernel = protect_dilation_px * 2 + 1
    if kernel % 2 == 0:
        kernel += 1
    protected_array = np.asarray(
        boundary.filter(ImageFilter.MaxFilter(kernel)),
        dtype=np.uint8,
    ) > 0
    allowed_region = np.ones(
        (base.height, base.width), dtype=bool
    )
    allowed_region_record = None
    if allowed_region_layer_id:
        semantic_path = Path(
            prepared["source"]["semantic_layers"]["path"]
        )
        layer_manifest = load_json(semantic_path)
        layers = [
            layer
            for layer in layer_manifest["layers"]
            if layer["layer_id"] == allowed_region_layer_id
        ]
        if len(layers) != 1:
            raise ValueError(
                "allowed region layer must resolve exactly once: "
                f"{allowed_region_layer_id}"
            )
        layer = layers[0]
        if layer["data"]["encoding"] != "npy":
            raise ValueError("allowed region layer must use NPY encoding")
        program_root = semantic_path.parents[2]
        region_path = program_root / layer["data"]["path"]
        region = np.load(region_path, allow_pickle=False)
        region_image = Image.fromarray(
            np.uint8(region > 0) * 255, mode="L"
        ).resize(base.size, Image.Resampling.NEAREST)
        allowed_region = np.asarray(region_image, dtype=np.uint8) > 0
        allowed_region_record = {
            "layer_id": allowed_region_layer_id,
            "path": str(region_path.resolve()),
            "source_sha256": layer["data"]["sha256"],
        }
    immutable_array = protected_array | ~allowed_region
    donors = []
    residuals = []
    for candidate in generated["candidates"]:
        path = experiment_root / candidate["path"]
        donor = Image.open(path).convert("RGB").resize(
            base.size, Image.Resampling.LANCZOS
        )
        residual = _float_rgb(donor) - _blur_array(donor, blur_radius)
        residuals.append(np.clip(residual, -residual_clip, residual_clip))
        donors.append((candidate, path))
    residual_stack = np.stack(residuals, axis=0)
    if not variant_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("projection variant ID must be path-safe")
    composite_root = (
        experiment_root
        / "composite"
        / "ensemble_material"
        / variant_id
    )
    composite_root.mkdir(parents=True, exist_ok=True)
    full_path = composite_root / "all_seed_median.png"
    full_image, full_metrics = _ensemble_image(
        base,
        residual_stack,
        immutable_array,
        residual_gain=residual_gain,
    )
    full_image.save(full_path, optimize=False)
    leave_one_out = []
    sheet_rows = []
    for index, (candidate, donor_path) in enumerate(donors):
        subset = np.delete(residual_stack, index, axis=0)
        composite, metrics = _ensemble_image(
            base,
            subset,
            immutable_array,
            residual_gain=residual_gain,
        )
        path = (
            composite_root
            / f"exclude_seed_{candidate['seed']}.png"
        )
        composite.save(path, optimize=False)
        leave_one_out.append(
            {
                "excluded_seed": candidate["seed"],
                "included_seed_count": int(subset.shape[0]),
                "composite": artifact_record(path, experiment_root),
                "metrics": metrics,
            }
        )
        sheet_rows.append(
            (
                f"exclude seed {candidate['seed']}",
                donor_path,
                path,
            )
        )
    sheet_path = (
        experiment_root
        / f"ensemble-material-comparison-{variant_id}.jpg"
    )
    _comparison_sheet(base_path, sheet_rows, sheet_path)
    leave_arrays = [
        _float_rgb(Image.open(experiment_root / item["composite"]["path"]))
        for item in leave_one_out
    ]
    pairwise_mae = []
    for first in range(len(leave_arrays)):
        for second in range(first + 1, len(leave_arrays)):
            pairwise_mae.append(
                float(
                    np.abs(
                        leave_arrays[first] - leave_arrays[second]
                    ).mean()
                )
            )
    low_base = _blur_array(base, blur_radius)
    low_composite = _blur_array(full_image, blur_radius)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": generated["experiment_id"],
        "case_id": generated["case_id"],
        "method_zh": (
            "每张 raw 供体先减去高斯低通并限幅，再对四个复现编号的残差逐像素"
            "取中位数。只在程序未保护区域加入该稳健残差；另做四张留一法结果，"
            "检查结论是否依赖某一个供体。"
        ),
        "settings": {
            "variant_id": variant_id,
            "blur_radius": blur_radius,
            "residual_gain": residual_gain,
            "residual_clip": residual_clip,
            "protect_dilation_px": protect_dilation_px,
            "aggregation": "pixelwise_median",
            "allowed_region_layer_id": allowed_region_layer_id,
        },
        "allowed_region_source": allowed_region_record,
        "allowed_pixel_fraction": round(
            float((~immutable_array).mean()), 8
        ),
        "raw_donors": [
            {
                "seed": candidate["seed"],
                "artifact": artifact_record(path, experiment_root),
            }
            for candidate, path in donors
        ],
        "full_ensemble": {
            "composite": artifact_record(full_path, experiment_root),
            "metrics": {
                **full_metrics,
                "low_frequency_mse_vs_program": round(
                    float(
                        np.square(low_composite - low_base).mean()
                    ),
                    6,
                ),
            },
        },
        "leave_one_out": leave_one_out,
        "leave_one_out_pairwise_mae_0_255": {
            "mean": round(float(np.mean(pairwise_mae)), 6),
            "maximum": round(float(np.max(pairwise_mae)), 6),
            "values": [round(value, 6) for value in pairwise_mae],
        },
        "comparison_sheet": artifact_record(sheet_path, experiment_root),
    }
    manifest["hard_checks"] = [
        {
            "name": "all_non_allowed_pixels_are_bit_exact",
            "passed": full_metrics[
                "protected_max_abs_difference_0_255"
            ]
            == 0
            and all(
                item["metrics"][
                    "protected_max_abs_difference_0_255"
                ]
                == 0
                for item in leave_one_out
            ),
            "evidence": {
                "full": full_metrics[
                    "protected_max_abs_difference_0_255"
                ],
                "leave_one_out": [
                    item["metrics"][
                        "protected_max_abs_difference_0_255"
                    ]
                    for item in leave_one_out
                ],
            },
        },
        {
            "name": "low_frequency_program_structure_is_preserved",
            "passed": manifest["full_ensemble"]["metrics"][
                "low_frequency_mse_vs_program"
            ]
            < 1.0,
            "evidence": manifest["full_ensemble"]["metrics"][
                "low_frequency_mse_vs_program"
            ],
        },
        {
            "name": "leave_one_out_composites_are_stable",
            "passed": manifest["leave_one_out_pairwise_mae_0_255"][
                "maximum"
            ]
            < 2.0,
            "evidence": manifest[
                "leave_one_out_pairwise_mae_0_255"
            ],
        },
    ]
    manifest_path = (
        experiment_root
        / "_work"
        / f"ensemble_projection_{variant_id}.json"
    )
    write_json(manifest_path, manifest)
    return manifest
