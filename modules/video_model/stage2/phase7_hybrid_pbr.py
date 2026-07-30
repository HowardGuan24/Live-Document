"""Build Phase 7 routes B and C with deterministic scientific state."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .framework.contracts import (
    artifact_record,
    load_json,
    sha256_path,
    write_json,
)


STAGE2_ROOT = Path(__file__).resolve().parent
PHASE2_ROOT = STAGE2_ROOT / "output/phase-2"
OUTPUT_ROOT = STAGE2_ROOT / "output/phase-7"
KEYFRAMES = (
    "00_start",
    "01_mechanism",
    "02_result",
    "03_end",
)


def _layer_path(
    case_id: str, keyframe_id: str, layer_id: str
) -> Path:
    root = PHASE2_ROOT / case_id / "keyframes" / keyframe_id
    manifest = load_json(root / "semantic_layers.json")
    layer = next(
        item
        for item in manifest["layers"]
        if item["layer_id"] == layer_id
    )
    return PHASE2_ROOT / case_id / layer["data"]["path"]


def _resize_array(
    array: np.ndarray,
    size: tuple[int, int],
    *,
    nearest: bool = False,
) -> np.ndarray:
    if array.dtype == np.float32 or array.dtype == np.float64:
        image = Image.fromarray(array.astype(np.float32), mode="F")
    else:
        image = Image.fromarray(array)
    method = (
        Image.Resampling.NEAREST
        if nearest
        else Image.Resampling.BILINEAR
    )
    return np.asarray(image.resize(size, method), dtype=np.float32)


def _soft_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(
        np.uint8(np.clip(mask, 0, 1) * 255), mode="L"
    )
    return (
        np.asarray(
            image.filter(ImageFilter.GaussianBlur(radius)),
            dtype=np.float32,
        )
        / 255.0
    )


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )


def _indicator_from_ph(ph: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(ph - 8.15) * 2.5))


def _chem_drop_mask(
    keyframe_id: str, size: tuple[int, int]
) -> np.ndarray:
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    if keyframe_id == "01_mechanism":
        scale_x = size[0] / 1024.0
        scale_y = size[1] / 576.0
        draw.ellipse(
            (
                505 * scale_x,
                230 * scale_y,
                519 * scale_x,
                250 * scale_y,
            ),
            fill=255,
        )
    return np.asarray(image, dtype=np.float32) / 255.0


def _chem_hybrid_variant(
    base: Image.Image,
    keyframe_id: str,
    variant: str,
) -> tuple[Image.Image, dict[str, float]]:
    base_array = np.asarray(base.convert("RGB"), dtype=np.float32)
    size = base.size
    program_liquid = (
        np.load(
            _layer_path(
                "CHEM-01",
                keyframe_id,
                "chem01_liquid_region",
            ),
            allow_pickle=False,
        )
        > 0
    )
    ph = np.load(
        _layer_path(
            "CHEM-01", keyframe_id, "chem01_ph_field"
        ),
        allow_pickle=False,
    ).astype(np.float32)
    ys, xs = np.nonzero(program_liquid)
    crop = ph[
        int(ys.min()) : int(ys.max()) + 1,
        int(xs.min()) : int(xs.max()) + 1,
    ]
    state = load_json(
        PHASE2_ROOT
        / "CHEM-01/keyframes"
        / keyframe_id
        / "state.json"
    )
    scale_x = size[0] / 1024.0
    scale_y = size[1] / 576.0
    x0 = round(350 * scale_x)
    x1 = round(675 * scale_x)
    y1 = round(493 * scale_y)
    y0 = round(
        (
            404
            + (
                float(state["liquid_level_y"])
                - 238.0
            )
            * 1.55
        )
        * scale_y
    )
    mapped_ph = _resize_array(crop, (x1 - x0, y1 - y0))
    ph_canvas = np.zeros((size[1], size[0]), dtype=np.float32)
    ph_canvas[y0:y1, x0:x1] = mapped_ph
    liquid_image = Image.new("L", size, 0)
    ImageDraw.Draw(liquid_image).polygon(
        [
            (round(350 * scale_x), y0),
            (round(675 * scale_x), y0),
            (round(652 * scale_x), y1),
            (round(370 * scale_x), y1),
        ],
        fill=255,
    )
    liquid = np.asarray(liquid_image, dtype=np.uint8) > 0
    ph = ph_canvas
    indicator = _indicator_from_ph(ph) * liquid
    feather = _soft_mask(liquid.astype(np.float32), 5.0)
    if variant == "state_tint":
        strength = 0.34
        optical_edge = 0.0
        add_drop = False
    elif variant == "optical_tint":
        strength = 0.48
        optical_edge = 0.30
        add_drop = False
    elif variant == "optical_tint_plus_drop":
        strength = 0.52
        optical_edge = 0.38
        add_drop = True
    else:
        raise ValueError(variant)
    alpha = np.clip(indicator * strength * feather, 0, 0.68)
    target = np.array([237.0, 73.0, 147.0], dtype=np.float32)
    target_luma = float(_luminance(target))
    base_luma = _luminance(base_array)
    target_field = np.clip(
        target[None, None, :]
        * (
            np.maximum(base_luma, 22.0)[:, :, None]
            / target_luma
        ),
        0,
        255,
    )
    output = (
        base_array * (1.0 - alpha[:, :, None])
        + target_field * alpha[:, :, None]
    )
    if optical_edge:
        gy, gx = np.gradient(indicator)
        edge = np.clip(
            np.sqrt(gx * gx + gy * gy) * 8.0, 0, 1
        )
        highlight = edge * feather * optical_edge
        output = output * (1.0 - highlight[:, :, None]) + (
            255.0 * highlight[:, :, None]
        )
    if add_drop:
        drop = _chem_drop_mask(keyframe_id, size)
        drop = _soft_mask(drop, 1.1) * float(indicator.max())
        output = output * (1.0 - drop[:, :, None] * 0.65) + (
            target[None, None, :] * drop[:, :, None] * 0.65
        )
        alpha = np.maximum(alpha, drop)
    output = np.clip(output, 0, 255)
    delta = np.abs(output - base_array)
    mutable = alpha > 0.005
    return Image.fromarray(np.uint8(np.rint(output)), mode="RGB"), {
        "outside_state_mask_max_difference_0_255": round(
            float(delta[~mutable].max(initial=0.0)), 6
        ),
        "inside_state_mask_mean_difference_0_255": round(
            float(delta[mutable].mean()) if mutable.any() else 0.0,
            6,
        ),
        "indicator_peak": round(float(indicator.max()), 6),
        "indicator_mean_in_liquid": round(
            float(indicator[liquid].mean()), 6
        ),
    }


def _bio_scientific_overlay(
    program: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    chromosome = (
        (program[..., 0] > 105)
        & (program[..., 2] > 110)
        & (program[..., 1] < 115)
    )
    poles = (
        (program[..., 2] > 105)
        & (program[..., 0] < 105)
        & (program[..., 1] < 135)
    )
    return chromosome, poles


def _bio_hybrid_variant(
    donor: Image.Image,
    keyframe_id: str,
    variant: str,
) -> tuple[Image.Image, dict[str, float]]:
    size = (640, 360)
    donor = donor.convert("RGB").resize(
        size, Image.Resampling.LANCZOS
    )
    donor_array = np.asarray(donor, dtype=np.float32)
    program = np.asarray(
        Image.open(
            PHASE2_ROOT
            / "BIO-01/keyframes"
            / keyframe_id
            / "clean.png"
        ).convert("RGB"),
        dtype=np.float32,
    )
    cell = (
        np.load(
            _layer_path(
                "BIO-01", keyframe_id, "bio01_cell_region"
            ),
            allow_pickle=False,
        )
        > 0
    )
    chromosome, poles = _bio_scientific_overlay(program)
    donor_blur = np.asarray(
        donor.filter(ImageFilter.GaussianBlur(5.0)),
        dtype=np.float32,
    )
    texture = donor_array - donor_blur
    texture /= max(float(texture.std()), 1.0)
    yy, xx = np.mgrid[0:360, 0:640]
    background = np.zeros_like(program)
    background[:] = np.array([22, 31, 41], dtype=np.float32)
    background += (
        np.sin(xx / 39.0)[:, :, None] * 2.0
        + np.cos(yy / 27.0)[:, :, None] * 2.0
    )
    if variant == "raw_underlay":
        material = donor_array
        membrane_strength = 0.30
    elif variant == "normalized_cytoplasm":
        material = (
            np.array([83, 145, 121], dtype=np.float32)
            + texture * np.array([8, 14, 11], dtype=np.float32)
        )
        membrane_strength = 0.55
    elif variant == "stable_material_plus_depth":
        center_x = 320.0
        center_y = 180.0
        radial = np.clip(
            1.0
            - np.sqrt(
                ((xx - center_x) / 290.0) ** 2
                + ((yy - center_y) / 150.0) ** 2
            ),
            0,
            1,
        )
        material = (
            np.array([66, 136, 111], dtype=np.float32)
            + texture * np.array([9, 17, 13], dtype=np.float32)
            + radial[:, :, None] * 24.0
        )
        membrane_strength = 0.85
    else:
        raise ValueError(variant)
    output = background.copy()
    output[cell] = material[cell]
    cell_image = Image.fromarray(np.uint8(cell) * 255, mode="L")
    outer = np.asarray(
        cell_image.filter(ImageFilter.MaxFilter(9)),
        dtype=np.float32,
    ) / 255.0
    inner = np.asarray(
        cell_image.filter(ImageFilter.MinFilter(9)),
        dtype=np.float32,
    ) / 255.0
    membrane = np.clip(outer - inner, 0, 1)
    membrane_color = np.array([151, 224, 181], dtype=np.float32)
    ma = membrane[:, :, None] * membrane_strength
    output = output * (1.0 - ma) + membrane_color * ma
    output[chromosome] = np.array(
        [205, 55, 215], dtype=np.float32
    )
    output[poles] = np.array([65, 128, 188], dtype=np.float32)
    output = np.clip(output, 0, 255)
    return Image.fromarray(np.uint8(np.rint(output)), mode="RGB"), {
        "program_cell_area_px": int(cell.sum()),
        "rendered_cell_area_px": int(cell.sum()),
        "chromosome_overlay_pixel_count": int(chromosome.sum()),
        "pole_overlay_pixel_count": int(poles.sum()),
        "outside_cell_background_hash_scope_px": int((~cell).sum()),
    }


def _mask_from_polygon(
    size: tuple[int, int], points: list[list[float]]
) -> np.ndarray:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).polygon(
        [tuple(point) for point in points], fill=255
    )
    return np.asarray(image, dtype=np.uint8) > 0


def _sample_piece_texture(
    donor: np.ndarray,
    points: list[list[float]],
    mask: np.ndarray,
) -> np.ndarray:
    p0, p1, p2 = (
        np.asarray(points[index], dtype=np.float32)
        for index in range(3)
    )
    transform = np.column_stack((p1 - p0, p2 - p0))
    inverse = np.linalg.inv(transform)
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    coordinates = np.stack(
        (xx.astype(np.float32) - p0[0], yy.astype(np.float32) - p0[1]),
        axis=-1,
    )
    uv = coordinates @ inverse.T
    tx = np.clip(
        np.rint(uv[..., 0] * (donor.shape[1] - 1)),
        0,
        donor.shape[1] - 1,
    ).astype(np.int32)
    ty = np.clip(
        np.rint(uv[..., 1] * (donor.shape[0] - 1)),
        0,
        donor.shape[0] - 1,
    ).astype(np.int32)
    return donor[ty, tx]


def _math_pbr_variant(
    donor: Image.Image,
    keyframe_id: str,
    variant: str,
) -> tuple[Image.Image, dict[str, Any]]:
    size = (640, 360)
    texture = np.asarray(
        donor.convert("RGB")
        .resize((512, 512), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    texture_luma = _luminance(texture)
    texture_luma = (
        texture_luma - texture_luma.mean()
    ) / max(float(texture_luma.std()), 1.0)
    neutral_texture = np.clip(
        178.0 + texture_luma[..., None] * 19.0, 80, 235
    )
    neutral_texture = np.repeat(neutral_texture, 3, axis=2)
    yy, xx = np.mgrid[0:360, 0:640]
    background = np.zeros((360, 640, 3), dtype=np.float32)
    background[:] = np.array([223, 215, 195], dtype=np.float32)
    background += (
        np.sin(xx / 44.0)[:, :, None] * 3.0
        + np.cos(yy / 58.0)[:, :, None] * 2.0
    )
    payload = load_json(
        _layer_path(
            "MATH-02", keyframe_id, "math02_piece_identity"
        )
    )
    colors = (
        np.array([187, 73, 55], dtype=np.float32),
        np.array([204, 128, 43], dtype=np.float32),
        np.array([50, 125, 118], dtype=np.float32),
        np.array([64, 87, 151], dtype=np.float32),
    )
    output = background.copy()
    remaining = (
        np.load(
            _layer_path(
                "MATH-02",
                keyframe_id,
                "math02_remaining_region",
            ),
            allow_pickle=False,
        )
        > 0
    )
    remaining_image = Image.fromarray(
        np.uint8(remaining) * 255, mode="L"
    )
    remaining_edge = (
        np.asarray(
            remaining_image.filter(ImageFilter.MaxFilter(7)),
            dtype=np.float32,
        )
        - np.asarray(
            remaining_image.filter(ImageFilter.MinFilter(7)),
            dtype=np.float32,
        )
    ) / 255.0
    output[remaining] = np.array(
        [202, 199, 168], dtype=np.float32
    )
    output = output * (
        1.0 - remaining_edge[:, :, None] * 0.14
    )
    masks = []
    for index, item in enumerate(payload["items"]):
        points = item["geometry"]["points"]
        mask = _mask_from_polygon(size, points)
        masks.append(mask)
        sampled = _sample_piece_texture(
            neutral_texture, points, mask
        )
        gray = _luminance(sampled) / 178.0
        colored = np.clip(
            colors[index][None, None, :] * gray[:, :, None],
            0,
            255,
        )
        if variant in {"beveled_wood", "studio_pbr"}:
            mask_image = Image.fromarray(
                np.uint8(mask) * 255, mode="L"
            )
            inner = np.asarray(
                mask_image.filter(ImageFilter.MinFilter(9)),
                dtype=np.float32,
            ) / 255.0
            bevel = mask.astype(np.float32) - inner
            colored += bevel[:, :, None] * 18.0
        if variant == "studio_pbr":
            shadow_image = Image.fromarray(
                np.uint8(mask) * 255, mode="L"
            ).transform(
                size,
                Image.Transform.AFFINE,
                (1, 0, -6, 0, 1, -8),
                resample=Image.Resampling.BILINEAR,
            )
            shadow = (
                np.asarray(
                    shadow_image.filter(
                        ImageFilter.GaussianBlur(5.0)
                    ),
                    dtype=np.float32,
                )
                / 255.0
                * 0.24
            )
            output *= 1.0 - shadow[:, :, None]
        output[mask] = np.clip(colored[mask], 0, 255)
    union = np.any(np.stack(masks, axis=0), axis=0)
    return Image.fromarray(
        np.uint8(np.rint(np.clip(output, 0, 255))), mode="RGB"
    ), {
        "object_count": len(masks),
        "program_piece_area_px": int(sum(mask.sum() for mask in masks)),
        "rendered_material_mask_area_px": int(
            sum(mask.sum() for mask in masks)
        ),
        "overlap_area_px": int(
            sum(mask.sum() for mask in masks) - union.sum()
        ),
        "identity_order": [
            item["object_id"] for item in payload["items"]
        ],
    }


def _draw_phys_sources(
    image: Image.Image, keyframe_id: str
) -> None:
    payload = load_json(
        _layer_path(
            "PHYS-01", keyframe_id, "phys01_source_identity"
        )
    )
    draw = ImageDraw.Draw(image)
    for item in payload["items"]:
        box = item["geometry"]["bbox_xyxy"]
        draw.ellipse(
            box,
            fill=(246, 151, 45),
            outline=(255, 248, 227),
            width=2,
        )


def _phys_pbr_variant(
    donor: Image.Image,
    keyframe_id: str,
    variant: str,
) -> tuple[Image.Image, dict[str, float]]:
    height = np.load(
        _layer_path(
            "PHYS-01", keyframe_id, "phys01_surface_height"
        ),
        allow_pickle=False,
    ).astype(np.float32)
    envelope = np.load(
        _layer_path(
            "PHYS-01",
            keyframe_id,
            "phys01_amplitude_envelope",
        ),
        allow_pickle=False,
    ).astype(np.float32)
    gy, gx = np.gradient(height)
    slope_scale = {
        "normal_diffuse": 10.0,
        "specular_water": 16.0,
        "refractive_water": 18.0,
    }[variant]
    normal = np.stack(
        (-gx * slope_scale, -gy * slope_scale, np.ones_like(height)),
        axis=-1,
    )
    normal /= np.maximum(
        np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6
    )
    light = np.array([-0.38, -0.48, 0.79], dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.clip(normal @ light, 0, 1)
    base = np.zeros((*height.shape, 3), dtype=np.float32)
    base[:] = np.array([38, 139, 177], dtype=np.float32)
    base *= (0.62 + diffuse[..., None] * 0.52)
    if variant in {"specular_water", "refractive_water"}:
        half_vector = (
            light + np.array([0, 0, 1], dtype=np.float32)
        )
        half_vector /= np.linalg.norm(half_vector)
        specular = np.clip(normal @ half_vector, 0, 1) ** 42
        base += specular[..., None] * 150.0
    if variant == "refractive_water":
        material = np.asarray(
            donor.convert("RGB").resize(
                (640, 360), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        material_luma = _luminance(material)
        material_luma = (
            material_luma - material_luma.mean()
        ) / max(float(material_luma.std()), 1.0)
        caustic = np.clip(
            material_luma * 5.0 + height * 170.0, -18, 26
        )
        base += caustic[..., None]
        fresnel = np.clip((1.0 - normal[..., 2]) * 2.6, 0, 1)
        base = (
            base * (1.0 - fresnel[..., None] * 0.25)
            + np.array([184, 224, 235], dtype=np.float32)
            * fresnel[..., None]
            * 0.25
        )
    image = Image.fromarray(
        np.uint8(np.rint(np.clip(base, 0, 255))), mode="RGB"
    )
    _draw_phys_sources(image, keyframe_id)
    output_luma = _luminance(np.asarray(image, dtype=np.float32))
    grad_magnitude = np.sqrt(gx * gx + gy * gy)
    return image, {
        "surface_height_min": round(float(height.min()), 6),
        "surface_height_max": round(float(height.max()), 6),
        "envelope_peak": round(float(envelope.max()), 6),
        "render_luminance_std_0_255": round(
            float(output_luma.std()), 6
        ),
        "height_gradient_to_luminance_correlation": round(
            float(
                np.corrcoef(
                    grad_magnitude.ravel(),
                    output_luma.ravel(),
                )[0, 1]
            ),
            6,
        ),
    }


def _sequence_sheet(
    output_root: Path,
    variants: tuple[str, ...],
) -> Path:
    thumb = (320, 180)
    label = 28
    sheet = Image.new(
        "RGB",
        (
            len(KEYFRAMES) * thumb[0],
            len(variants) * (thumb[1] + label),
        ),
        (12, 31, 37),
    )
    draw = ImageDraw.Draw(sheet)
    for row, variant in enumerate(variants):
        y = row * (thumb[1] + label)
        for column, keyframe in enumerate(KEYFRAMES):
            path = (
                output_root
                / "variants"
                / variant
                / f"{keyframe}.png"
            )
            image = Image.open(path).convert("RGB").resize(thumb)
            sheet.paste(image, (column * thumb[0], y))
        draw.text(
            (8, y + thumb[1] + 7),
            variant.replace("_", " ").upper()
            + " · START → MECHANISM → RESULT → END",
            fill=(234, 245, 240),
        )
    path = output_root / "all-variants-sequence.jpg"
    sheet.save(path, quality=92, subsampling=0)
    return path


def _build_case(
    *,
    route_id: str,
    case_id: str,
    variants: tuple[str, ...],
    renderer: Any,
    donor_path: Path,
) -> dict[str, Any]:
    root = OUTPUT_ROOT / route_id / case_id
    root.mkdir(parents=True, exist_ok=True)
    donor = Image.open(donor_path).convert("RGB")
    records = []
    for variant, keyframe_id in itertools.product(
        variants, KEYFRAMES
    ):
        image, metrics = renderer(donor, keyframe_id, variant)
        path = (
            root
            / "variants"
            / variant
            / f"{keyframe_id}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, optimize=False)
        records.append(
            {
                "variant": variant,
                "keyframe_id": keyframe_id,
                "output": artifact_record(path, root),
                "metrics": metrics,
            }
        )
    sheet = _sequence_sheet(root, variants)
    manifest = {
        "schema_version": "1.0",
        "route_id": route_id,
        "case_id": case_id,
        "status": "generated_for_review",
        "model_runs": {"image": 0, "video": 0},
        "donor": artifact_record(donor_path, STAGE2_ROOT),
        "variants": list(variants),
        "records": records,
        "comparison": artifact_record(sheet, root),
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def _variant_hashes() -> dict[str, str]:
    result = {}
    for route in ("route-b", "route-c"):
        root = OUTPUT_ROOT / route
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/variants/*/*.png")):
            result[
                str(path.relative_to(OUTPUT_ROOT))
            ] = sha256_path(path)
    return result


def run() -> dict[str, Any]:
    previous_hashes = _variant_hashes()
    route_b = [
        _build_case(
            route_id="route-b",
            case_id="CHEM-01",
            variants=(
                "state_tint",
                "optical_tint",
                "optical_tint_plus_drop",
            ),
            renderer=_chem_hybrid_variant,
            donor_path=(
                OUTPUT_ROOT
                / "route-a/experiments/"
                "EXP-P7-A-chem-01-00_start/raw/"
                "semantic_control_065/seed_7101.png"
            ),
        ),
        _build_case(
            route_id="route-b",
            case_id="BIO-01",
            variants=(
                "raw_underlay",
                "normalized_cytoplasm",
                "stable_material_plus_depth",
            ),
            renderer=_bio_hybrid_variant,
            donor_path=(
                STAGE2_ROOT
                / "output/phase-3/EXP-20260729-013/raw/"
                "t2i_dense_cell_and_chromatids/seed_3104.png"
            ),
        ),
    ]
    route_c = [
        _build_case(
            route_id="route-c",
            case_id="MATH-02",
            variants=(
                "mapped_wood",
                "beveled_wood",
                "studio_pbr",
            ),
            renderer=_math_pbr_variant,
            donor_path=(
                STAGE2_ROOT
                / "output/phase-3/EXP-20260729-012/raw/"
                "t2i_wood_material_donor/seed_3101.png"
            ),
        ),
        _build_case(
            route_id="route-c",
            case_id="PHYS-01",
            variants=(
                "normal_diffuse",
                "specular_water",
                "refractive_water",
            ),
            renderer=_phys_pbr_variant,
            donor_path=(
                STAGE2_ROOT
                / "output/phase-3/EXP-20260729-005/raw/"
                "t2i_material_donor/seed_3101.png"
            ),
        ),
    ]
    current_hashes = _variant_hashes()
    unchanged = sum(
        previous_hashes.get(path) == digest
        for path, digest in current_hashes.items()
    )
    replay_checked = len(previous_hashes) == len(
        current_hashes
    ) == 48
    result = {
        "schema_version": "1.0",
        "status": "generated_for_review",
        "route_b": [
            {
                "case_id": item["case_id"],
                "variants": item["variants"],
            }
            for item in route_b
        ],
        "route_c": [
            {
                "case_id": item["case_id"],
                "variants": item["variants"],
            }
            for item in route_c
        ],
        "output_count": sum(
            len(item["records"]) for item in route_b + route_c
        ),
        "determinism_replay": {
            "status": (
                "passed"
                if replay_checked
                and unchanged == len(current_hashes)
                else (
                    "first_generation"
                    if not previous_hashes
                    else "failed"
                )
            ),
            "previous_output_count": len(previous_hashes),
            "current_output_count": len(current_hashes),
            "unchanged_sha256_count": unchanged,
            "passed": bool(
                replay_checked
                and unchanged == len(current_hashes)
            ),
        },
    }
    write_json(
        OUTPUT_ROOT / "hybrid-pbr-manifest.json", result
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run()
    print(
        f"Phase 7 routes B/C: {result['status']} · "
        f"{result['output_count']} rendered keyframes"
    )


if __name__ == "__main__":
    main()
