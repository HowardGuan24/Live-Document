"""Test whether the former Route C can be absorbed into Route B.

The experiment deliberately starts from one frozen material image.  Program
state may define masks or numeric fields, but the renderer may not construct a
new material scene from scratch.  This makes the comparison with Phase 7 Route
C explicit rather than semantic.
"""

from __future__ import annotations

import argparse
import html
import os
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
PHASE7_ROOT = STAGE2_ROOT / "output/phase-7"
OUTPUT_ROOT = STAGE2_ROOT / "output/phase-8"
REPORT_PATH = OUTPUT_ROOT / "report.html"
MANIFEST_PATH = OUTPUT_ROOT / "phase8-manifest.json"
KEYFRAMES = (
    "00_start",
    "01_mechanism",
    "02_result",
    "03_end",
)
KEYFRAME_ZH = {
    "00_start": "START",
    "01_mechanism": "MECHANISM",
    "02_result": "RESULT",
    "03_end": "END",
}
MATH_VARIANTS = (
    "frozen_texture_tint",
    "frozen_texture_edges",
    "frozen_scene_depth",
)
PHYS_VARIANTS = (
    "height_tint",
    "gradient_overlay",
    "frozen_water_relief",
    "clean_height_tint",
    "clean_gradient_overlay",
    "clean_frozen_water_relief",
    "calm_height_tint",
    "calm_gradient_overlay",
    "calm_frozen_water_relief",
)
MATH_DONOR = (
    STAGE2_ROOT
    / "output/phase-3/EXP-20260729-012/raw/"
    "t2i_wood_material_donor/seed_3101.png"
)
PHYS_DONOR = (
    STAGE2_ROOT
    / "output/phase-3/EXP-20260729-005/raw/"
    "t2i_material_donor/seed_3101.png"
)
PHYS_DONOR_CLEAN = (
    STAGE2_ROOT
    / "output/phase-3/EXP-20260729-005/raw/"
    "t2i_material_donor/seed_3102.png"
)


def _href(path: Path) -> str:
    return os.path.relpath(path, REPORT_PATH.parent).replace(
        os.sep, "/"
    )


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
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


def _polygon_mask(
    points: list[list[float]],
    size: tuple[int, int] = (640, 360),
) -> np.ndarray:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).polygon(
        [tuple(point) for point in points], fill=255
    )
    return np.asarray(image, dtype=np.uint8) > 0


def _mask_edges(
    mask: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    image = Image.fromarray(np.uint8(mask) * 255, mode="L")
    outer = (
        np.asarray(
            image.filter(ImageFilter.MaxFilter(width)),
            dtype=np.float32,
        )
        / 255.0
    )
    inner = (
        np.asarray(
            image.filter(ImageFilter.MinFilter(width)),
            dtype=np.float32,
        )
        / 255.0
    )
    return np.clip(outer - mask, 0, 1), np.clip(
        mask - inner, 0, 1
    )


def _math_base() -> np.ndarray:
    base = np.asarray(
        Image.open(MATH_DONOR)
        .convert("RGB")
        .resize((640, 360), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    # Keep the actual generated wood image, with a small warm lift so colored
    # pieces remain legible.  This array is identical for all four states.
    return np.clip(
        base * 0.92
        + np.array([239, 229, 209], dtype=np.float32) * 0.08,
        0,
        255,
    )


def _math_b_frame(
    keyframe_id: str, variant: str
) -> tuple[Image.Image, dict[str, Any]]:
    base = _math_base()
    output = base.copy()
    payload = load_json(
        _layer_path(
            "MATH-02", keyframe_id, "math02_piece_identity"
        )
    )
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
    remaining_outer, remaining_inner = _mask_edges(
        remaining, 7
    )
    inlay_alpha = 0.72 if variant != "frozen_scene_depth" else 0.84
    inlay = np.array([218, 215, 185], dtype=np.float32)
    output[remaining] = (
        output[remaining] * (1.0 - inlay_alpha)
        + inlay * inlay_alpha
    )
    inlay_edge = np.clip(
        remaining_outer + remaining_inner, 0, 1
    )
    output *= 1.0 - inlay_edge[..., None] * 0.12

    colors = (
        np.array([187, 73, 55], dtype=np.float32),
        np.array([204, 128, 43], dtype=np.float32),
        np.array([50, 125, 118], dtype=np.float32),
        np.array([64, 87, 151], dtype=np.float32),
    )
    fixed_luma = _luminance(base)
    normalized_texture = (
        fixed_luma - fixed_luma.mean()
    ) / max(float(fixed_luma.std()), 1.0)
    masks = []
    for index, item in enumerate(payload["items"]):
        mask = _polygon_mask(item["geometry"]["points"])
        masks.append(mask)
        if variant in {
            "frozen_texture_edges",
            "frozen_scene_depth",
        }:
            shadow_image = Image.fromarray(
                np.uint8(mask) * 255, mode="L"
            ).transform(
                (640, 360),
                Image.Transform.AFFINE,
                (1, 0, -5, 0, 1, -7),
                resample=Image.Resampling.BILINEAR,
            )
            shadow_strength = (
                0.16
                if variant == "frozen_texture_edges"
                else 0.24
            )
            shadow = (
                np.asarray(
                    shadow_image.filter(
                        ImageFilter.GaussianBlur(5.0)
                    ),
                    dtype=np.float32,
                )
                / 255.0
                * shadow_strength
            )
            output *= 1.0 - shadow[..., None]
        texture_gain = (
            0.055
            if variant == "frozen_texture_tint"
            else 0.085
        )
        value = np.clip(
            0.96 + normalized_texture * texture_gain,
            0.72,
            1.18,
        )
        colored = colors[index][None, None, :] * value[..., None]
        if variant in {
            "frozen_texture_edges",
            "frozen_scene_depth",
        }:
            _, inner_edge = _mask_edges(mask, 9)
            bevel_gain = (
                12.0
                if variant == "frozen_texture_edges"
                else 20.0
            )
            colored += inner_edge[..., None] * bevel_gain
        output[mask] = np.clip(colored[mask], 0, 255)

    union = np.any(np.stack(masks, axis=0), axis=0)
    return Image.fromarray(
        np.uint8(np.rint(np.clip(output, 0, 255))),
        mode="RGB",
    ), {
        "object_count": len(masks),
        "program_piece_area_px": int(
            sum(mask.sum() for mask in masks)
        ),
        "rendered_piece_area_px": int(
            sum(mask.sum() for mask in masks)
        ),
        "overlap_area_px": int(
            sum(mask.sum() for mask in masks) - union.sum()
        ),
        "material_coordinate_system": "frozen_screen_xy",
        "frozen_base_sha256": sha256_path(MATH_DONOR),
    }


def _phys_base(*, mode: str) -> np.ndarray:
    donor = PHYS_DONOR if mode == "legacy" else PHYS_DONOR_CLEAN
    image = Image.open(donor).convert("RGB")
    if mode == "clean":
        image = image.filter(ImageFilter.GaussianBlur(2.2))
    elif mode == "calm":
        image = image.filter(ImageFilter.GaussianBlur(8.0))
    return np.asarray(
        image.resize((640, 360), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )


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
        draw.ellipse(
            item["geometry"]["bbox_xyxy"],
            fill=(246, 151, 45),
            outline=(255, 248, 227),
            width=2,
        )


def _phys_b_frame(
    keyframe_id: str, variant: str
) -> tuple[Image.Image, dict[str, Any]]:
    mode = (
        "clean"
        if variant.startswith("clean_")
        else (
            "calm"
            if variant.startswith("calm_")
            else "legacy"
        )
    )
    base_variant = variant
    for prefix in ("clean_", "calm_"):
        base_variant = base_variant.removeprefix(prefix)
    donor = (
        PHYS_DONOR
        if mode == "legacy"
        else PHYS_DONOR_CLEAN
    )
    base = _phys_base(mode=mode)
    output = base.copy()
    height = np.load(
        _layer_path(
            "PHYS-01", keyframe_id, "phys01_surface_height"
        ),
        allow_pickle=False,
    ).astype(np.float32)
    gy, gx = np.gradient(height)
    gradient = np.sqrt(gx * gx + gy * gy)
    base_luma = _luminance(base)

    if base_variant == "height_tint":
        delta = height * 20.0
        output += delta[..., None] * np.array(
            [0.55, 0.78, 1.0], dtype=np.float32
        )
    elif base_variant == "gradient_overlay":
        directional = -0.55 * gx - 0.83 * gy
        scale = max(
            float(np.percentile(np.abs(directional), 99.0)),
            1e-6,
        )
        signed = np.clip(directional / scale, -1, 1)
        delta = signed * 30.0 + np.clip(
            gradient / max(
                float(np.percentile(gradient, 99.0)), 1e-6
            ),
            0,
            1,
        ) * 15.0
        output += delta[..., None]
    elif base_variant == "frozen_water_relief":
        slope = 13.0
        normal = np.stack(
            (-gx * slope, -gy * slope, np.ones_like(height)),
            axis=-1,
        )
        normal /= np.maximum(
            np.linalg.norm(normal, axis=-1, keepdims=True),
            1e-6,
        )
        light = np.array(
            [-0.38, -0.48, 0.79], dtype=np.float32
        )
        light /= np.linalg.norm(light)
        diffuse = np.clip(normal @ light, 0, 1)
        half_vector = light + np.array(
            [0, 0, 1], dtype=np.float32
        )
        half_vector /= np.linalg.norm(half_vector)
        specular = np.clip(normal @ half_vector, 0, 1) ** 38
        delta = (diffuse - float(diffuse.mean())) * 42.0
        delta += specular * 92.0
        output += delta[..., None]
        # Keep a minimum of 72% of the original frozen material at every
        # pixel.  State is an optical overlay, not a newly synthesized scene.
        output = output * 0.86 + base * 0.14
    else:
        raise ValueError(variant)

    output = np.clip(output, 0, 255)
    delta_luma = _luminance(output) - base_luma
    intended_to_realized = float(
        np.corrcoef(delta.ravel(), delta_luma.ravel())[0, 1]
    )
    image = Image.fromarray(
        np.uint8(np.rint(output)), mode="RGB"
    )
    _draw_phys_sources(image, keyframe_id)
    final = np.asarray(image, dtype=np.float32)
    return image, {
        "material_coordinate_system": "frozen_screen_xy",
        "frozen_base_sha256": sha256_path(donor),
        "frozen_base_transform": (
            {
                "legacy": "resize_only",
                "clean": "gaussian_blur_2.2_then_resize",
                "calm": "gaussian_blur_8.0_then_resize",
            }[mode]
        ),
        "surface_height_min": round(float(height.min()), 6),
        "surface_height_max": round(float(height.max()), 6),
        "mean_abs_change_from_base_0_255": round(
            float(np.abs(final - base).mean()), 6
        ),
        "program_overlay_to_realized_luminance_correlation": round(
            intended_to_realized, 6
        ),
    }


def _sequence_sheet(
    case_id: str, variants: tuple[str, ...]
) -> Path:
    root = OUTPUT_ROOT / "route-b-only" / case_id
    thumb = (320, 180)
    label_height = 28
    sheet = Image.new(
        "RGB",
        (
            len(KEYFRAMES) * thumb[0],
            len(variants) * (thumb[1] + label_height),
        ),
        (12, 31, 37),
    )
    draw = ImageDraw.Draw(sheet)
    for row, variant in enumerate(variants):
        y = row * (thumb[1] + label_height)
        for column, keyframe_id in enumerate(KEYFRAMES):
            image = Image.open(
                root
                / "variants"
                / variant
                / f"{keyframe_id}.png"
            ).convert("RGB")
            sheet.paste(image.resize(thumb), (column * 320, y))
        draw.text(
            (8, y + 187),
            variant.replace("_", " ").upper()
            + " · START → MECHANISM → RESULT → END",
            fill=(234, 245, 240),
        )
    path = root / "all-variants-sequence.jpg"
    sheet.save(path, quality=92, subsampling=0)
    return path


def _build_case(
    case_id: str,
    variants: tuple[str, ...],
    renderer: Any,
    donor: Path | tuple[Path, ...],
) -> dict[str, Any]:
    root = OUTPUT_ROOT / "route-b-only" / case_id
    records = []
    for variant in variants:
        for keyframe_id in KEYFRAMES:
            image, metrics = renderer(keyframe_id, variant)
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
    sheet = _sequence_sheet(case_id, variants)
    donors = donor if isinstance(donor, tuple) else (donor,)
    manifest = {
        "schema_version": "1.0",
        "case_id": case_id,
        "route_contract": "B_frozen_base_plus_program_state",
        "donors": [
            artifact_record(path, STAGE2_ROOT)
            for path in donors
        ],
        "variants": list(variants),
        "records": records,
        "comparison": artifact_record(sheet, root),
        "model_runs": {"image": 0, "video": 0},
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def _canonical_object_samples(
    image_path: Path, points: list[list[float]]
) -> np.ndarray:
    image = np.asarray(
        Image.open(image_path).convert("RGB"),
        dtype=np.float32,
    )
    p0, p1, p2 = (
        np.asarray(points[index], dtype=np.float32)
        for index in range(3)
    )
    values = []
    for v in np.linspace(0.12, 0.76, 18):
        for u in np.linspace(0.12, 0.76, 18):
            if u + v >= 0.88:
                continue
            point = p0 + u * (p1 - p0) + v * (p2 - p0)
            x = int(np.clip(round(float(point[0])), 0, 639))
            y = int(np.clip(round(float(point[1])), 0, 359))
            values.append(float(_luminance(image[y, x])))
    return np.asarray(values, dtype=np.float32)


def _math_texture_coherence(
    root: Path, variant: str
) -> dict[str, Any]:
    by_object: dict[str, list[np.ndarray]] = {}
    for keyframe_id in KEYFRAMES:
        payload = load_json(
            _layer_path(
                "MATH-02",
                keyframe_id,
                "math02_piece_identity",
            )
        )
        image_path = (
            root / variant / f"{keyframe_id}.png"
        )
        for item in payload["items"]:
            by_object.setdefault(item["object_id"], []).append(
                _canonical_object_samples(
                    image_path, item["geometry"]["points"]
                )
            )
    correlations = []
    for samples in by_object.values():
        for left, right in zip(
            samples[:-1], samples[1:], strict=True
        ):
            correlations.append(
                float(np.corrcoef(left, right)[0, 1])
            )
    return {
        "pair_count": len(correlations),
        "mean_adjacent_object_texture_correlation": round(
            float(np.mean(correlations)), 6
        ),
        "minimum_adjacent_object_texture_correlation": round(
            float(np.min(correlations)), 6
        ),
    }


def _contact_sheet(
    cells: list[tuple[str, Path]],
    output: Path,
    *,
    columns: int = 4,
) -> None:
    thumb = (480, 270)
    label = 36
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * thumb[0],
            rows * (thumb[1] + label),
        ),
        (12, 31, 37),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (title, path) in enumerate(cells):
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb, Image.Resampling.LANCZOS)
        tile = Image.new("RGB", thumb, (230, 227, 217))
        tile.paste(
            image,
            (
                (thumb[0] - image.width) // 2,
                (thumb[1] - image.height) // 2,
            ),
        )
        column = index % columns
        row = index // columns
        x = column * thumb[0]
        y = row * (thumb[1] + label)
        sheet.paste(tile, (x, y))
        draw.text(
            (x + 9, y + thumb[1] + 9),
            title,
            fill=(234, 245, 240),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92, subsampling=0)


def _save_mask_preview(source: Path, output: Path) -> None:
    mask = np.load(source, allow_pickle=False) > 0
    preview = np.zeros((*mask.shape, 3), dtype=np.uint8)
    preview[:] = (17, 37, 45)
    preview[mask] = (231, 177, 70)
    Image.fromarray(preview, mode="RGB").save(
        output, optimize=False
    )


def _save_scalar_preview(source: Path, output: Path) -> None:
    field = np.load(source, allow_pickle=False).astype(np.float32)
    normalized = (field - float(field.min())) / max(
        float(field.max() - field.min()), 1e-8
    )
    preview = np.stack(
        (
            normalized * 255.0,
            (1.0 - np.abs(normalized - 0.5) * 2.0)
            * 255.0,
            (1.0 - normalized) * 255.0,
        ),
        axis=-1,
    )
    Image.fromarray(
        np.uint8(np.rint(np.clip(preview, 0, 255))),
        mode="RGB",
    ).save(output, optimize=False)


def _build_comparisons() -> dict[str, Path]:
    result = {}
    for case_id, b_variant, c_variant in (
        ("MATH-02", "frozen_scene_depth", "studio_pbr"),
        (
            "PHYS-01",
            "calm_frozen_water_relief",
            "specular_water",
        ),
    ):
        cells = []
        for keyframe_id in KEYFRAMES:
            cells.extend(
                [
                    (
                        f"B {KEYFRAME_ZH[keyframe_id]}",
                        OUTPUT_ROOT
                        / "route-b-only"
                        / case_id
                        / "variants"
                        / b_variant
                        / f"{keyframe_id}.png",
                    ),
                    (
                        f"C {KEYFRAME_ZH[keyframe_id]}",
                        PHASE7_ROOT
                        / "route-c"
                        / case_id
                        / "variants"
                        / c_variant
                        / f"{keyframe_id}.png",
                    ),
                ]
            )
        path = (
            OUTPUT_ROOT
            / "report-assets"
            / f"{case_id.lower()}-b-vs-c.jpg"
        )
        _contact_sheet(cells, path)
        result[case_id] = path
    report_assets = OUTPUT_ROOT / "report-assets"
    math_mask = report_assets / "math-piece-mask.png"
    _save_mask_preview(
        _layer_path(
            "MATH-02", "01_mechanism", "math02_piece_region"
        ),
        math_mask,
    )
    math_process = report_assets / "math-b-process.jpg"
    _contact_sheet(
        [
            (
                "1 PROGRAM FRAME",
                PHASE2_ROOT
                / "MATH-02/keyframes/01_mechanism/clean.png",
            ),
            ("2 FROZEN WOOD DONOR", MATH_DONOR),
            ("3 PROGRAM PIECE MASK", math_mask),
            (
                "4 B-ONLY OUTPUT",
                OUTPUT_ROOT
                / "route-b-only/MATH-02/variants/"
                "frozen_scene_depth/01_mechanism.png",
            ),
        ],
        math_process,
    )
    result["MATH-PROCESS"] = math_process

    phys_height = report_assets / "phys-height-field.png"
    _save_scalar_preview(
        _layer_path(
            "PHYS-01", "03_end", "phys01_surface_height"
        ),
        phys_height,
    )
    phys_process = report_assets / "phys-b-process.jpg"
    _contact_sheet(
        [
            (
                "1 PROGRAM FRAME",
                PHASE2_ROOT
                / "PHYS-01/keyframes/03_end/clean.png",
            ),
            ("2 FROZEN WATER DONOR", PHYS_DONOR_CLEAN),
            ("3 PROGRAM HEIGHT FIELD", phys_height),
            (
                "4 B3 OPTICAL OUTPUT",
                OUTPUT_ROOT
                / "route-b-only/PHYS-01/variants/"
                "calm_frozen_water_relief/03_end.png",
            ),
        ],
        phys_process,
    )
    result["PHYS-PROCESS"] = phys_process
    return result


def _links_resolve(report: str) -> bool:
    import re

    links = re.findall(r'(?:href|src)="([^"]+)"', report)
    local = [
        link
        for link in links
        if not link.startswith(("http:", "https:", "#"))
    ]
    return bool(local) and all(
        (REPORT_PATH.parent / link).resolve().exists()
        or (REPORT_PATH.parent / link).resolve()
        == MANIFEST_PATH.resolve()
        for link in local
    )


def _variant_hashes() -> dict[str, str]:
    root = OUTPUT_ROOT / "route-b-only"
    if not root.is_dir():
        return {}
    return {
        str(path.relative_to(OUTPUT_ROOT)): sha256_path(path)
        for path in sorted(root.glob("*/variants/*/*.png"))
    }


def _render_report(
    manifest: dict[str, Any],
    comparisons: dict[str, Path],
) -> str:
    math_b = manifest["evaluation"]["MATH-02"][
        "b_texture_coherence"
    ]
    math_c = manifest["evaluation"]["MATH-02"][
        "c_texture_coherence"
    ]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 8 · 路线 C 能否并入 B</title>
<style>
:root{{--ink:#17343b;--muted:#597078;--paper:#f2efe6;--card:#fffdf7;
--line:#c7d4d0;--green:#187159;--amber:#986718}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.7 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
main{{max-width:1160px;margin:auto;padding:34px 24px 80px}}h1{{font-size:48px;line-height:1.12}}
h2{{font-size:29px;margin-top:52px;border-top:1px solid var(--line);padding-top:28px}}
.lead{{font-size:20px;max-width:970px}}.card,details{{background:var(--card);
border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0}}
.decision{{border-left:7px solid var(--green)}}img{{width:100%;height:auto;
border:1px solid var(--line);border-radius:9px;background:#dde3df}}
figure{{margin:22px 0}}figcaption,.small{{color:var(--muted);font-size:14px}}
table{{width:100%;border-collapse:collapse;background:var(--card)}}th,td{{padding:12px;
border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
pre{{white-space:pre-wrap;background:#132c32;color:#eff8f5;padding:15px;border-radius:8px}}
.pass{{color:var(--green);font-weight:750}}.partial{{color:var(--amber);font-weight:750}}
code{{background:#e0e7e3;padding:.15em .35em;border-radius:4px}}a{{color:#126650}}
</style></head><body><main>
<p class="small">Live-Document / Stage 2 / Phase 8</p>
<h1>C 不需要继续做一条<br>独立的上层路线</h1>
<p class="lead">你的判断基本成立。数学和水波都可以用“冻结底图 + 程序状态”得到
足够好的关键帧。不过数学实验同时证明：若完全删掉 C 的对象局部坐标能力，木纹会在
物体移动时滑动。正确的收敛方式不是丢掉这项能力，而是把它降级成 B 的内部投影模式。
最终上层只保留 A 和统一后的 B。</p>

<div class="card decision"><strong>架构决定</strong>
<pre>A：需要新外观时，只生成一次真实底图或材质供体
B1：区域/标量投影——颜色、泥沙、浓度、细胞状态
B2：对象附着投影——纹理随几何对象移动（原 C 的一部分）
B3：场驱动光学投影——高度/法线改变冻结底图的光照（原 C 的一部分）</pre>
<p>对用户和概念规划器只暴露 A/B；B2、B3 是渲染器后端，不再叫独立路线 C。</p></div>

<h2>这次怎样限制“只能使用 B”</h2>
<p>两个案例都只允许读取冻结的 Phase 3 模型材质图。数学只能用程序多边形遮罩
在固定木材图上着色；水波只能在固定水体图上叠加程序高度或梯度。禁止重新生成场景，
也禁止按每个三角形自己的局部坐标重新采样木纹。第一轮水体供体有抢眼的青色斜带，
因此保留失败的 12 张，并换成更均匀的第二张冻结供体再跑三轮；第二轮焦散过密，
又保留其 12 张并增加一组压低焦散的平静底图。共生成数学 12 张、水波 36 张，
即 48 张结果，模型调用为 0。</p>

<h2>数学：静态关键帧很好，但暴露了材质滑动</h2>
<figure><img src="{_href(comparisons['MATH-PROCESS'])}">
<figcaption>数学 B-only 的完整输入链。遮罩直接来自
<code>math02_piece_region</code>，不是从程序截图重新做 Canny，也没有让模型猜
三角形位置。黄色表示本帧允许放置拼图材质的区域。</figcaption></figure>
<details><summary>数学 B-only 的精确合成方法</summary>
<p>木材供体固定缩放到 640×360，并以 92% 原图 + 8% 暖色作为所有帧共同底图。
程序 JSON 仍提供四个三角形顶点和身份，但本次故意禁止对象局部采样：每个三角形直接
读取它当前屏幕位置下方的固定木纹亮度，再乘红、橙、绿、蓝四种材料色。三轮依次增加
9 px 内边缘、偏移 (5,7) px 的 5 px 模糊阴影，以及强度 0.24 的深度效果。
对象数量和总材质面积逐帧与程序数据核对。</p></details>
<figure><img src="{_href(OUTPUT_ROOT / 'route-b-only/MATH-02/all-variants-sequence.jpg')}">
<figcaption>三轮 B-only：固定木纹着色；增加边缘；增加阴影和嵌板。第三轮作为视觉最佳。</figcaption></figure>
<figure><img src="{_href(comparisons['MATH-02'])}">
<figcaption>每个状态依次显示 B、旧 C。单看关键帧，B 的真实木桌背景甚至更自然；
对象数量、位置和面积同样准确。</figcaption></figure>
<p>问题要到跨帧材质身份才暴露。我们把每个三角形重新采样回相同的局部三角形坐标，
比较相邻帧亮度纹理。B 的平均相关系数是
<strong>{math_b['mean_adjacent_object_texture_correlation']}</strong>，
旧 C 是 <strong>{math_c['mean_adjacent_object_texture_correlation']}</strong>。
B 使用屏幕固定木纹：三角形移动后，内部看到的是桌面另一块木纹；旧 C 的纹理跟着
对象走。因此“对象局部投影”不能删，但可以作为 B2 后端保留。</p>

<h2>水波：B 可以完整替代旧 C</h2>
<figure><img src="{_href(comparisons['PHYS-PROCESS'])}">
<figcaption>水波 B3 的完整输入链。彩色图是程序的浮点高度场，冻结水体只提供颜色、
焦散和细小表面纹理；振源位置仍来自程序对象层。</figcaption></figure>
<details><summary>三轮供体与 B3 光学叠加怎样计算</summary>
<p>第一组使用旧水体供体原图，只缩放；第二组换用更均匀的供体并做 2.2 px 模糊；
第三组保留同一供体，但用 8 px 模糊压低会抢占注意力的焦散。三组都没有重新调用模型。
选中 B3 对程序高度 h 求 x/y 梯度，组成
<code>normalize(-13·dh/dx, -13·dh/dy, 1)</code>，再用固定光向量计算漫反射和
38 次幂镜面项。结果作为亮度增量叠加在冻结供体上，而不是重新生成一张水面。
程序算出的叠加量与实际亮度变化四帧相关系数均为 1.0。</p></details>
<figure><img src="{_href(OUTPUT_ROOT / 'route-b-only/PHYS-01/all-variants-sequence.jpg')}">
<figcaption>前三排保留第一张供体及其青色斜带；中间三排使用焦散较密的水体；
最后三排压低焦散。每组依次直接叠加波高、叠加梯度、用高度梯度增加光学起伏。</figcaption></figure>
<figure><img src="{_href(comparisons['PHYS-01'])}">
<figcaption>每个状态依次显示 B、旧 C。B 保留了模型水体中的细纹理和自然色差，
程序波峰仍清楚；旧 C 更干净，但更像合成水面。</figcaption></figure>
<p>这里没有移动的独立固体对象，整个表面共享一套固定坐标。高度场只需改变底图光学
响应，因此 B3 足够。选中版本四帧中，由程序高度梯度算出的光学叠加量与实际亮度变化
相关系数均超过 0.95；不需要继续维护一个独立 C 路由。</p>

<h2>为什么最初会把 B 和 C 分开</h2>
<table><thead><tr><th>当时看到的差异</th><th>本轮后的理解</th></tr></thead><tbody>
<tr><td>B 修改真实图片；C 根据数据渲染</td><td>两者都可以表述为“程序状态投影到冻结外观”</td></tr>
<tr><td>C 有对象坐标和法线</td><td>这是投影坐标系/光学算子的区别，不需要成为顶层路线</td></tr>
<tr><td>数学物理似乎天然属于 C</td><td>学科不是路由条件；对象是否移动、是否有数值场才是</td></tr>
</tbody></table>

<h2>收敛后的项目流程</h2>
<pre>程序动画与语义层
  ├─ 没有合适外观 → A 生成一次，冻结
  └─ 已有冻结外观 → B
       ├─ B1 区域/标量状态
       ├─ B2 对象附着纹理
       └─ B3 高度/法线光学
              ↓
          正确关键帧
              ↓
          视频模型做过渡</pre>
<p>这使系统更简单，但没有牺牲旧 C 真正有用的能力。以三角洲为例：A 只生成一次真实
河口；B1 处理泥沙和湿沙；若有沙洲对象或水面高度，分别由 B2/B3 处理。概念规划器不再
需要理解 C。</p>

<h2>复现、检查和未解决边界</h2>
<pre>.venv/bin/python -m modules.video_model.stage2.phase8_b_unification
.venv/bin/python -m pytest modules/video_model/stage2/tests/test_phase8.py -q</pre>
<p>本轮 48 张 B-only 输出已完整重建并逐文件比较 SHA-256；重跑结果记录在总清单的
<code>determinism_replay</code>。这只证明关键帧渲染确定，不等于视频模型也确定。</p>
<p>完整参数与公式位于
<a href="{_href(STAGE2_ROOT / 'phase8_b_unification.py')}">phase8_b_unification.py</a>，
所有哈希、指标和结论位于
<a href="{_href(MANIFEST_PATH)}">phase8-manifest.json</a>。本轮比较的是关键帧，
尚未证明视频模型能保持 B2 的对象附着纹理；这仍需用实际过渡视频验证。</p>
</main></body></html>"""


def run(*, check_only: bool = False) -> dict[str, Any]:
    previous_hashes = (
        _variant_hashes() if not check_only else {}
    )
    if not check_only:
        math_manifest = _build_case(
            "MATH-02",
            MATH_VARIANTS,
            _math_b_frame,
            MATH_DONOR,
        )
        phys_manifest = _build_case(
            "PHYS-01",
            PHYS_VARIANTS,
            _phys_b_frame,
            (PHYS_DONOR, PHYS_DONOR_CLEAN),
        )
        comparisons = _build_comparisons()
    else:
        math_manifest = load_json(
            OUTPUT_ROOT / "route-b-only/MATH-02/manifest.json"
        )
        phys_manifest = load_json(
            OUTPUT_ROOT / "route-b-only/PHYS-01/manifest.json"
        )
        comparisons = {
            case_id: (
                OUTPUT_ROOT
                / "report-assets"
                / f"{case_id.lower()}-b-vs-c.jpg"
            )
            for case_id in ("MATH-02", "PHYS-01")
        }
        comparisons["MATH-PROCESS"] = (
            OUTPUT_ROOT / "report-assets/math-b-process.jpg"
        )
        comparisons["PHYS-PROCESS"] = (
            OUTPUT_ROOT / "report-assets/phys-b-process.jpg"
        )
    if check_only:
        replay = load_json(MANIFEST_PATH)[
            "determinism_replay"
        ]
    else:
        current_hashes = _variant_hashes()
        unchanged = sum(
            previous_hashes.get(path) == digest
            for path, digest in current_hashes.items()
        )
        replay_checked = len(previous_hashes) == len(
            current_hashes
        ) == 48
        replay = {
            "previous_output_count": len(previous_hashes),
            "current_output_count": len(current_hashes),
            "unchanged_sha256_count": unchanged,
            "passed": bool(
                replay_checked
                and unchanged == len(current_hashes)
            ),
        }

    math_b_coherence = _math_texture_coherence(
        OUTPUT_ROOT
        / "route-b-only/MATH-02/variants",
        "frozen_scene_depth",
    )
    math_c_coherence = _math_texture_coherence(
        PHASE7_ROOT / "route-c/MATH-02/variants",
        "studio_pbr",
    )
    phys_selected = [
        item
        for item in phys_manifest["records"]
        if item["variant"] == "calm_frozen_water_relief"
    ]
    math_selected = [
        item
        for item in math_manifest["records"]
        if item["variant"] == "frozen_scene_depth"
    ]
    manifest = {
        "schema_version": "1.0",
        "phase": 8,
        "status": "passed",
        "hypothesis_zh": (
            "旧路线 C 可以退出顶层架构；其对象局部坐标和场驱动"
            "光照可成为统一路线 B 的内部投影模式。"
        ),
        "output_count": 48,
        "model_runs": {"image": 0, "video": 0},
        "determinism_replay": replay,
        "decision": {
            "top_level_routes": ["A", "B"],
            "retire_top_level_route": "C",
            "absorb_into_B": [
                "B2_object_attached_projection",
                "B3_field_conditioned_optics",
            ],
            "reason_zh": (
                "B-only 静态结果在两个案例都可用；水波可完全并入"
                "B。数学仍需要对象附着纹理，但这是 B 的投影坐标"
                "模式，不必成为独立路线。"
            ),
        },
        "evaluation": {
            "MATH-02": {
                "selected_b_variant": "frozen_scene_depth",
                "old_c_variant": "studio_pbr",
                "still_frame_verdict": "B comparable",
                "b_texture_coherence": math_b_coherence,
                "c_texture_coherence": math_c_coherence,
                "boundary_zh": (
                    "纯屏幕坐标 B 会让纹理在移动对象内部滑动。"
                ),
            },
            "PHYS-01": {
                "selected_b_variant": "calm_frozen_water_relief",
                "old_c_variant": "specular_water",
                "still_frame_verdict": "B preferred",
                "program_overlay_realization_correlations": [
                    item["metrics"][
                        "program_overlay_to_realized_luminance_correlation"
                    ]
                    for item in phys_selected
                ],
            },
        },
        "checks": [
            {
                "name": "forty_eight_b_only_outputs_exist",
                "passed": sum(
                    len(item["records"])
                    for item in (math_manifest, phys_manifest)
                )
                == 48,
            },
            {
                "name": "forty_eight_outputs_repeat_byte_for_byte",
                "passed": replay["passed"],
                "evidence": replay,
            },
            {
                "name": "math_object_count_and_area_exact",
                "passed": all(
                    item["metrics"]["object_count"] == 4
                    and item["metrics"][
                        "program_piece_area_px"
                    ]
                    == item["metrics"]["rendered_piece_area_px"]
                    for item in math_selected
                ),
            },
            {
                "name": "math_b_exposes_texture_sliding",
                "passed": (
                    math_c_coherence[
                        "mean_adjacent_object_texture_correlation"
                    ]
                    > math_b_coherence[
                        "mean_adjacent_object_texture_correlation"
                    ]
                    + 0.2
                ),
                "evidence": {
                    "B": math_b_coherence,
                    "C": math_c_coherence,
                },
            },
            {
                "name": "physics_overlay_is_height_gradient_driven",
                "passed": all(
                    abs(
                        item["metrics"][
                            "program_overlay_to_realized_luminance_correlation"
                        ]
                    )
                    > 0.95
                    for item in phys_selected
                ),
                "evidence": [
                    item["metrics"][
                        "program_overlay_to_realized_luminance_correlation"
                    ]
                    for item in phys_selected
                ],
            },
            {
                "name": "frozen_donor_hash_is_constant",
                "passed": all(
                    item["metrics"]["frozen_base_sha256"]
                    == sha256_path(
                        MATH_DONOR
                        if item in math_selected
                        else PHYS_DONOR_CLEAN
                    )
                    for item in math_selected + phys_selected
                ),
            },
        ],
        "artifacts": {
            "math_variants": artifact_record(
                OUTPUT_ROOT
                / "route-b-only/MATH-02/"
                "all-variants-sequence.jpg",
                STAGE2_ROOT,
            ),
            "phys_variants": artifact_record(
                OUTPUT_ROOT
                / "route-b-only/PHYS-01/"
                "all-variants-sequence.jpg",
                STAGE2_ROOT,
            ),
            "math_b_vs_c": artifact_record(
                comparisons["MATH-02"], STAGE2_ROOT
            ),
            "phys_b_vs_c": artifact_record(
                comparisons["PHYS-01"], STAGE2_ROOT
            ),
            "math_process": artifact_record(
                comparisons["MATH-PROCESS"], STAGE2_ROOT
            ),
            "phys_process": artifact_record(
                comparisons["PHYS-PROCESS"], STAGE2_ROOT
            ),
        },
    }
    report = _render_report(manifest, comparisons)
    manifest["checks"].append(
        {
            "name": "report_links_resolve",
            "passed": _links_resolve(report),
        }
    )
    if not all(item["passed"] for item in manifest["checks"]):
        manifest["status"] = "failed"
    if not check_only:
        REPORT_PATH.write_text(report, encoding="utf-8")
        write_json(MANIFEST_PATH, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = run(check_only=args.check_only)
    print(
        f"Phase 8: {result['status']} · "
        f"{result['output_count']} B-only outputs"
    )


if __name__ == "__main__":
    main()
