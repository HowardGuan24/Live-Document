"""Deterministic, data-driven State Renderer B operators.

The renderer never calls an image model.  A frozen appearance image provides
material and lighting; versioned semantic layers provide the state.  Operator
implementations contain no Case IDs or final Case coordinates.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from modules.video_model.stage3.framework.contracts import (
    file_record,
    load_json,
    sha256_path,
    write_json,
)
from modules.video_model.stage3.framework.geometry import (
    keyframe_semantics,
    layer_data_path,
)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )


def _decorrelated_highpass(
    anchor: np.ndarray,
    *,
    radius: float,
    seed: int,
    smoothing_radius: float,
) -> np.ndarray:
    """Keep donor residual statistics while destroying donor coordinates."""
    blurred = np.asarray(
        Image.fromarray(np.uint8(np.clip(anchor, 0, 255))).filter(
            ImageFilter.GaussianBlur(radius)
        ),
        dtype=np.float32,
    )
    residual = anchor - blurred
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(residual.shape[0] * residual.shape[1])
    shuffled = residual.reshape(-1, residual.shape[2])[permutation].reshape(
        residual.shape
    )
    if smoothing_radius > 0:
        channels = []
        for channel in range(shuffled.shape[2]):
            encoded = np.uint8(
                np.rint(np.clip(shuffled[..., channel] * 6.0 + 128.0, 0, 255))
            )
            channels.append(
                (
                    np.asarray(
                        Image.fromarray(encoded, mode="L").filter(
                            ImageFilter.GaussianBlur(smoothing_radius)
                        ),
                        dtype=np.float32,
                    )
                    - 128.0
                )
                / 6.0
            )
        shuffled = np.stack(channels, axis=-1)
    shuffled /= max(float(shuffled.std()), 1.0)
    return np.clip(shuffled, -2.5, 2.5)


def _bbox(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    if "bbox_xyxy" in geometry:
        return tuple(float(value) for value in geometry["bbox_xyxy"])
    points = geometry.get("points", [])
    if not points:
        raise ValueError(f"geometry has no bounds: {geometry}")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _layer(
    contract: dict[str, Any],
    semantic: dict[str, Any],
    layer_id: str,
) -> dict[str, Any]:
    matches = [
        item
        for item in semantic["layers"]
        if item["layer_id"] == layer_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{contract['case_id']} expected one layer {layer_id}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _load_layer_data(
    contract: dict[str, Any],
    semantic: dict[str, Any],
    layer_id: str,
    repo_root: Path,
) -> tuple[Any, dict[str, Any], Path]:
    record = _layer(contract, semantic, layer_id)
    path = layer_data_path(contract, record, repo_root)
    if record["data"]["encoding"] == "npy":
        value = np.load(path, allow_pickle=False)
    elif record["data"]["encoding"] == "json":
        value = load_json(path)
    else:
        raise ValueError(
            f"unsupported layer encoding {record['data']['encoding']}"
        )
    return value, record, path


def _resize_array(
    array: np.ndarray,
    size: tuple[int, int],
    *,
    nearest: bool = False,
) -> np.ndarray:
    mode = "F" if np.issubdtype(array.dtype, np.floating) else None
    image = Image.fromarray(
        array.astype(np.float32) if mode == "F" else array,
        mode=mode,
    )
    return np.asarray(
        image.resize(
            size,
            Image.Resampling.NEAREST
            if nearest
            else Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )


def _preprocess_anchor(path: Path, spec: dict[str, Any]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    mode = spec["mode"]
    if mode == "identity":
        return image
    if mode == "resize":
        return image.resize(
            tuple(spec["size"]), Image.Resampling.LANCZOS
        )
    if mode == "gaussian_blur_then_resize":
        return image.filter(
            ImageFilter.GaussianBlur(float(spec["radius"]))
        ).resize(tuple(spec["size"]), Image.Resampling.LANCZOS)
    raise ValueError(f"unsupported anchor preprocess: {mode}")


def _base_canvas(
    anchor: np.ndarray,
    spec: dict[str, Any] | None,
) -> np.ndarray:
    """Create the immutable canvas while keeping the anchor as material data.

    Existing plans omit ``base_canvas`` and therefore retain the historical
    behavior where the appearance anchor is also the initial canvas.  A plan
    may instead declare a neutral canvas when the frozen image is only a
    material donor.  This prevents donor geometry from leaking into output.
    """
    if spec is None or spec["mode"] == "anchor":
        return anchor.copy()
    if spec["mode"] not in {
        "constant_with_low_frequency_variation",
        "constant_with_highpass_statistics",
        "constant_with_shuffled_highpass_statistics",
    }:
        raise ValueError(f"unsupported base canvas mode: {spec['mode']}")
    height, width = anchor.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.broadcast_to(
        np.asarray(spec["rgb"], dtype=np.float32),
        anchor.shape,
    ).copy()
    amplitude = float(spec.get("variation_amplitude", 0.0))
    if amplitude:
        period_x = float(spec.get("variation_period_x_px", 39.0))
        period_y = float(spec.get("variation_period_y_px", 27.0))
        variation = amplitude * (
            np.sin(xx / period_x) + np.cos(yy / period_y)
        )
        base += variation[..., None]
    if spec["mode"] == "constant_with_highpass_statistics":
        radius = float(spec.get("highpass_radius_px", 7.0))
        blurred = np.asarray(
            Image.fromarray(np.uint8(np.clip(anchor, 0, 255))).filter(
                ImageFilter.GaussianBlur(radius)
            ),
            dtype=np.float32,
        )
        residual = anchor - blurred
        residual /= max(float(residual.std()), 1.0)
        residual = np.clip(residual, -2.5, 2.5)
        gain = np.asarray(
            spec.get("texture_gain_rgb", [4.0, 4.0, 4.0]),
            dtype=np.float32,
        )
        base += residual * gain[None, None, :]
    elif spec["mode"] == "constant_with_shuffled_highpass_statistics":
        residual = _decorrelated_highpass(
            anchor,
            radius=float(spec.get("highpass_radius_px", 7.0)),
            seed=int(spec.get("shuffle_seed", 0)),
            smoothing_radius=float(spec.get("smoothing_radius_px", 0.65)),
        )
        gain = np.asarray(
            spec.get("texture_gain_rgb", [4.0, 4.0, 4.0]),
            dtype=np.float32,
        )
        base += residual * gain[None, None, :]
    return np.clip(base, 0, 255)


def _identity_payload(
    contract: dict[str, Any],
    semantic: dict[str, Any],
    layer_id: str,
    repo_root: Path,
) -> tuple[dict[str, Any], Path]:
    value, _, path = _load_layer_data(
        contract, semantic, layer_id, repo_root
    )
    if not isinstance(value, dict) or "items" not in value:
        raise ValueError(f"identity layer {layer_id} has no items")
    return value, path


def _find_identity_object(
    payload: dict[str, Any], class_id: str
) -> dict[str, Any]:
    matches = [
        item
        for item in payload["items"]
        if item.get("class_id") == class_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one identity object {class_id}, found {len(matches)}"
        )
    return matches[0]


def _semantic_identity(
    contract: dict[str, Any],
    semantic: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    record = next(
        item
        for item in semantic["layers"]
        if item["layer_type"] == "object_identity"
    )
    return load_json(layer_data_path(contract, record, repo_root))


def _detect_horizontal_surface(
    anchor: np.ndarray,
    bbox: list[int],
    relative_y: list[float],
    side_inset: float,
) -> tuple[int, dict[str, Any]]:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    xa = int(round(x0 + width * side_inset))
    xb = int(round(x1 - width * side_inset))
    ya = int(round(y0 + height * relative_y[0]))
    yb = int(round(y0 + height * relative_y[1]))
    gray = _luminance(anchor)
    vertical_gradient = np.abs(np.diff(gray, axis=0))
    score = vertical_gradient[ya:yb, xa:xb].mean(axis=1)
    local = int(np.argmax(score))
    surface_y = ya + local
    return surface_y, {
        "search_bbox_xyxy": [xa, ya, xb, yb],
        "selected_y": surface_y,
        "selected_mean_vertical_gradient": round(float(score[local]), 6),
        "method": "maximum mean vertical luminance gradient in provider-declared container search band",
    }


def _contained_scalar(
    image: np.ndarray,
    anchor: np.ndarray,
    contract: dict[str, Any],
    semantic: dict[str, Any],
    keyframe_id: str,
    config: dict[str, Any],
    stage3_root: Path,
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    projection = config["projection"]
    gate = load_json(stage3_root / projection["geometry_gate"])
    target_object = next(
        item
        for item in gate["rendered_objects"]
        if item["class_id"] == projection["container_class_id"]
    )
    target_bbox = target_object["output_bbox_xyxy"]
    region, _, region_path = _load_layer_data(
        contract,
        semantic,
        config["region_layer_id"],
        repo_root,
    )
    scalar, _, scalar_path = _load_layer_data(
        contract,
        semantic,
        config["scalar_layer_id"],
        repo_root,
    )
    region = region > 0
    scalar = scalar.astype(np.float32)
    ys, xs = np.nonzero(region)
    if not len(xs):
        raise ValueError("contained scalar region is empty")
    source_region_bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]

    initial_semantic, _, _ = keyframe_semantics(
        contract, projection["initial_keyframe_id"], repo_root
    )
    initial_region, _, _ = _load_layer_data(
        contract,
        initial_semantic,
        config["region_layer_id"],
        repo_root,
    )
    initial_ys, initial_xs = np.nonzero(initial_region > 0)
    initial_identity = _semantic_identity(
        contract, initial_semantic, repo_root
    )
    source_container = _find_identity_object(
        initial_identity, projection["container_class_id"]
    )
    source_container_bbox = _bbox(source_container["geometry"])
    target_surface_y, surface_derivation = _detect_horizontal_surface(
        anchor,
        target_bbox,
        projection["surface_search_relative_y"],
        float(projection["side_inset_relative_x"]),
    )
    source_delta_y = int(ys.min()) - int(initial_ys.min())
    scale_y = (
        (target_bbox[3] - target_bbox[1])
        / max(source_container_bbox[3] - source_container_bbox[1], 1.0)
    )
    current_surface_y = int(
        round(target_surface_y + source_delta_y * scale_y)
    )
    bottom = int(
        round(
            target_bbox[3]
            - (target_bbox[3] - target_bbox[1])
            * float(projection["bottom_inset_relative_y"])
        )
    )
    inset = float(projection["side_inset_relative_x"])
    width = target_bbox[2] - target_bbox[0]
    left_top = int(round(target_bbox[0] + width * inset))
    right_top = int(round(target_bbox[2] - width * inset))
    left_bottom = int(round(target_bbox[0] + width * (inset + 0.045)))
    right_bottom = int(round(target_bbox[2] - width * (inset + 0.045)))
    target_mask_image = Image.new(
        "L", (image.shape[1], image.shape[0]), 0
    )
    ImageDraw.Draw(target_mask_image).polygon(
        [
            (left_top, current_surface_y),
            (right_top, current_surface_y),
            (right_bottom, bottom),
            (left_bottom, bottom),
        ],
        fill=255,
    )
    target_mask = (
        np.asarray(target_mask_image, dtype=np.float32) / 255.0
    )

    crop = scalar[
        source_region_bbox[1] : source_region_bbox[3] + 1,
        source_region_bbox[0] : source_region_bbox[2] + 1,
    ]
    mapped = _resize_array(
        crop,
        (
            right_top - left_top + 1,
            bottom - current_surface_y + 1,
        ),
    )
    scalar_canvas = np.zeros(
        (image.shape[0], image.shape[1]), dtype=np.float32
    )
    scalar_canvas[
        current_surface_y : bottom + 1,
        left_top : right_top + 1,
    ] = mapped
    transfer = config["transfer"]
    if transfer["kind"] != "sigmoid":
        raise ValueError("only sigmoid scalar transfer is implemented")
    indicator = 1.0 / (
        1.0
        + np.exp(
            -(
                scalar_canvas - float(transfer["center"])
            )
            * float(transfer["slope"])
        )
    )
    indicator *= target_mask
    if float(indicator.max()) < 1e-4:
        return image, np.zeros(target_mask.shape, dtype=bool), {
            "operator_type": "scalar_transfer",
            "changed": False,
            "indicator_peak": round(float(indicator.max()), 8),
            "indicator_mean_in_region": round(
                float(indicator[target_mask > 0].mean()), 8
            ),
            "source_region": file_record(region_path, repo_root),
            "source_scalar": file_record(scalar_path, repo_root),
            "target_surface_derivation": surface_derivation,
        }
    feather = (
        np.asarray(
            target_mask_image.filter(
                ImageFilter.GaussianBlur(
                    float(transfer["feather_radius_px"])
                )
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    alpha = np.clip(
        indicator
        * feather
        * float(transfer["maximum_alpha"]),
        0,
        float(transfer["maximum_alpha"]),
    )
    target_rgb = np.asarray(
        transfer["target_rgb"], dtype=np.float32
    )
    if transfer.get("preserve_luminance", False):
        target_luma = max(float(_luminance(target_rgb)), 1.0)
        base_luma = _luminance(image)
        target_field = np.clip(
            target_rgb[None, None, :]
            * (
                np.maximum(base_luma, 20.0)[:, :, None]
                / target_luma
            ),
            0,
            255,
        )
    else:
        target_field = np.broadcast_to(target_rgb, image.shape)
    output = (
        image * (1.0 - alpha[:, :, None])
        + target_field * alpha[:, :, None]
    )
    mutable = alpha > 1e-8
    return output, mutable, {
        "operator_type": "scalar_transfer",
        "changed": True,
        "indicator_peak": round(float(indicator.max()), 8),
        "indicator_mean_in_region": round(
            float(indicator[target_mask > 0].mean()), 8
        ),
        "source_region": file_record(region_path, repo_root),
        "source_scalar": file_record(scalar_path, repo_root),
        "source_region_bbox_xyxy": source_region_bbox,
        "target_region_bbox_xyxy": [
            left_top,
            current_surface_y,
            right_top,
            bottom,
        ],
        "target_surface_derivation": surface_derivation,
        "source_liquid_top_delta_px": source_delta_y,
    }


def _project_geometry(
    geometry: dict[str, Any],
    mode: str,
    source_canvas: tuple[int, int],
    output_size: tuple[int, int],
    *,
    source_anchor_bbox: tuple[float, float, float, float] | None = None,
    target_anchor_bbox: list[int] | None = None,
) -> dict[str, Any]:
    sx = output_size[0] / source_canvas[0]
    sy = output_size[1] / source_canvas[1]

    def point(x: float, y: float) -> tuple[float, float]:
        if mode == "canvas_scale":
            return x * sx, y * sy
        if mode == "relative_to_anchor_object":
            assert source_anchor_bbox is not None
            assert target_anchor_bbox is not None
            u = (x - source_anchor_bbox[0]) / max(
                source_anchor_bbox[2] - source_anchor_bbox[0], 1e-6
            )
            v = (y - source_anchor_bbox[1]) / max(
                source_anchor_bbox[3] - source_anchor_bbox[1], 1e-6
            )
            return (
                target_anchor_bbox[0]
                + u * (target_anchor_bbox[2] - target_anchor_bbox[0]),
                target_anchor_bbox[1]
                + v * (target_anchor_bbox[3] - target_anchor_bbox[1]),
            )
        raise ValueError(f"unsupported object projection: {mode}")

    output = dict(geometry)
    if "bbox_xyxy" in geometry:
        x0, y0, x1, y1 = geometry["bbox_xyxy"]
        a = point(float(x0), float(y0))
        b = point(float(x1), float(y1))
        output["bbox_xyxy"] = [a[0], a[1], b[0], b[1]]
    elif "points" in geometry:
        output["points"] = [
            list(point(float(x), float(y)))
            for x, y in geometry["points"]
        ]
    else:
        raise ValueError("object geometry cannot be projected")
    return output


def _object_overlay(
    image: np.ndarray,
    contract: dict[str, Any],
    semantic: dict[str, Any],
    config: dict[str, Any],
    stage3_root: Path,
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload, path = _identity_payload(
        contract,
        semantic,
        config["identity_layer_id"],
        repo_root,
    )
    include = set(config["include_class_ids"])
    items = [
        item
        for item in payload["items"]
        if item.get("class_id") in include
    ]
    source_canvas = (
        int(semantic["canvas"]["width"]),
        int(semantic["canvas"]["height"]),
    )
    output_size = (image.shape[1], image.shape[0])
    projection = config["projection"]
    source_anchor_bbox = None
    target_anchor_bbox = None
    if projection["mode"] == "relative_to_anchor_object":
        source_anchor = _find_identity_object(
            payload, projection["anchor_class_id"]
        )
        source_anchor_bbox = _bbox(source_anchor["geometry"])
        gate = load_json(stage3_root / projection["geometry_gate"])
        target_anchor = next(
            item
            for item in gate["rendered_objects"]
            if item["class_id"] == projection["anchor_class_id"]
        )
        target_anchor_bbox = target_anchor["output_bbox_xyxy"]
    overlay = Image.new("RGBA", output_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    style = config["style"]
    alpha = int(round(float(style["alpha"]) * 255))
    fill = tuple(style["fill_rgb"]) + (alpha,)
    outline = tuple(style["outline_rgb"]) + (alpha,)
    projected_items = []
    for item in items:
        geometry = _project_geometry(
            item["geometry"],
            projection["mode"],
            source_canvas,
            output_size,
            source_anchor_bbox=source_anchor_bbox,
            target_anchor_bbox=target_anchor_bbox,
        )
        split_colors = style.get("split_fill_rgb")
        if split_colors:
            object_mask = Image.new("L", output_size, 0)
            mask_draw = ImageDraw.Draw(object_mask)
            if "bbox_xyxy" in geometry:
                mask_draw.ellipse(tuple(geometry["bbox_xyxy"]), fill=alpha)
            elif "points" in geometry:
                mask_draw.polygon(
                    [tuple(value) for value in geometry["points"]], fill=alpha
                )
            bounds = _bbox(geometry)
            axis = style.get("split_axis", "x")
            fraction = float(style.get("split_fraction", 0.5))
            colored = Image.new("RGBA", output_size, (0, 0, 0, 0))
            colored_draw = ImageDraw.Draw(colored)
            if axis == "x":
                split = bounds[0] + (bounds[2] - bounds[0]) * fraction
                rectangles = [
                    (bounds[0], bounds[1], split, bounds[3]),
                    (split, bounds[1], bounds[2], bounds[3]),
                ]
            elif axis == "y":
                split = bounds[1] + (bounds[3] - bounds[1]) * fraction
                rectangles = [
                    (bounds[0], bounds[1], bounds[2], split),
                    (bounds[0], split, bounds[2], bounds[3]),
                ]
            else:
                raise ValueError("split_axis must be x or y")
            for rectangle, rgb in zip(rectangles, split_colors):
                colored_draw.rectangle(rectangle, fill=tuple(rgb) + (alpha,))
            colored.putalpha(object_mask)
            overlay.alpha_composite(colored)
            if "bbox_xyxy" in geometry:
                draw.ellipse(
                    tuple(geometry["bbox_xyxy"]), outline=outline, width=2
                )
            else:
                draw.polygon(
                    [tuple(value) for value in geometry["points"]],
                    outline=outline,
                )
        elif style.get("facet_fill_rgb") and "points" in geometry:
            points = [tuple(value) for value in geometry["points"]]
            center = (
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            )
            facet_colors = style["facet_fill_rgb"]
            for facet_index, point in enumerate(points):
                next_point = points[(facet_index + 1) % len(points)]
                rgb = facet_colors[facet_index % len(facet_colors)]
                draw.polygon(
                    [center, point, next_point],
                    fill=tuple(rgb) + (alpha,),
                )
            draw.line(points + [points[0]], fill=outline, width=2)
            highlight = style.get("facet_highlight_rgb")
            if highlight:
                draw.line(
                    [points[0], center, points[-1]],
                    fill=tuple(highlight) + (alpha,),
                    width=1,
                )
        elif "bbox_xyxy" in geometry:
            box = tuple(geometry["bbox_xyxy"])
            draw.ellipse(box, fill=fill, outline=outline, width=2)
        elif "points" in geometry:
            draw.polygon(
                [tuple(value) for value in geometry["points"]],
                fill=fill,
                outline=outline,
            )
        projected_items.append(
            {
                "object_id": item["object_id"],
                "class_id": item["class_id"],
                "geometry": geometry,
            }
        )
    rgba = np.asarray(overlay, dtype=np.float32)
    a = rgba[..., 3] / 255.0
    output = image * (1 - a[..., None]) + rgba[..., :3] * a[..., None]
    return output, a > 0, {
        "operator_type": "object_overlay",
        "source_identity": file_record(path, repo_root),
        "object_count": len(items),
        "object_counts_by_class": {
            class_id: sum(
                item["class_id"] == class_id
                for item in projected_items
            )
            for class_id in sorted(include)
        },
        "projected_items": projected_items,
    }


def _mask_edges(mask: np.ndarray, width: int) -> np.ndarray:
    image = Image.fromarray(np.uint8(mask) * 255, mode="L")
    inner = (
        np.asarray(
            image.filter(ImageFilter.MinFilter(width)),
            dtype=np.float32,
        )
        / 255.0
    )
    return np.clip(mask.astype(np.float32) - inner, 0, 1)


def _region_fill(
    image: np.ndarray,
    contract: dict[str, Any],
    semantic: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    value, _, path = _load_layer_data(
        contract,
        semantic,
        config["region_layer_id"],
        repo_root,
    )
    mask = value > 0
    if mask.shape != image.shape[:2]:
        mask = (
            _resize_array(
                np.uint8(mask) * 255,
                (image.shape[1], image.shape[0]),
                nearest=True,
            )
            > 0
        )
    output = image.copy()
    alpha = float(config["alpha"])
    fill = np.asarray(config["fill_rgb"], dtype=np.float32)
    output[mask] = output[mask] * (1 - alpha) + fill * alpha
    edge = _mask_edges(mask, int(config["edge_width_px"]))
    output *= (
        1.0
        + edge[..., None] * float(config["inner_edge_gain"])
    )
    return output, mask, {
        "operator_type": "region_fill",
        "source_region": file_record(path, repo_root),
        "region_area_px": int(mask.sum()),
    }


def _raster_overlay(
    image: np.ndarray,
    contract: dict[str, Any],
    semantic: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Draw a typed semantic raster without interpreting screenshot edges."""
    value, record, path = _load_layer_data(
        contract, semantic, config["layer_id"], repo_root
    )
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("raster_overlay requires a two-dimensional layer")
    if array.shape != image.shape[:2]:
        array = _resize_array(
            array, (image.shape[1], image.shape[0]), nearest=True
        )
    threshold = float(config.get("threshold", 0.0))
    mask = array > threshold
    clip = config.get("clip_normalized_xyxy")
    if clip is not None:
        height, width = mask.shape
        x0 = int(round(float(clip[0]) * width))
        y0 = int(round(float(clip[1]) * height))
        x1 = int(round(float(clip[2]) * width))
        y1 = int(round(float(clip[3]) * height))
        allowed = np.zeros_like(mask)
        allowed[
            max(0, y0) : min(height, y1),
            max(0, x0) : min(width, x1),
        ] = True
        mask &= allowed
    width_px = int(config.get("width_px", 1))
    if width_px > 1:
        if width_px % 2 == 0:
            width_px += 1
        mask = np.asarray(
            Image.fromarray(np.uint8(mask) * 255).filter(
                ImageFilter.MaxFilter(width_px)
            )
        ) > 0
    output = image.copy()
    shadow_alpha = float(config.get("shadow_alpha", 0.0))
    shadow_mask = np.zeros_like(mask, dtype=np.float32)
    if shadow_alpha:
        shadow_mask = (
            np.asarray(
                Image.fromarray(np.uint8(mask) * 255).filter(
                    ImageFilter.GaussianBlur(
                        float(config.get("shadow_radius_px", 2.0))
                    )
                ),
                dtype=np.float32,
            )
            / 255.0
        )
        offset = config.get("shadow_offset_xy", [1, 1])
        shadow_mask = np.roll(
            shadow_mask,
            (int(offset[1]), int(offset[0])),
            axis=(0, 1),
        )
        shadow_rgb = np.asarray(
            config.get("shadow_rgb", [20, 25, 25]), dtype=np.float32
        )
        shadow_strength = np.clip(
            shadow_mask * shadow_alpha, 0.0, 1.0
        )
        output = (
            output * (1.0 - shadow_strength[..., None])
            + shadow_rgb * shadow_strength[..., None]
        )
    color = np.asarray(config["rgb"], dtype=np.float32)
    line_alpha = mask.astype(np.float32) * float(config.get("alpha", 1.0))
    output = output * (1.0 - line_alpha[..., None]) + color * line_alpha[..., None]
    return np.clip(output, 0, 255), mask | (shadow_mask > 0), {
        "operator_type": "raster_overlay",
        "source_layer_type": record["layer_type"],
        "source_raster": file_record(path, repo_root),
        "source_pixel_count": int((array > threshold).sum()),
        "rendered_pixel_count": int(mask.sum()),
        "clip_normalized_xyxy": clip,
        "width_px": width_px,
    }


def _connected_component_count(mask: np.ndarray) -> int:
    """Count 8-connected regions without adding a runtime dependency."""
    remaining = mask.astype(bool).copy()
    height, width = remaining.shape
    count = 0
    while remaining.any():
        y, x = np.argwhere(remaining)[0]
        count += 1
        queue: deque[tuple[int, int]] = deque([(int(y), int(x))])
        remaining[y, x] = False
        while queue:
            cy, cx = queue.popleft()
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if remaining[ny, nx]:
                        remaining[ny, nx] = False
                        queue.append((ny, nx))
    return count


def _region_material(
    image: np.ndarray,
    anchor: np.ndarray,
    contract: dict[str, Any],
    semantic: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fill a program region with appearance-only information from an anchor."""
    value, _, path = _load_layer_data(
        contract,
        semantic,
        config["region_layer_id"],
        repo_root,
    )
    mask = value > 0
    if mask.shape != image.shape[:2]:
        mask = (
            _resize_array(
                np.uint8(mask) * 255,
                (image.shape[1], image.shape[0]),
                nearest=True,
            )
            > 0
        )
    material_source_path = config.get("material_source_path")
    if material_source_path:
        material_path = repo_root / material_source_path
        material_anchor = np.asarray(
            Image.open(material_path)
            .convert("RGB")
            .resize(
                (anchor.shape[1], anchor.shape[0]),
                Image.Resampling.LANCZOS,
            ),
            dtype=np.float32,
        )
    else:
        material_path = None
        material_anchor = anchor
    mode = config["transfer_mode"]
    if mode == "raw_underlay":
        material = material_anchor.copy()
    elif mode == "translucent_tint":
        # Keep scene-scale reflection and bench detail from an empty-vessel
        # appearance anchor, then add only a restrained optical tint inside
        # the program-owned region.  This is useful for transparent media:
        # replacing every pixel with a flat RGB value makes a photograph look
        # like a diagram, while copying the raw anchor alone makes the liquid
        # invisible.
        tint = np.asarray(config["base_rgb"], dtype=np.float32)
        tint_alpha = float(config.get("tint_alpha", 0.2))
        if not 0.0 <= tint_alpha <= 1.0:
            raise ValueError("translucent_tint tint_alpha must be in [0, 1]")
        material = (
            material_anchor * (1.0 - tint_alpha) + tint * tint_alpha
        )
        depth_radius = float(config.get("depth_radius_px", 18.0))
        depth_gain = float(config.get("depth_gain", 0.0))
        if depth_gain:
            depth = (
                np.asarray(
                    Image.fromarray(np.uint8(mask) * 255).filter(
                        ImageFilter.GaussianBlur(depth_radius)
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )
            material += depth[..., None] * depth_gain
    elif mode in {"highpass_statistics", "shuffled_highpass_statistics"}:
        radius = float(config["highpass_radius_px"])
        blurred = np.asarray(
            Image.fromarray(
                np.uint8(np.clip(material_anchor, 0, 255)), mode="RGB"
            ).filter(ImageFilter.GaussianBlur(radius)),
            dtype=np.float32,
        )
        if mode == "shuffled_highpass_statistics":
            residual = _decorrelated_highpass(
                material_anchor,
                radius=radius,
                seed=int(config.get("shuffle_seed", 0)),
                smoothing_radius=float(
                    config.get("smoothing_radius_px", 0.65)
                ),
            )
        else:
            residual = material_anchor - blurred
            residual /= max(float(residual.std()), 1.0)
            residual = np.clip(residual, -2.5, 2.5)
        base_rgb = np.asarray(config["base_rgb"], dtype=np.float32)
        texture_gain = np.asarray(
            config["texture_gain_rgb"], dtype=np.float32
        )
        material = base_rgb[None, None, :] + residual * texture_gain
        mask_image = Image.fromarray(
            np.uint8(mask) * 255, mode="L"
        )
        depth = (
            np.asarray(
                mask_image.filter(
                    ImageFilter.GaussianBlur(
                        float(config["depth_radius_px"])
                    )
                ),
                dtype=np.float32,
            )
            / 255.0
        )
        material += (
            depth[..., None] * float(config["depth_gain"])
        )
    else:
        raise ValueError(f"unsupported region material mode: {mode}")

    output = image.copy()
    shadow_alpha = float(config.get("shadow_alpha", 0.0))
    if shadow_alpha:
        shadow = (
            np.asarray(
                Image.fromarray(np.uint8(mask) * 255).filter(
                    ImageFilter.GaussianBlur(
                        float(config.get("shadow_blur_radius_px", 2.0))
                    )
                ),
                dtype=np.float32,
            )
            / 255.0
        )
        offset = config.get("shadow_offset_xy", [2, 2])
        shadow = np.roll(
            shadow,
            (int(offset[1]), int(offset[0])),
            axis=(0, 1),
        )
        shadow_mix = np.clip(shadow * shadow_alpha, 0.0, 1.0)
        shadow_rgb = np.asarray(
            config.get("shadow_rgb", [52, 58, 56]), dtype=np.float32
        )
        output = (
            output * (1.0 - shadow_mix[..., None])
            + shadow_rgb * shadow_mix[..., None]
        )
    material_alpha = float(config.get("material_alpha", 1.0))
    if not 0.0 <= material_alpha <= 1.0:
        raise ValueError("region material_alpha must be in [0, 1]")
    output[mask] = (
        output[mask] * (1.0 - material_alpha)
        + material[mask] * material_alpha
    )
    membrane_width = int(config["membrane_width_px"])
    if membrane_width < 3 or membrane_width % 2 == 0:
        raise ValueError("membrane_width_px must be an odd integer >= 3")
    mask_image = Image.fromarray(np.uint8(mask) * 255, mode="L")
    outer = (
        np.asarray(
            mask_image.filter(ImageFilter.MaxFilter(membrane_width)),
            dtype=np.float32,
        )
        / 255.0
    )
    inner = (
        np.asarray(
            mask_image.filter(ImageFilter.MinFilter(membrane_width)),
            dtype=np.float32,
        )
        / 255.0
    )
    membrane = np.clip(outer - inner, 0, 1)
    membrane_color = np.asarray(
        config["membrane_rgb"], dtype=np.float32
    )
    alpha = membrane * float(config["membrane_alpha"])
    output = (
        output * (1.0 - alpha[..., None])
        + membrane_color[None, None, :] * alpha[..., None]
    )
    mutable = mask | (membrane > 0)
    return np.clip(output, 0, 255), mutable, {
        "operator_type": "region_material",
        "source_region": file_record(path, repo_root),
        "transfer_mode": mode,
        "region_area_px": int(mask.sum()),
        "connected_component_count": _connected_component_count(mask),
        "appearance_source_usage": (
            "full_rgb_underlay"
            if mode == "raw_underlay"
            else "anchor_reflections_plus_program_region_tint"
            if mode == "translucent_tint"
            else "normalized_high_frequency_statistics_only"
        ),
        "material_source": (
            file_record(material_path, repo_root)
            if material_path is not None
            else "appearance_anchor"
        ),
        "material_alpha": material_alpha,
        "shadow_alpha": shadow_alpha,
    }


def _identity_stroke(
    image: np.ndarray,
    contract: dict[str, Any],
    semantic: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Draw identity-preserving polylines from structured program objects."""
    payload, path = _identity_payload(
        contract,
        semantic,
        config["identity_layer_id"],
        repo_root,
    )
    source_canvas = (
        int(semantic["canvas"]["width"]),
        int(semantic["canvas"]["height"]),
    )
    output_size = (image.shape[1], image.shape[0])
    sx = output_size[0] / source_canvas[0]
    sy = output_size[1] / source_canvas[1]
    style_field = config["style_field"]
    styles = config["styles"]
    overlay = Image.new("RGBA", output_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    item_records = []
    parent_counts: Counter[str] = Counter()
    destination_counts: Counter[str] = Counter()
    lineage_unit_count = 0
    for item in payload["items"]:
        geometry = item["geometry"]
        if geometry.get("kind") != "polyline":
            raise ValueError("identity_stroke requires polyline geometry")
        style_key = str(item.get(style_field, "default"))
        if style_key not in styles:
            raise ValueError(f"missing identity style: {style_key}")
        style = styles[style_key]
        points = [
            (float(x) * sx, float(y) * sy)
            for x, y in geometry["points"]
        ]
        width = max(1, int(round(float(style["width_px"]) * min(sx, sy))))
        glow_width = max(
            width,
            int(round(float(style.get("glow_width_px", width)) * min(sx, sy))),
        )
        glow_rgb = tuple(style.get("glow_rgb", style["rgb"]))
        glow_alpha = int(round(float(style.get("glow_alpha", 0.0)) * 255))
        if glow_alpha:
            draw.line(
                points,
                fill=glow_rgb + (glow_alpha,),
                width=glow_width,
                joint="curve",
            )
        draw.line(
            points,
            fill=tuple(style["rgb"]) + (255,),
            width=width,
            joint="curve",
        )
        parent_id = item.get("parent_id")
        if parent_id:
            parent_counts[parent_id] += 1
        destination_counts[str(item.get("destination", "unassigned"))] += 1
        lineage_unit_count += int(style.get("lineage_units", 1))
        item_records.append(
            {
                "object_id": item["object_id"],
                "parent_id": parent_id,
                "destination": item.get("destination"),
                "style_key": style_key,
                "points": [[round(x, 4), round(y, 4)] for x, y in points],
            }
        )
    rgba = np.asarray(overlay, dtype=np.float32)
    alpha = rgba[..., 3] / 255.0
    output = image * (1 - alpha[..., None]) + rgba[..., :3] * alpha[..., None]
    return output, alpha > 0, {
        "operator_type": "identity_stroke",
        "source_identity": file_record(path, repo_root),
        "object_count": len(item_records),
        "object_ids": [item["object_id"] for item in item_records],
        "lineage_unit_count": lineage_unit_count,
        "children_per_parent": dict(sorted(parent_counts.items())),
        "destination_counts": dict(sorted(destination_counts.items())),
        "projected_items": item_records,
    }


def _polygon_mask(
    size: tuple[int, int], points: list[list[float]]
) -> np.ndarray:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).polygon(
        [tuple(point) for point in points], fill=255
    )
    return np.asarray(image, dtype=np.uint8) > 0


def _polygon_area(points: list[list[float]]) -> float:
    array = np.asarray(points, dtype=np.float64)
    x = array[:, 0]
    y = array[:, 1]
    return float(
        abs(
            np.dot(x, np.roll(y, -1))
            - np.dot(y, np.roll(x, -1))
        )
        * 0.5
    )


def _bilinear_rgb(image: np.ndarray, x: float, y: float) -> np.ndarray:
    x = float(np.clip(x, 0, image.shape[1] - 1))
    y = float(np.clip(y, 0, image.shape[0] - 1))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, image.shape[1] - 1)
    y1 = min(y0 + 1, image.shape[0] - 1)
    dx = x - x0
    dy = y - y0
    return (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x1] * dx * (1 - dy)
        + image[y1, x0] * (1 - dx) * dy
        + image[y1, x1] * dx * dy
    )


def _object_material(
    image: np.ndarray,
    anchor: np.ndarray,
    contract: dict[str, Any],
    semantic: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload, path = _identity_payload(
        contract,
        semantic,
        config["identity_layer_id"],
        repo_root,
    )
    include = set(config["include_class_ids"])
    items = [
        item
        for item in payload["items"]
        if item.get("class_id") in include
    ]
    palette = [
        np.asarray(value, dtype=np.float32)
        for value in config["palette_rgb"]
    ]
    texture_source_path = config.get("texture_source_path")
    if texture_source_path:
        texture_path = repo_root / texture_source_path
        texture_image = np.asarray(
            Image.open(texture_path)
            .convert("RGB")
            .resize((anchor.shape[1], anchor.shape[0]), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    else:
        texture_path = None
        texture_image = anchor
    texture = _luminance(texture_image)
    texture = (texture - texture.mean()) / max(float(texture.std()), 1.0)
    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    output = image.copy()
    masks = []
    samples: dict[str, list[float]] = {}
    expected_area = 0.0
    geometry_area = 0.0
    for index, item in enumerate(items):
        points = [
            [
                float(point[0])
                * image.shape[1]
                / semantic["canvas"]["width"],
                float(point[1])
                * image.shape[0]
                / semantic["canvas"]["height"],
            ]
            for point in item["geometry"]["points"]
        ]
        mask = _polygon_mask(
            (image.shape[1], image.shape[0]), points
        )
        masks.append(mask)
        shadow_alpha = float(config.get("shadow_alpha", 0.0))
        if shadow_alpha:
            shadow_image = Image.fromarray(np.uint8(mask) * 255).filter(
                ImageFilter.GaussianBlur(
                    float(config.get("shadow_blur_radius_px", 2.0))
                )
            )
            shadow = np.asarray(shadow_image, dtype=np.float32) / 255.0
            offset = config.get("shadow_offset_xy", [2, 2])
            shadow = np.roll(
                shadow,
                (int(offset[1]), int(offset[0])),
                axis=(0, 1),
            )
            alpha = np.clip(shadow * shadow_alpha, 0.0, 1.0)
            shadow_rgb = np.asarray(
                config.get("shadow_rgb", [52, 58, 56]), dtype=np.float32
            )
            output = (
                output * (1.0 - alpha[..., None])
                + shadow_rgb * alpha[..., None]
            )
        p0, p1, p2 = (
            np.asarray(points[i], dtype=np.float32) for i in range(3)
        )
        transform = np.column_stack((p1 - p0, p2 - p0))
        inverse = np.linalg.inv(transform)
        coordinates = np.stack(
            (xx.astype(np.float32) - p0[0], yy.astype(np.float32) - p0[1]),
            axis=-1,
        )
        uv = coordinates @ inverse.T
        tx = np.clip(
            np.rint(uv[..., 0] * (anchor.shape[1] - 1)),
            0,
            anchor.shape[1] - 1,
        ).astype(np.int32)
        ty = np.clip(
            np.rint(uv[..., 1] * (anchor.shape[0] - 1)),
            0,
            anchor.shape[0] - 1,
        ).astype(np.int32)
        local_texture = texture[ty, tx]
        value = np.clip(
            0.96
            + local_texture * float(config["texture_gain"]),
            0.72,
            1.18,
        )
        colored = palette[index % len(palette)][None, None, :] * value[..., None]
        bevel = _mask_edges(mask, int(config["bevel_width_px"]))
        colored += bevel[..., None] * float(config["bevel_gain"])
        facet_factors = config.get("facet_light_factors")
        if facet_factors and len(points) >= 3:
            center = [
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            ]
            for facet_index, point in enumerate(points):
                triangle = _polygon_mask(
                    (image.shape[1], image.shape[0]),
                    [center, point, points[(facet_index + 1) % len(points)]],
                )
                factor = float(
                    facet_factors[facet_index % len(facet_factors)]
                )
                colored[triangle] *= factor
        output[mask] = np.clip(colored[mask], 0, 255)
        fixed_uv = [
            (0.18, 0.18),
            (0.35, 0.18),
            (0.18, 0.35),
            (0.52, 0.18),
            (0.18, 0.52),
            (0.34, 0.34),
        ]
        sample_values = []
        for u, v in fixed_uv:
            point = p0 + u * (p1 - p0) + v * (p2 - p0)
            sample_values.append(
                round(
                    float(
                        _luminance(
                            _bilinear_rgb(
                                output,
                                float(point[0]),
                                float(point[1]),
                            )
                        )
                    ),
                    6,
                )
            )
        samples[item["object_id"]] = sample_values
        expected_area += float(item.get("area_px2", mask.sum()))
        geometry_area += _polygon_area(points)
    union = (
        np.any(np.stack(masks, axis=0), axis=0)
        if masks
        else np.zeros(image.shape[:2], dtype=bool)
    )
    interiors = [
        np.asarray(
            Image.fromarray(np.uint8(mask) * 255).filter(
                ImageFilter.MinFilter(3)
            )
        )
        > 0
        for mask in masks
    ]
    interior_sum = (
        np.stack(interiors, axis=0).sum(axis=0)
        if interiors
        else np.zeros(image.shape[:2], dtype=np.uint8)
    )
    rendered_area = int(sum(mask.sum() for mask in masks))
    return output, union, {
        "operator_type": "object_material",
        "source_identity": file_record(path, repo_root),
        "object_count": len(items),
        "object_ids": [item["object_id"] for item in items],
        "expected_area_px2": round(expected_area, 6),
        "geometry_area_px2": round(geometry_area, 6),
        "rendered_area_px": rendered_area,
        "area_relative_error": round(
            abs(geometry_area - expected_area)
            / max(expected_area, 1.0),
            8,
        ),
        "interior_overlap_area_px": int((interior_sum > 1).sum()),
        "material_coordinate_system": config[
            "material_coordinate_system"
        ],
        "texture_source": (
            file_record(texture_path, repo_root)
            if texture_path is not None
            else "appearance_anchor"
        ),
        "object_local_luminance_samples": samples,
        "shadow_alpha": float(config.get("shadow_alpha", 0.0)),
    }


def _height_normal(
    image: np.ndarray,
    contract: dict[str, Any],
    semantic: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, _, path = _load_layer_data(
        contract,
        semantic,
        config["height_layer_id"],
        repo_root,
    )
    height = height.astype(np.float32)
    if height.shape != image.shape[:2]:
        height = _resize_array(
            height, (image.shape[1], image.shape[0])
        )
    gy, gx = np.gradient(height)
    slope = float(config["slope"])
    normal = np.stack(
        (-gx * slope, -gy * slope, np.ones_like(height)),
        axis=-1,
    )
    normal /= np.maximum(
        np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6
    )
    light = np.asarray(config["light_xyz"], dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.clip(normal @ light, 0, 1)
    half_vector = light + np.array([0, 0, 1], dtype=np.float32)
    half_vector /= np.linalg.norm(half_vector)
    specular = (
        np.clip(normal @ half_vector, 0, 1)
        ** float(config["specular_power"])
    )
    intended = (
        (diffuse - float(diffuse.mean()))
        * float(config["diffuse_gain"])
        + specular * float(config["specular_gain"])
    )
    effected = np.clip(image + intended[..., None], 0, 255)
    retention = float(config["base_retention"])
    output = effected * retention + image * (1.0 - retention)
    realized = _luminance(output) - _luminance(image)
    correlation = float(
        np.corrcoef(intended.ravel(), realized.ravel())[0, 1]
    )
    mutable = np.abs(output - image).max(axis=2) > 0
    return output, mutable, {
        "operator_type": "height_normal",
        "source_height": file_record(path, repo_root),
        "height_min": round(float(height.min()), 8),
        "height_max": round(float(height.max()), 8),
        "program_shading_to_realized_luminance_correlation": round(
            correlation, 8
        ),
        "mean_abs_change_0_255": round(
            float(np.abs(output - image).mean()), 8
        ),
    }


def _state_value(state: dict[str, Any], dotted_path: str) -> float:
    value: Any = state
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"state field is missing: {dotted_path}")
        value = value[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"state field is not numeric: {dotted_path}")
    return float(value)


def _scalar_field_overlay(
    image: np.ndarray,
    contract: dict[str, Any],
    semantic: dict[str, Any],
    state: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Render a program scalar field as a soft volume or sparse streaks.

    The core does not attach a scientific meaning to the values.  A versioned
    Case plan supplies the color, threshold, optional state modulation and
    allowed normalized canvas region.
    """
    field, _, path = _load_layer_data(
        contract,
        semantic,
        config["scalar_layer_id"],
        repo_root,
    )
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError("scalar_field_overlay requires a two-dimensional field")
    if field.shape != image.shape[:2]:
        field = _resize_array(
            field,
            (image.shape[1], image.shape[0]),
        )
    value_min, value_max = [
        float(value) for value in config.get("value_range", [0.0, 1.0])
    ]
    if value_max <= value_min:
        raise ValueError("scalar overlay value_range must increase")
    normalized = np.clip(
        (field - value_min) / (value_max - value_min), 0.0, 1.0
    )
    threshold = float(config["threshold"])
    if not 0.0 <= threshold < 1.0:
        raise ValueError("scalar overlay threshold must be in [0, 1)")
    strength = np.clip(
        (normalized - threshold) / max(1.0 - threshold, 1e-6),
        0.0,
        1.0,
    ) ** float(config.get("gamma", 1.0))

    state_modulation = config.get("state_modulation")
    state_value = None
    state_factor = 1.0
    if state_modulation:
        state_value = _state_value(state, state_modulation["field"])
        lower = float(state_modulation["input_range"][0])
        upper = float(state_modulation["input_range"][1])
        if upper <= lower:
            raise ValueError("state modulation input_range must increase")
        state_factor = float(
            np.clip((state_value - lower) / (upper - lower), 0.0, 1.0)
        ) ** float(state_modulation.get("gamma", 1.0))
        strength *= state_factor

    height, width = field.shape
    clip = config.get("clip_normalized_xyxy", [0.0, 0.0, 1.0, 1.0])
    x0 = int(round(float(clip[0]) * width))
    y0 = int(round(float(clip[1]) * height))
    x1 = int(round(float(clip[2]) * width))
    y1 = int(round(float(clip[3]) * height))
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("invalid scalar overlay clip_normalized_xyxy")
    allowed = np.zeros((height, width), dtype=bool)
    allowed[y0:y1, x0:x1] = True
    strength *= allowed

    mode = config["render_mode"]
    color = np.asarray(config["color_rgb"], dtype=np.float32)
    maximum_alpha = float(config["maximum_alpha"])
    if not 0.0 <= maximum_alpha <= 1.0:
        raise ValueError("maximum_alpha must be in [0, 1]")
    record: dict[str, Any] = {
        "operator_type": "scalar_field_overlay",
        "source_scalar": file_record(path, repo_root),
        "render_mode": mode,
        "state_modulation_field": (
            state_modulation["field"] if state_modulation else None
        ),
        "state_modulation_value": state_value,
        "state_modulation_factor": round(state_factor, 8),
        "field_min": round(float(field.min()), 8),
        "field_max": round(float(field.max()), 8),
    }

    if mode == "soft_tint":
        alpha_image = Image.fromarray(
            np.uint8(np.rint(strength * maximum_alpha * 255)),
            mode="L",
        )
        blur_radius = float(config.get("blur_radius_px", 0.0))
        if blur_radius:
            alpha_image = alpha_image.filter(
                ImageFilter.GaussianBlur(blur_radius)
            )
        alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
        output = (
            image * (1.0 - alpha[..., None])
            + color[None, None, :] * alpha[..., None]
        )
        mutable = alpha > 0
        record["active_pixel_count"] = int(mutable.sum())
    elif mode == "streaks":
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        grid_x = int(config["grid_step_px"][0])
        grid_y = int(config["grid_step_px"][1])
        dx = int(config["line_delta_px"][0])
        dy = int(config["line_delta_px"][1])
        line_width = int(config["line_width_px"])
        if min(grid_x, grid_y, line_width) <= 0:
            raise ValueError("streak grid and line width must be positive")
        streak_count = 0
        for y in range(y0 + grid_y // 2, y1, grid_y):
            for x in range(x0 + grid_x // 2, x1, grid_x):
                local = float(strength[y, x])
                if local <= 0:
                    continue
                # Stable spatial thinning avoids a synthetic full grid while
                # keeping identical inputs byte-reproducible.
                hashed = ((x * 73856093) ^ (y * 19349663)) & 1023
                if hashed / 1023.0 > local:
                    continue
                alpha = int(round(255 * maximum_alpha * (0.35 + 0.65 * local)))
                draw.line(
                    (x, y, x + dx, y + dy),
                    fill=tuple(int(v) for v in color) + (alpha,),
                    width=line_width,
                )
                streak_count += 1
        rgba = np.asarray(overlay, dtype=np.float32)
        alpha = rgba[..., 3] / 255.0
        output = (
            image * (1.0 - alpha[..., None])
            + rgba[..., :3] * alpha[..., None]
        )
        mutable = alpha > 0
        record["streak_count"] = streak_count
        record["active_pixel_count"] = int(mutable.sum())
    else:
        raise ValueError(f"unsupported scalar field overlay mode: {mode}")

    weights = strength * allowed
    weight_sum = float(weights.sum())
    if weight_sum > 1e-8:
        yy, xx = np.mgrid[:height, :width]
        record["weighted_centroid_normalized_xy"] = [
            round(float((weights * xx).sum() / weight_sum / width), 8),
            round(float((weights * yy).sum() / weight_sum / height), 8),
        ]
    else:
        record["weighted_centroid_normalized_xy"] = None
    return np.clip(output, 0, 255), mutable, record


def render_plan(
    plan: dict[str, Any],
    stage3_root: Path,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    contract = load_json(stage3_root / plan["contract"])
    if contract["case_id"] != plan["case_id"]:
        raise ValueError("plan and contract Case IDs differ")
    anchor_path = stage3_root / plan["anchor"]["path"]
    anchor_image = _preprocess_anchor(
        anchor_path, plan["anchor"]["preprocess"]
    )
    anchor = np.asarray(anchor_image, dtype=np.float32)
    base = _base_canvas(anchor, plan.get("base_canvas"))
    anchor_sha = sha256_path(anchor_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_anchor_path = output_dir / "frozen_anchor.png"
    anchor_image.save(prepared_anchor_path, optimize=False)
    prepared_base_path = output_dir / "frozen_base_canvas.png"
    Image.fromarray(
        np.uint8(np.rint(np.clip(base, 0, 255))), mode="RGB"
    ).save(prepared_base_path, optimize=False)
    records = []
    for keyframe_id in plan["keyframe_ids"]:
        semantic, semantic_path, keyframe = keyframe_semantics(
            contract, keyframe_id, repo_root
        )
        state = load_json(repo_root / keyframe["state"]["path"])
        image = base.copy()
        mutable = np.zeros(image.shape[:2], dtype=bool)
        operator_records = []
        for operator in plan["operators"]:
            operator_type = operator["operator_type"]
            before = image.copy()
            if operator_type == "scalar_transfer":
                image, changed, metrics = _contained_scalar(
                    image,
                    anchor,
                    contract,
                    semantic,
                    keyframe_id,
                    operator["config"],
                    stage3_root,
                    repo_root,
                )
            elif operator_type == "object_overlay":
                image, changed, metrics = _object_overlay(
                    image,
                    contract,
                    semantic,
                    operator["config"],
                    stage3_root,
                    repo_root,
                )
            elif operator_type == "region_fill":
                image, changed, metrics = _region_fill(
                    image,
                    contract,
                    semantic,
                    operator["config"],
                    repo_root,
                )
            elif operator_type == "raster_overlay":
                image, changed, metrics = _raster_overlay(
                    image,
                    contract,
                    semantic,
                    operator["config"],
                    repo_root,
                )
            elif operator_type == "object_material":
                image, changed, metrics = _object_material(
                    image,
                    anchor,
                    contract,
                    semantic,
                    operator["config"],
                    repo_root,
                )
            elif operator_type == "height_normal":
                image, changed, metrics = _height_normal(
                    image,
                    contract,
                    semantic,
                    operator["config"],
                    repo_root,
                )
            elif operator_type == "region_material":
                image, changed, metrics = _region_material(
                    image,
                    anchor,
                    contract,
                    semantic,
                    operator["config"],
                    repo_root,
                )
            elif operator_type == "identity_stroke":
                image, changed, metrics = _identity_stroke(
                    image,
                    contract,
                    semantic,
                    operator["config"],
                    repo_root,
                )
            elif operator_type == "scalar_field_overlay":
                image, changed, metrics = _scalar_field_overlay(
                    image,
                    contract,
                    semantic,
                    state,
                    operator["config"],
                    repo_root,
                )
            else:
                raise ValueError(
                    f"unsupported operator {operator_type}"
                )
            metrics["operator_id"] = operator["operator_id"]
            metrics["mean_abs_change_from_previous_0_255"] = round(
                float(np.abs(image - before).mean()), 8
            )
            mutable |= changed
            operator_records.append(metrics)
        image = np.uint8(np.rint(np.clip(image, 0, 255)))
        frame_path = output_dir / "frames" / f"{keyframe_id}.png"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(
            frame_path, optimize=False
        )
        mutable_path = output_dir / "mutable" / f"{keyframe_id}.png"
        mutable_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(mutable) * 255, mode="L").save(
            mutable_path, optimize=False
        )
        # The saved base canvas is rounded to uint8.  Compare output pixels to
        # that exact raster, not to the pre-save float canvas; otherwise a
        # harmless fractional background value is misreported as a 1/255
        # outside-region mutation.
        base_pixels = np.uint8(np.rint(np.clip(base, 0, 255)))
        delta = np.abs(
            image.astype(np.int16) - base_pixels.astype(np.int16)
        )
        outside = ~mutable
        records.append(
            {
                "keyframe_id": keyframe_id,
                "source_state": keyframe["state"],
                "source_semantics": file_record(
                    semantic_path, repo_root
                ),
                "output": file_record(frame_path, repo_root),
                "mutable_mask": file_record(mutable_path, repo_root),
                "operator_records": operator_records,
                "metrics": {
                    "mutable_area_px": int(mutable.sum()),
                    "outside_mutable_max_difference_0_255": int(
                        delta[outside].max(initial=0)
                    ),
                    "inside_mutable_mean_difference_0_255": round(
                        float(delta[mutable].mean())
                        if mutable.any()
                        else 0.0,
                        8,
                    ),
                },
            }
        )
    manifest = {
        "schema_version": "1.0",
        "plan_id": plan["plan_id"],
        "case_id": plan["case_id"],
        "route_contract": "B_frozen_anchor_plus_program_state",
        "anchor": {
            "source": file_record(anchor_path, repo_root),
            "source_sha256_before_preprocess": anchor_sha,
            "preprocess": plan["anchor"]["preprocess"],
            "prepared": file_record(prepared_anchor_path, repo_root),
        },
        "base_canvas": {
            "spec": plan.get("base_canvas", {"mode": "anchor"}),
            "prepared": file_record(prepared_base_path, repo_root),
        },
        "operator_ids": [
            item["operator_id"] for item in plan["operators"]
        ],
        "records": records,
        "model_runs": {"image_candidates": 0, "video_candidates": 0},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
