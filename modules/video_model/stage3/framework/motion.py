"""Case-agnostic motion-contract prompt compilation and video measurements."""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


def compile_motion_prompt(
    segment: dict[str, Any],
    *,
    guidance_level: str,
) -> dict[str, str]:
    """Compile L0 or L1 text without knowing a case ID."""

    if guidance_level == "L0":
        positive = segment["brief_prompt"]
        negative = (
            "camera movement, scene change, abrupt cut, flicker, "
            "jitter, watermark"
        )
    elif guidance_level == "L1":
        inventory = ", ".join(
            f"exactly {item['count']} {item['class'].replace('_', ' ')}"
            for item in segment["entities"]
        )
        events = " Then ".join(segment["ordered_events"])
        trends = "; ".join(segment["state_trends"])
        invariants = ", ".join(segment["invariants"])
        positive = (
            f"{segment['scene_lock']} Keep {inventory}. {events} "
            f"State constraints: {trends}. Preserve {invariants}. "
            "End exactly at the supplied last frame. One continuous "
            "physically or mathematically understandable transition. "
            "Locked camera."
        )
        negative = (
            ", ".join(segment["forbidden"])
            + ", camera movement, pan, zoom, tilt, perspective change, "
            "scene change, abrupt cut, flicker, jitter, ghosting, "
            "uncontracted object birth or loss, watermark"
        )
    else:
        raise ValueError(f"prompt compiler does not accept {guidance_level}")
    return {"positive": positive, "negative": negative}


def decode_video(path: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = [
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(video=0)
        ]
        fps = float(stream.average_rate)
        return (
            {
                "width": int(stream.width),
                "height": int(stream.height),
                "fps": fps,
                "frame_count": len(frames),
                "duration_seconds": len(frames) / fps,
            },
            frames,
        )


def encode_video(
    frames: list[np.ndarray],
    path: Path,
    *,
    fps: float,
) -> None:
    import av

    if not frames:
        raise ValueError("cannot encode an empty frame sequence")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(
            "libx264", rate=Fraction(str(fps)).limit_denominator(1000)
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for array in frames:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def concatenate_segment_videos(
    sources: list[Path],
    output: Path,
) -> dict[str, Any]:
    """Join FLF segments and remove duplicated boundary frames."""

    all_frames: list[np.ndarray] = []
    records = []
    fps: float | None = None
    for index, source in enumerate(sources):
        info, frames = decode_video(source)
        if fps is None:
            fps = info["fps"]
        elif abs(info["fps"] - fps) > 1e-6:
            raise ValueError("segment FPS mismatch")
        if index:
            frames = frames[1:]
        records.append(
            {
                "source": str(source),
                "source_frame_count": info["frame_count"],
                "kept_frame_count": len(frames),
            }
        )
        all_frames.extend(frames)
    assert fps is not None
    encode_video(all_frames, output, fps=fps)
    return {
        "segments": records,
        "assembled_frame_count": len(all_frames),
        "fps": fps,
        "boundary_policy": "drop the first frame of every segment after the first",
    }


def fitted_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        ImageOps.fit(
            Image.open(path).convert("RGB"),
            size,
            method=Image.Resampling.BICUBIC,
            centering=(0.5, 0.5),
        )
    )


def _components(
    mask: np.ndarray,
    minimum_pixels: int,
    *,
    connectivity: int = 4,
) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    values: list[np.ndarray] = []
    neighbors = (
        ((-1, 0), (1, 0), (0, -1), (0, 1))
        if connectivity == 4
        else (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    )
    for start_y, start_x in zip(*np.nonzero(mask & ~seen)):
        if seen[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        seen[start_y, start_x] = True
        points: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            points.append((y, x))
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                    and not seen[ny, nx]
                ):
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if len(points) >= minimum_pixels:
            values.append(np.asarray(points, dtype=np.int32))
    return values


def audit_scalar_decay(
    frames: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    height, width = frames[0].shape[:2]
    x0, y0, x1, y1 = config["roi_normalized_xyxy"]
    box = (
        int(round(x0 * width)),
        int(round(y0 * height)),
        int(round(x1 * width)),
        int(round(y1 * height)),
    )
    counts = []
    scores = []
    for frame in frames:
        crop = frame[box[1] : box[3], box[0] : box[2]].astype(np.int16)
        mask = (
            (crop[:, :, 0] >= int(config["r_min"]))
            & (
                crop[:, :, 0] - crop[:, :, 1]
                >= int(config["minimum_r_minus_g"])
            )
            & (
                crop[:, :, 2] - crop[:, :, 1]
                >= int(config["minimum_b_minus_g"])
            )
        )
        counts.append(int(mask.sum()))
        score = np.maximum(
            crop[:, :, 0] - crop[:, :, 1]
            - int(config["minimum_r_minus_g"]),
            0,
        )
        scores.append(float(score[mask].sum()))
    initial = max(counts[0], 1)
    peak_fraction = max(counts) / initial
    final_fraction = counts[-1] / initial
    step_limit = (
        float(config["maximum_step_increase_fraction_of_initial"])
        * initial
    )
    maximum_step = max(
        (right - left for left, right in zip(counts, counts[1:])),
        default=0,
    )
    passed = (
        counts[0] >= int(config["minimum_initial_pixels"])
        and final_fraction
        <= float(config["maximum_final_fraction_of_initial"])
        and peak_fraction
        <= float(config["maximum_peak_fraction_of_initial"])
        and maximum_step <= step_limit
    )
    return {
        "name": "localized_scalar_mass_decays_without_regrowth",
        "passed": passed,
        "roi_pixel_xyxy": list(box),
        "colored_pixel_counts": counts,
        "integrated_color_scores": scores,
        "initial_count": counts[0],
        "final_fraction_of_initial": round(final_fraction, 6),
        "peak_fraction_of_initial": round(peak_fraction, 6),
        "maximum_step_increase_pixels": int(maximum_step),
        "allowed_step_increase_pixels": round(step_limit, 3),
    }


def audit_fixed_sources(
    frames: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    height, width = frames[0].shape[:2]
    radius = int(config["source_search_radius_px"])
    color = config["source_rgb_ranges"]
    records = []
    maximum_drift = 0.0
    all_visible = True
    for frame_index, frame in enumerate(frames):
        detected = []
        for nx, ny in config["source_centers_normalized_xy"]:
            expected_x, expected_y = nx * width, ny * height
            left = max(0, int(round(expected_x)) - radius)
            right = min(width, int(round(expected_x)) + radius + 1)
            top = max(0, int(round(expected_y)) - radius)
            bottom = min(height, int(round(expected_y)) + radius + 1)
            crop = frame[top:bottom, left:right]
            mask = (
                (crop[:, :, 0] >= int(color["r_min"]))
                & (crop[:, :, 1] >= int(color["g_min"]))
                & (crop[:, :, 1] <= int(color["g_max"]))
                & (crop[:, :, 2] <= int(color["b_max"]))
            )
            ys, xs = np.nonzero(mask)
            visible = len(xs) >= int(config["minimum_source_pixels"])
            all_visible = all_visible and visible
            if visible:
                center_x = float(xs.mean() + left)
                center_y = float(ys.mean() + top)
                drift = float(
                    np.hypot(center_x - expected_x, center_y - expected_y)
                )
                maximum_drift = max(maximum_drift, drift)
                center = [round(center_x, 3), round(center_y, 3)]
            else:
                drift = None
                center = None
            detected.append(
                {
                    "visible": visible,
                    "detected_xy": center,
                    "pixel_count": int(len(xs)),
                    "drift_px": round(drift, 4)
                    if drift is not None
                    else None,
                }
            )
        records.append({"frame": frame_index, "sources": detected})
    passed = all_visible and maximum_drift <= float(
        config["maximum_source_drift_px"]
    )
    return {
        "name": "two_declared_sources_remain_visible_and_fixed",
        "passed": passed,
        "all_visible": all_visible,
        "maximum_drift_px": round(maximum_drift, 4),
        "threshold_px": config["maximum_source_drift_px"],
        "records": records,
    }


def audit_radial_progress(
    frames: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Estimate outward front extent from high-pass energy around each center."""

    height, width = frames[0].shape[:2]
    y_grid, x_grid = np.mgrid[:height, :width]
    sample_indices = np.linspace(
        0, len(frames) - 1, min(9, len(frames)), dtype=int
    ).tolist()
    center_extents: list[list[float]] = [
        [] for _ in config["source_centers_normalized_xy"]
    ]
    for frame_index in sample_indices:
        gray = frames[frame_index].astype(np.float32).mean(axis=2)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        energy = gx + gy
        for center_index, (nx, ny) in enumerate(
            config["source_centers_normalized_xy"]
        ):
            radius = np.hypot(x_grid - nx * width, y_grid - ny * height)
            radial = []
            for value in range(12, int(min(width, height) * 0.42)):
                annulus = (radius >= value - 1) & (radius < value + 1)
                radial.append((value, float(np.mean(energy[annulus]))))
            values = np.asarray([item[1] for item in radial])
            threshold = max(
                float(np.percentile(values, 70)),
                float(np.median(values) + 0.35 * np.std(values)),
            )
            active = [value for value, score in radial if score >= threshold]
            center_extents[center_index].append(
                float(max(active) if active else 12)
            )
    progress = [
        values[-1] - values[0] for values in center_extents
    ]
    passed = all(
        value >= float(config["minimum_outward_progress_px"])
        for value in progress
    )
    return {
        "name": "radial_field_reaches_farther_from_both_sources",
        "passed": passed,
        "sample_indices": sample_indices,
        "extent_by_center_px": center_extents,
        "progress_by_center_px": progress,
        "minimum_progress_px": config["minimum_outward_progress_px"],
        "note_zh": "对每个固定点源做同心圆高通能量扫描，比较首尾最外显著波纹半径；它验证总体向外传播，不宣称恢复完整相位。",
    }


def _triangle_color_masks(frame: np.ndarray) -> list[np.ndarray]:
    rgb = frame.astype(np.int16)
    red = (
        (rgb[:, :, 0] - rgb[:, :, 1] >= 25)
        & (rgb[:, :, 0] - rgb[:, :, 2] >= 20)
    )
    orange = (
        (rgb[:, :, 0] - rgb[:, :, 2] >= 25)
        & (rgb[:, :, 1] - rgb[:, :, 2] >= 12)
        & (rgb[:, :, 0] >= rgb[:, :, 1])
    )
    teal = (
        (rgb[:, :, 1] - rgb[:, :, 0] >= 8)
        & (rgb[:, :, 2] - rgb[:, :, 0] >= 5)
    )
    blue = (
        (rgb[:, :, 2] - rgb[:, :, 0] >= 12)
        & (rgb[:, :, 2] - rgb[:, :, 1] >= 5)
    )
    return [red & ~orange, orange, teal & ~blue, blue]


def audit_rigid_identity(
    frames: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    areas_by_identity: list[list[int]] = [[], [], [], []]
    component_counts: list[list[int]] = [[], [], [], []]
    for frame in frames:
        for index, mask in enumerate(_triangle_color_masks(frame)):
            components = _components(mask, minimum_pixels=30)
            areas_by_identity[index].append(
                int(max((len(item) for item in components), default=0))
            )
            component_counts[index].append(len(components))
    deviations = []
    identities_present = []
    for areas in areas_by_identity:
        baseline = max(areas[0], 1)
        deviations.append(
            max(abs(value - baseline) / baseline for value in areas)
        )
        identities_present.append(all(value > 0 for value in areas))
    threshold = float(config["maximum_area_deviation_fraction"])
    passed = all(identities_present) and max(deviations) <= threshold
    return {
        "name": "four_colored_rigid_identities_preserve_area",
        "passed": passed,
        "largest_component_area_by_identity": areas_by_identity,
        "component_count_by_identity": component_counts,
        "maximum_area_deviation_fraction_by_identity": [
            round(value, 6) for value in deviations
        ],
        "threshold": threshold,
        "note_zh": "按四种冻结颜色分别追踪最大连通块；若对象形变、褪色、复制或融合，面积或连通块数量会偏离。",
    }


def audit_object_division(
    frames: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Audit one colored region dividing while identity marks stay allocated."""
    region_counts = []
    identity_counts = []
    identity_pixels = []
    left_identity_pixels = []
    right_identity_pixels = []
    for frame in frames:
        rgb = frame.astype(np.int16)
        region_color = config["region_color"]
        region_mask = (
            (rgb[:, :, 1] >= int(region_color["g_min"]))
            & (
                rgb[:, :, 1] - rgb[:, :, 0]
                >= int(region_color["minimum_g_minus_r"])
            )
            & (
                rgb[:, :, 1] - rgb[:, :, 2]
                >= int(region_color["minimum_g_minus_b"])
            )
        )
        region_components = _components(
            region_mask,
            minimum_pixels=int(config["minimum_region_component_pixels"]),
            connectivity=8,
        )
        region_counts.append(len(region_components))

        identity_color = config["identity_color"]
        identity_mask = (
            (rgb[:, :, 0] >= int(identity_color["r_min"]))
            & (rgb[:, :, 2] >= int(identity_color["b_min"]))
            & (
                rgb[:, :, 0] - rgb[:, :, 1]
                >= int(identity_color["minimum_r_minus_g"])
            )
            & (
                rgb[:, :, 2] - rgb[:, :, 1]
                >= int(identity_color["minimum_b_minus_g"])
            )
        )
        identity_components = _components(
            identity_mask,
            minimum_pixels=int(
                config["minimum_identity_component_pixels"]
            ),
        )
        identity_counts.append(len(identity_components))
        identity_pixels.append(int(identity_mask.sum()))
        midpoint = identity_mask.shape[1] // 2
        left_identity_pixels.append(
            int(identity_mask[:, :midpoint].sum())
        )
        right_identity_pixels.append(
            int(identity_mask[:, midpoint:].sum())
        )

    division_indices = [
        index for index, value in enumerate(region_counts) if value == 2
    ]
    first_division = (
        division_indices[0] if division_indices else None
    )
    topology_valid = (
        region_counts[0] == 1
        and region_counts[-1] == 2
        and all(value in (1, 2) for value in region_counts)
        and first_division is not None
        and all(value == 2 for value in region_counts[first_division:])
    )
    initial_identity = max(identity_pixels[0], 1)
    identity_mass_ratio = [
        value / initial_identity for value in identity_pixels
    ]
    mass_valid = (
        min(identity_mass_ratio)
        >= float(config["minimum_identity_mass_fraction"])
        and max(identity_mass_ratio)
        <= float(config["maximum_identity_mass_fraction"])
    )
    final_allocation = (
        left_identity_pixels[-1]
        >= int(config["minimum_final_identity_pixels_per_side"])
        and right_identity_pixels[-1]
        >= int(config["minimum_final_identity_pixels_per_side"])
    )
    return {
        "name": "one_region_divides_once_and_identity_marks_reach_both_sides",
        "passed": topology_valid and mass_valid and final_allocation,
        "region_component_count_by_frame": region_counts,
        "identity_component_count_by_frame": identity_counts,
        "identity_pixel_count_by_frame": identity_pixels,
        "identity_mass_fraction_of_initial": [
            round(value, 6) for value in identity_mass_ratio
        ],
        "left_identity_pixels_by_frame": left_identity_pixels,
        "right_identity_pixels_by_frame": right_identity_pixels,
        "first_two_region_frame": first_division,
        "subchecks": {
            "topology_valid": topology_valid,
            "identity_mass_valid": mass_valid,
            "final_left_right_allocation_valid": final_allocation,
        },
        "note_zh": (
            "绿色大连通区代表程序细胞区域；洋红色连通块代表身份标记。"
            "该门禁验证一个区域只分裂一次、遗传物质没有整体消失或爆增，"
            "并在终点分配到左右两侧；它不把像素连通块数量冒充染色体生物学计数。"
        ),
    }


def audit_advected_scalar_event(
    frames: list[np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Track one colored identity and a localized transient scalar event."""
    height, width = frames[0].shape[:2]
    px0, py0, px1, py1 = config["parcel_roi_normalized_xyxy"]
    parcel_box = (
        int(round(px0 * width)),
        int(round(py0 * height)),
        int(round(px1 * width)),
        int(round(py1 * height)),
    )
    rx0, ry0, rx1, ry1 = config["event_roi_normalized_xyxy"]
    event_box = (
        int(round(rx0 * width)),
        int(round(ry0 * height)),
        int(round(rx1 * width)),
        int(round(ry1 * height)),
    )
    parcel_centers: list[list[float] | None] = []
    parcel_pixels: list[int] = []
    event_pixels: list[int] = []
    event_centers: list[list[float] | None] = []
    parcel_color = config["parcel_color"]
    event_color = config["event_color"]
    for frame in frames:
        rgb = frame.astype(np.int16)
        parcel_crop = rgb[
            parcel_box[1] : parcel_box[3], parcel_box[0] : parcel_box[2]
        ]
        parcel_mask = (
            (parcel_crop[:, :, 0] >= int(parcel_color["r_min"]))
            & (parcel_crop[:, :, 1] >= int(parcel_color["g_min"]))
            & (parcel_crop[:, :, 1] <= int(parcel_color["g_max"]))
            & (parcel_crop[:, :, 2] <= int(parcel_color["b_max"]))
            & (
                parcel_crop[:, :, 0] - parcel_crop[:, :, 1]
                >= int(parcel_color["minimum_r_minus_g"])
            )
        )
        ys, xs = np.nonzero(parcel_mask)
        parcel_pixels.append(int(len(xs)))
        parcel_centers.append(
            [
                round(float(xs.mean() + parcel_box[0]), 4),
                round(float(ys.mean() + parcel_box[1]), 4),
            ]
            if len(xs)
            else None
        )

        event_crop = rgb[event_box[1] : event_box[3], event_box[0] : event_box[2]]
        event_mask = (
            (event_crop[:, :, 2] >= int(event_color["b_min"]))
            & (
                event_crop[:, :, 2] - event_crop[:, :, 0]
                >= int(event_color["minimum_b_minus_r"])
            )
            & (
                event_crop[:, :, 1] - event_crop[:, :, 0]
                >= int(event_color["minimum_g_minus_r"])
            )
        )
        ys, xs = np.nonzero(event_mask)
        event_pixels.append(int(len(xs)))
        event_centers.append(
            [
                round(float(xs.mean() + event_box[0]), 4),
                round(float(ys.mean() + event_box[1]), 4),
            ]
            if len(xs)
            else None
        )

    visible = all(
        value >= int(config["minimum_parcel_pixels"])
        for value in parcel_pixels
    )
    x_values = [
        float(value[0]) for value in parcel_centers if value is not None
    ]
    progress = x_values[-1] - x_values[0] if len(x_values) == len(frames) else 0.0
    maximum_backward_step = max(
        (left - right for left, right in zip(x_values, x_values[1:])),
        default=0.0,
    )
    parcel_passed = (
        visible
        and progress >= float(config["minimum_parcel_progress_px"])
        and maximum_backward_step <= float(config["maximum_parcel_backward_step_px"])
    )

    peak_index = int(np.argmax(event_pixels))
    peak_count = max(event_pixels)
    initial_fraction = event_pixels[0] / max(peak_count, 1)
    final_fraction = event_pixels[-1] / max(peak_count, 1)
    peak_fraction = peak_index / max(len(frames) - 1, 1)
    peak_center = event_centers[peak_index]
    event_passed = (
        peak_count >= int(config["minimum_peak_event_pixels"])
        and initial_fraction <= float(config["maximum_initial_fraction_of_peak"])
        and final_fraction <= float(config["maximum_final_fraction_of_peak"])
        and float(config["peak_time_fraction_range"][0])
        <= peak_fraction
        <= float(config["peak_time_fraction_range"][1])
        and peak_center is not None
        and peak_center[0] / width
        <= float(config["maximum_peak_centroid_normalized_x"])
    )
    return {
        "name": "identity_advects_and_localized_scalar_event_rises_then_falls",
        "passed": parcel_passed and event_passed,
        "parcel_pixel_count_by_frame": parcel_pixels,
        "parcel_center_xy_by_frame": parcel_centers,
        "parcel_progress_px": round(progress, 4),
        "maximum_parcel_backward_step_px": round(maximum_backward_step, 4),
        "event_pixel_count_by_frame": event_pixels,
        "event_center_xy_by_frame": event_centers,
        "peak_event_frame": peak_index,
        "peak_time_fraction": round(peak_fraction, 6),
        "initial_fraction_of_peak": round(initial_fraction, 6),
        "final_fraction_of_peak": round(final_fraction, 6),
        "subchecks": {
            "parcel_visible_and_advects": parcel_passed,
            "localized_event_rises_on_declared_side_then_falls": event_passed,
        },
        "note_zh": (
            "在冻结颜色范围内追踪同一身份标记的质心，并在声明区域统计短暂标量事件。"
            "它检查方向、事件峰值、空间侧别和终点衰减，不把整幅图亮度变化冒充机制。"
        ),
    }


def endpoint_metrics(
    frames: list[np.ndarray],
    first_path: Path,
    last_path: Path,
) -> dict[str, Any]:
    height, width = frames[0].shape[:2]
    references = [
        fitted_rgb(first_path, (width, height)),
        fitted_rgb(last_path, (width, height)),
    ]
    output = {}
    for label, frame, reference in (
        ("first", frames[0], references[0]),
        ("last", frames[-1], references[1]),
    ):
        delta = frame.astype(np.float32) - reference.astype(np.float32)
        output[label] = {
            "mean_absolute_pixel_error_0_255": round(
                float(np.mean(np.abs(delta))), 4
            )
        }
    return output


def consecutive_metrics(frames: list[np.ndarray]) -> dict[str, float]:
    values = [
        float(
            np.mean(
                np.abs(
                    right.astype(np.float32) - left.astype(np.float32)
                )
            )
        )
        for left, right in zip(frames, frames[1:])
    ]
    return {
        "mean": round(float(np.mean(values)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "maximum": round(float(max(values)), 4),
    }


def audit_sparse_checkpoints(
    frames: list[np.ndarray],
    checkpoints: list[tuple[int, Path]],
    *,
    maximum_mae_0_255: float,
) -> dict[str, Any]:
    height, width = frames[0].shape[:2]
    records = []
    for frame_index, path in checkpoints:
        reference = fitted_rgb(path, (width, height))
        mae = float(
            np.mean(
                np.abs(
                    frames[frame_index].astype(np.float32)
                    - reference.astype(np.float32)
                )
            )
        )
        records.append(
            {
                "frame_index": frame_index,
                "reference": str(path),
                "mean_absolute_pixel_error_0_255": round(mae, 4),
            }
        )
    maximum = max(
        item["mean_absolute_pixel_error_0_255"] for item in records
    )
    return {
        "name": "declared_sparse_checkpoint_frames_are_followed",
        "passed": maximum <= maximum_mae_0_255,
        "records": records,
        "maximum_mae_0_255": maximum,
        "threshold": maximum_mae_0_255,
        "note_zh": "在合同时间点把模型帧与同一张已验收关键帧逐像素比较；用于判断分段引导是否真的提高中间状态忠实度。",
    }


def audit_video(
    video_path: Path,
    *,
    first_path: Path,
    last_path: Path,
    config: dict[str, Any],
    sparse_checkpoints: list[tuple[int, Path]] | None = None,
) -> dict[str, Any]:
    info, frames = decode_video(video_path)
    endpoints = endpoint_metrics(frames, first_path, last_path)
    consecutive = consecutive_metrics(frames)
    endpoint_limit = float(config["maximum_endpoint_mae_0_255"])
    jump_limit = float(config["maximum_consecutive_frame_mae_0_255"])
    checks = [
        {
            "name": "first_and_last_frames_follow_inputs",
            "passed": all(
                value["mean_absolute_pixel_error_0_255"]
                <= endpoint_limit
                for value in endpoints.values()
            ),
            "evidence": {
                "metrics": endpoints,
                "maximum_mae_0_255": endpoint_limit,
            },
        },
        {
            "name": "no_abrupt_pixel_jump",
            "passed": consecutive["maximum"] <= jump_limit,
            "evidence": {
                **consecutive,
                "maximum_allowed": jump_limit,
            },
        },
    ]
    audit_type = config["audit_type"]
    if audit_type == "scalar_decay":
        checks.append(audit_scalar_decay(frames, config))
    elif audit_type == "radial_field":
        checks.extend(
            [
                audit_fixed_sources(frames, config),
                audit_radial_progress(frames, config),
            ]
        )
    elif audit_type == "rigid_identity":
        checks.append(audit_rigid_identity(frames, config))
    elif audit_type == "object_division":
        checks.append(audit_object_division(frames, config))
    elif audit_type == "advected_scalar_event":
        checks.append(audit_advected_scalar_event(frames, config))
    else:
        raise ValueError(f"unknown G4 audit type: {audit_type}")
    if sparse_checkpoints:
        checks.append(
            audit_sparse_checkpoints(
                frames,
                sparse_checkpoints,
                maximum_mae_0_255=endpoint_limit,
            )
        )
    return {
        "video": info,
        "endpoint_metrics": endpoints,
        "consecutive_metrics": consecutive,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
