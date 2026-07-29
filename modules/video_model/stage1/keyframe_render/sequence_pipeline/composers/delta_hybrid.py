"""Compose delta semantic layers over a fixed visual anchor."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ...first_frame_test import _contact_sheet
from ..schema import resolve_stage_path
from ..utils import (
    image_record,
    save_gray,
    sha256,
    stable_hash,
    write_json,
)


def _load_gray(path: str) -> np.ndarray:
    return (
        np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255
    )


def _sand_texture(
    anchor: np.ndarray,
    state_frame: int,
    settings: dict[str, Any],
) -> np.ndarray:
    left, top, right, bottom = settings["sand_reference_box"]
    source = anchor[top:bottom, left:right]
    mean = np.mean(source, axis=(0, 1))
    std = np.std(source, axis=(0, 1))
    rng = np.random.default_rng(state_frame)
    noise = rng.normal(
        0.0, 1.0, (anchor.shape[0], anchor.shape[1])
    ).astype(np.float32)
    fine = cv2.GaussianBlur(noise, (0, 0), 2.2)
    broad = cv2.GaussianBlur(noise, (0, 0), 18.0)
    texture = fine * 0.65 + broad * 1.7
    texture /= max(float(np.std(texture)), 1e-6)
    wet_mix = float(settings["wet_sand_color_mix"])
    wet_mean = mean * (1.0 - wet_mix) + np.asarray(
        settings["wet_sand_color_rgb"], dtype=np.float32
    ) * wet_mix
    return np.clip(
        wet_mean[None, None, :]
        + texture[..., None]
        * std[None, None, :]
        * float(settings["wet_sand_texture_strength"]),
        0,
        255,
    )


def _colored_layer(
    color: tuple[float, float, float],
    alpha: np.ndarray,
) -> Image.Image:
    rgb = np.zeros((*alpha.shape, 3), dtype=np.float32)
    rgb[:] = np.asarray(color, dtype=np.float32)
    rgb *= alpha[..., None]
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 255)), mode="RGB")


def compose_sequence(
    spec: dict[str, Any],
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    prepare_path = output_root / "_work" / "manifests" / "prepare.json"
    if not prepare_path.is_file():
        raise FileNotFoundError("run --prepare before --compose")
    prepared = json.loads(prepare_path.read_text(encoding="utf-8"))
    anchor_path = resolve_stage_path(spec["paths"]["visual_anchor"])
    signature = stable_hash(
        {
            "prepare_input_signature": prepared.get("input_signature"),
            "anchor_sha256": sha256(anchor_path),
            "settings": spec["composite"],
            "code_sha256": sha256(Path(__file__)),
        }
    )
    manifest_path = (
        output_root / "_work" / "manifests" / "compose.json"
    )
    if manifest_path.is_file() and not force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        final_paths = [
            previous.get("keyframes", {})
            .get(item["id"], {})
            .get("final", {})
            .get("path")
            for item in spec["keyframes"]
        ]
        if (
            previous.get("input_signature") == signature
            and all(path and Path(path).is_file() for path in final_paths)
        ):
            previous["cache"] = {"reused": True}
            write_json(manifest_path, previous)
            return previous
    anchor_image = Image.open(anchor_path).convert("RGB")
    size = (int(spec["canvas"]["width"]), int(spec["canvas"]["height"]))
    if anchor_image.size != size:
        raise ValueError(
            f"visual anchor is {anchor_image.size}, expected {size}"
        )
    anchor = np.asarray(anchor_image, dtype=np.float32)
    settings = spec["composite"]
    luminance = (
        0.2126 * anchor[..., 0]
        + 0.7152 * anchor[..., 1]
        + 0.0722 * anchor[..., 2]
    )
    luminance_factor = np.clip(luminance / 128.0, 0.68, 1.22)
    composite_root = output_root / "_work" / "composite"
    review_root = output_root / "review" / "mechanism_constrained"
    final_root = output_root / "final"
    for path in (composite_root, review_root, final_root):
        path.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "status": "composed",
        "sequence_id": spec["sequence_id"],
        "classification": (
            "hybrid: fixed selected SDXL visual anchor plus deterministic "
            "mechanism-derived suspended sediment, underwater deposit, "
            "and emergent wet-sand layers"
        ),
        "raw_candidate_usage": (
            "The current-stage raw SDXL candidates are retained for "
            "comparison but are not copied into final pixels because they "
            "change the full scene. The Stage 1.2 selected model image is "
            "the visual anchor for every final frame."
        ),
        "input_signature": signature,
        "cache": {"reused": False},
        "anchor": image_record(anchor_path),
        "keyframes": {},
    }
    final_panels: list[tuple[str, Path]] = [
        (
            f"0 | {spec['anchor']['id']}",
            final_root / spec["anchor"]["output_filename"],
        )
    ]
    anchor_target = final_root / spec["anchor"]["output_filename"]
    if not anchor_target.is_file():
        shutil.copyfile(anchor_path, anchor_target)

    for index, keyframe in enumerate(spec["keyframes"], start=1):
        keyframe_id = keyframe["id"]
        entry = prepared["keyframes"][keyframe_id]
        layers = entry["semantic_layers"]
        density = _load_gray(layers["suspended_density"]["path"])
        deposit = _load_gray(layers["underwater_deposit"]["path"])
        new_land_binary = _load_gray(
            layers["new_land_binary"]["path"]
        )
        new_land_alpha = _load_gray(layers["new_land_alpha"]["path"])
        flow_paths = _load_gray(layers["flow_paths"]["path"])

        result = anchor.copy()
        deposit_color = (
            np.asarray(
                settings["deposit_color_rgb"], dtype=np.float32
            )
            * luminance_factor[..., None]
        )
        deposit_max_alpha = float(settings["deposit_max_alpha"])
        deposit_alpha = np.clip(
            deposit * deposit_max_alpha, 0.0, deposit_max_alpha
        )
        result = (
            result * (1.0 - deposit_alpha[..., None])
            + deposit_color * deposit_alpha[..., None]
        )

        sediment_color = (
            np.asarray(
                settings["sediment_color_rgb"], dtype=np.float32
            )
            * luminance_factor[..., None]
        )
        sediment_max_alpha = float(settings["sediment_max_alpha"])
        sediment_alpha = np.clip(
            density * sediment_max_alpha, 0.0, sediment_max_alpha
        )
        result = (
            result * (1.0 - sediment_alpha[..., None])
            + sediment_color * sediment_alpha[..., None]
        )

        flow_strength = float(
            keyframe.get("composite_overrides", {}).get(
                "flow_path_strength", 0.0
            )
        )
        flow_alpha = np.clip(
            flow_paths
            * float(settings["flow_path_max_alpha"])
            * flow_strength,
            0.0,
            1.0,
        )
        flow_noise = np.random.default_rng(
            int(keyframe["state_frame"]) + 901
        ).normal(0.0, 1.0, flow_paths.shape).astype(np.float32)
        flow_noise = cv2.GaussianBlur(flow_noise, (0, 0), 18.0)
        flow_noise /= max(float(flow_noise.std()), 1e-6)
        flow_alpha *= np.clip(0.88 + flow_noise * 0.06, 0.72, 1.0)
        flow_color = np.asarray(
            settings["flow_path_color_rgb"], dtype=np.float32
        )
        flow_color_mix = float(settings["flow_target_color_mix"])
        flow_target = (
            anchor * (1.0 - flow_color_mix)
            + flow_color * flow_color_mix
        )
        result = (
            result * (1.0 - flow_alpha[..., None])
            + flow_target * flow_alpha[..., None]
        )
        flow_highlight_alpha = (
            np.clip((flow_paths - 0.52) / 0.48, 0.0, 1.0)
            * float(settings["flow_highlight_max_alpha"])
            * flow_strength
        )
        flow_highlight = np.asarray(
            settings["flow_highlight_color_rgb"], dtype=np.float32
        )
        result = (
            result * (1.0 - flow_highlight_alpha[..., None])
            + flow_highlight * flow_highlight_alpha[..., None]
        )

        sand = _sand_texture(
            anchor, int(keyframe["state_frame"]), settings
        )
        land_max_alpha = float(settings["new_land_max_alpha"])
        land_alpha = np.clip(
            new_land_alpha * land_max_alpha, 0.0, land_max_alpha
        )
        result = (
            result * (1.0 - land_alpha[..., None])
            + sand * land_alpha[..., None]
        )
        binary = np.uint8(new_land_binary > 0.5)
        rim = cv2.dilate(binary, np.ones((7, 7), np.uint8)) - cv2.erode(
            binary, np.ones((5, 5), np.uint8)
        )
        rim_alpha = (
            np.float32(rim > 0)
            * new_land_alpha
            * float(settings["wet_rim_max_alpha"])
        )
        wet_rim = np.asarray(
            settings["wet_rim_color_rgb"], dtype=np.float32
        )
        result = (
            result * (1.0 - rim_alpha[..., None])
            + wet_rim * rim_alpha[..., None]
        )
        outer = cv2.dilate(
            binary, np.ones((13, 13), np.uint8)
        ) - cv2.dilate(binary, np.ones((3, 3), np.uint8))
        waterline_alpha = (
            cv2.GaussianBlur(np.float32(outer > 0), (0, 0), 2.0)
            * (1.0 - new_land_alpha)
            * float(settings["waterline_max_alpha"])
        )
        waterline = np.asarray(
            settings["waterline_color_rgb"], dtype=np.float32
        )
        result = (
            result * (1.0 - waterline_alpha[..., None])
            + waterline * waterline_alpha[..., None]
        )
        allowed = np.clip(
            np.maximum.reduce(
                (
                    deposit_alpha,
                    sediment_alpha,
                    flow_alpha,
                    flow_highlight_alpha,
                    land_alpha,
                    rim_alpha,
                    waterline_alpha,
                )
            ),
            0.0,
            1.0,
        )
        result[allowed <= 1e-6] = anchor[allowed <= 1e-6]
        result_uint8 = np.uint8(np.clip(result, 0, 255))

        output_path = review_root / f"{keyframe_id}.png"
        final_path = final_root / keyframe["output_filename"]
        Image.fromarray(result_uint8, mode="RGB").save(output_path)
        shutil.copyfile(output_path, final_path)

        allowed_path = composite_root / f"{keyframe_id}_allowed_region.png"
        difference_path = composite_root / f"{keyframe_id}_difference.png"
        deposit_color_path = composite_root / f"{keyframe_id}_deposit.png"
        sediment_color_path = (
            composite_root / f"{keyframe_id}_suspended.png"
        )
        sand_path = composite_root / f"{keyframe_id}_new_land_texture.png"
        flow_path = composite_root / f"{keyframe_id}_flow_paths.png"
        waterline_path = composite_root / f"{keyframe_id}_waterline.png"
        save_gray(allowed_path, allowed)
        difference = np.max(
            np.abs(result_uint8.astype(np.float32) - anchor), axis=2
        )
        save_gray(difference_path, np.clip(difference / 96.0, 0.0, 1.0))
        _colored_layer(
            tuple(settings["deposit_color_rgb"]), deposit_alpha
        ).save(
            deposit_color_path
        )
        _colored_layer(
            tuple(settings["sediment_color_rgb"]), sediment_alpha
        ).save(
            sediment_color_path
        )
        _colored_layer(
            tuple(settings["flow_path_color_rgb"]),
            np.maximum(flow_alpha, flow_highlight_alpha),
        ).save(flow_path)
        _colored_layer(
            tuple(settings["waterline_color_rgb"]), waterline_alpha
        ).save(waterline_path)
        Image.fromarray(
            np.uint8(np.clip(sand * land_alpha[..., None], 0, 255)),
            mode="RGB",
        ).save(sand_path)
        static = allowed <= 1e-6
        static_difference = np.abs(
            result_uint8.astype(np.int16) - anchor.astype(np.int16)
        )[static]
        land_region = new_land_binary > 0.5
        near_ring = (
            cv2.dilate(
                np.uint8(land_region), np.ones((25, 25), np.uint8)
            )
            > 0
        ) & ~(
            cv2.dilate(
                np.uint8(land_region), np.ones((7, 7), np.uint8)
            )
            > 0
        )
        if land_region.any() and near_ring.any():
            land_mean = result_uint8[land_region].mean(axis=0)
            ring_mean = result_uint8[near_ring].mean(axis=0)
            new_land_rgb_contrast = float(
                np.linalg.norm(land_mean - ring_mean)
            )
        else:
            new_land_rgb_contrast = 0.0
        flow_region = flow_paths > 0.20
        flow_difference = np.max(
            np.abs(result_uint8.astype(np.float32) - anchor), axis=2
        )
        mean_flow_path_difference = (
            float(flow_difference[flow_region].mean())
            if flow_region.any() and flow_strength > 0
            else 0.0
        )
        manifest["keyframes"][keyframe_id] = {
            "display_frame": keyframe["display_frame"],
            "state_frame": keyframe["state_frame"],
            "meaning": keyframe["meaning"],
            "inputs": layers,
            "method": {
                "underwater_deposit": (
                    f"Blend toward RGB{tuple(settings['deposit_color_rgb'])}, "
                    "modulated by anchor luminance, at maximum alpha "
                    f"{settings['deposit_max_alpha']}."
                ),
                "suspended_sediment": (
                    f"Blend toward RGB{tuple(settings['sediment_color_rgb'])}, "
                    "modulated by anchor luminance, at maximum alpha "
                    f"{settings['sediment_max_alpha']}."
                ),
                "new_land": (
                    "Generate deterministic wet-sand texture from the "
                    "anchor's existing sand statistics and composite only "
                    "inside the projected new_land region."
                ),
                "visible_flow_paths": (
                    "Trace two lanes from mechanism flow samples around "
                    "emergent land, then blend only water color and soft "
                    "surface highlights without drawing arrows."
                ),
            },
            "allowed_region": image_record(
                allowed_path,
                meaning="只有白色区域允许相对视觉锚点发生变化",
            ),
            "difference": image_record(
                difference_path,
                meaning="越亮表示最终关键帧相对视觉锚点变化越大",
            ),
            "colored_layers": {
                "underwater_deposit": image_record(deposit_color_path),
                "suspended_sediment": image_record(
                    sediment_color_path
                ),
                "new_land_texture": image_record(sand_path),
                "visible_flow_paths": image_record(flow_path),
                "waterline": image_record(waterline_path),
            },
            "result": image_record(
                output_path,
                classification="mechanism-constrained hybrid keyframe",
            ),
            "final": image_record(final_path),
            "static_region_max_difference_0_255": (
                int(static_difference.max())
                if static_difference.size
                else 0
            ),
            "new_land_rgb_contrast": round(
                new_land_rgb_contrast, 3
            ),
            "flow_path_strength": flow_strength,
            "visible_flow_path_count": layers["flow_paths"][
                "path_count"
            ],
            "mean_flow_path_difference_0_255": round(
                mean_flow_path_difference, 3
            ),
        }
        final_panels.append((f"{index} | {keyframe_id}", final_path))

    _contact_sheet(
        final_panels,
        output_root / "sequence-contact-sheet.jpg",
        columns=3,
    )
    write_json(manifest_path, manifest)
    return manifest
