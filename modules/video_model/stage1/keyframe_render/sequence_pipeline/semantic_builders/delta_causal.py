"""Build visible semantic layers for the delta-causal state adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..projection import Projection
from ..utils import image_record, save_gray


def suspended_density(
    particles: list[dict[str, Any]],
    projection: Projection,
    new_land_alpha: np.ndarray,
) -> np.ndarray:
    density = np.zeros(
        (projection.height, projection.width), dtype=np.float32
    )
    for particle in particles:
        px, py = projection.particle_xy(
            float(particle["x"]), float(particle["y"])
        )
        x = int(np.clip(round(px), 0, projection.width - 1))
        y = int(np.clip(round(py), 0, projection.height - 1))
        density[y, x] += 1.0
    density = cv2.GaussianBlur(
        density,
        (0, 0),
        sigmaX=20,
        sigmaY=24,
        borderType=cv2.BORDER_REFLECT,
    )
    positive = density[density > 0]
    scale = float(np.quantile(positive, 0.99)) if positive.size else 1.0
    density = np.clip(density / max(scale, 1e-8), 0.0, 1.0)
    density = np.power(density, 0.72)
    return (
        density
        * projection.water_region()
        * (1.0 - np.clip(new_land_alpha, 0.0, 1.0))
    )


def new_land_layers(
    field: list[list[int]], projection: Projection
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(field, dtype=np.float32)
    projected = projection.project_sea_field(
        source, interpolation=cv2.INTER_LINEAR
    )
    softened = cv2.GaussianBlur(projected, (0, 0), 2.0)
    binary = np.float32(softened >= 0.22)
    alpha = cv2.GaussianBlur(binary, (0, 0), 2.2)
    return binary, np.clip(alpha, 0.0, 1.0)


def underwater_deposit(
    field: list[list[float]],
    projection: Projection,
    maximum_thickness: float,
    new_land_alpha: np.ndarray,
) -> np.ndarray:
    source = np.asarray(field, dtype=np.float32)
    projected = projection.project_sea_field(
        source, interpolation=cv2.INTER_CUBIC
    )
    normalized = np.clip(
        projected / max(maximum_thickness, 1e-8), 0.0, 1.0
    )
    normalized = np.power(normalized, 0.58)
    normalized = cv2.GaussianBlur(normalized, (0, 0), 4.0)
    return (
        normalized
        * projection.water_region()
        * (1.0 - np.clip(new_land_alpha, 0.0, 1.0))
    )


def flow_audit_image(
    samples: list[list[float]],
    projection: Projection,
) -> Image.Image:
    image = Image.new("RGB", (projection.width, projection.height), "#102d36")
    draw = ImageDraw.Draw(image)
    for x, y, flow_x, flow_y, speed in samples:
        if float(speed) <= 0:
            continue
        start = projection.particle_xy(float(x), float(y))
        end_projection = projection.particle_xy(
            float(x) + float(flow_x) * 6,
            float(y) + float(flow_y) * 6,
        )
        dx = end_projection[0] - start[0]
        dy = end_projection[1] - start[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        scale = 15.0 / length
        end = (start[0] + dx * scale, start[1] + dy * scale)
        color = (
            int(70 + 130 * min(1.0, float(speed) / 1.34)),
            210,
            225,
        )
        draw.line((*start, *end), fill=color, width=2)
    return image


def build_semantic_layers(
    record: dict[str, Any],
    projection: Projection,
    output_root: Path,
    maximum_thickness: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    binary_land, land_alpha = new_land_layers(
        record["new_land"], projection
    )
    density = suspended_density(
        record["particles"], projection, land_alpha
    )
    deposit = underwater_deposit(
        record["thickness"],
        projection,
        maximum_thickness,
        land_alpha,
    )
    layer_root = output_root / "_work" / "semantic_layers"
    paths = {
        "suspended_density": (
            layer_root / "suspended_density" / f"{record['id']}.png"
        ),
        "underwater_deposit": (
            layer_root / "underwater_deposit" / f"{record['id']}.png"
        ),
        "new_land_binary": (
            layer_root / "new_land" / f"{record['id']}_binary.png"
        ),
        "new_land_alpha": (
            layer_root / "new_land" / f"{record['id']}_alpha.png"
        ),
        "flow_audit": (
            layer_root / "flow_audit" / f"{record['id']}.png"
        ),
    }
    save_gray(paths["suspended_density"], density)
    save_gray(paths["underwater_deposit"], deposit)
    save_gray(paths["new_land_binary"], binary_land)
    save_gray(paths["new_land_alpha"], land_alpha)
    paths["flow_audit"].parent.mkdir(parents=True, exist_ok=True)
    flow_audit_image(record["flow_samples"], projection).save(
        paths["flow_audit"]
    )
    manifest = {
        "suspended_density": image_record(
            paths["suspended_density"],
            meaning="白色越亮，表示该处悬浮泥沙越集中",
            model_input=False,
        ),
        "underwater_deposit": image_record(
            paths["underwater_deposit"],
            meaning="白色越亮，表示水底累积厚度相对越大",
            normalization_maximum=maximum_thickness,
            model_input=False,
        ),
        "new_land_binary": image_record(
            paths["new_land_binary"],
            meaning="白色表示机制状态中已经露出水面的新生陆地",
            model_input=False,
        ),
        "new_land_alpha": image_record(
            paths["new_land_alpha"],
            meaning="新生陆地合成时使用的柔和边界",
            model_input=False,
        ),
        "flow_audit": image_record(
            paths["flow_audit"],
            meaning="只用于核对水流是否绕行，不输入模型也不进入成品",
            model_input=False,
        ),
    }
    arrays = {
        "suspended_density": density,
        "underwater_deposit": deposit,
        "new_land_binary": binary_land,
        "new_land_alpha": land_alpha,
        "geometry_source": np.maximum(
            np.asarray(record["land"], dtype=np.uint8),
            np.asarray(record["new_land"], dtype=np.uint8),
        ),
        "hard_boundary": binary_land,
    }
    return manifest, arrays


def prepare_context(
    records: dict[str, dict[str, Any]],
    spec: dict[str, Any],
    projection: Projection,
) -> dict[str, Any]:
    del spec, projection
    return {
        "maximum_thickness_for_normalization": max(
            float(record["stats"]["max_thickness"])
            for record in records.values()
        )
    }


def build_layers(
    record: dict[str, Any],
    projection: Projection,
    output_root: Path,
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    return build_semantic_layers(
        record,
        projection,
        output_root,
        float(context["maximum_thickness_for_normalization"]),
    )
