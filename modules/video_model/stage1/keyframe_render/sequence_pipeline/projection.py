"""Configurable projection from mechanism coordinates to visual canvas."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class Projection:
    def __init__(self, config: dict[str, Any], size: tuple[int, int]):
        if config["type"] != "piecewise_river_to_sea":
            raise ValueError(f"unsupported projection: {config['type']}")
        self.config = config
        self.width, self.height = size
        self.mechanism_width = int(config["mechanism_width"])
        self.mechanism_height = int(config["mechanism_height"])
        self.coastline_x = float(config["coastline_x"])
        self.center_y = float(config["river_center_y"])
        self.half_width = float(config["river_half_width"])
        self.mouth_x = float(config["mouth_x_ratio"]) * self.width
        self.river_x = np.linspace(0.0, self.mouth_x, 512)
        curve = config["river_curve"]
        center = np.full(
            self.river_x.shape,
            float(curve["center_y_ratio"]) * self.height,
        )
        for component in curve["sinusoids"]:
            center += float(component["amplitude"]) * np.sin(
                self.river_x / float(component["period"])
                + float(component["phase"])
            )
        progress = self.river_x / self.mouth_x
        half = float(curve["half_width_start"]) + (
            float(curve["half_width_end"])
            - float(curve["half_width_start"])
        ) * progress ** float(curve["half_width_power"])
        self.river_center = center
        self.river_half_width = half

    def particle_xy(self, x: float, y: float) -> tuple[float, float]:
        if x <= self.coastline_x:
            px = np.clip(
                x / self.coastline_x * self.mouth_x,
                0,
                self.mouth_x,
            )
            center = float(
                np.interp(px, self.river_x, self.river_center)
            )
            half = float(
                np.interp(px, self.river_x, self.river_half_width)
            )
            lateral = np.clip(
                (y - self.center_y) / self.half_width, -1.5, 1.5
            )
            return float(px), center + float(lateral) * half * 0.78
        sea_progress = (x - self.coastline_x) / (
            self.mechanism_width - self.coastline_x
        )
        px = self.mouth_x + sea_progress * (self.width - self.mouth_x)
        py = y / self.mechanism_height * self.height
        return float(px), float(py)

    def project_sea_field(
        self,
        field: np.ndarray,
        *,
        interpolation: int,
    ) -> np.ndarray:
        if field.shape != (
            self.mechanism_height,
            self.mechanism_width,
        ):
            raise ValueError(
                f"unexpected mechanism field shape: {field.shape}"
            )
        coast = int(round(self.coastline_x))
        sea = field[:, coast:]
        sea_width = self.width - int(round(self.mouth_x))
        resized = cv2.resize(
            sea.astype(np.float32),
            (sea_width, self.height),
            interpolation=interpolation,
        )
        result = np.zeros((self.height, self.width), dtype=np.float32)
        result[:, self.width - sea_width :] = resized
        return result

    def water_region(self) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        upper = self.river_center - self.river_half_width
        lower = self.river_center + self.river_half_width
        polygon = np.vstack(
            (
                np.column_stack((self.river_x, upper)),
                np.column_stack((self.river_x[::-1], lower[::-1])),
            )
        )
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
        mask[:, int(round(self.mouth_x)) :] = 255
        return cv2.GaussianBlur(mask, (0, 0), 2.5).astype(np.float32) / 255

