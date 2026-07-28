from __future__ import annotations

import numpy as np
from PIL import Image

from modules.video_model.stage1.keyframe_render.enhance import project_texture


def test_projection_uses_exclusive_masks_and_reduces_boundary_weight() -> None:
    height, width = 48, 64
    base_array = np.zeros((height, width, 3), dtype=np.uint8)
    base_array[:, :32] = (80, 130, 90)
    base_array[:, 32:] = (50, 110, 150)
    proposal_array = np.random.default_rng(7).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8
    )
    masks = {
        "land": np.indices((height, width))[1] < 32,
        "water": np.indices((height, width))[1] >= 32,
    }
    output, metadata = project_texture(
        Image.fromarray(base_array),
        Image.fromarray(proposal_array),
        masks,
    )
    assert output.size == (width, height)
    assert metadata["exclusive_exhaustive_masks"]
    assert not metadata["proposal_color_layout_used"]
    assert all(
        record["boundary_model_weight"] < record["interior_model_weight"]
        for record in metadata["categories"].values()
    )


def test_projection_rejects_overlapping_masks() -> None:
    image = Image.new("RGB", (16, 16), "gray")
    masks = {
        "a": np.ones((16, 16), dtype=bool),
        "b": np.ones((16, 16), dtype=bool),
    }
    try:
        project_texture(image, image, masks)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping masks must be rejected")
