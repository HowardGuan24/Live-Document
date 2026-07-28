import numpy as np

from modules.video_model.stage1.keyframe_render.controlnet_line_test import (
    SETTINGS,
    detailed_lineart,
)


def test_detailed_lineart_is_binary_and_has_expected_size() -> None:
    image = detailed_lineart()
    array = np.asarray(image)
    assert image.size == (SETTINGS["width"], SETTINGS["height"])
    assert set(np.unique(array)) == {0, 255}


def test_detailed_lineart_has_structural_information_without_filling_canvas() -> None:
    array = np.asarray(detailed_lineart()) > 0
    edge_fraction = float(array.mean())
    assert 0.01 < edge_fraction < 0.15
    assert array[:, :100].any()
    assert array[:, int(SETTINGS["width"] * 0.7) :].any()
