import numpy as np

from modules.video_model.stage1.keyframe_render.first_frame_test import (
    SOURCE_DISPLAY_FRAME,
    SOURCE_STATE_FRAME,
    _load_source_state,
    _plume_density,
)


def test_first_frame_state_is_exact_early_stage() -> None:
    state, selection, config = _load_source_state()
    assert selection["display_frame"] == SOURCE_DISPLAY_FRAME == 25
    assert selection["state_frame"] == SOURCE_STATE_FRAME == 36
    assert selection["state_stats"]["suspended_particles"] == 516
    assert selection["state_stats"]["settled_particles"] == 2
    assert selection["state_stats"]["underwater_deposit_cells"] == 50
    assert selection["state_stats"]["new_land_cells"] == 0
    assert len(state["particles"]) == 516
    assert config.grid_width == 96
    assert config.grid_height == 64


def test_plume_density_is_continuous_and_bounded() -> None:
    state, _, config = _load_source_state()
    density = _plume_density(state, config)
    assert density.shape == (64, 96)
    assert np.isfinite(density).all()
    assert float(density.min()) >= 0.0
    assert float(density.max()) <= 1.0
    assert np.count_nonzero(density > 0.05) > 50
