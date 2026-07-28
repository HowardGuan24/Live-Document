import json

import numpy as np

from modules.video_model.stage1.causal_delta.config import MECHANISM_ROOT
from modules.video_model.stage1.causal_delta.validate import load_states
from modules.video_model.stage1.keyframe_render.transport_pair_test import (
    SETTINGS,
    _select_mechanism_states,
    _soft_sediment_density,
    natural_sparse_canny,
)


def test_natural_sparse_canny_is_binary_and_sparse() -> None:
    image = natural_sparse_canny()
    array = np.asarray(image)
    assert image.size == (SETTINGS["width"], SETTINGS["height"])
    assert set(np.unique(array)) == {0, 255}
    edge_fraction = float((array > 0).mean())
    assert 0.001 < edge_fraction < 0.02


def test_natural_sparse_canny_has_river_and_coast_segments() -> None:
    array = np.asarray(natural_sparse_canny()) > 0
    width = SETTINGS["width"]
    assert array[:, : int(width * 0.2)].any()
    assert array[:, int(width * 0.4) : int(width * 0.48)].any()
    assert not array[:, int(width * 0.65) :].any()


def test_selected_mechanism_states_stop_before_coast() -> None:
    selections = _select_mechanism_states()
    first = selections["in_channel"]
    second = selections["at_outlet"]
    assert first["particles_at_or_beyond_coast"] == 0
    assert second["particles_at_or_beyond_coast"] == 0
    assert first["particle_max_x"] < second["particle_max_x"]
    assert second["particle_max_x"] < second["coastline_x"]
    assert first["state_stats"]["underwater_deposit_cells"] == 0
    assert second["state_stats"]["underwater_deposit_cells"] == 0


def test_soft_sediment_front_advances_to_outlet() -> None:
    states = load_states(MECHANISM_ROOT / "states.jsonl")
    config = json.loads(
        (MECHANISM_ROOT / "simulation_config.json").read_text(
            encoding="utf-8"
        )
    )
    size = (SETTINGS["width"], SETTINGS["height"])
    first = _soft_sediment_density(states[21], config, size)
    second = _soft_sediment_density(states[27], config, size)
    first_x = np.where((first > 0.1).any(axis=0))[0]
    second_x = np.where((second > 0.1).any(axis=0))[0]
    mouth_x = int(round(size[0] * 0.43))
    assert int(first_x.max()) < mouth_x - 70
    assert int(second_x.max()) >= mouth_x
    assert int(second_x.max()) > int(first_x.max()) + 80
