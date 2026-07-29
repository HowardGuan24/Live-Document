from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from modules.video_model.stage1.keyframe_render.sequence_pipeline.adapters.delta_causal import (
    load_selected_states,
)
from modules.video_model.stage1.keyframe_render.sequence_pipeline.cli import (
    prepare,
)
from modules.video_model.stage1.keyframe_render.sequence_pipeline.projection import (
    Projection,
)
from modules.video_model.stage1.keyframe_render.sequence_pipeline.schema import (
    default_output_root,
    load_spec,
)
from modules.video_model.stage1.keyframe_render.sequence_pipeline.semantic_builders.delta_causal import (
    new_land_layers,
)


SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "delta_sequence_spec.json"
)
SMOKE_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "delta_sequence_smoke_spec.json"
)


def _components(binary: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(binary.astype(np.uint8))
    return count - 1


def test_sequence_spec_has_four_ordered_mechanism_states() -> None:
    spec = load_spec(SPEC_PATH)
    assert len(spec["keyframes"]) == 4
    assert [item["display_frame"] for item in spec["keyframes"]] == [
        32,
        64,
        78,
        97,
    ]
    assert [item["state_frame"] for item in spec["keyframes"]] == [
        49,
        100,
        107,
        119,
    ]


def test_new_land_connectivity_matches_mechanism_story() -> None:
    spec = load_spec(SPEC_PATH)
    records, _ = load_selected_states(spec)
    projection = Projection(
        spec["projection"],
        (spec["canvas"]["width"], spec["canvas"]["height"]),
    )
    expected = {
        "decelerated_plume": 0,
        "underwater_accumulation": 0,
        "sandbar_emergence": 2,
        "rerouted_flow": 1,
    }
    for keyframe_id, component_count in expected.items():
        binary, _ = new_land_layers(
            records[keyframe_id]["new_land"], projection
        )
        assert _components(binary) == component_count


def test_projected_new_land_stays_in_sea_and_grows() -> None:
    spec = load_spec(SPEC_PATH)
    records, _ = load_selected_states(spec)
    projection = Projection(
        spec["projection"],
        (spec["canvas"]["width"], spec["canvas"]["height"]),
    )
    areas: list[int] = []
    for keyframe_id in ("sandbar_emergence", "rerouted_flow"):
        binary, _ = new_land_layers(
            records[keyframe_id]["new_land"], projection
        )
        ys, xs = np.where(binary > 0)
        assert len(xs) > 0
        assert int(xs.min()) >= int(round(projection.mouth_x))
        assert int(xs.max()) < projection.width
        assert int(ys.min()) >= 0
        assert int(ys.max()) < projection.height
        areas.append(int(binary.sum()))
    assert areas[1] > areas[0]


def test_schema_does_not_hardcode_adapter_name(tmp_path: Path) -> None:
    data = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    data["state_adapter"] = "another_program"
    data["semantic_builder"] = "another_program"
    data["composer"] = "another_strategy"
    path = tmp_path / "generic-spec.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_spec(path)
    assert loaded["state_adapter"] == "another_program"
    assert loaded["semantic_builder"] == "another_program"
    assert loaded["composer"] == "another_strategy"


def test_output_directory_is_config_driven() -> None:
    formal = load_spec(SPEC_PATH)
    smoke = load_spec(SMOKE_SPEC_PATH)
    assert default_output_root(formal).name == "delta_sequence"
    assert default_output_root(smoke).parts[-2:] == (
        "_smoke",
        "delta_sequence_prepare_smoke",
    )


def test_one_keyframe_smoke_prepare_writes_report_and_reuses(
    tmp_path: Path,
) -> None:
    spec = load_spec(SMOKE_SPEC_PATH)
    output_root = tmp_path / "smoke"
    first = prepare(spec, output_root, force=True)
    assert list(first["keyframes"]) == ["accumulation_start"]
    audit = output_root / "prepare-audit.html"
    assert audit.is_file()
    text = audit.read_text(encoding="utf-8")
    assert "display 40 / state 50" in text
    second = prepare(spec, output_root)
    assert second["cache"]["reused"]
