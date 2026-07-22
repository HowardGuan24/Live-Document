import json
import shutil
from pathlib import Path

import pytest

from modules.animation_engine.schema import DSLValidationError, load_animation_spec, parse_animation_spec


ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "modules" / "animation_engine" / "examples" / "data_flow.json"
FORMULA_EXAMPLE = ROOT / "modules" / "animation_engine" / "examples" / "formula_derivation.json"


def test_example_dsl_is_valid():
    spec = load_animation_spec(EXAMPLE)

    assert spec.id == "data_flow_demo"
    assert len(spec.objects) == 10
    assert spec.timeline[-1].action == "wait"
    assert spec.output.formats == ["mp4", "gif"]


def test_legacy_explanation_spec_is_normalized():
    spec = parse_animation_spec(
        {
            "source_text": "Gradient descent follows the negative gradient.",
            "goal": "Show the parameter approaching a minimum.",
            "objects": ["curve", "point"],
            "steps": ["show_curve", "move_parameter"],
        }
    )

    assert spec.id == "legacy_explanation"
    assert any(item.id == "title" for item in spec.objects)
    assert any(action.action == "highlight" for action in spec.timeline)


def test_unknown_action_is_rejected():
    with pytest.raises(DSLValidationError, match="unsupported"):
        parse_animation_spec(
            {
                "objects": [{"id": "dot", "type": "point"}],
                "timeline": [{"action": "teleport", "target": "dot"}],
            }
        )


def test_unknown_reference_is_rejected():
    with pytest.raises(DSLValidationError, match="unknown target"):
        parse_animation_spec(
            {
                "objects": [{"id": "dot", "type": "point"}],
                "timeline": [{"action": "create", "target": "missing"}],
            }
        )


def test_move_requires_destination():
    with pytest.raises(DSLValidationError, match="to is required"):
        parse_animation_spec(
            {
                "objects": [{"id": "dot", "type": "point"}],
                "timeline": [{"action": "move", "target": "dot"}],
            }
        )


def test_formula_derivation_dsl_is_valid():
    spec = load_animation_spec(FORMULA_EXAMPLE)

    formula_objects = [item for item in spec.objects if item.type == "formula"]
    actions = [action.action for action in spec.timeline if action.action]
    assert len(formula_objects) == 4
    assert "formula_transform" in actions or any(
        child.action == "formula_transform"
        for action in spec.timeline
        for child in action.parallel
    )
    assert "highlight_parts" in actions


def test_formula_transform_requires_formula_objects():
    with pytest.raises(DSLValidationError, match="must be a formula object"):
        parse_animation_spec(
            {
                "objects": [
                    {"id": "source", "type": "text", "content": "x = 1"},
                    {"id": "target", "type": "text", "content": "x = 2"},
                ],
                "timeline": [
                    {
                        "action": "formula_transform",
                        "target": "source",
                        "replacement": "target",
                    }
                ],
            }
        )


def test_highlight_parts_requires_non_empty_string_list():
    with pytest.raises(DSLValidationError, match="non-empty array of strings"):
        parse_animation_spec(
            {
                "objects": [
                    {
                        "id": "equation",
                        "type": "formula",
                        "content": "x = 1",
                        "render_mode": "text",
                    }
                ],
                "timeline": [
                    {"action": "highlight_parts", "target": "equation", "parts": []}
                ],
            }
        )


@pytest.mark.integration
@pytest.mark.parametrize("example", [EXAMPLE, FORMULA_EXAMPLE])
def test_end_to_end_render(example):
    pytest.importorskip("manim")
    try:
        import imageio_ffmpeg

        imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        if shutil.which("ffmpeg") is None:
            pytest.skip("FFmpeg is unavailable")

    from modules.animation_engine.pipeline import generate_animation

    # Keep render artifacts under the repository because some sandboxed Windows
    # environments deny access to pytest's system temporary directory.
    result = generate_animation(example, ROOT / "outputs" / "test_runs")
    assert result["status"] == "completed"
    assert Path(result["outputs"]["mp4"]).stat().st_size > 0
    assert Path(result["outputs"]["gif"]).stat().st_size > 0
    assert result["metrics"]["gif_frames"] > 1
