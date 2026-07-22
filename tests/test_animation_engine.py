import json
import shutil
from pathlib import Path

import pytest

from modules.animation_engine.schema import DSLValidationError, load_animation_spec, parse_animation_spec


ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "modules" / "animation_engine" / "examples" / "data_flow.json"


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


@pytest.mark.integration
def test_end_to_end_render():
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
    result = generate_animation(EXAMPLE, ROOT / "outputs" / "test_runs")
    assert result["status"] == "completed"
    assert Path(result["outputs"]["mp4"]).stat().st_size > 0
    assert Path(result["outputs"]["gif"]).stat().st_size > 0
    assert result["metrics"]["gif_frames"] > 1
