from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from modules.video_model.generate_explainer_video import DEFAULT_MODEL, run
from modules.video_model.prompting import InputSpec, build_prompts, validate_input


ROOT = Path(__file__).resolve().parents[1]


class PromptingTests(unittest.TestCase):
    def test_prompts_use_all_required_fields(self) -> None:
        spec = InputSpec(
            source_text="SOURCE_SENTINEL",
            visual_goal="GOAL_SENTINEL",
            video_prompt="PROMPT_SENTINEL",
            content_type="data_flow",
        )
        positive, negative = build_prompts(spec)
        self.assertIn("SOURCE_SENTINEL", positive)
        self.assertIn("GOAL_SENTINEL", positive)
        self.assertIn("PROMPT_SENTINEL", positive)
        self.assertIn("source", positive.lower())
        self.assertIn("camera", negative.lower())

    def test_aliases_are_normalized(self) -> None:
        data = {
            "source_text": "a",
            "visual_goal": "b",
            "video_prompt": "c",
            "content_type": "dataflow",
        }
        self.assertEqual(validate_input(data).content_type, "data_flow")

    def test_empty_required_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_text"):
            validate_input({"source_text": "", "visual_goal": "b", "video_prompt": "c"})


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe not installed")
class PipelineTests(unittest.TestCase):
    def make_args(self, output_dir: Path, **overrides: object) -> Namespace:
        values = {
            "input": str(ROOT / "samples" / "gradient_descent.json"),
            "output_dir": str(output_dir),
            "backend": "procedural",
            "model_id": DEFAULT_MODEL,
            "duration": 3.0,
            "fps": 8,
            "width": 320,
            "height": 256,
            "seed": 42,
            "inference_steps": 2,
            "guidance_scale": 5.0,
            "cpu_offload": False,
            "loop_mode": "none",
            "no_gif": False,
            "run_name": "test-run",
            "overwrite": False,
            "failure_note": None,
        }
        values.update(overrides)
        return Namespace(**values)

    def test_procedural_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "outputs"
            args = self.make_args(output_dir)
            code, metadata_path = run(args)
            self.assertEqual(code, 0)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "success")
            self.assertEqual(metadata["chosen_workflow"]["backend"], "procedural")
            self.assertTrue(metadata["quality_review"]["automated_checks"]["duration_in_range"])
            for key in ("video", "webm", "gif"):
                path = Path(metadata["outputs"][key]["path"])
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_failure_still_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "outputs"
            args = self.make_args(output_dir, input=str(Path(directory) / "missing.json"))
            code, metadata_path = run(args)
            self.assertEqual(code, 1)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failure")
            self.assertEqual(metadata["failure"]["type"], "ValueError")
            self.assertIn("does not exist", metadata["failure"]["message"])


if __name__ == "__main__":
    unittest.main()
