from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from modules.video_model import backends
from modules.video_model import generate_explainer_video as generator_module
from modules.video_model.backends import GenerationSettings
from modules.video_model.benchmarks.runner import aggregate_results, load_cases
from modules.video_model.benchmarks.experiment_runner import load_experiments
from modules.video_model.config import load_profile, validate_profile_data
from modules.video_model.evaluation import HUMAN_COLUMNS, load_rubric, validate_human_template
from modules.video_model.generate_explainer_video import DEFAULT_MODEL, run
from modules.video_model.generate_explainer_video import TIMING_KEYS, candidate_run_name
from modules.video_model.prompting import (
    DEFAULT_PROMPT_TEMPLATE_VERSION,
    InputSpec,
    build_prompts,
    validate_input,
)


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

    def test_prompt_template_version_is_explicit_and_validated(self) -> None:
        spec = InputSpec("source", "goal", "visual", "process")
        current, _ = build_prompts(spec, DEFAULT_PROMPT_TEMPLATE_VERSION)
        legacy, _ = build_prompts(spec, "explainer-v1")
        self.assertNotEqual(current, legacy)
        with self.assertRaisesRegex(ValueError, "unknown prompt template version"):
            build_prompts(spec, "missing-version")


class ConfigurationTests(unittest.TestCase):
    def test_profiles_load_with_ltx_constraints(self) -> None:
        profiles = [load_profile(name) for name in ("fast", "balanced", "quality")]
        self.assertEqual([profile.name for profile in profiles], ["fast", "balanced", "quality"])
        for profile in profiles:
            self.assertEqual(profile.width % 32, 0)
            self.assertEqual(profile.height % 32, 0)
            self.assertEqual((profile.num_frames - 1) % 8, 0)

    def test_invalid_profile_is_rejected(self) -> None:
        bad = load_profile("fast").to_dict()
        bad["width"] = 500
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            validate_profile_data(bad, "fast")

    def test_candidate_naming_is_stable(self) -> None:
        self.assertEqual(candidate_run_name("batch one", 2, 77), "batch-one-c02-s77")
        long_name = candidate_run_name("x" * 150, 3, 99)
        self.assertEqual(len(long_name), 100)
        self.assertTrue(long_name.endswith("-c03-s99"))
        with self.assertRaisesRegex(ValueError, "start at 1"):
            candidate_run_name("batch", 0, 77)


class EnvironmentTests(unittest.TestCase):
    def test_ltx_environment_detects_rocm_install(self) -> None:
        fake_torch = {
            "torch_installed": True,
            "torch_version": "test",
            "accelerator_available": True,
            "rocm_version": "7.2",
            "gpu_name": "AMD test GPU",
            "gpu_memory_gib": 48.0,
        }
        with (
            patch.object(backends, "_torch_environment", return_value=fake_torch),
            patch.object(backends.importlib.util, "find_spec", return_value=object()),
            patch.object(backends, "_package_version", return_value="test"),
        ):
            environment = backends.ltx_environment()
        self.assertTrue(environment["available"])
        self.assertIsNone(environment["reason"])

    def test_ltx_environment_rejects_non_rocm_torch(self) -> None:
        fake_torch = {
            "torch_installed": True,
            "torch_version": "test",
            "accelerator_available": True,
            "rocm_version": None,
            "gpu_name": "other GPU",
            "gpu_memory_gib": 48.0,
        }
        with (
            patch.object(backends, "_torch_environment", return_value=fake_torch),
            patch.object(backends.importlib.util, "find_spec", return_value=object()),
            patch.object(backends, "_package_version", return_value="test"),
        ):
            environment = backends.ltx_environment()
        self.assertFalse(environment["available"])
        self.assertIn("ROCm", environment["reason"])


class RuntimeFallbackTests(unittest.TestCase):
    def test_any_backend_substitution_has_fallback_status(self) -> None:
        self.assertEqual(generator_module._success_status("wan", True), "success_fallback")
        self.assertEqual(generator_module._success_status("procedural", False), "success_fallback")
        self.assertEqual(generator_module._success_status("ltx", False), "success_model")

    def test_auto_continues_after_model_generation_failure(self) -> None:
        class FailingRunner:
            backend = "ltx"

            def generate(self, *args: object) -> dict[str, object]:
                raise RuntimeError("synthetic inference failure")

        class GoodRunner:
            backend = "wan"

            def generate(self, *args: object) -> dict[str, object]:
                return {"backend": "wan", "timing_seconds": {}}

        settings = GenerationSettings(
            duration=4.0,
            fps=12,
            width=512,
            height=320,
            seed=1,
            inference_steps=4,
            guidance_scale=1.0,
            model_id="test",
            num_frames=49,
        )
        fallback_settings = GenerationSettings(
            **{**settings.__dict__, "guidance_scale": 5.0, "inference_steps": 30}
        )
        good = GoodRunner()
        with (
            patch.object(
                generator_module,
                "_select_and_load_runner",
                return_value=(
                    good,
                    fallback_settings,
                    1.0,
                    {"model_id": "test"},
                    [{"backend": "wan", "status": "loaded"}],
                    {"used": False, "reason": None},
                ),
            ),
            patch.object(generator_module, "release_runner"),
        ):
            result = generator_module._generate_with_auto_fallback(
                Namespace(backend="auto"),
                {},
                1,
                {},
                FailingRunner(),
                settings,
                object(),
                "positive",
                "negative",
                Path("raw.mp4"),
                [{"backend": "ltx", "status": "loaded"}],
            )
        runner, actual_settings, backend_info, load_seconds, fallback, attempts, load_info = result
        self.assertIs(runner, good)
        self.assertIs(actual_settings, fallback_settings)
        self.assertEqual(backend_info["backend"], "wan")
        self.assertEqual(load_seconds, 1.0)
        self.assertTrue(fallback["used"])
        self.assertIn("synthetic inference failure", fallback["reason"])
        self.assertEqual(attempts[-1]["backend"], "wan")
        self.assertEqual(load_info["model_id"], "test")


class BenchmarkAndEvaluationTests(unittest.TestCase):
    def test_all_fixed_benchmark_cases_load(self) -> None:
        speed = load_cases(ROOT / "benchmarks" / "speed_cases")
        quality = load_cases(ROOT / "benchmarks" / "quality_cases")
        self.assertEqual(len(speed), 4)
        self.assertGreaterEqual(len(quality), 8)

    def test_aggregation_excludes_fallback_from_model_successes(self) -> None:
        base = {
            "backend": "ltx",
            "profile": "fast",
            "total_seconds": 10.0,
            "warm_inference_seconds": 8.0,
            "peak_gpu_memory_gib": 9.0,
        }
        summary = aggregate_results(
            [{**base, "status": "success_model"}, {**base, "status": "success_fallback"}]
        )
        group = summary["groups"][0]
        self.assertEqual(group["runs"], 2)
        self.assertEqual(group["successes"], 1)
        self.assertEqual(group["fallback_successes"], 1)
        self.assertEqual(group["failures"], 0)
        self.assertEqual(group["mean_total_seconds"], 10.0)

    def test_controlled_experiments_change_only_declared_variable(self) -> None:
        experiments = load_experiments()
        self.assertEqual(
            set(experiments),
            {"exp_model_reuse", "exp_steps_4_vs_7", "exp_resolution_512_vs_640"},
        )

    def test_rubric_and_human_template_validate(self) -> None:
        rubric = load_rubric()
        self.assertEqual(len(rubric["dimensions"]), 6)
        template = ROOT / "evaluations" / "human_pairwise_template.csv"
        self.assertEqual(validate_human_template(template), [])
        self.assertEqual(template.read_text(encoding="utf-8").splitlines()[0].split(","), HUMAN_COLUMNS)


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
            self.assertEqual(metadata["status"], "success_fallback")
            self.assertTrue(metadata["fallback"]["used"] is False)
            self.assertEqual(metadata["chosen_workflow"]["backend"], "procedural")
            self.assertEqual(set(metadata["timing_seconds"]), set(TIMING_KEYS))
            self.assertEqual(metadata["prompt_template_version"], DEFAULT_PROMPT_TEMPLATE_VERSION)
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
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["failure"]["type"], "ValueError")
            self.assertIn("does not exist", metadata["failure"]["message"])

    def test_auto_fallback_records_requested_and_actual_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "outputs"
            args = self.make_args(
                output_dir,
                backend="auto",
                run_name="auto-fallback",
                no_gif=True,
            )
            unavailable = {"available": False, "reason": "disabled for unit test"}
            with (
                patch.object(backends, "ltx_environment", return_value=unavailable),
                patch.object(backends, "wan_environment", return_value=unavailable),
            ):
                code, metadata_path = run(args)
            self.assertEqual(code, 0)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["requested_backend"], "auto")
            self.assertEqual(metadata["actual_backend"], "procedural")
            self.assertEqual(metadata["status"], "success_fallback")
            self.assertTrue(metadata["fallback"]["used"])
            self.assertIn("ltx", metadata["fallback"]["reason"].lower())
            self.assertIn("wan", metadata["fallback"]["reason"].lower())


if __name__ == "__main__":
    unittest.main()
