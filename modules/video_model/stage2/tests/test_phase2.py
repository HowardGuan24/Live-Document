from __future__ import annotations

import hashlib
import io
import re

from modules.video_model.stage2.cases.sentinel_programs import (
    KEYFRAME_PROGRESS,
    PROGRAMS,
)
from modules.video_model.stage2.framework.contracts import load_json
from modules.video_model.stage2.framework.program_runner import (
    FRAME_COUNT,
    validate_program_tree,
)
from modules.video_model.stage2.phase2 import (
    MANIFEST_PATH,
    OUTPUT_ROOT,
    REPORT_PATH,
    SENTINEL_IDS,
    build_phase2,
)


def _image_digest(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def test_phase2_generated_outputs_are_current_and_passed() -> None:
    manifest = build_phase2(check_only=True)
    assert manifest["status"] == "passed"
    assert manifest["program_count"] == 5
    assert manifest["frame_count"] == 245
    assert manifest["keyframe_count"] == 20
    assert manifest["model_runs"] == {"image": 0, "video": 0}
    assert manifest["automatic_next_action"] == "advance_to_phase_3"


def test_all_saved_program_trees_validate() -> None:
    for case_id in SENTINEL_IDS:
        manifest = validate_program_tree(OUTPUT_ROOT / case_id)
        assert manifest["status"] == "passed"
        assert manifest["timeline"]["frame_count"] == FRAME_COUNT
        assert manifest["model_runs"] == {"image": 0, "video": 0}


def test_case_mechanisms_pass_on_four_keyframes() -> None:
    for case_id in SENTINEL_IDS:
        program = PROGRAMS[case_id]
        samples = [program.sample(progress) for progress in KEYFRAME_PROGRESS]
        checks = program.validate(samples)
        assert checks
        assert all(item["passed"] for item in checks)


def test_keyframe_programs_and_layers_are_byte_deterministic() -> None:
    for case_id in SENTINEL_IDS:
        program = PROGRAMS[case_id]
        first = [program.sample(progress) for progress in KEYFRAME_PROGRESS]
        second = [program.sample(progress) for progress in KEYFRAME_PROGRESS]
        assert [
            _image_digest(sample.clean_frame) for sample in first
        ] == [_image_digest(sample.clean_frame) for sample in second]
        assert [
            _image_digest(sample.program_frame) for sample in first
        ] == [_image_digest(sample.program_frame) for sample in second]
        for first_sample, second_sample in zip(first, second):
            assert [
                (layer.layer_id, layer.layer_type)
                for layer in first_sample.layers
            ] == [
                (layer.layer_id, layer.layer_type)
                for layer in second_sample.layers
            ]


def test_annotations_are_explicitly_post_generation() -> None:
    for case_id in SENTINEL_IDS:
        manifest = load_json(
            OUTPUT_ROOT / case_id / "program_manifest.json"
        )
        for keyframe in manifest["keyframes"]:
            annotations = [
                layer
                for layer in keyframe["layers"]
                if layer["layer_type"] == "annotation"
            ]
            assert len(annotations) == 1
            assert annotations[0]["model_input_policy"] == "never"
            assert annotations[0]["used_as_model_input"] is False


def test_titration_mixing_spreads_as_peak_and_integral_fall() -> None:
    program = PROGRAMS["CHEM-01"]
    states = [
        program.sample(progress).state
        for progress in (1 / 3, 5 / 12, 1 / 2, 7 / 12, 2 / 3)
    ]
    spread = [state["plume_spread_factor"] for state in states]
    peak = [state["plume_peak_amplitude"] for state in states]
    integral = [
        state["plume_integrated_proxy"] for state in states
    ]
    assert all(b >= a for a, b in zip(spread, spread[1:]))
    assert all(b <= a for a, b in zip(peak, peak[1:]))
    assert all(b <= a for a, b in zip(integral, integral[1:]))
    assert spread[-1] > spread[0]
    assert peak[-1] == 0


def test_phase2_report_has_no_missing_local_links() -> None:
    assert MANIFEST_PATH.is_file()
    report = REPORT_PATH.read_text(encoding="utf-8")
    targets = re.findall(r'(?:href|src|poster)="([^"]+)"', report)
    missing = []
    for target in targets:
        if target.startswith(("#", "http://", "https://")):
            continue
        if not (REPORT_PATH.parent / target).resolve().exists():
            missing.append(target)
    assert not missing
    for phrase in (
        "先把科学机制画对",
        "这不是生图结果",
        "每根箭头",
        "下载可复现原始数据",
        "advance_to_phase_3",
    ):
        assert phrase in report
