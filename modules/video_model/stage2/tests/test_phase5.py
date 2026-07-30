from modules.video_model.stage2.framework.contracts import load_json
from modules.video_model.stage2.phase5 import (
    MANIFEST_PATH,
    MOTION_ORDER,
    build_phase5,
)


def test_phase5_cumulative_report_is_current() -> None:
    manifest = build_phase5(check_only=True)
    assert manifest["phase"] == 5
    assert manifest["motion_classes_total"] == 5
    assert manifest["motion_classes_passed"] >= 1
    assert manifest["model_runs"]["image"] == 0
    assert manifest["model_runs"]["video"] >= 1


def test_phase5_motion_status_uses_frozen_motion_classes() -> None:
    manifest = load_json(MANIFEST_PATH)
    assert tuple(
        item["motion_class"] for item in manifest["motion_status"]
    ) == MOTION_ORDER


def test_reviewed_phase5_videos_pass_declared_hard_checks() -> None:
    manifest = load_json(MANIFEST_PATH)
    for experiment in manifest["experiments"]:
        run = load_json(
            MANIFEST_PATH.parents[2] / experiment["run"]["path"]
        )
        assert run["model_runs"]["image"] == 0
        assert run["model_runs"]["video"] in {0, 1}
        review = load_json(
            MANIFEST_PATH.parents[2] / experiment["review"]["path"]
        )
        if review["motion_class_passed"]:
            assert all(item["passed"] for item in run["hard_checks"])
