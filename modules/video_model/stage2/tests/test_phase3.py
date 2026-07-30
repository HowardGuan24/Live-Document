from pathlib import Path

from modules.video_model.stage2.framework.contracts import load_json
from modules.video_model.stage2.phase3 import (
    EXPECTED_CASES,
    EXPECTED_EXPERIMENT_IDS,
    LEDGER_PATH,
    PHASE_ROOT,
    REPORT_PATH,
    check_phase3,
)


def test_phase3_is_current_complete_and_budgeted() -> None:
    manifest = check_phase3()
    assert manifest["status"] == "passed"
    assert manifest["phase_complete"] is True
    assert manifest["automatic_next_action"] == "advance_to_phase_4"
    assert manifest["experiment_count"] == 15
    assert manifest["candidate_count"] == 112
    assert manifest["model_runs"] == {
        "new_image_candidates": 92,
        "reused_image_candidates": 20,
        "video_candidates": 0,
    }
    assert set(manifest["case_ids"]) == EXPECTED_CASES
    assert all(manifest["hard_gates"].values())


def test_every_experiment_keeps_raw_evidence_and_review() -> None:
    for experiment_id in EXPECTED_EXPERIMENT_IDS:
        root = PHASE_ROOT / experiment_id
        generated = load_json(root / "_work" / "generate.json")
        blind_map = load_json(root / "_work" / "blind_map.json")
        assert len(generated["candidates"]) == len(blind_map)
        assert (root / "candidates-blind.jpg").is_file()
        assert (root / "candidates-labeled.jpg").is_file()
        for candidate in generated["candidates"]:
            path = Path(candidate["path"])
            assert path.parts[0] == "raw"
            assert (root / path).is_file()


def test_raw_composite_and_final_are_separate_classes() -> None:
    for experiment_id in EXPECTED_EXPERIMENT_IDS:
        root = PHASE_ROOT / experiment_id
        assert (root / "raw").is_dir()
        assert (root / "composite").is_dir()
        assert (root / "final").is_dir()
    assert (
        PHASE_ROOT
        / "EXP-20260729-012/composite/ensemble_material/gain_070"
        / "all_seed_median.png"
    ).is_file()


def test_accepted_material_projections_pass_machine_gates() -> None:
    wave = load_json(
        PHASE_ROOT
        / "EXP-20260729-005/_work/ensemble_projection_gain_030.json"
    )
    math = load_json(
        PHASE_ROOT
        / "EXP-20260729-012/_work/ensemble_projection_gain_070.json"
    )
    assert all(item["passed"] for item in wave["hard_checks"])
    assert all(item["passed"] for item in math["hard_checks"])
    assert (
        math["full_ensemble"]["metrics"][
            "protected_max_abs_difference_0_255"
        ]
        == 0
    )


def test_ledger_and_beginner_report_cover_all_experiments() -> None:
    ledger = load_json(LEDGER_PATH)
    phase3 = [
        item for item in ledger["experiments"] if item["phase"] == 3
    ]
    assert [item["experiment_id"] for item in phase3] == list(
        EXPECTED_EXPERIMENT_IDS
    )
    report = REPORT_PATH.read_text(encoding="utf-8")
    for required_text in (
        "第一次接手项目",
        "语义层",
        "控制图（conditioning）",
        "seed",
        "raw",
        "composite",
        "Phase 3 判定：passed",
    ):
        assert required_text in report
    for experiment_id in EXPECTED_EXPERIMENT_IDS:
        assert experiment_id in report

