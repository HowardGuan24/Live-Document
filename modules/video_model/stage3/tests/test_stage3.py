from pathlib import Path
import re

import numpy as np
from PIL import Image

from modules.video_model.stage3.framework.contracts import (
    load_json,
    sha256_path,
    verify_file_record,
)
from modules.video_model.stage3.framework.motion import (
    compile_motion_prompt,
)


STAGE3 = Path(__file__).resolve().parents[1]
REPO_ROOT = STAGE3.parents[2]


def test_phase3_frozen_prompt_and_position_selector() -> None:
    assert (
        sha256_path(STAGE3 / "prompt_lexicon_v4.json")
        == "ac19526bb5bbedf51f58d6e1eb83926a78d45e43022b2583552c58041ac59bf6"
    )
    selection = load_json(
        STAGE3
        / "output/phase-3/selection-v4-position-audit/selection.json"
    )
    assert selection["eligible_count"] == 1
    assert (
        selection["selected_candidate_id"]
        == "auto_control_080-s7101"
    )


def test_phase4_core_is_case_agnostic_and_model_free() -> None:
    source = (STAGE3 / "framework/state_renderer.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "CHEM-01",
        "PHYS-01",
        "MATH-02",
        "BIO-01",
        "GEO-02",
        "diffusers",
        "ControlNetModel",
        "torch",
    ):
        assert token not in source


def test_phase4_gate_and_contract_smoke_pass() -> None:
    gate = load_json(STAGE3 / "output/phase-4/g3.json")
    assert gate["passed"]
    assert all(
        value["passed"] for value in gate["cohorts"].values()
    )
    assert all(item["passed"] for item in gate["contract_smoke"])
    assert gate["model_runs"] == {
        "image_candidates": 0,
        "video_candidates": 0,
    }


def test_chem_initial_anchor_and_stable_pixels() -> None:
    manifest = load_json(
        STAGE3 / "output/phase-4/CHEM-01/manifest.json"
    )
    anchor = np.asarray(
        Image.open(
            REPO_ROOT / manifest["anchor"]["prepared"]["path"]
        ).convert("RGB")
    )
    start = np.asarray(
        Image.open(
            REPO_ROOT / manifest["records"][0]["output"]["path"]
        ).convert("RGB")
    )
    assert np.array_equal(anchor, start)
    assert [
        item["metrics"][
            "outside_mutable_max_difference_0_255"
        ]
        for item in manifest["records"]
    ] == [0, 0, 0, 0]


def test_math_object_material_and_area_are_stable() -> None:
    gate = load_json(STAGE3 / "output/phase-4/g3.json")
    checks = {
        item["name"]: item
        for item in gate["cohorts"]["MATH-02"]["checks"]
    }
    assert checks["piece_area_preserved"]["evidence"] == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert (
        checks["object_local_material_binding_stable"][
            "evidence"
        ]["minimum"]
        == 1.0
    )


def test_accepted_baseline_records_resolve() -> None:
    accepted = load_json(STAGE3 / "baselines/accepted.json")
    for record in accepted["records"]:
        verify_file_record(record, REPO_ROOT)


def test_motion_core_is_case_agnostic_and_runtime_capability_is_honest() -> None:
    source = (STAGE3 / "framework/motion.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "CHEM-01",
        "PHYS-01",
        "MATH-02",
        "BIO-01",
        "GEO-02",
    ):
        assert token not in source
    policy = load_json(STAGE3 / "motion_guidance.json")
    assert policy["runtime"]["native_image_guide_indices"] == [0, -1]
    assert not policy["runtime"]["supports_native_middle_frame"]
    assert not policy["runtime"]["supports_program_video_conditioning"]


def test_motion_prompt_levels_change_only_text_detail() -> None:
    policy = load_json(STAGE3 / "motion_guidance.json")
    segment = policy["segments"][0]
    brief = compile_motion_prompt(segment, guidance_level="L0")
    structured = compile_motion_prompt(segment, guidance_level="L1")
    assert brief["positive"] == segment["brief_prompt"]
    assert len(structured["positive"]) > len(brief["positive"])
    assert "State constraints:" in structured["positive"]
    assert "End exactly at the supplied last frame." in structured["positive"]


def test_phase5_motion_defaults_are_explicit_and_honest() -> None:
    decisions = load_json(
        STAGE3 / "output/phase-5/guidance-decisions.json"
    )
    defaults = {
        item["motion_class"]: item["default"]
        for item in decisions["defaults"]
    }
    assert defaults == {
        "liquid_mixing": "deterministic_program_animation",
        "continuous_field_propagation": "L1",
        "rigid_motion_exact_identity": "deterministic_program_animation",
    }
    assert decisions["L3"]["status"] == "unsupported"


def test_release_policy_claims_validated_candidate_not_production_1_0() -> None:
    policy = load_json(STAGE3 / "release_policy.json")
    assert policy["release_class"] == "validated_stage3_candidate"
    assert policy["stage3_exit_passed"]
    assert not policy["production_1_0_ready"]
    maturity = {
        item["case_id"]: item["release_maturity"]
        for item in policy["case_matrix"]
    }
    assert maturity["BIO-01"] == "validated_with_fallback"
    assert maturity["GEO-02"] == "validated_with_fallback"
    assert maturity["GEO-01"] == "validated"
    assert len(maturity) == 10
    assert policy["production_1_0_blockers"]


def test_phase10_release_covers_all_cases_and_embeds_report_media() -> None:
    manifest = load_json(
        STAGE3 / "output/phase-10-release/release-manifest.json"
    )
    audit = load_json(STAGE3 / "output/phase-10-release/release-audit.json")
    assert manifest["stage3_exit_passed"]
    assert len(manifest["formal_cases"]) == 10
    assert manifest["route_counts"] == {
        "LTX_L1": 2,
        "materialized_deterministic": 8,
    }
    assert manifest["report_embedded_media_count"] == 26
    assert audit["passed"]
    assert audit["counts"]["formal_cases"] == 10


def test_phase6_rerun_keeps_target_success_separate_from_phase_exit() -> None:
    manifest = load_json(
        STAGE3 / "output/phase-6-rerun-1/phase6-rerun-manifest.json"
    )
    assert manifest["status"] == "target_completed_phase_not_passed"
    assert manifest["target"] == {
        "case_id": "BIO-01",
        "g2_g3_image": "passed",
        "L1_video": "rejected",
        "L2_video": "rejected",
        "full_timeline_state_renderer_fallback": "passed_case_specific",
    }
    assert not manifest["phase_exit"]["passed"]
    assert (
        load_json(
            STAGE3 / "output/phase-6-rerun-1/BIO-01/g3-machine.json"
        )["passed"]
    )
    assert (
        load_json(
            STAGE3 / "output/phase-6-rerun-1/visual-review.json"
        )["passed"]
    )
    assert not load_json(
        STAGE3 / "output/phase-6-rerun-1/BIO-01/video/L1/g4.json"
    )["passed"]
    assert not load_json(
        STAGE3 / "output/phase-6-rerun-1/BIO-01/video/L2/g4.json"
    )["passed"]
    assert load_json(
        STAGE3
        / "output/phase-6-rerun-1/BIO-01/video/deterministic/g4.json"
    )["passed"]


def test_phase6_rerun_supersedes_invalid_old_exit_and_logs_verdicts() -> None:
    ledger = load_json(STAGE3 / "experiments/ledger.json")
    verdicts = {
        item["experiment_id"]: item["verdict"]
        for item in ledger["experiments"]
    }
    assert (
        verdicts["EXP-S3-20260731-019"]
        == "superseded_invalid_phase_exit"
    )
    assert verdicts["EXP-S3-20260731-020"] == "accepted_core"
    assert verdicts["EXP-S3-20260731-021"] == "rejected"
    assert verdicts["EXP-S3-20260731-022"] == "rejected"
    assert (
        verdicts["EXP-S3-20260731-023"]
        == "accepted_case_specific"
    )


def test_phase11_has_ten_readable_teaching_timelines() -> None:
    manifest = load_json(
        STAGE3 / "output/phase-11-pedagogy/phase11-manifest.json"
    )
    assert manifest["passed"]
    assert manifest["case_count"] == 10
    assert set(manifest["cases"]) == {
        "MATH-01", "MATH-02", "PHYS-01", "PHYS-02", "CHEM-01",
        "CHEM-02", "BIO-01", "BIO-02", "GEO-01", "GEO-02",
    }
    for case_id in manifest["cases"]:
        result = load_json(
            STAGE3
            / f"output/phase-11-pedagogy/cases/{case_id}/result.json"
        )
        assert result["passed"]
        assert 8.5 <= result["timeline_compile"]["duration_seconds"] <= 12.0
        assert result["video_check"]["passed"]
        verify_file_record(result["video"], REPO_ROOT)
        verify_file_record(result["stage_contact_sheet"], REPO_ROOT)


def test_phase11_repairs_have_mechanism_specific_hard_gates() -> None:
    expected = {
        "CHEM-01": {
            "persistent_endpoint_color_has_multiple_transition_frames",
            "indicator_does_not_change_in_one_frame",
        },
        "CHEM-02": {
            "no_crystal_before_nucleation_threshold",
            "crystal_identity_count_is_monotonic_and_ends_at_four",
            "solute_mass_is_conserved_on_display_clock",
        },
        "GEO-01": {
            "four_oxbow_mechanisms_occur_in_order",
            "isolation_requires_breach_and_most_of_both_plugs",
            "main_channel_stays_connected_and_exactly_one_oxbow_remains",
            "sediment_plugs_are_absent_before_sealing_and_grow_monotonically",
        },
    }
    for case_id, names in expected.items():
        value = load_json(
            STAGE3
            / f"output/phase-11-pedagogy/cases/{case_id}/mechanism-audit.json"
        )
        checks = {item["name"]: item for item in value["checks"]}
        assert value["passed"]
        assert names <= set(checks)
        assert all(checks[name]["passed"] for name in names)


def test_phase11_replay_and_embedded_report_pass() -> None:
    replay = load_json(
        STAGE3 / "output/phase-11-pedagogy/reproducibility-audit.json"
    )
    release = load_json(
        STAGE3 / "output/phase-11-pedagogy/release-manifest.json"
    )
    assert replay["passed"]
    assert replay["scope"] == ["CHEM-01", "CHEM-02", "GEO-01"]
    assert all(
        item["run_1_tree_sha256"] == item["run_2_tree_sha256"]
        for item in replay["checks"]
    )
    report = (
        STAGE3 / "output/phase-11-pedagogy/report.html"
    ).read_text(encoding="utf-8")
    media = re.findall(r"\b(?:src)=['\"]([^'\"]+)", report)
    assert release["passed"]
    assert len(media) == release["report_embedded_media_count"] == 35
    assert all(item.startswith("data:") for item in media)


def test_pedagogy_compiler_is_case_agnostic() -> None:
    source = (STAGE3 / "framework/pedagogy.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "MATH-01", "MATH-02", "PHYS-01", "PHYS-02", "CHEM-01",
        "CHEM-02", "BIO-01", "BIO-02", "GEO-01", "GEO-02",
    ):
        assert token not in source
