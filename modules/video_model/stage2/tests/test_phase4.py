from modules.video_model.stage2.framework.contracts import load_json
from modules.video_model.stage2.cases.remaining_programs import (
    PROGRAMS,
)
from modules.video_model.stage2.phase4 import (
    CASE_IDS,
    MANIFEST_PATH,
    PROGRAM_ROOT,
    build_phase4_programs,
)
from modules.video_model.stage2.phase4_routes import (
    OUTPUT_PATH as ROUTE_PLAN_PATH,
    build_route_plan,
)


def test_phase4_remaining_program_milestone_is_current() -> None:
    manifest = build_phase4_programs(check_only=True)
    assert manifest["status"] == "passed"
    assert manifest["phase_complete"] is True
    assert manifest["program_count"] == 5
    assert manifest["frame_count"] == 245
    assert manifest["keyframe_count"] == 20
    assert manifest["program_model_runs"] == {"image": 0, "video": 0}
    assert manifest["model_runs"] == {"image": 24, "video": 0}
    assert len(manifest["route_smoke_experiments"]) == 6
    assert (
        manifest["automatic_next_action"]
        == "run_phase5_motion_class_video_smokes"
    )
    assert tuple(manifest["case_ids"]) == CASE_IDS


def test_remaining_programs_pass_mechanism_checks() -> None:
    for case_id in CASE_IDS:
        manifest = load_json(PROGRAM_ROOT / case_id / "program_manifest.json")
        assert manifest["phase"] == 4
        assert manifest["status"] == "passed"
        mechanism = [
            item
            for item in manifest["checks"]
            if item["name"]
            not in {
                "normalized_progress_is_monotonic",
                "fixed_rgb_canvas",
                "semantic_layer_contract_is_stable",
                "annotations_are_post_generation_only",
                "video_was_encoded_from_program_frames",
                "zero_generative_model_runs",
                "case_id_matches_plugin",
            }
        ]
        assert mechanism
        assert all(item["passed"] for item in mechanism)


def test_route_plan_covers_ten_cases_without_case_id_branches() -> None:
    plan = build_route_plan(check_only=True)
    assert len(plan["routes"]) == 10
    assert sum(plan["route_counts"].values()) == 10
    assert plan["model_runs"] == {"image": 0, "video": 0}
    selector = (
        ROUTE_PLAN_PATH.parents[2]
        / "framework"
        / "route_selector.py"
    ).read_text(encoding="utf-8")
    for case_id in [route["case_id"] for route in plan["routes"]]:
        assert case_id not in selector


def test_geo01_raster_has_connected_main_channel_and_isolated_oxbow() -> None:
    program = PROGRAMS["GEO-01"]
    before = program.sample(2 / 3).state
    after = program.sample(1.0).state
    assert before["water_component_count"] == 1
    assert before["main_channel_components"] == 1
    assert before["isolated_oxbow_count"] == 0
    assert after["water_component_count"] == 2
    assert after["main_channel_components"] == 1
    assert after["isolated_oxbow_count"] == 1


def test_phase4_manifest_exists_after_route_freeze() -> None:
    assert MANIFEST_PATH.is_file()
    manifest = load_json(MANIFEST_PATH)
    assert manifest["route_plan"]["path"] == "route-plan.json"
