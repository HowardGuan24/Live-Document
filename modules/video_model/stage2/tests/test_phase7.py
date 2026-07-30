from modules.video_model.stage2.framework.contracts import load_json
from modules.video_model.stage2.phase7 import (
    MANIFEST_PATH,
    REPORT_PATH,
    REVIEW_PATH,
    build_phase7,
    select_appearance_route,
)


def test_phase7_manifest_and_inventory_are_current() -> None:
    manifest = build_phase7(check_only=True)
    assert manifest["status"] == "passed_with_documented_boundaries"
    assert manifest["counts"]["route_a_model_images"] == 36
    assert manifest["counts"]["route_b_c_renders"] == 48
    assert manifest["counts"]["total_visual_outputs"] == 84
    assert all(item["passed"] for item in manifest["checks"])


def test_phase7_route_selector_uses_data_properties() -> None:
    assert (
        select_appearance_route(
            has_exact_geometry_or_field=True,
            has_frozen_real_base=True,
            state_is_local_or_semantic=True,
        )
        == "C_exact_renderer"
    )
    assert (
        select_appearance_route(
            has_exact_geometry_or_field=False,
            has_frozen_real_base=True,
            state_is_local_or_semantic=True,
        )
        == "B_frozen_base_projection"
    )
    assert (
        select_appearance_route(
            has_exact_geometry_or_field=False,
            has_frozen_real_base=False,
            state_is_local_or_semantic=True,
        )
        == "A_once_then_B"
    )


def test_phase7_keeps_route_boundaries_and_failures() -> None:
    review = load_json(REVIEW_PATH)
    assert review["route_a"]["status"] == "partial"
    assert (
        review["route_b"]["status"]
        == "accepted_for_material_cases"
    )
    assert (
        review["route_c"]["status"]
        == "accepted_for_exact_geometry_and_fields"
    )
    assert review["route_a"]["preflight_failure"]
    assert review["route_b"]["cases"][1]["rejected"] == [
        "raw_underlay"
    ]
    assert review["route_c"]["cases"][1]["rejected"] == [
        "refractive_water"
    ]


def test_phase7_report_explains_the_full_process() -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    required = (
        "为什么 Phase 6 看不出“材质增强”",
        "mask（遮罩）",
        "conditioning（控制条件）",
        "模型实际看到的黑白结构控制图",
        "完整提示词与六张候选",
        "路线 B：冻结真实底图",
        "路线 C：对象坐标 / 物理场",
        "失败记录：我们具体学到了什么",
        "怎样完整复现",
        "本阶段没有宣称完成的部分",
    )
    assert all(item in text for item in required)
    assert REPORT_PATH.stat().st_size > 25_000
    manifest = load_json(MANIFEST_PATH)
    assert next(
        item
        for item in manifest["checks"]
        if item["name"] == "report_links_resolve"
    )["passed"]
