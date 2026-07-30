from modules.video_model.stage2.phase8_b_unification import (
    MANIFEST_PATH,
    REPORT_PATH,
    run,
)
from modules.video_model.stage2.framework.contracts import load_json


def test_phase8_b_only_experiment_passes() -> None:
    manifest = run(check_only=True)
    assert manifest["status"] == "passed"
    assert manifest["output_count"] == 48
    assert manifest["model_runs"] == {"image": 0, "video": 0}
    assert all(item["passed"] for item in manifest["checks"])


def test_phase8_retires_c_only_as_a_top_level_route() -> None:
    manifest = load_json(MANIFEST_PATH)
    decision = manifest["decision"]
    assert decision["top_level_routes"] == ["A", "B"]
    assert decision["retire_top_level_route"] == "C"
    assert decision["absorb_into_B"] == [
        "B2_object_attached_projection",
        "B3_field_conditioned_optics",
    ]


def test_phase8_report_explains_the_non_visual_boundary() -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    required = (
        "C 不需要继续做一条",
        "材质滑动",
        "B1：区域/标量投影",
        "B2：对象附着投影",
        "B3：场驱动光学投影",
        "水波：B 可以完整替代旧 C",
        "尚未证明视频模型能保持",
    )
    assert all(item in text for item in required)
