from pathlib import Path

from modules.video_model.stage2.framework.contracts import load_json
from modules.video_model.stage2.phase6 import (
    CORE_FILES,
    MANIFEST_PATH,
    REPORT_PATH,
    STAGE2_ROOT,
    build_phase6,
)


def test_phase6_release_manifest_is_current() -> None:
    manifest = build_phase6(check_only=True)
    assert manifest["status"] == "passed"
    assert manifest["release_version"] == "0.1.0"
    assert manifest["automatic_next_action"] == "release_complete"
    assert all(item["passed"] for item in manifest["checks"])


def test_phase6_covers_release_policy_sets() -> None:
    manifest = load_json(MANIFEST_PATH)
    assert manifest["contract_regression"]["case_count"] == 10
    assert manifest["contract_regression"]["program_frame_count"] == 490
    assert len(manifest["image_representatives"]) == 5
    assert len(manifest["video_representatives"]) == 5
    assert all(
        item["status"] == "passed"
        for item in manifest["image_representatives"]
    )
    assert all(
        item["status"] == "passed"
        for item in manifest["video_representatives"]
    )


def test_phase6_material_regressions_are_visible_and_protected() -> None:
    root = MANIFEST_PATH.parent / "image-regressions"
    for case_id in ("CHEM-01", "BIO-01", "GEO-02"):
        manifest = load_json(root / case_id / "manifest.json")
        checks = {
            item["name"]: item for item in manifest["hard_checks"]
        }
        assert checks[
            "all_non_allowed_and_protected_pixels_are_exact"
        ]["passed"]
        assert checks[
            "material_enhancement_is_not_an_unchanged_no_op"
        ]["passed"]
        assert all(item["passed"] for item in manifest["hard_checks"])


def test_phase6_actual_video_tokenizer_evidence_passes() -> None:
    manifest = load_json(MANIFEST_PATH)
    token = manifest["token_integrity"]
    assert token["status"] == "passed"
    assert token["experiment_count"] == 8
    assert (
        token["measurement"]["tokenizer_class"]
        == "LTXAVGemmaTokenizer"
    )
    for item in token["experiments"]:
        assert item["passed"]
        assert (
            item["positive"]["content_tokens_including_bos"] <= 1024
        )
        assert (
            item["negative"]["content_tokens_including_bos"] <= 1024
        )


def test_released_core_has_no_benchmark_case_id() -> None:
    blocked = ("MATH-", "PHYS-", "CHEM-", "BIO-", "GEO-")
    for relative in CORE_FILES:
        text = (STAGE2_ROOT / relative).read_text(encoding="utf-8")
        assert not any(prefix in text for prefix in blocked)


def test_final_report_explains_process_in_plain_language() -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    required = (
        "一段动画究竟怎样生成",
        "模型当时实际看到的控制图",
        "完整正向与负向图片提示词",
        "四个 raw 模型候选",
        "视频模型不是每种运动都值得用",
        "字符数不再冒充 token 数",
        "通用核心与案例插件",
        "失败没有被删除",
        "从仓库根目录重跑",
    )
    assert all(item in text for item in required)
    assert "最终报告全部链接可打开" in text
    assert "final_report_links_resolve</td>" not in text
    assert Path(REPORT_PATH).stat().st_size > 20_000
