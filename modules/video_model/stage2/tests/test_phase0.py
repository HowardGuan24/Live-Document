from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from modules.video_model.stage2.phase0 import (
    CONFIG_PATHS,
    DISCIPLINES,
    OUTPUT_ROOT,
    build_phase0,
    load_json,
    validate_all,
)


def test_phase0_validation_passes() -> None:
    state = validate_all()
    assert len(state["checks"]) == 15
    assert all(check["passed"] for check in state["checks"])
    assert len(state["baseline_records"]) == 14


def test_registry_has_two_cases_and_one_sentinel_per_discipline() -> None:
    registry = load_json(CONFIG_PATHS["case_registry"])
    cases = registry["cases"]
    assert len(cases) == 10
    assert len({case["case_id"] for case in cases}) == 10
    assert Counter(case["discipline"] for case in cases) == Counter(
        {discipline: 2 for discipline in DISCIPLINES}
    )
    assert Counter(
        case["discipline"] for case in cases if case["sentinel"]
    ) == Counter({discipline: 1 for discipline in DISCIPLINES})


def test_scoring_dimensions_each_sum_to_one_hundred() -> None:
    scoring = load_json(CONFIG_PATHS["scoring_protocol"])
    assert (
        sum(
            item["weight"]
            for item in scoring["image_and_sequence_dimensions"]
        )
        == 100
    )
    assert (
        sum(item["weight"] for item in scoring["video_dimensions"])
        == 100
    )


def test_baseline_hashes_are_well_formed() -> None:
    baseline = load_json(CONFIG_PATHS["stage1_baseline"])
    assert len(baseline["assets"]) == 14
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])
        for asset in baseline["assets"]
    )


def test_generated_phase0_outputs_are_current() -> None:
    manifest = build_phase0(check_only=True)
    assert manifest["status"] == "passed"
    assert manifest["model_runs"] == {"image": 0, "video": 0}
    assert manifest["case_summary"] == {
        "new_case_count": 10,
        "discipline_count": 5,
        "sentinel_count": 5,
        "historical_regression_count": 1,
    }


def test_report_contains_every_case_and_plain_language_explanations() -> None:
    report_path = OUTPUT_ROOT / "report.html"
    report = report_path.read_text(encoding="utf-8")
    registry = load_json(CONFIG_PATHS["case_registry"])
    assert all(case["case_id"] in report for case in registry["cases"])
    for phrase in (
        "“冻结”是什么意思？",
        "Loop 不是无限抽卡",
        "先过硬门禁，再谈好不好看",
        "Stage 1 提供起跑线，不是像素答案",
    ):
        assert phrase in report


def test_every_local_report_link_exists() -> None:
    report_path = OUTPUT_ROOT / "report.html"
    report = report_path.read_text(encoding="utf-8")
    targets = re.findall(r'(?:href|src)="([^"]+)"', report)
    missing: list[str] = []
    for target in targets:
        if target.startswith(("#", "http://", "https://")):
            continue
        resolved = (report_path.parent / target).resolve()
        if not resolved.exists():
            missing.append(target)
    assert not missing

