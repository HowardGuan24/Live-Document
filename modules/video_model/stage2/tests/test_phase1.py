from __future__ import annotations

import re
import tempfile
from collections import Counter
from pathlib import Path

from modules.video_model.stage2.framework.contracts import (
    load_json,
    sha256_path,
    validate_schema_documents,
)
from modules.video_model.stage2.framework.fixture_builder import (
    build_fixture,
)
from modules.video_model.stage2.phase1 import (
    FIXTURES_ROOT,
    OUTPUT_ROOT,
    REGISTRY_PATH,
    TEMPLATE_PATH,
    build_phase1,
)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_path(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_all_contract_schema_documents_are_present() -> None:
    records = validate_schema_documents()
    assert len(records) == 5
    assert {
        "concept_spec.schema.json",
        "sequence_spec.schema.json",
        "semantic_layers.schema.json",
        "fixture_manifest.schema.json",
        "score_record.schema.json",
    } == {record["name"] for record in records}


def test_phase1_generated_outputs_are_current() -> None:
    manifest = build_phase1(check_only=True)
    assert manifest["status"] == "passed"
    assert manifest["fixture_count"] == 10
    assert manifest["state_count"] == 40
    assert manifest["semantic_layer_count"] == 44
    assert manifest["model_runs"] == {"image": 0, "video": 0}
    assert len(manifest["checks"]) == 8


def test_every_fixture_has_four_states_and_explicit_classification() -> None:
    registry = load_json(REGISTRY_PATH)
    for case in registry["cases"]:
        fixture_root = FIXTURES_ROOT / case["case_id"]
        manifest = load_json(fixture_root / "fixture_manifest.json")
        states = [
            line
            for line in (
                fixture_root / manifest["states"]["path"]
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        assert len(states) == 4
        assert manifest["classification"] == (
            "model-free contract fixture, "
            "not a finished scientific animation"
        )
        assert manifest["model_runs"] == {"image": 0, "video": 0}


def test_control_routes_are_balanced_and_wave_does_not_force_canny() -> None:
    registry = load_json(REGISTRY_PATH)
    routes = {}
    for case in registry["cases"]:
        manifest = load_json(
            FIXTURES_ROOT / case["case_id"] / "fixture_manifest.json"
        )
        routes[case["case_id"]] = manifest["control"]["route"]
    assert Counter(routes.values()) == {
        "sparse_hard_boundary_candidate": 5,
        "off": 5,
    }
    assert routes["PHYS-01"] == "off"
    assert routes["GEO-02"] == "off"


def test_annotation_layers_are_never_model_inputs() -> None:
    registry = load_json(REGISTRY_PATH)
    for case in registry["cases"]:
        layers = load_json(
            FIXTURES_ROOT / case["case_id"] / "semantic_layers.json"
        )["layers"]
        annotations = [
            layer
            for layer in layers
            if layer["layer_type"] == "annotation"
        ]
        assert len(annotations) == 1
        assert annotations[0]["model_input_policy"] == "never"
        assert annotations[0]["used_as_model_input"] is False


def test_fixture_builder_is_byte_deterministic_for_all_cases() -> None:
    registry = load_json(REGISTRY_PATH)
    templates = load_json(TEMPLATE_PATH)["cases"]
    with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
        first_root = Path(first_dir)
        second_root = Path(second_dir)
        for case in registry["cases"]:
            build_fixture(
                case,
                templates[case["case_id"]],
                first_root / case["case_id"],
            )
            build_fixture(
                case,
                templates[case["case_id"]],
                second_root / case["case_id"],
            )
        assert _tree_hashes(first_root) == _tree_hashes(second_root)


def test_phase1_report_has_no_missing_local_links() -> None:
    report_path = OUTPUT_ROOT / "report.html"
    report = report_path.read_text(encoding="utf-8")
    targets = re.findall(r'(?:href|src)="([^"]+)"', report)
    missing = []
    for target in targets:
        if target.startswith(("#", "http://", "https://")):
            continue
        if not (report_path.parent / target).resolve().exists():
            missing.append(target)
    assert not missing
    for phrase in (
        "契约已实现，科学动画尚未实现",
        "不再把所有中间图都叫 mask",
        "有硬边界才保存线稿候选",
        "Phase 2：五个哨兵案例的真实确定性程序",
    ):
        assert phrase in report

