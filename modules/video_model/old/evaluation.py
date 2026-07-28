"""Offline-compatible AI and human review formats for explanation videos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUBRIC = MODULE_ROOT / "evaluations" / "quality_rubric.json"
SCORE_DIMENSIONS = {
    "semantic_faithfulness",
    "explanation_clarity",
    "object_consistency",
    "camera_stability",
    "visual_simplicity",
    "loop_suitability",
}
FAILURE_LABELS = {
    "object_drift",
    "object_duplication",
    "object_disappearance",
    "semantic_error",
    "motion_unclear",
    "camera_drift",
    "visual_clutter",
    "bad_loop",
    "prompt_ignored",
    "unreadable_generated_text",
    "temporal_flicker",
}
HUMAN_COLUMNS = [
    "case_id",
    "candidate_a",
    "candidate_b",
    "clearer",
    "more_faithful",
    "more_visually_stable",
    "candidate_a_unusable",
    "candidate_b_unusable",
    "comment",
]


def load_rubric(path: str | Path = DEFAULT_RUBRIC) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    dimensions = set(data.get("dimensions", {}))
    if dimensions != SCORE_DIMENSIONS:
        raise ValueError(f"rubric dimensions must be exactly {sorted(SCORE_DIMENSIONS)}")
    for name, dimension in data["dimensions"].items():
        anchors = dimension.get("anchors", {})
        if set(anchors) != {"1", "2", "3", "4", "5"}:
            raise ValueError(f"rubric dimension '{name}' must define anchors 1-5")
    if set(data.get("failure_labels", [])) != FAILURE_LABELS:
        raise ValueError("rubric failure labels do not match the V2 schema")
    return data


def validate_ai_review(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("AI review must be a JSON object")
    for field in ("reviewer", "scores", "detected_failures", "evidence", "confidence"):
        if field not in data:
            raise ValueError(f"AI review is missing '{field}'")
    reviewer = data["reviewer"]
    if not isinstance(reviewer, dict) or set(reviewer) != {"provider", "model", "version"}:
        raise ValueError("AI review reviewer must contain provider, model, and version")
    if not all(isinstance(value, str) and value.strip() for value in reviewer.values()):
        raise ValueError("AI reviewer provider, model, and version must be non-empty strings")
    if not isinstance(data["scores"], dict):
        raise ValueError("AI review scores must be an object")
    if set(data["scores"]) != SCORE_DIMENSIONS:
        raise ValueError("AI review scores must contain all six rubric dimensions")
    for name, score in data["scores"].items():
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"AI review score '{name}' must be an integer from 1 to 5")
    if not isinstance(data["detected_failures"], list) or not all(
        isinstance(item, str) for item in data["detected_failures"]
    ):
        raise ValueError("AI review detected_failures must be a string array")
    failures = set(data["detected_failures"])
    if not failures <= FAILURE_LABELS:
        raise ValueError(f"unknown AI review failure labels: {sorted(failures - FAILURE_LABELS)}")
    confidence = data["confidence"]
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("AI review confidence must be between 0 and 1")
    if not isinstance(data["evidence"], str) or not data["evidence"].strip():
        raise ValueError("AI review evidence must be a non-empty string")
    return data


def make_ai_review_packet(
    metadata_path: str | Path,
    sampled_frames: list[str] | None = None,
    rubric_path: str | Path = DEFAULT_RUBRIC,
) -> dict[str, Any]:
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = metadata.get("input") or {}
    return {
        "schema_version": "1.0",
        "run_id": metadata.get("run_id"),
        "metadata_path": str(metadata_path.resolve()),
        "source_text": source.get("source_text"),
        "visual_goal": source.get("visual_goal"),
        "generated_video": metadata.get("outputs", {}).get("video", {}).get("path"),
        "sampled_frames": sampled_frames or [],
        "rubric": load_rubric(rubric_path),
        "review_instruction": (
            "Return JSON matching ai_review_schema.json. Treat the review as advisory, cite visible evidence, "
            "and do not infer correctness from file metadata."
        ),
    }


def validate_human_template(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HUMAN_COLUMNS:
            raise ValueError(f"human review columns must be: {', '.join(HUMAN_COLUMNS)}")
        rows = list(reader)
    for index, row in enumerate(rows, start=2):
        for field in ("clearer", "more_faithful", "more_visually_stable"):
            if row[field] and row[field] not in {"A", "B", "tie"}:
                raise ValueError(f"row {index} field '{field}' must be A, B, tie, or empty")
        for field in ("candidate_a_unusable", "candidate_b_unusable"):
            if row[field].lower() not in {"", "true", "false"}:
                raise ValueError(f"row {index} field '{field}' must be true, false, or empty")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or validate offline quality review records.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    packet = subparsers.add_parser("make-ai-packet")
    packet.add_argument("metadata")
    packet.add_argument("--frame", action="append")
    packet.add_argument("--output", required=True)
    ai_validate = subparsers.add_parser("validate-ai-review")
    ai_validate.add_argument("review")
    human_validate = subparsers.add_parser("validate-human-csv")
    human_validate.add_argument("csv")
    args = parser.parse_args(argv)
    if args.command == "make-ai-packet":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(make_ai_review_packet(args.metadata, args.frame), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif args.command == "validate-ai-review":
        validate_ai_review(json.loads(Path(args.review).read_text(encoding="utf-8")))
    else:
        validate_human_template(args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
