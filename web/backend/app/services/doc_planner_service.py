"""Wrap the repo's doc_planner pipeline: text -> LearningSpec list."""

from __future__ import annotations

import json
from typing import Any

# config.py already inserts REPO_ROOT into sys.path on import
from app.config import REPO_ROOT  # noqa: F401  (side-effect: sys.path)
from modules.doc_planner.parser import parse_document
from modules.doc_planner.generator import generate_spec


def plan_document(text: str) -> list[dict[str, Any]]:
    """Run the rule-based document planning pipeline and return LearningSpec dicts."""
    segments = parse_document(text)
    specs = []
    for seg in segments:
        spec = generate_spec(seg)
        if spec is not None:
            specs.append(json.loads(spec.to_json()))
    return specs


def first_suitable(text: str) -> dict[str, Any] | None:
    """Return the first spec that is suitable for animation, else None."""
    for spec in plan_document(text):
        if spec.get("fallback_reason") is None and spec.get("learning_goal"):
            return spec
    return None
