"""Small built-in text-to-LearningSpec planner used by the Web API."""

from __future__ import annotations

import re
from typing import Any


def plan_document(text: str) -> list[dict[str, Any]]:
    """Create one compact LearningSpec without the removed legacy planner."""
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return []

    clauses = [
        clause.strip(" ，,；;：:")
        for clause in re.split(r"[。！？!?；;\n]+", normalized)
        if clause.strip(" ，,；;：:")
    ]
    if not clauses:
        clauses = [normalized]

    entities = []
    for clause in clauses[:6]:
        label = clause[:24]
        if label and label not in entities:
            entities.append(label)

    causal_steps = [
        {"cause": clauses[index][:46], "change": clauses[index + 1][:46]}
        for index in range(min(len(clauses) - 1, 5))
    ]
    return [{
        "learning_goal": clauses[0][:120],
        "entities": entities,
        "state_variables": [],
        "causal_steps": causal_steps,
        "invariants": [],
        "comprehension_questions": [],
        "fallback_reason": None,
    }]


def first_suitable(text: str) -> dict[str, Any] | None:
    """Return the first spec that is suitable for animation, else None."""
    for spec in plan_document(text):
        if spec.get("fallback_reason") is None and spec.get("learning_goal"):
            return spec
    return None
