"""Optional case evaluators loaded by name from the sequence specification."""

from __future__ import annotations

import importlib
from typing import Any, Callable


CaseEvaluator = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]],
    list[dict[str, Any]],
]


def get_case_evaluator(name: str) -> CaseEvaluator:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"invalid case evaluator name: {name}")
    module = importlib.import_module(f"{__name__}.{name}")
    evaluator = getattr(module, "evaluate_case", None)
    if not callable(evaluator):
        raise ValueError(f"case evaluator {name!r} has no evaluate_case()")
    return evaluator
