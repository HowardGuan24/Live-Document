"""Load case semantic builders behind a small generic interface."""

from __future__ import annotations

import importlib
from types import ModuleType


def get_semantic_builder(name: str) -> ModuleType:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"invalid semantic builder name: {name}")
    module = importlib.import_module(f"{__name__}.{name}")
    for method in ("prepare_context", "build_layers"):
        if not callable(getattr(module, method, None)):
            raise ValueError(
                f"semantic builder {name!r} has no {method}()"
            )
    return module
