"""Load case adapters without teaching the generic pipeline their fields."""

from __future__ import annotations

import importlib
from typing import Any, Callable


Adapter = Callable[
    [dict[str, Any]],
    tuple[dict[str, dict[str, Any]], dict[str, Any]],
]


def get_state_adapter(name: str) -> Adapter:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"invalid state adapter name: {name}")
    module = importlib.import_module(f"{__name__}.{name}")
    adapter = getattr(module, "load_selected_states", None)
    if not callable(adapter):
        raise ValueError(
            f"state adapter {name!r} has no load_selected_states()"
        )
    return adapter
