"""Load a case composition strategy by specification name."""

from __future__ import annotations

import importlib
from typing import Any


def compose_with(
    name: str,
    spec: dict[str, Any],
    output_root: Any,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"invalid composer name: {name}")
    module = importlib.import_module(f"{__name__}.{name}")
    function = getattr(module, "compose_sequence", None)
    if not callable(function):
        raise ValueError(f"composer {name!r} has no compose_sequence()")
    return function(spec, output_root, force=force)
