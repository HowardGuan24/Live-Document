"""Dispatch mechanism-constrained composition without case layer names."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .composers import compose_with


def compose_sequence(
    spec: dict[str, Any],
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    return compose_with(
        spec["composer"], spec, output_root, force=force
    )
