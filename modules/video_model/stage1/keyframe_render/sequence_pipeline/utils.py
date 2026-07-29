"""Small file and image helpers shared by the sequence pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.uint8(np.clip(array, 0.0, 1.0) * 255),
        mode="L",
    ).save(path)


def image_record(path: Path, **extra: Any) -> dict[str, Any]:
    with Image.open(path) as image:
        record = {
            "path": str(path.resolve()),
            "relative_name": path.name,
            "sha256": sha256(path),
            "size": list(image.size),
            "mode": image.mode,
        }
    record.update(extra)
    return record
