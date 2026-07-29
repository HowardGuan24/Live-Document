"""Compose visible prompt parts and validate both SDXL tokenizers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import CLIPTokenizer

from ..enhance import model_paths
from .utils import write_json


def build_prompt_record(
    spec: dict[str, Any],
    keyframe: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    positive = f"{spec['common_visual']}, {keyframe['mechanism_delta']}"
    negative = f"{spec['common_negative']}, {keyframe['stage_forbidden']}"
    counts: dict[str, dict[str, int]] = {}
    base_path = model_paths()["sdxl_base"]
    for subfolder in ("tokenizer", "tokenizer_2"):
        tokenizer = CLIPTokenizer.from_pretrained(
            str(base_path),
            subfolder=subfolder,
            local_files_only=True,
        )
        pos_count = len(
            tokenizer(
                positive,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        neg_count = len(
            tokenizer(
                negative,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        limit = int(tokenizer.model_max_length)
        if pos_count > limit or neg_count > limit:
            raise ValueError(
                f"prompt exceeds {subfolder}: "
                f"{pos_count}/{neg_count} > {limit}"
            )
        counts[subfolder] = {
            "positive": pos_count,
            "negative": neg_count,
            "limit": limit,
        }
    record = {
        "keyframe_id": keyframe["id"],
        "common_visual": spec["common_visual"],
        "mechanism_delta": keyframe["mechanism_delta"],
        "stage_forbidden": keyframe["stage_forbidden"],
        "common_negative": spec["common_negative"],
        "positive_combined": positive,
        "negative_combined": negative,
        "token_counts": counts,
    }
    prompt_root = output_root / "_work" / "prompts"
    write_json(
        prompt_root / "prompt_parts" / f"{keyframe['id']}.json",
        record,
    )
    combined_root = prompt_root / "combined"
    combined_root.mkdir(parents=True, exist_ok=True)
    (combined_root / f"{keyframe['id']}_positive.txt").write_text(
        positive + "\n", encoding="utf-8"
    )
    (combined_root / f"{keyframe['id']}_negative.txt").write_text(
        negative + "\n", encoding="utf-8"
    )
    return record

