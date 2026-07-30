"""Audit Phase 5 prompts with the exact tokenizer embedded in the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from safetensors import safe_open


STAGE2_ROOT = Path(__file__).resolve().parent
COMFY_ROOT = Path("/persistent/ComfyUI")
TEXT_ENCODER = (
    COMFY_ROOT
    / "models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"
)
SOURCE_ROOT = STAGE2_ROOT / "phase5_experiments"
RUN_ROOT = STAGE2_ROOT / "output/phase-5/experiments"
OUTPUT_PATH = (
    STAGE2_ROOT / "output/phase-6/video-token-integrity.json"
)
RELEASE_SOFT_LIMIT = 1024


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _token_record(tokenizer: Any, text: str) -> dict[str, Any]:
    tokenized = tokenizer.tokenize_with_weights(text)
    rows = tokenized["gemma3_12b"]
    ids = [
        int(token_id)
        for row in rows
        for token_id, _weight in row
    ]
    content = [token_id for token_id in ids if token_id != 0]
    return {
        "content_tokens_including_bos": len(content),
        "padded_tokens_consumed_by_encoder": len(ids),
        "row_count": len(rows),
        "row_lengths": [len(row) for row in rows],
        "padding_tokens": len(ids) - len(content),
        "first_non_padding_token_id": (
            content[0] if content else None
        ),
        "bos_token_present": bool(content and content[0] == 2),
        "release_soft_limit": RELEASE_SOFT_LIMIT,
        "would_exceed_release_soft_limit": (
            len(content) > RELEASE_SOFT_LIMIT
        ),
        "text_sha256": _sha256_bytes(text.encode("utf-8")),
        "character_count": len(text),
    }


def build_token_audit() -> dict[str, Any]:
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    from comfy.text_encoders.lt import LTXAVGemmaTokenizer

    with safe_open(
        str(TEXT_ENCODER), framework="pt", device="cpu"
    ) as handle:
        sentencepiece = handle.get_tensor("spiece_model")
    sentencepiece_bytes = sentencepiece.numpy().tobytes()
    tokenizer = LTXAVGemmaTokenizer(
        tokenizer_data={"spiece_model": sentencepiece}
    )

    experiments = []
    for spec_path in sorted(SOURCE_ROOT.glob("EXP-P5-*/spec.json")):
        spec = _load_json(spec_path)
        if "prompt" not in spec:
            continue
        run_path = (
            RUN_ROOT
            / spec["experiment_id"]
            / "_work"
            / "run.json"
        )
        run = _load_json(run_path)
        if run["model_runs"]["video"] != 1:
            continue
        records = {
            name: _token_record(tokenizer, spec["prompt"][name])
            for name in ("positive", "negative")
        }
        experiments.append(
            {
                "experiment_id": spec["experiment_id"],
                "case_id": spec["case_id"],
                "text_encoder_declared": spec["model"][
                    "text_encoder"
                ],
                "positive": records["positive"],
                "negative": records["negative"],
                "passed": all(
                    not record[
                        "would_exceed_release_soft_limit"
                    ]
                    and record["row_count"] == 1
                    and record["padded_tokens_consumed_by_encoder"]
                    == RELEASE_SOFT_LIMIT
                    and record["bos_token_present"]
                    for record in records.values()
                ),
            }
        )
    result = {
        "schema_version": "1.0",
        "status": (
            "passed"
            if experiments
            and all(item["passed"] for item in experiments)
            else "failed"
        ),
        "measurement": {
            "tokenizer_class": "LTXAVGemmaTokenizer",
            "inner_tokenizer_class": "Gemma3_12BTokenizer",
            "embedding_key": "gemma3_12b",
            "padding_side": "left",
            "padding_token_id": 0,
            "bos_token_id": 2,
            "minimum_padded_length": RELEASE_SOFT_LIMIT,
            "maximum_length_declared_by_comfyui": 99999999,
            "release_soft_limit": RELEASE_SOFT_LIMIT,
            "template_mode": (
                "skip_template=True, matching CLIPTextEncode default"
            ),
            "explanation_zh": (
                "直接从登记的 Gemma safetensors 读取 spiece_model，"
                "调用 ComfyUI 的 LTXAVGemmaTokenizer；content token "
                "排除左侧 pad=0，但包含实际 BOS=2。"
            ),
        },
        "tokenizer_asset": {
            "text_encoder_path": str(TEXT_ENCODER),
            "embedded_key": "spiece_model",
            "embedded_size_bytes": len(sentencepiece_bytes),
            "embedded_sha256": _sha256_bytes(
                sentencepiece_bytes
            ),
        },
        "experiment_count": len(experiments),
        "experiments": experiments,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            result, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = build_token_audit()
    print(
        "Phase 6 video token audit: "
        f"{result['status']} · "
        f"{result['experiment_count']} model video prompts"
    )


if __name__ == "__main__":
    main()
