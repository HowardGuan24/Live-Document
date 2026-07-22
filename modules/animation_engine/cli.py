"""Command-line interface for the deterministic animation engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manim_renderer import RendererDependencyError
from .pipeline import generate_animation
from .schema import DSLValidationError, load_animation_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render animation DSL JSON to MP4 and GIF")
    parser.add_argument("input", help="Path to an animation DSL or ExplanationSpec JSON file")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Artifact root directory")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and normalize input without starting Manim",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.validate_only:
            spec = load_animation_spec(args.input)
            print(json.dumps({"status": "valid", "id": spec.id}, ensure_ascii=False, indent=2))
            return 0
        result = generate_animation(args.input, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (DSLValidationError, RendererDependencyError, RuntimeError, ValueError) as exc:
        print(f"animation generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
