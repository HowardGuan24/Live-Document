"""Generic CLI for one spec-driven Phase 3 image experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .framework.contracts import load_json
from .framework.image_experiment import (
    generate_experiment,
    prepare_experiment,
)
from .framework.material_projection import (
    build_ensemble_material_projection,
    build_material_projection,
)


STAGE2_ROOT = Path(__file__).resolve().parent
PHASE_ROOT = STAGE2_ROOT / "output" / "phase-3"


def resolve_spec(experiment_id: str) -> dict[str, Any]:
    path = STAGE2_ROOT / "experiments" / experiment_id / "spec.json"
    spec = load_json(path)
    if spec["experiment_id"] != experiment_id:
        raise ValueError("experiment ID does not match spec directory")
    resolved = json.loads(json.dumps(spec))
    for field in ("clean_frame", "semantic_layers"):
        resolved["source"][field] = str(
            (STAGE2_ROOT / spec["source"][field]).resolve()
        )
    for configuration in resolved["configurations"]:
        reuse = configuration.get("reuse_from")
        if reuse:
            reuse["experiment_root"] = str(
                (STAGE2_ROOT / reuse["experiment_root"]).resolve()
            )
    resolved["control_overrides"] = {
        name: str((STAGE2_ROOT / value).resolve())
        for name, value in spec.get("control_overrides", {}).items()
    }
    planned = len(spec["configurations"]) * len(spec["render"]["seeds"])
    if planned != spec["budget"]["actual_planned_image_candidates"]:
        raise ValueError("planned image count does not match matrix")
    if (
        spec["budget"].get("maximum_new_generation", planned)
        > spec["budget"]["maximum_new_image_candidates"]
    ):
        raise ValueError("planned new image generation exceeds budget")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--project-material", action="store_true")
    parser.add_argument("--project-material-ensemble", action="store_true")
    parser.add_argument("--projection-variant")
    parser.add_argument("--residual-gain", type=float)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not (
        args.prepare
        or args.generate
        or args.project_material
        or args.project_material_ensemble
    ):
        parser.error(
            "choose --prepare, --generate, --project-material, "
            "or --project-material-ensemble"
        )
    spec = resolve_spec(args.experiment)
    root = PHASE_ROOT / args.experiment
    if args.prepare:
        prepared = prepare_experiment(spec, root)
        print(
            f"{args.experiment}: prepared "
            f"{len(prepared['controls'])} controls"
        )
    if args.generate:
        metadata = generate_experiment(spec, root, force=args.force)
        print(
            f"{args.experiment}: {metadata['status']} · "
            f"{metadata['cache']['generated']} generated · "
            f"{metadata['cache']['reused']} reused"
        )
    if args.project_material:
        settings = spec.get("projection")
        if not settings:
            raise ValueError("experiment spec has no projection settings")
        manifest = build_material_projection(
            root,
            blur_radius=float(settings["blur_radius"]),
            residual_gain=float(settings["residual_gain"]),
            protect_dilation_px=int(settings["protect_dilation_px"]),
        )
        print(
            f"{args.experiment}: projected "
            f"{len(manifest['records'])} composites · "
            f"{sum(item['passed'] for item in manifest['hard_checks'])}/"
            f"{len(manifest['hard_checks'])} hard checks passed"
        )
    if args.project_material_ensemble:
        settings = spec.get("projection")
        if not settings:
            raise ValueError("experiment spec has no projection settings")
        manifest = build_ensemble_material_projection(
            root,
            blur_radius=float(settings["blur_radius"]),
            residual_gain=float(
                args.residual_gain
                if args.residual_gain is not None
                else settings["residual_gain"]
            ),
            residual_clip=float(settings["residual_clip"]),
            protect_dilation_px=int(settings["protect_dilation_px"]),
            variant_id=(
                args.projection_variant
                or settings.get("variant_id", "default")
            ),
            allowed_region_layer_id=settings.get(
                "allowed_region_layer_id"
            ),
        )
        print(
            f"{args.experiment}: projected robust ensemble · "
            f"{sum(item['passed'] for item in manifest['hard_checks'])}/"
            f"{len(manifest['hard_checks'])} hard checks passed"
        )


if __name__ == "__main__":
    main()
