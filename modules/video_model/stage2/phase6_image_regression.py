"""Build all zero-new-model Phase 6 image regressions."""

from __future__ import annotations

from pathlib import Path

from .framework.contracts import load_json
from .framework.release_image_regression import (
    build_release_image_regression,
)


STAGE2_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = STAGE2_ROOT / "phase6_image_routes.json"
OUTPUT_ROOT = STAGE2_ROOT / "output/phase-6/image-regressions"


def main() -> None:
    config = load_json(CONFIG_PATH)
    passed = 0
    for route in config["new_zero_model_regressions"]:
        manifest = build_release_image_regression(
            stage2_root=STAGE2_ROOT,
            config=route,
            output_root=OUTPUT_ROOT / route["case_id"],
        )
        passed += manifest["status"] == "passed"
        print(
            f"{route['case_id']}: {manifest['status']} · "
            f"{len(manifest['hard_checks'])} image hard gates · "
            "new_model_runs=0"
        )
    if passed != len(config["new_zero_model_regressions"]):
        raise SystemExit("a Phase 6 image regression failed")


if __name__ == "__main__":
    main()
