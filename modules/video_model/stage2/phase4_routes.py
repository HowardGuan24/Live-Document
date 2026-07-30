"""Freeze the ten-case Phase 4 image-route plan."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .framework.contracts import load_json, sha256_path, write_json
from .framework.route_selector import select_image_route


STAGE2_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = STAGE2_ROOT / "case_registry.json"
OUTPUT_PATH = STAGE2_ROOT / "output" / "phase-4" / "route-plan.json"


def build_route_plan(*, check_only: bool = False) -> dict[str, Any]:
    if check_only:
        plan = load_json(OUTPUT_PATH)
        if sha256_path(REGISTRY_PATH) != plan["source"]["sha256"]:
            raise ValueError("case registry changed after route plan")
        rebuilt = [
            {
                "case_id": case["case_id"],
                "sentinel": case["sentinel"],
                **select_image_route(case),
            }
            for case in load_json(REGISTRY_PATH)["cases"]
        ]
        if rebuilt != plan["routes"]:
            raise ValueError("route selector output changed")
        return plan

    registry = load_json(REGISTRY_PATH)
    routes = [
        {
            "case_id": case["case_id"],
            "sentinel": case["sentinel"],
            **select_image_route(case),
        }
        for case in registry["cases"]
    ]
    plan = {
        "schema_version": "1.0",
        "phase": 4,
        "classification": (
            "data-type route plan; no image or video generation"
        ),
        "source": {
            "path": REGISTRY_PATH.relative_to(STAGE2_ROOT).as_posix(),
            "sha256": sha256_path(REGISTRY_PATH),
        },
        "selector": {
            "path": "framework/route_selector.py",
            "sha256": sha256_path(
                STAGE2_ROOT / "framework/route_selector.py"
            ),
        },
        "routes": routes,
        "route_counts": {
            route_id: sum(
                route["route_id"] == route_id for route in routes
            )
            for route_id in sorted(
                {route["route_id"] for route in routes}
            )
        },
        "model_runs": {"image": 0, "video": 0},
        "automatic_next_action": "run_remaining_case_route_smoke_experiments",
    }
    write_json(OUTPUT_PATH, plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    plan = build_route_plan(check_only=args.check)
    print(
        f"Phase 4 routes: {len(plan['routes'])} cases · "
        f"{len(plan['route_counts'])} route classes · "
        f"next={plan['automatic_next_action']}"
    )


if __name__ == "__main__":
    main()
