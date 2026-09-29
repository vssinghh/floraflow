"""Self-Contained CLI Entrypoint for Data Collection (`python -m floraflow.collection`).

Supports both multi-camera RGB-D/vision demonstration collection (default)
and low-dimensional state demonstration collection (`--state`).
"""

from __future__ import annotations

import argparse

from floraflow.collection.collector import (
    generate_demonstrations,
    generate_vision_demonstrations,
)

__all__ = ["generate_demonstrations", "generate_vision_demonstrations", "main"]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m floraflow.collection",
        description="Collect Franka plant watering expert demonstrations (vision or state)",
    )
    parser.add_argument(
        "--state",
        action="store_true",
        help="Collect low-dimensional state-only demonstrations instead of multi-camera RGB vision",
    )
    parser.add_argument("--num-demos", type=int, default=100, help="Number of demonstrations to collect")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="HDF5 destination path (defaults to datasets/watering_demos_vision_100.h5 or datasets/watering_demos_100.h5)",
    )
    parser.add_argument("--start-seed", type=int, default=0, help="Initial random seed")
    parser.add_argument("--widened-bounds", action="store_true", help="Sample from expanded workspace bounds (state mode)")
    parser.add_argument("--narrow-bounds", action="store_true", help="Use narrow nominal bounds instead of widened (vision mode)")
    parser.add_argument("--resolution", type=int, default=128, help="Square camera resolution in pixels")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["third_person_cam", "overhead_cam", "wrist_cam"],
        help="Names of cameras to capture",
    )
    parser.add_argument(
        "--domain-rand",
        action="store_true",
        help="Enable Sim-to-Real visual and physical domain randomization during collection",
    )
    args = parser.parse_args()

    suffix = "_dr" if args.domain_rand else ""
    if args.state:
        out_path = args.output or f"datasets/watering_demos{suffix}_{args.num_demos}.h5"
        generate_demonstrations(
            num_demos=args.num_demos,
            output_path=out_path,
            start_seed=args.start_seed,
            widened_bounds=args.widened_bounds,
            domain_rand=args.domain_rand,
        )
    else:
        out_path = args.output or f"datasets/watering_demos_vision{suffix}_{args.num_demos}.h5"
        generate_vision_demonstrations(
            num_demos=args.num_demos,
            output_path=out_path,
            start_seed=args.start_seed,
            widened_bounds=not args.narrow_bounds,
            resolution=(args.resolution, args.resolution),
            cameras=tuple(args.cameras),
            domain_rand=args.domain_rand,
        )


if __name__ == "__main__":
    main()
