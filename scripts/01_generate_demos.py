"""CLI Entrypoint: Collect State-Based Expert Demonstrations.

Delegates to floraflow.collection.collector.generate_demonstrations.
"""

from __future__ import annotations

import argparse

from floraflow.collection.collector import generate_demonstrations

__all__ = ["generate_demonstrations", "main"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Franka watering expert demonstrations")
    parser.add_argument("--num-demos", type=int, default=100, help="Number of demonstrations to collect")
    parser.add_argument("--output", type=str, default="datasets/watering_demos_100.h5", help="HDF5 output path")
    parser.add_argument("--start-seed", type=int, default=0, help="Initial random seed")
    parser.add_argument("--widened-bounds", action="store_true", help="Sample from expanded workspace bounds")
    args = parser.parse_args()

    generate_demonstrations(
        num_demos=args.num_demos,
        output_path=args.output,
        start_seed=args.start_seed,
        widened_bounds=args.widened_bounds,
    )


if __name__ == "__main__":
    main()
