"""Clean CLI Entrypoint for Multi-Camera Vision Demonstration Collection."""

from __future__ import annotations
from floraflow.collection.collector import generate_vision_demonstrations
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Franka visual plant watering expert demonstrations")
    parser.add_argument("--num-demos", type=int, default=100, help="Number of demonstrations to collect")
    parser.add_argument("--output", type=str, default="datasets/watering_demos_vision_100.h5", help="HDF5 destination path")
    parser.add_argument("--start-seed", type=int, default=0, help="Initial random seed")
    parser.add_argument("--narrow-bounds", action="store_true", help="Use narrow nominal bounds instead of widened")
    parser.add_argument("--resolution", type=int, default=128, help="Square camera resolution in pixels")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["third_person_cam", "overhead_cam", "wrist_cam"],
        help="Names of cameras to capture",
    )
    args = parser.parse_args()

    generate_vision_demonstrations(
        num_demos=args.num_demos,
        output_path=args.output,
        start_seed=args.start_seed,
        widened_bounds=not args.narrow_bounds,
        resolution=(args.resolution, args.resolution),
        cameras=tuple(args.cameras),
    )


if __name__ == "__main__":
    main()
