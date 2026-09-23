"""Generate synthetic expert demonstrations with synchronized multi-view RGB frames.

Executes the kinematic expert pour planner in DeskWateringEnv with camera rendering enabled
and stores compressed HDF5 demonstration archives for vision-based imitation learning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np

from floraflow.env.desk_env import DeskWateringEnv
from floraflow.expert.pour_planner import PourExpertPlanner


def generate_vision_demonstrations(
    num_demos: int = 100,
    output_path: str = "data/watering_demos_vision_100.h5",
    start_seed: int = 0,
    widened_bounds: bool = True,
    resolution: Tuple[int, int] = (128, 128),
) -> None:
    """Collect and persist visual demonstrations with multi-camera streams.

    Args:
        num_demos: Target number of successful demonstrations to collect.
        output_path: Destination path for the HDF5 archive.
        start_seed: Initial random seed.
        widened_bounds: Whether to sample from widened tabletop workspace bounds.
        resolution: Camera image resolution (width, height).
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    can_range = None
    plant_range = None
    if widened_bounds:
        can_range = {
            "x": (0.42, 0.62),
            "y": (0.05, 0.28),
        }
        plant_range = {
            "x": (0.32, 0.52),
            "y": (-0.32, -0.12),
        }
        print("Using WIDENED workspace bounds for visual demonstration collection.")

    cameras = ("third_person_cam", "overhead_cam")
    env = DeskWateringEnv(
        control_hz=20,
        can_pos_range=can_range,
        plant_pos_range=plant_range,
        include_rgb=True,
        rgb_cameras=cameras,
        rgb_resolution=resolution,
    )
    planner = PourExpertPlanner(env)

    print(f"Initializing visual demonstration collection: {num_demos} episodes target.")
    print(f"Cameras active: {cameras} at {resolution[0]}x{resolution[1]}")
    print(f"Destination: {out_file}")

    total_samples = 0
    successful_episodes = 0
    t0 = time.time()

    with h5py.File(out_file, "w") as f:
        data_grp = f.create_group("data")

        seed = start_seed
        demo_idx = 0

        while demo_idx < num_demos:
            t_ep_start = time.time()
            obs = env.reset(seed=seed)
            init_can_pos = obs["can_pos"].copy()
            init_plant_pos = obs["plant_pos"].copy()

            obs_history, action_history, success = planner.plan_and_execute()

            if not success:
                print(f"Warning: Seed {seed} failed validation. Skipping seed.")
                seed += 1
                continue

            ep_len = len(action_history)
            demo_grp = data_grp.create_group(f"demo_{demo_idx}")

            # Pack observation arrays
            obs_grp = demo_grp.create_group("obs")

            # Store camera image streams
            for cam in cameras:
                cam_key = f"rgb_{cam}"
                stacked_cam = np.stack([o[cam_key] for o in obs_history], axis=0)
                obs_grp.create_dataset(
                    cam_key,
                    data=stacked_cam,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, resolution[1], resolution[0], 3),
                )

            # Store proprioception and state features
            state_keys = [
                "arm_qpos",
                "arm_qvel",
                "gripper_width",
                "ee_pos",
                "ee_quat",
                "can_pos",
                "can_quat",
                "grip_pos",
                "spout_pos",
                "plant_pos",
                "relative_spout_to_plant",
                "tilt_angle_deg",
                "particles_in_pot",
            ]
            for key in state_keys:
                stacked = np.stack([o[key] for o in obs_history], axis=0)
                obs_grp.create_dataset(
                    key,
                    data=stacked,
                    compression="gzip",
                    compression_opts=4,
                )

            # Pack action array
            actions_stacked = np.stack(action_history, axis=0).astype(np.float32)
            demo_grp.create_dataset(
                "actions",
                data=actions_stacked,
                compression="gzip",
                compression_opts=4,
            )

            # Metadata attributes
            demo_grp.attrs["num_samples"] = ep_len
            demo_grp.attrs["seed"] = seed
            demo_grp.attrs["success"] = success
            demo_grp.attrs["can_pos_init"] = init_can_pos
            demo_grp.attrs["plant_pos_init"] = init_plant_pos
            demo_grp.attrs["language_instruction"] = "Grasp the watering can and water the desk plant"

            total_samples += ep_len
            successful_episodes += 1
            demo_idx += 1
            seed += 1

            ep_time = time.time() - t_ep_start
            if demo_idx % 10 == 0 or demo_idx == num_demos:
                elapsed = time.time() - t0
                hz = demo_idx / elapsed
                print(
                    f"Progress: [{demo_idx:3d}/{num_demos}] episodes collected "
                    f"({total_samples:5d} transitions, {hz:.2f} eps/sec, {ep_time:.2f}s/ep)"
                )

        f.attrs["total_episodes"] = successful_episodes
        f.attrs["total_samples"] = total_samples
        f.attrs["control_hz"] = 20
        f.attrs["physics_dt"] = 0.002
        f.attrs["task_name"] = "desk_plant_watering_vision"
        f.attrs["env_class"] = "DeskWateringEnv"
        f.attrs["cameras"] = list(cameras)
        f.attrs["resolution"] = list(resolution)

    elapsed_total = time.time() - t0
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(
        f"\nVisual collection complete: {successful_episodes} episodes, "
        f"{total_samples} transitions stored at {out_file} ({file_size_mb:.1f} MB) in {elapsed_total:.1f}s."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Franka visual plant watering expert demonstrations")
    parser.add_argument("--num-demos", type=int, default=100, help="Number of demonstrations to collect")
    parser.add_argument("--output", type=str, default="data/watering_demos_vision_100.h5", help="HDF5 destination path")
    parser.add_argument("--start-seed", type=int, default=0, help="Initial random seed")
    parser.add_argument("--narrow-bounds", action="store_true", help="Use narrow nominal bounds instead of widened")
    parser.add_argument("--resolution", type=int, default=128, help="Square camera resolution in pixels")
    args = parser.parse_args()

    generate_vision_demonstrations(
        num_demos=args.num_demos,
        output_path=args.output,
        start_seed=args.start_seed,
        widened_bounds=not args.narrow_bounds,
        resolution=(args.resolution, args.resolution),
    )


if __name__ == "__main__":
    main()
