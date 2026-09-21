"""Generate synthetic expert demonstrations for Franka desk plant watering.

Executes the kinematic expert planner in the MuJoCo simulation environment
and persists trajectories as an HDF5 dataset for imitation learning.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Dict, List

import h5py
import numpy as np

from floraflow.env.desk_env import DeskWateringEnv
from floraflow.expert.pour_planner import PourExpertPlanner


def generate_demonstrations(
    num_demos: int = 100,
    output_path: str = "data/watering_demos_100.h5",
    start_seed: int = 0,
) -> None:
    """Generate and store expert demonstrations.

    Args:
        num_demos: Total number of successful episodes to collect.
        output_path: Destination path for the HDF5 archive.
        start_seed: Starting random seed.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    env = DeskWateringEnv(control_hz=20)
    planner = PourExpertPlanner(env)

    print(f"Initializing demonstration collection: {num_demos} episodes target.")
    print(f"Output file destination: {out_file}")

    total_samples = 0
    successful_episodes = 0
    t0 = time.time()

    with h5py.File(out_file, "w") as f:
        data_grp = f.create_group("data")

        seed = start_seed
        demo_idx = 0

        while demo_idx < num_demos:
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
            keys = [
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

            for key in keys:
                stacked = np.stack([o[key] for o in obs_history], axis=0)
                obs_grp.create_dataset(key, data=stacked, compression="gzip")

            # Pack action array
            actions_stacked = np.stack(action_history, axis=0).astype(np.float32)
            demo_grp.create_dataset("actions", data=actions_stacked, compression="gzip")

            # Episode metadata attributes
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

            if demo_idx % 10 == 0 or demo_idx == num_demos:
                elapsed = time.time() - t0
                hz = demo_idx / elapsed
                print(
                    f"Progress: [{demo_idx:3d}/{num_demos}] episodes collected "
                    f"({total_samples:5d} transitions, {hz:.2f} eps/sec)"
                )

        f.attrs["total_episodes"] = successful_episodes
        f.attrs["total_samples"] = total_samples
        f.attrs["control_hz"] = 20
        f.attrs["physics_dt"] = 0.002
        f.attrs["task_name"] = "desk_plant_watering"
        f.attrs["env_class"] = "DeskWateringEnv"

    elapsed_total = time.time() - t0
    print(
        f"Data collection complete: {successful_episodes} episodes, "
        f"{total_samples} samples stored at {out_file} in {elapsed_total:.1f}s."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Franka watering expert demonstrations")
    parser.add_argument("--num-demos", type=int, default=100, help="Number of demonstrations to collect")
    parser.add_argument("--output", type=str, default="data/watering_demos_100.h5", help="HDF5 output path")
    parser.add_argument("--start-seed", type=int, default=0, help="Initial random seed")
    args = parser.parse_args()

    generate_demonstrations(
        num_demos=args.num_demos,
        output_path=args.output,
        start_seed=args.start_seed,
    )


if __name__ == "__main__":
    main()
