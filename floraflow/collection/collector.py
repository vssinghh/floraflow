"""Demonstration Data Collection Pipeline for State and Multi-Camera Vision Trajectories.

Executes the kinematic expert planner in DeskWateringEnv and persists validated,
collision-free demonstration archives as compressed HDF5 files inside datasets/.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Tuple

import h5py
import numpy as np

from floraflow.common.env import DeskWateringEnv
from floraflow.collection.pour_planner import PourExpertPlanner


STATE_KEYS = [
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


def generate_demonstrations(
    num_demos: int = 100,
    output_path: str = "datasets/watering_demos_100.h5",
    start_seed: int = 0,
    widened_bounds: bool = False,
    domain_rand: bool = False,
) -> None:
    """Generate and store state-based expert demonstrations.

    Args:
        num_demos: Total number of successful episodes to collect.
        output_path: Destination path for the HDF5 archive.
        start_seed: Starting random seed.
        widened_bounds: Whether to sample from expanded desk workspace bounds.
        domain_rand: Whether to apply Sim-to-Real visual and physical domain randomization.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    can_range = None
    plant_range = None
    if widened_bounds:
        can_range = {
            "x": (0.42, 0.62),
            "y": (0.120, 0.28),
        }
        plant_range = {
            "x": (0.32, 0.52),
            "y": (-0.32, -0.170),
        }
        print("Using WIDENED workspace bounds for collection.")

    env = DeskWateringEnv(
        control_hz=20,
        can_pos_range=can_range,
        plant_pos_range=plant_range,
        domain_rand=domain_rand,
    )
    planner = PourExpertPlanner(env)

    print(f"Initializing demonstration collection: {num_demos} episodes target (domain_rand={domain_rand}).")
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

            if env.has_initial_collision() or not env.is_valid_spawn(
                (float(init_can_pos[0]), float(init_can_pos[1])),
                (float(init_plant_pos[0]), float(init_plant_pos[1])),
            ):
                print(f"Warning: Seed {seed} had initial spawn collision or insufficient clearance. Skipping seed.")
                seed += 1
                continue

            obs_history, action_history, success = planner.plan_and_execute()

            if not success:
                print(f"Warning: Seed {seed} failed validation. Skipping seed.")
                seed += 1
                continue

            ep_len = len(action_history)
            demo_grp = data_grp.create_group(f"demo_{demo_idx}")
            obs_grp = demo_grp.create_group("obs")

            for key in STATE_KEYS:
                stacked = np.stack([o[key] for o in obs_history], axis=0)
                obs_grp.create_dataset(key, data=stacked, compression="gzip")

            actions_stacked = np.stack(action_history, axis=0).astype(np.float32)
            demo_grp.create_dataset("actions", data=actions_stacked, compression="gzip")

            demo_grp.attrs["num_samples"] = ep_len
            demo_grp.attrs["seed"] = seed
            demo_grp.attrs["success"] = success
            demo_grp.attrs["can_pos_init"] = init_can_pos
            demo_grp.attrs["plant_pos_init"] = init_plant_pos
            demo_grp.attrs["domain_rand"] = bool(domain_rand)
            if env.last_domain_params is not None:
                demo_grp.attrs["domain_params_json"] = json.dumps(env.last_domain_params)
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
        f.attrs["domain_rand"] = bool(domain_rand)

    elapsed_total = time.time() - t0
    print(
        f"Data collection complete: {successful_episodes} episodes, "
        f"{total_samples} samples stored at {out_file} in {elapsed_total:.1f}s."
    )


def generate_vision_demonstrations(
    num_demos: int = 100,
    output_path: str = "datasets/watering_demos_vision_100.h5",
    start_seed: int = 0,
    widened_bounds: bool = True,
    resolution: Tuple[int, int] = (128, 128),
    cameras: Tuple[str, ...] = ("third_person_cam", "overhead_cam", "wrist_cam"),
    domain_rand: bool = False,
) -> None:
    """Collect and persist visual demonstrations with multi-camera streams.

    Args:
        num_demos: Target number of successful demonstrations to collect.
        output_path: Destination path for the HDF5 archive.
        start_seed: Initial random seed.
        widened_bounds: Whether to sample from widened tabletop workspace bounds.
        resolution: Camera image resolution (width, height).
        cameras: Tuple of camera names to capture.
        domain_rand: Whether to apply Sim-to-Real visual and physical domain randomization.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    can_range = None
    plant_range = None
    if widened_bounds:
        can_range = {
            "x": (0.42, 0.62),
            "y": (0.120, 0.28),
        }
        plant_range = {
            "x": (0.32, 0.52),
            "y": (-0.32, -0.170),
        }
        print("Using WIDENED workspace bounds for visual demonstration collection.")

    cameras = tuple(cameras)
    env = DeskWateringEnv(
        control_hz=20,
        can_pos_range=can_range,
        plant_pos_range=plant_range,
        include_rgb=True,
        rgb_cameras=cameras,
        rgb_resolution=resolution,
        domain_rand=domain_rand,
    )
    planner = PourExpertPlanner(env)

    print(f"Initializing visual demonstration collection: {num_demos} episodes target (domain_rand={domain_rand}).")
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

            if env.has_initial_collision() or not env.is_valid_spawn(
                (float(init_can_pos[0]), float(init_can_pos[1])),
                (float(init_plant_pos[0]), float(init_plant_pos[1])),
            ):
                print(f"Warning: Seed {seed} had initial spawn collision or insufficient clearance. Skipping seed.")
                seed += 1
                continue

            obs_history, action_history, success = planner.plan_and_execute()

            if not success:
                print(f"Warning: Seed {seed} failed validation. Skipping seed.")
                seed += 1
                continue

            ep_len = len(action_history)
            demo_grp = data_grp.create_group(f"demo_{demo_idx}")
            obs_grp = demo_grp.create_group("obs")

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

            for key in STATE_KEYS:
                stacked = np.stack([o[key] for o in obs_history], axis=0)
                obs_grp.create_dataset(
                    key,
                    data=stacked,
                    compression="gzip",
                    compression_opts=4,
                )

            actions_stacked = np.stack(action_history, axis=0).astype(np.float32)
            demo_grp.create_dataset(
                "actions",
                data=actions_stacked,
                compression="gzip",
                compression_opts=4,
            )

            demo_grp.attrs["num_samples"] = ep_len
            demo_grp.attrs["seed"] = seed
            demo_grp.attrs["success"] = success
            demo_grp.attrs["can_pos_init"] = init_can_pos
            demo_grp.attrs["plant_pos_init"] = init_plant_pos
            demo_grp.attrs["domain_rand"] = bool(domain_rand)
            if env.last_domain_params is not None:
                demo_grp.attrs["domain_params_json"] = json.dumps(env.last_domain_params)
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
        f.attrs["domain_rand"] = bool(domain_rand)

    elapsed_total = time.time() - t0
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(
        f"\nVisual collection complete: {successful_episodes} episodes, "
        f"{total_samples} transitions stored at {out_file} ({file_size_mb:.1f} MB) in {elapsed_total:.1f}s."
    )
