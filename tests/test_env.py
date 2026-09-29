"""Unit tests for MuJoCo Desk Watering Simulation Environment."""

from __future__ import annotations

import numpy as np
import pytest

from floraflow.common.env import DeskWateringEnv


def test_env_initialization() -> None:
    """Verify simulation environment loads and initializes correct dimensions."""
    env = DeskWateringEnv()
    assert env.model is not None
    assert env.data is not None
    assert env.model.nq > 0
    assert env.model.nv > 0
    assert env.control_hz == 20
    assert env.physics_dt == 0.002
    assert env.substeps == 25


def test_env_reset_deterministic() -> None:
    """Verify reset is repeatable and deterministic with a fixed seed."""
    env = DeskWateringEnv()
    obs1 = env.reset(seed=42)
    can_pos1 = obs1["can_pos"].copy()
    plant_pos1 = obs1["plant_pos"].copy()

    obs2 = env.reset(seed=42)
    can_pos2 = obs2["can_pos"].copy()
    plant_pos2 = obs2["plant_pos"].copy()

    np.testing.assert_allclose(can_pos1, can_pos2, atol=1e-5)
    np.testing.assert_allclose(plant_pos1, plant_pos2, atol=1e-5)


def test_env_observation_schema() -> None:
    """Verify observation dictionary contains required physical signals."""
    env = DeskWateringEnv()
    obs = env.reset(seed=0)

    expected_keys = [
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
    for k in expected_keys:
        assert k in obs, f"Missing observation key: {k}"

    assert obs["arm_qpos"].shape == (7,)
    assert obs["arm_qvel"].shape == (7,)
    assert obs["ee_pos"].shape == (3,)
    assert obs["ee_quat"].shape == (4,)
    assert obs["tilt_angle_deg"].shape == (1,)
    assert obs["particles_in_pot"].shape == (1,)


def test_env_step_contract() -> None:
    """Verify action execution updates simulation and returns valid types."""
    env = DeskWateringEnv()
    obs = env.reset(seed=0)
    init_qpos = obs["arm_qpos"].copy()

    action = np.concatenate([init_qpos, [1.0]]).astype(np.float32)
    next_obs, reward, done, info = env.step(action)

    assert isinstance(next_obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "success" in info
    assert "is_pouring" in info
    assert "spout_dist" in info
    assert "tilt_deg" in info


def test_spawn_clearance_and_collision_validation() -> None:
    """Verify geometric spawn clearance rules and physics contact collision checks."""
    from floraflow.evaluation.evaluator import generate_ood_configurations

    env = DeskWateringEnv()

    # Valid nominal spawn
    assert DeskWateringEnv.is_valid_spawn((0.52, 0.16), (0.42, -0.22))

    # Reject objects too close to each other (< 0.22 m)
    assert not DeskWateringEnv.is_valid_spawn((0.45, 0.09), (0.45, -0.10))

    # Reject watering can too close to right finger (cy < 0.09 m)
    assert not DeskWateringEnv.is_valid_spawn((0.48, 0.04), (0.42, -0.22))

    # Reject plant pot too close to left finger (py > -0.14 m)
    assert not DeskWateringEnv.is_valid_spawn((0.52, 0.16), (0.45, -0.09))

    # Forcing plant pot onto the left finger at Step 0 triggers has_initial_collision
    env.reset(seed=0, can_xy=(0.52, 0.16), plant_xy=(0.45, -0.095))
    assert env.has_initial_collision()

    # Standard reset and all OOD configurations must be 100% collision-free
    obs = env.reset(seed=0)
    assert not env.has_initial_collision()
    assert abs(float(obs["gripper_width"][0]) - 0.08) < 1e-3

    for difficulty in ("mild", "hard", "extreme"):
        seeds, can_xys, plant_xys = generate_ood_configurations(
            num_episodes=50, base_seed=200, difficulty=difficulty
        )
        for s, c_xy, p_xy in zip(seeds, can_xys, plant_xys):
            assert DeskWateringEnv.is_valid_spawn(c_xy, p_xy), (
                f"Invalid {difficulty} OOD spawn for seed {s}: can={c_xy}, plant={p_xy}"
            )
    # Also verify physics settling on all 50 hard OOD configurations has zero collisions
    _, hard_cans, hard_plants = generate_ood_configurations(
        num_episodes=50, base_seed=200, difficulty="hard"
    )
    for idx, (c_xy, p_xy) in enumerate(zip(hard_cans, hard_plants)):
        env.reset(seed=200 + idx, can_xy=c_xy, plant_xy=p_xy)
        assert not env.has_initial_collision(), f"Hard OOD seed {200 + idx} had initial collision"


def test_stored_hdf5_datasets_clean_spawns() -> None:
    """Verify that all HDF5 demonstration archives in datasets/ have valid clearance and unperturbed Step 0 grippers."""
    from pathlib import Path
    import h5py

    data_dir = Path(__file__).resolve().parent.parent / "datasets"
    if not data_dir.exists():
        return

    for h5_path in sorted(data_dir.glob("*.h5")):
        with h5py.File(h5_path, "r") as f:
            if "data" not in f:
                continue
            for demo_key in f["data"].keys():
                grp = f["data"][demo_key]
                c_init = grp.attrs["can_pos_init"][:2]
                p_init = grp.attrs["plant_pos_init"][:2]
                gw0 = float(grp["obs/gripper_width"][0][0])
                assert DeskWateringEnv.is_valid_spawn(
                    (float(c_init[0]), float(c_init[1])),
                    (float(p_init[0]), float(p_init[1])),
                ), f"{h5_path.name}/{demo_key} failed is_valid_spawn: can={c_init}, plant={p_init}"
                assert abs(gw0 - 0.08) <= 1e-3, (
                    f"{h5_path.name}/{demo_key} had deflected Step 0 gripper_width={gw0:.4f}"
                )


def test_env_domain_randomization_and_restore(tmp_path) -> None:
    """Verify Sim-to-Real domain randomization sampling, application, nominal restore, and 4-split manifest."""
    from floraflow.evaluation.evaluator import load_or_create_sim2real_benchmark

    env = DeskWateringEnv(domain_rand=True)
    nom_light_pos = env._init_light_pos.copy()
    nom_can_mass = float(env._init_body_mass[env._can_body_id])

    # 1. Reset with domain_rand=True must perturb lighting, camera extrinsics, and physical mass
    env.reset(seed=300)
    assert env.last_domain_params is not None
    assert "can_mass_scale" in env.last_domain_params
    assert "cam_pos_deltas" in env.last_domain_params
    assert not np.allclose(env.model.light_pos, nom_light_pos)
    assert abs(float(env.model.body_mass[env._can_body_id]) - nom_can_mass) > 1e-5

    # 2. Reset with domain_rand=False must restore exact nominal studio parameters
    env.domain_rand = False
    env.reset(seed=300)
    assert env.last_domain_params is None
    np.testing.assert_allclose(env.model.light_pos, nom_light_pos, atol=1e-6)
    assert abs(float(env.model.body_mass[env._can_body_id]) - nom_can_mass) < 1e-6

    # 3. Verify 4-split Sim-to-Real benchmark manifest generation & reload
    manifest_file = tmp_path / "test_sim2real_manifest.json"
    manifest = load_or_create_sim2real_benchmark(manifest_path=str(manifest_file))
    assert manifest_file.exists()
    assert set(manifest.keys()) == {"clean_id", "clean_ood", "dr_id", "dr_ood"}
    assert len(manifest["clean_id"]) == 20
    assert len(manifest["clean_ood"]) == 50
    assert len(manifest["dr_id"]) == 20
    assert len(manifest["dr_ood"]) == 50
    assert manifest["dr_id"][0]["domain_params"] is not None
    assert manifest["clean_id"][0]["domain_params"] is None


