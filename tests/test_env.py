"""Unit tests for MuJoCo Desk Watering Simulation Environment."""

from __future__ import annotations

import numpy as np
import pytest

from floraflow.env.desk_env import DeskWateringEnv


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
