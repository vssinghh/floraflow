"""Unit tests for DLS Inverse Kinematics, trajectory generation, and expert planner."""

from __future__ import annotations

import numpy as np
import pytest

from floraflow.env.desk_env import DeskWateringEnv
from floraflow.expert.ik_solver import IKSolver
from floraflow.expert.trajectory import interpolate_joint_trajectory, minimum_jerk_scaling


def test_minimum_jerk_boundary_conditions() -> None:
    """Verify minimum jerk polynomial satisfies zero velocity boundary conditions."""
    duration = 2.0
    s0, ds0, dds0 = minimum_jerk_scaling(0.0, duration)
    assert s0 == pytest.approx(0.0, abs=1e-6)
    assert ds0 == pytest.approx(0.0, abs=1e-6)

    s1, ds1, dds1 = minimum_jerk_scaling(duration, duration)
    assert s1 == pytest.approx(1.0, abs=1e-6)
    assert ds1 == pytest.approx(0.0, abs=1e-6)


def test_joint_interpolation_step_count() -> None:
    """Verify joint trajectory interpolator returns valid trajectory endpoints."""
    q0 = np.zeros(7)
    q1 = np.ones(7)
    traj = interpolate_joint_trajectory(q0, q1, duration=1.0, control_hz=20)
    assert len(traj) == 21  # 20 intervals + 1 endpoint
    np.testing.assert_allclose(traj[0], q0, atol=1e-5)
    np.testing.assert_allclose(traj[-1], q1, atol=1e-5)


def test_ik_solver_convergence() -> None:
    """Verify DLS IK solver converges on a standard tabletop Cartesian point."""
    env = DeskWateringEnv()
    obs = env.reset(seed=0)
    ik = IKSolver(
        env.model,
        site_name="pinch",
        pos_tol=5e-3,
        rot_tol=0.15,
        rot_weight=0.04,
        max_iterations=300,
    )
    target_pos = np.array([0.50, 0.0, 0.55])
    R_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

    res = ik.solve(target_pos, target_rot=R_down, q_init=obs["arm_qpos"])
    assert res.success
    assert res.pos_error < 5e-3
