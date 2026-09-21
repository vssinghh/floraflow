"""Deterministic 6-DoF Expert Trajectory Planner for Plant Watering.

Executes a pick, high lift, spatial transport, pour tilt, hold, and upright
sequence using minimum-jerk interpolation and DLS Inverse Kinematics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from floraflow.env.desk_env import DeskWateringEnv
from floraflow.expert.ik_solver import IKSolver
from floraflow.expert.trajectory import interpolate_joint_trajectory


class PourExpertPlanner:
    """Expert trajectory planner that reliably executes plant watering."""

    def __init__(
        self,
        env: DeskWateringEnv,
        lift_altitude: float = 0.62,
        pour_altitude: float = 0.58,
        pour_angle_deg: float = 50.0,
        hold_duration_sec: float = 1.5,
    ) -> None:
        self.env = env
        self.ik = IKSolver(
            env.model,
            site_name="pinch",
            pos_tol=4e-3,
            rot_tol=0.15,
            rot_weight=0.04,
            max_iterations=300,
        )
        self.lift_altitude = lift_altitude
        self.pour_altitude = pour_altitude
        self.pour_angle_rad = float(np.radians(pour_angle_deg))
        self.hold_duration_sec = hold_duration_sec
        self.R_down = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )

        # Precompute tilt rotation matrix around Y axis
        theta = -self.pour_angle_rad
        c, s = np.cos(theta), np.sin(theta)
        Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
        self.R_tilt = Ry @ self.R_down

    def plan_and_execute(
        self,
    ) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], bool]:
        """Execute full expert watering demonstration.

        Returns:
            trajectory_obs: List of state observation dictionaries.
            trajectory_actions: List of executed 8D actions [target_q(7), grip_cmd].
            success: Whether the demonstration achieved the pour condition.
        """
        obs_history: List[Dict[str, np.ndarray]] = []
        action_history: List[np.ndarray] = []

        obs = self.env.get_obs()
        q_init = obs["arm_qpos"].copy().astype(np.float64)
        gp = obs["grip_pos"].copy().astype(np.float64)
        pp = obs["plant_pos"].copy().astype(np.float64)

        # 1. Pregrasp and Grasp IK solutions
        res_pre = self.ik.solve(gp + np.array([0.0, 0.0, 0.08]), self.R_down, q_init=q_init)
        res_grasp = self.ik.solve(gp + np.array([0.0, 0.0, 0.005]), self.R_down, q_init=res_pre.q)

        # 2. Lift high: clears table surface and pot walls
        p_lift = np.array([gp[0], gp[1], self.lift_altitude], dtype=np.float64)
        res_lift = self.ik.solve(p_lift, self.R_down, q_init=res_grasp.q)

        # 3. Transport high over target plant pot
        p_over = np.array([pp[0] + 0.08, pp[1], self.lift_altitude], dtype=np.float64)
        res_over = self.ik.solve(p_over, self.R_down, q_init=res_lift.q)

        # 4. Descend to pouring altitude
        p_pour = np.array([pp[0] + 0.08, pp[1], self.pour_altitude], dtype=np.float64)
        res_pour = self.ik.solve(p_pour, self.R_down, q_init=res_over.q)

        # 5. Pitch tilt to execute pour
        res_tilt = self.ik.solve(p_pour, self.R_tilt, q_init=res_pour.q)

        # Execution Phase A: Pregrasp approach (open gripper: +1.0)
        for q in interpolate_joint_trajectory(q_init, res_pre.q, 0.8, self.env.control_hz):
            act = np.concatenate([q, [1.0]]).astype(np.float32)
            obs, _, _, _ = self.env.step(act)
            obs_history.append(obs)
            action_history.append(act)

        # Execution Phase B: Descend to handle (open gripper: +1.0)
        for q in interpolate_joint_trajectory(res_pre.q, res_grasp.q, 0.6, self.env.control_hz):
            act = np.concatenate([q, [1.0]]).astype(np.float32)
            obs, _, _, _ = self.env.step(act)
            obs_history.append(obs)
            action_history.append(act)

        # Execution Phase C: Clamp fingers firmly around handle (closed gripper: -1.0)
        for _ in range(15):
            act = np.concatenate([res_grasp.q, [-1.0]]).astype(np.float32)
            obs, _, _, _ = self.env.step(act)
            obs_history.append(obs)
            action_history.append(act)

        # Execution Phase D: High lift (closed gripper: -1.0)
        for q in interpolate_joint_trajectory(res_grasp.q, res_lift.q, 1.0, self.env.control_hz):
            act = np.concatenate([q, [-1.0]]).astype(np.float32)
            obs, _, _, _ = self.env.step(act)
            obs_history.append(obs)
            action_history.append(act)

        # Execution Phase E: High transport towards plant (closed gripper: -1.0)
        for q in interpolate_joint_trajectory(res_lift.q, res_over.q, 1.5, self.env.control_hz):
            act = np.concatenate([q, [-1.0]]).astype(np.float32)
            obs, _, _, _ = self.env.step(act)
            obs_history.append(obs)
            action_history.append(act)

        # Execution Phase F: Descend to pour altitude (closed gripper: -1.0)
        for q in interpolate_joint_trajectory(res_over.q, res_pour.q, 0.6, self.env.control_hz):
            act = np.concatenate([q, [-1.0]]).astype(np.float32)
            obs, _, _, _ = self.env.step(act)
            obs_history.append(obs)
            action_history.append(act)

        # Execution Phase G: Pitch tilt (closed gripper: -1.0)
        for q in interpolate_joint_trajectory(res_pour.q, res_tilt.q, 0.8, self.env.control_hz):
            act = np.concatenate([q, [-1.0]]).astype(np.float32)
            obs, _, _, _ = self.env.step(act)
            obs_history.append(obs)
            action_history.append(act)

        # Execution Phase H: Hold pour orientation
        hold_steps = int(round(self.hold_duration_sec * self.env.control_hz))
        success_recorded = False
        for _ in range(hold_steps):
            act = np.concatenate([res_tilt.q, [-1.0]]).astype(np.float32)
            obs, _, _, info = self.env.step(act)
            obs_history.append(obs)
            action_history.append(act)
            if info["success"]:
                success_recorded = True

        # Execution Phase I: Return to upright pose
        for q in interpolate_joint_trajectory(res_tilt.q, res_pour.q, 0.8, self.env.control_hz):
            act = np.concatenate([q, [-1.0]]).astype(np.float32)
            obs, _, _, _ = self.env.step(act)
            obs_history.append(obs)
            action_history.append(act)

        return obs_history, action_history, success_recorded
