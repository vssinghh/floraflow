"""Inverse Kinematics solver for 7-DoF Franka Panda using Damped Least-Squares (DLS)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import mujoco


@dataclass
class IKResult:
    """Result container for Inverse Kinematics solve."""
    q: np.ndarray
    success: bool
    iterations: int
    pos_error: float
    rot_error: Optional[float]
    final_pos: np.ndarray
    final_rot: np.ndarray


def rotmat_to_rot6d(rot: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix R in SO(3) to Zhou et al. 6D continuous representation [r1, r2]."""
    r = np.asarray(rot, dtype=np.float32).reshape(3, 3)
    return np.concatenate([r[:, 0], r[:, 1]], axis=-1).astype(np.float32)


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Recover orthogonal 3x3 rotation matrix R in SO(3) from 6D vector via Gram-Schmidt."""
    v = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1 = v[0:3]
    a2 = v[3:6]
    b1 = a1 / max(float(np.linalg.norm(a1)), 1e-8)
    u2 = a2 - float(np.dot(b1, a2)) * b1
    b2 = u2 / max(float(np.linalg.norm(u2)), 1e-8)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


class IKSolver:
    """Damped Least-Squares (DLS) Inverse Kinematics solver with nullspace posture control."""

    DEFAULT_Q_NOMINAL = np.array([0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398], dtype=np.float64)

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "pinch",
        damping: float = 0.05,
        step_size: float = 0.6,
        max_iterations: int = 150,
        pos_tol: float = 2e-3,
        rot_tol: float = 0.05,
        rot_weight: float = 0.25,
        nullspace_weight: float = 0.05,
        max_step: float = 0.2,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in MuJoCo model.")

        self.damping = damping
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.rot_weight = rot_weight
        self.nullspace_weight = nullspace_weight
        self.max_step = max_step

        self.n_arm_joints = 7
        self.joint_ranges = np.zeros((self.n_arm_joints, 2), dtype=np.float64)
        for i in range(self.n_arm_joints):
            self.joint_ranges[i] = model.jnt_range[i]

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector Cartesian position and rotation matrix for joint configuration q."""
        for i in range(self.n_arm_joints):
            self.data.qpos[i] = q[i]
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        rot = self.data.site_xmat[self.site_id].copy().reshape(3, 3)
        return pos, rot

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
        q_nominal: Optional[np.ndarray] = None,
    ) -> IKResult:
        """Solve DLS IK for target position and optional orientation."""
        target_pos = np.asarray(target_pos, dtype=np.float64)
        if target_rot is not None:
            target_rot = np.asarray(target_rot, dtype=np.float64)

        if q_init is None:
            q = self.DEFAULT_Q_NOMINAL.copy()
        else:
            q = np.asarray(q_init, dtype=np.float64).copy()

        if q_nominal is None:
            q_nominal = self.DEFAULT_Q_NOMINAL

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        pos_err_val = float("inf")
        rot_err_val: Optional[float] = None
        current_pos = np.zeros(3)
        current_rot = np.eye(3)

        for iteration in range(self.max_iterations):
            for i in range(self.n_arm_joints):
                self.data.qpos[i] = q[i]
            mujoco.mj_forward(self.model, self.data)

            current_pos = self.data.site_xpos[self.site_id].copy()
            current_rot = self.data.site_xmat[self.site_id].copy().reshape(3, 3)

            pos_err = target_pos - current_pos
            pos_err_val = float(np.linalg.norm(pos_err))

            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            J_pos = jacp[:, :self.n_arm_joints]

            if target_rot is not None:
                R_err = target_rot @ current_rot.T
                rot_err_vec = 0.5 * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])
                rot_err_val = float(np.linalg.norm(rot_err_vec))
                if pos_err_val < self.pos_tol and rot_err_val < self.rot_tol:
                    return IKResult(q, True, iteration, pos_err_val, rot_err_val, current_pos, current_rot)

                J_rot = jacr[:, :self.n_arm_joints]
                J = np.vstack([J_pos, self.rot_weight * J_rot])
                e = np.concatenate([pos_err, self.rot_weight * rot_err_vec])
            else:
                if pos_err_val < self.pos_tol:
                    return IKResult(q, True, iteration, pos_err_val, None, current_pos, current_rot)
                J = J_pos
                e = pos_err

            # DLS formulation
            m = J.shape[0]
            lambda_sq = self.damping ** 2
            J_JT = J @ J.T + lambda_sq * np.eye(m)
            J_dagger = J.T @ np.linalg.inv(J_JT)
            delta_q_task = J_dagger @ e

            # Nullspace posture regularization
            N = np.eye(self.n_arm_joints) - J_dagger @ J
            delta_q_null = N @ (self.nullspace_weight * (q_nominal - q))

            delta_q = delta_q_task + delta_q_null
            step_norm = np.linalg.norm(delta_q)
            if step_norm > self.max_step:
                delta_q = (delta_q / step_norm) * self.max_step

            q = q + self.step_size * delta_q
            q = np.clip(q, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

        success = pos_err_val < self.pos_tol and (rot_err_val is None or rot_err_val < self.rot_tol)
        return IKResult(q, success, self.max_iterations, pos_err_val, rot_err_val, current_pos, current_rot)
