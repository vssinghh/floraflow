"""Minimum-jerk trajectory interpolation in joint space and Cartesian space."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np


def minimum_jerk_scaling(t: float, duration: float) -> Tuple[float, float, float]:
    """Compute quintic polynomial scaling s(tau), s_dot(tau), s_ddot(tau)."""
    if duration <= 0.0:
        return 1.0, 0.0, 0.0

    tau = np.clip(t / duration, 0.0, 1.0)
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau

    s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
    s_dot = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / duration
    s_ddot = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (duration * duration)

    return float(s), float(s_dot), float(s_ddot)


def interpolate_joint_trajectory(
    q_start: np.ndarray,
    q_end: np.ndarray,
    duration: float,
    control_hz: int = 20,
) -> List[np.ndarray]:
    """Generate smooth sequence of joint positions between q_start and q_end."""
    n_steps = max(1, int(round(duration * control_hz)))
    trajectory = []
    for step in range(n_steps + 1):
        t = step / control_hz
        s, _, _ = minimum_jerk_scaling(t, duration)
        q_t = (1.0 - s) * q_start + s * q_end
        trajectory.append(q_t)
    return trajectory
