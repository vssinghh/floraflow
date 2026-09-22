"""MuJoCo Desk Watering Environment.

Provides a realistic tabletop simulation with Franka Panda arm, potted plant,
hollow watering can, and discrete water particles.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np


class DeskWateringEnv:
    """Desk environment for 6-DoF plant watering manipulation."""

    DEFAULT_CAN_POS_RANGE = {
        "x": (0.48, 0.56),
        "y": (0.12, 0.22),
    }

    DEFAULT_PLANT_POS_RANGE = {
        "x": (0.38, 0.46),
        "y": (-0.26, -0.18),
    }

    def __init__(
        self,
        xml_path: Optional[str] = None,
        control_hz: int = 20,
        physics_dt: float = 0.002,
        can_pos_range: Optional[Dict[str, Tuple[float, float]]] = None,
        plant_pos_range: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """Initialize the MuJoCo simulation environment."""
        self.can_pos_range = can_pos_range if can_pos_range is not None else self.DEFAULT_CAN_POS_RANGE
        self.plant_pos_range = plant_pos_range if plant_pos_range is not None else self.DEFAULT_PLANT_POS_RANGE
        if xml_path is None:
            pkg_root = Path(__file__).resolve().parent.parent.parent
            xml_path = str(pkg_root / "assets" / "scenes" / "desk_scene.xml")

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Scene XML file not found at: {xml_path}")

        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.control_hz = control_hz
        self.physics_dt = physics_dt
        self.model.opt.timestep = physics_dt
        self.substeps = int(round((1.0 / control_hz) / physics_dt))

        # Query body, joint, site, and actuator IDs
        self._arm_joint_ids = [self.model.joint(f"joint{i}").id for i in range(1, 8)]
        self._finger_joint_ids = [
            self.model.joint("finger_joint1").id,
            self.model.joint("finger_joint2").id,
        ]
        self._arm_qpos_adr = [self.model.jnt_qposadr[jid] for jid in self._arm_joint_ids]
        self._finger_qpos_adr = [self.model.jnt_qposadr[jid] for jid in self._finger_joint_ids]

        self._can_body_id = self.model.body("watering_can").id
        self._plant_body_id = self.model.body("potted_plant").id
        self._can_jnt_id = self.model.joint("watering_can_joint").id
        self._can_qpos_adr = self.model.jnt_qposadr[self._can_jnt_id]

        self._grip_site_id = self.model.site("can_grip_site").id
        self._spout_site_id = self.model.site("can_spout_tip").id
        self._plant_site_id = self.model.site("plant_pot_center").id

        # Water particle bodies
        self.n_particles = 8
        self._particle_jnt_ids = [
            self.model.joint(f"particle_joint_{i}").id for i in range(self.n_particles)
        ]
        self._particle_qpos_adrs = [
            self.model.jnt_qposadr[jid] for jid in self._particle_jnt_ids
        ]

        self.renderer: Optional[mujoco.Renderer] = None
        self.current_step = 0

    def reset(
        self,
        seed: Optional[int] = None,
        can_xy: Optional[Tuple[float, float]] = None,
        plant_xy: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Reset environment with optional seed and randomized object positions."""
        rng = np.random.default_rng(seed)

        # Reset simulation state
        mujoco.mj_resetData(self.model, self.data)

        # Robot default home pose: pre-positioned above desk
        default_qpos = np.array([0.0, -0.35, 0.0, -2.1, 0.0, 1.75, 0.785])
        for adr, val in zip(self._arm_qpos_adr, default_qpos):
            self.data.qpos[adr] = val

        # Open gripper
        for adr in self._finger_qpos_adr:
            self.data.qpos[adr] = 0.04

        # Randomize or assign watering can position
        if can_xy is None:
            can_x = rng.uniform(*self.can_pos_range["x"])
            can_y = rng.uniform(*self.can_pos_range["y"])
        else:
            can_x, can_y = can_xy

        # Can rests on table surface at Z = 0.405 m
        self.data.qpos[self._can_qpos_adr : self._can_qpos_adr + 3] = [can_x, can_y, 0.405]
        self.data.qpos[self._can_qpos_adr + 3 : self._can_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # Randomize or assign plant position
        if plant_xy is None:
            plant_x = rng.uniform(*self.plant_pos_range["x"])
            plant_y = rng.uniform(*self.plant_pos_range["y"])
        else:
            plant_x, plant_y = plant_xy

        self.model.body_pos[self._plant_body_id] = [plant_x, plant_y, 0.40]

        # Place water particles inside the watering can reservoir
        particle_offsets = [
            (-0.01, -0.01, 0.02),
            (0.01, -0.01, 0.02),
            (-0.01, 0.01, 0.02),
            (0.01, 0.01, 0.02),
            (0.0, 0.0, 0.02),
            (-0.01, 0.0, 0.035),
            (0.01, 0.0, 0.035),
            (0.0, -0.01, 0.035),
        ]
        for adr, (dx, dy, dz) in zip(self._particle_qpos_adrs, particle_offsets):
            self.data.qpos[adr : adr + 3] = [can_x + dx, can_y + dy, 0.405 + dz]
            self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # Zero out velocities and set position servo targets
        self.data.qvel[:] = 0.0
        for i, val in enumerate(default_qpos):
            self.data.ctrl[i] = val
        if self.model.nu > 7:
            self.data.ctrl[7] = 255.0

        # Step forward 50 physics steps to settle contacts
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self.current_step = 0
        return self.get_obs()

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Compute structured observation dictionary."""
        # Arm state
        arm_qpos = np.array([self.data.qpos[adr] for adr in self._arm_qpos_adr], dtype=np.float32)
        arm_qvel = np.array([self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self._arm_joint_ids], dtype=np.float32)

        # Gripper width
        f1 = self.data.qpos[self._finger_qpos_adr[0]]
        f2 = self.data.qpos[self._finger_qpos_adr[1]]
        gripper_width = np.array([f1 + f2], dtype=np.float32)

        # End-effector (Franka hand body)
        ee_body_id = self.model.body("hand").id
        ee_pos = self.data.xpos[ee_body_id].copy().astype(np.float32)
        ee_mat = self.data.xmat[ee_body_id].copy().reshape(3, 3)
        ee_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_quat, ee_mat.flatten())

        # Object positions and sites
        can_pos = self.data.xpos[self._can_body_id].copy().astype(np.float32)
        can_mat = self.data.xmat[self._can_body_id].copy().reshape(3, 3)
        can_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(can_quat, can_mat.flatten())

        grip_pos = self.data.site_xpos[self._grip_site_id].copy().astype(np.float32)
        spout_pos = self.data.site_xpos[self._spout_site_id].copy().astype(np.float32)
        plant_pos = self.data.site_xpos[self._plant_site_id].copy().astype(np.float32)

        # Relative vector: spout to plant
        relative_spout_to_plant = (spout_pos - plant_pos).astype(np.float32)

        # Pour angle: pitch angle from vertical Z axis
        # Upright can has Z axis along [0, 0, 1]
        can_z_axis = can_mat[:, 2]
        dot_product = np.clip(can_z_axis[2], -1.0, 1.0)
        tilt_angle_rad = float(np.arccos(dot_product))
        tilt_angle_deg = float(np.degrees(tilt_angle_rad))

        # Count particles landed inside pot radius (<0.07 m from pot center in xy, above soil)
        particles_in_pot = 0
        for i in range(self.n_particles):
            p_pos = self.data.body(f"water_particle_{i}").xpos
            d_xy = np.linalg.norm(p_pos[:2] - plant_pos[:2])
            dz = p_pos[2] - plant_pos[2]
            if d_xy < 0.07 and dz > -0.02 and dz < 0.10:
                particles_in_pot += 1

        return {
            "arm_qpos": arm_qpos,
            "arm_qvel": arm_qvel,
            "gripper_width": gripper_width,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "can_pos": can_pos,
            "can_quat": can_quat,
            "grip_pos": grip_pos,
            "spout_pos": spout_pos,
            "plant_pos": plant_pos,
            "relative_spout_to_plant": relative_spout_to_plant,
            "tilt_angle_deg": np.array([tilt_angle_deg], dtype=np.float32),
            "particles_in_pot": np.array([particles_in_pot], dtype=np.int32),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step simulation forward by one control interval.

        Action: 8-dimensional vector
        - action[0:7]: target arm joint positions or delta positions
        - action[7]: gripper command (+1.0 open, -1.0 close)
        """
        target_qpos = action[:7]
        gripper_cmd = action[7]

        # Gripper mapped from [-1, 1] to actuator8 ctrlrange [0.0, 255.0]
        # 255.0 is fully open (~0.04m per finger), 0.0 is closed
        target_finger = 255.0 if gripper_cmd > 0.0 else 0.0

        # Apply controls to Panda actuators
        # Actuators 0-6: arm joints (position servos)
        # Actuator 7: fingers
        for i in range(7):
            self.data.ctrl[i] = target_qpos[i]

        if self.model.nu > 7:
            self.data.ctrl[7] = target_finger

        # Step physics forward
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1
        obs = self.get_obs()

        # Compute pour metrics
        spout_dist = float(np.linalg.norm(obs["spout_pos"] - obs["plant_pos"]))
        tilt_deg = float(obs["tilt_angle_deg"][0])
        particles = int(obs["particles_in_pot"][0])

        is_pouring = (tilt_deg >= 40.0) and (spout_dist <= 0.14)
        success = (particles >= 2) or (is_pouring and spout_dist <= 0.14)

        reward = float(particles * 10.0 + (1.0 if is_pouring else 0.0) - spout_dist)
        done = self.current_step >= 250

        info = {
            "is_pouring": is_pouring,
            "tilt_deg": tilt_deg,
            "spout_dist": spout_dist,
            "particles_in_pot": particles,
            "success": success,
        }

        return obs, reward, done, info

    def render(self, camera: str = "third_person_cam", width: int = 640, height: int = 480) -> np.ndarray:
        """Render RGB image from requested camera."""
        if self.renderer is None or self.renderer.width != width or self.renderer.height != height:
            self.renderer = mujoco.Renderer(self.model, height, width)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()
