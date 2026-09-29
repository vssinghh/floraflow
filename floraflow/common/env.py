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

    MIN_OBJECT_DISTANCE = 0.22
    MIN_CAN_Y_CLEARANCE = 0.120
    MAX_PLANT_Y_CLEARANCE = -0.170

    def __init__(
        self,
        xml_path: Optional[str] = None,
        control_hz: int = 20,
        physics_dt: float = 0.002,
        can_pos_range: Optional[Dict[str, Tuple[float, float]]] = None,
        plant_pos_range: Optional[Dict[str, Tuple[float, float]]] = None,
        include_rgb: bool = False,
        rgb_cameras: Tuple[str, ...] = ("third_person_cam",),
        rgb_resolution: Tuple[int, int] = (128, 128),
        domain_rand: bool = False,
    ) -> None:
        """Initialize the MuJoCo simulation environment."""
        self.can_pos_range = can_pos_range if can_pos_range is not None else self.DEFAULT_CAN_POS_RANGE
        self.plant_pos_range = plant_pos_range if plant_pos_range is not None else self.DEFAULT_PLANT_POS_RANGE
        self.include_rgb = include_rgb
        self.rgb_cameras = rgb_cameras
        self.rgb_resolution = rgb_resolution
        self.domain_rand = domain_rand
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
        self._arm_dof_adrs = [self.model.jnt_dofadr[jid] for jid in self._arm_joint_ids]
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
        self._robot_body_ids = {
            bid
            for bid in range(self.model.nbody)
            if (self.model.body(bid).name or "").startswith(("link", "hand", "left_finger", "right_finger"))
        }

        self._grip_site_id = self.model.site("can_grip_site").id
        self._spout_site_id = self.model.site("can_spout_tip").id
        self._plant_site_id = self.model.site("plant_pot_center").id

        # Query geom, material, and camera IDs for domain randomization
        self._can_geom_ids = [
            gid
            for gid in range(self.model.ngeom)
            if int(self.model.geom_bodyid[gid]) == self._can_body_id
        ]
        self._pot_geom_ids = [
            self.model.geom(name).id
            for name in ("pot_base", "pot_wall_n", "pot_wall_s", "pot_wall_e", "pot_wall_w")
        ]
        self._wood_table_mat_id = self.model.material("wood_table").id
        self._leaf_mat_id = self.model.material("leaf_mat").id
        self._soil_mat_id = self.model.material("soil_mat").id
        self._cam_ids = {
            cam_name: self.model.camera(cam_name).id
            for cam_name in ("third_person_cam", "overhead_cam", "wrist_cam", "pour_cam")
            if self.model.camera(cam_name) is not None
        }

        # Snapshot nominal MuJoCo model parameters for clean restoration
        self._init_light_pos = self.model.light_pos.copy()
        self._init_light_dir = self.model.light_dir.copy()
        self._init_light_diffuse = self.model.light_diffuse.copy()
        self._init_light_specular = self.model.light_specular.copy()
        self._init_cam_pos = self.model.cam_pos.copy()
        self._init_cam_quat = self.model.cam_quat.copy()
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_mat_rgba = self.model.mat_rgba.copy()
        self._init_body_mass = self.model.body_mass.copy()
        self._init_body_inertia = self.model.body_inertia.copy()
        self._init_dof_damping = self.model.dof_damping.copy()

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
        self.last_domain_params: Optional[Dict[str, Any]] = None

    def restore_nominal_domain(self) -> None:
        """Restore all visual and physical MuJoCo model parameters to their nominal studio values."""
        self.model.light_pos[:] = self._init_light_pos
        self.model.light_dir[:] = self._init_light_dir
        self.model.light_diffuse[:] = self._init_light_diffuse
        self.model.light_specular[:] = self._init_light_specular
        self.model.cam_pos[:] = self._init_cam_pos
        self.model.cam_quat[:] = self._init_cam_quat
        self.model.geom_rgba[:] = self._init_geom_rgba
        self.model.mat_rgba[:] = self._init_mat_rgba
        self.model.body_mass[:] = self._init_body_mass
        self.model.body_inertia[:] = self._init_body_inertia
        self.model.dof_damping[:] = self._init_dof_damping
        self.last_domain_params = None

    def sample_domain_params(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Sample a deterministic JSON-serializable dictionary of Sim-to-Real domain perturbations."""
        # 1. Lighting position, direction, intensity, and color temperature
        light_pos_delta = rng.uniform(
            low=[-0.30, -0.30, -0.20],
            high=[0.30, 0.30, 0.20],
            size=self._init_light_pos.shape,
        )
        light_dir_delta = rng.uniform(
            low=-0.25,
            high=0.25,
            size=self._init_light_dir.shape,
        )
        light_intensity_scale = float(rng.uniform(0.55, 1.35))
        light_rgb_tint = rng.uniform(0.85, 1.15, size=(3,))

        # 2. Object and tabletop appearance (RGB colors)
        base_can_rgb = self._init_geom_rgba[self._can_geom_ids[0], :3]
        can_rgb = np.clip(
            base_can_rgb * rng.uniform(0.65, 1.35, size=(3,)) + rng.uniform(-0.08, 0.08, size=(3,)),
            0.05,
            0.95,
        )
        base_pot_rgb = self._init_geom_rgba[self._pot_geom_ids[0], :3]
        pot_rgb = np.clip(
            base_pot_rgb * rng.uniform(0.65, 1.35, size=(3,)) + rng.uniform(-0.08, 0.08, size=(3,)),
            0.05,
            0.95,
        )
        base_leaf_rgb = self._init_mat_rgba[self._leaf_mat_id, :3]
        leaf_rgb = np.clip(
            base_leaf_rgb * rng.uniform(0.70, 1.30, size=(3,)) + rng.uniform(-0.06, 0.06, size=(3,)),
            0.05,
            0.95,
        )
        desk_rgb_scale = np.clip(rng.uniform(0.65, 1.25, size=(3,)), 0.45, 1.35)

        # 3. Camera mount extrinsics jitter (position & small axis-angle orientation)
        cam_pos_deltas: Dict[str, list[float]] = {}
        cam_euler_deltas: Dict[str, list[float]] = {}
        for cam_name in ("third_person_cam", "overhead_cam", "wrist_cam"):
            if cam_name not in self._cam_ids:
                continue
            if cam_name == "wrist_cam":
                pos_lim, rot_lim = 0.004, 0.010
            else:
                pos_lim, rot_lim = 0.012, 0.020
            cam_pos_deltas[cam_name] = [
                float(x) for x in rng.uniform(-pos_lim, pos_lim, size=(3,))
            ]
            cam_euler_deltas[cam_name] = [
                float(x) for x in rng.uniform(-rot_lim, rot_lim, size=(3,))
            ]

        # 4. Physical dynamics (payload mass & arm joint damping)
        can_mass_scale = float(rng.uniform(0.70, 2.00))
        arm_damping_scale = [float(x) for x in rng.uniform(0.80, 1.25, size=(7,))]

        return {
            "light_pos_delta": [[float(v) for v in row] for row in light_pos_delta],
            "light_dir_delta": [[float(v) for v in row] for row in light_dir_delta],
            "light_intensity_scale": light_intensity_scale,
            "light_rgb_tint": [float(v) for v in light_rgb_tint],
            "can_rgb": [float(v) for v in can_rgb],
            "pot_rgb": [float(v) for v in pot_rgb],
            "leaf_rgb": [float(v) for v in leaf_rgb],
            "desk_rgb_scale": [float(v) for v in desk_rgb_scale],
            "cam_pos_deltas": cam_pos_deltas,
            "cam_euler_deltas": cam_euler_deltas,
            "can_mass_scale": can_mass_scale,
            "arm_damping_scale": arm_damping_scale,
        }

    def apply_domain_params(self, params: Dict[str, Any]) -> None:
        """Apply a dictionary of Sim-to-Real domain perturbations onto the active MuJoCo model."""
        self.restore_nominal_domain()

        # 1. Lighting
        light_pos_delta = np.array(params["light_pos_delta"], dtype=np.float64)
        self.model.light_pos[:] = self._init_light_pos + light_pos_delta

        light_dir = self._init_light_dir + np.array(params["light_dir_delta"], dtype=np.float64)
        norms = np.linalg.norm(light_dir, axis=1, keepdims=True)
        self.model.light_dir[:] = light_dir / np.clip(norms, 1e-6, None)

        scale = float(params["light_intensity_scale"])
        tint = np.array(params["light_rgb_tint"], dtype=np.float32)[None, :]
        self.model.light_diffuse[:] = np.clip(self._init_light_diffuse * scale * tint, 0.05, 1.50)

        # 2. Object and tabletop appearance
        can_rgb = np.array(params["can_rgb"], dtype=np.float32)
        for gid in self._can_geom_ids:
            self.model.geom_rgba[gid, :3] = can_rgb

        pot_rgb = np.array(params["pot_rgb"], dtype=np.float32)
        for gid in self._pot_geom_ids:
            self.model.geom_rgba[gid, :3] = pot_rgb

        self.model.mat_rgba[self._leaf_mat_id, :3] = np.array(params["leaf_rgb"], dtype=np.float32)
        self.model.mat_rgba[self._wood_table_mat_id, :3] = np.array(params["desk_rgb_scale"], dtype=np.float32)

        # 3. Camera mount extrinsics
        cam_pos_deltas = params.get("cam_pos_deltas", {})
        cam_euler_deltas = params.get("cam_euler_deltas", {})
        for cam_name, cid in self._cam_ids.items():
            if cam_name in cam_pos_deltas:
                self.model.cam_pos[cid] = self._init_cam_pos[cid] + np.array(
                    cam_pos_deltas[cam_name], dtype=np.float64
                )
            if cam_name in cam_euler_deltas:
                euler = np.array(cam_euler_deltas[cam_name], dtype=np.float64)
                angle = float(np.linalg.norm(euler))
                if angle > 1e-8:
                    axis = euler / angle
                    dq = np.zeros(4, dtype=np.float64)
                    mujoco.mju_axisAngle2Quat(dq, axis, angle)
                    q_new = np.zeros(4, dtype=np.float64)
                    mujoco.mju_mulQuat(q_new, self._init_cam_quat[cid], dq)
                    self.model.cam_quat[cid] = q_new

        # 4. Physical dynamics
        mass_scale = float(params.get("can_mass_scale", 1.0))
        self.model.body_mass[self._can_body_id] = self._init_body_mass[self._can_body_id] * mass_scale
        self.model.body_inertia[self._can_body_id] = self._init_body_inertia[self._can_body_id] * mass_scale

        damping_scales = params.get("arm_damping_scale", [1.0] * 7)
        for dof_adr, d_scale in zip(self._arm_dof_adrs, damping_scales):
            self.model.dof_damping[dof_adr] = self._init_dof_damping[dof_adr] * float(d_scale)

        self.last_domain_params = params

    @classmethod
    def is_valid_spawn(
        cls,
        can_xy: Tuple[float, float],
        plant_xy: Tuple[float, float],
    ) -> bool:
        """Validate that a (can_xy, plant_xy) spawn pair has safe physical clearance.

        Ensures:
        1. Minimum center-to-center distance between watering can and plant pot (>= 22 cm)
           so their 3D geometry (spout, handle, rim, foliage) never touch or overlap.
        2. Safe Y clearance from the robot's initial home hand and fingers at y = 0.00 m
           (right finger at y = +0.04 m, left finger at y = -0.04 m) so neither object
           spawns inside or hooking the open gripper fingers.
        """
        cx, cy = float(can_xy[0]), float(can_xy[1])
        px, py = float(plant_xy[0]), float(plant_xy[1])

        dist_xy = float(np.hypot(cx - px, cy - py))
        if dist_xy < cls.MIN_OBJECT_DISTANCE:
            return False

        if cy < cls.MIN_CAN_Y_CLEARANCE:
            return False

        if py > cls.MAX_PLANT_Y_CLEARANCE:
            return False

        return True

    def has_initial_collision(self) -> bool:
        """Check whether the settled reset state has any illegal contacts or finger deflection.

        At Step 0 after settling, the robot arm and fingers must have zero contacts
        with any scene object, the watering can and plant pot must not touch each other,
        and the open gripper width must remain unperturbed (~0.08 m).
        """
        f1 = float(self.data.qpos[self._finger_qpos_adr[0]])
        f2 = float(self.data.qpos[self._finger_qpos_adr[1]])
        if abs((f1 + f2) - 0.08) > 1e-3:
            return True

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[con.geom1])
            b2 = int(self.model.geom_bodyid[con.geom2])
            if b1 in self._robot_body_ids or b2 in self._robot_body_ids:
                return True
            if (b1 == self._can_body_id and b2 == self._plant_body_id) or (
                b1 == self._plant_body_id and b2 == self._can_body_id
            ):
                return True
        return False

    def reset(
        self,
        seed: Optional[int] = None,
        can_xy: Optional[Tuple[float, float]] = None,
        plant_xy: Optional[Tuple[float, float]] = None,
        domain_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Reset environment with optional seed, object positions, and Sim-to-Real domain randomization."""
        rng = np.random.default_rng(seed)
        if domain_params is not None:
            self.apply_domain_params(domain_params)
        elif self.domain_rand:
            dr_seed = (seed + 1_000_000) if seed is not None else None
            dr_rng = np.random.default_rng(dr_seed)
            sampled_params = self.sample_domain_params(dr_rng)
            self.apply_domain_params(sampled_params)
        else:
            self.restore_nominal_domain()

        default_qpos = np.array([0.0, -0.35, 0.0, -2.1, 0.0, 1.75, 0.785])
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

        max_attempts = 100 if (can_xy is None or plant_xy is None) else 1
        for attempt in range(max_attempts):
            # Randomize or assign watering can position
            if can_xy is None:
                can_x = float(rng.uniform(*self.can_pos_range["x"]))
                can_y = float(rng.uniform(*self.can_pos_range["y"]))
            else:
                can_x, can_y = float(can_xy[0]), float(can_xy[1])

            # Randomize or assign plant position
            if plant_xy is None:
                plant_x = float(rng.uniform(*self.plant_pos_range["x"]))
                plant_y = float(rng.uniform(*self.plant_pos_range["y"]))
            else:
                plant_x, plant_y = float(plant_xy[0]), float(plant_xy[1])

            if (can_xy is None or plant_xy is None) and attempt < max_attempts - 1:
                if not self.is_valid_spawn((can_x, can_y), (plant_x, plant_y)):
                    continue

            # Reset simulation state
            mujoco.mj_resetData(self.model, self.data)

            # Robot default home pose: pre-positioned above desk
            for adr, val in zip(self._arm_qpos_adr, default_qpos):
                self.data.qpos[adr] = val

            # Open gripper
            for adr in self._finger_qpos_adr:
                self.data.qpos[adr] = 0.04

            # Can rests on table surface at Z = 0.405 m
            self.data.qpos[self._can_qpos_adr : self._can_qpos_adr + 3] = [can_x, can_y, 0.405]
            self.data.qpos[self._can_qpos_adr + 3 : self._can_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

            self.model.body_pos[self._plant_body_id] = [plant_x, plant_y, 0.40]

            # Place water particles inside the watering can reservoir
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

            if not self.has_initial_collision():
                break

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

        obs_dict = {
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

        if self.include_rgb:
            w, h = self.rgb_resolution
            for cam in self.rgb_cameras:
                obs_dict[f"rgb_{cam}"] = self.render(camera=cam, width=w, height=h)

        return obs_dict

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
