"""HDF5 State and Multi-Camera Vision Dataset Loaders with Temporal Action Chunking.

Loads demonstration trajectories with state features or synchronized RGB camera frames
and robot proprioception, computing normalization statistics and serving batches for training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import mujoco
import numpy as np
import torch
from torch.utils.data import Dataset

from floraflow.common.kinematics import IKSolver, rotmat_to_rot6d

OBS_BASE_KEYS = [
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
]

TOTAL_DEMO_HORIZON = 174.0


def extract_observation_vector(
    obs_dict: Dict[str, np.ndarray],
    step_idx: int = 0,
) -> np.ndarray:
    """Extract and concatenate state features into a 52-dimensional observation vector.

    Includes base physical measurements plus explicit spatial offset vectors
    and normalized trajectory phase progress.
    """
    parts: List[np.ndarray] = []
    for k in OBS_BASE_KEYS:
        val = obs_dict[k]
        if val.ndim == 0:
            parts.append(val.reshape(1))
        elif val.ndim == 1:
            parts.append(val)
        else:
            parts.append(val.flatten())

    ee_pos = obs_dict["ee_pos"]
    grip_pos = obs_dict["grip_pos"]
    spout_pos = obs_dict["spout_pos"]
    plant_pos = obs_dict["plant_pos"]

    vec_ee_to_grip = (grip_pos - ee_pos).astype(np.float32)
    vec_ee_to_plant = (plant_pos - ee_pos).astype(np.float32)
    vec_spout_to_plant = (plant_pos - spout_pos).astype(np.float32)
    parts.extend([vec_ee_to_grip, vec_ee_to_plant, vec_spout_to_plant])

    progress = np.array([min(step_idx / TOTAL_DEMO_HORIZON, 1.0)], dtype=np.float32)
    parts.append(progress)

    return np.concatenate(parts, axis=0).astype(np.float32)


class WateringDemonstrationDataset(Dataset):
    """PyTorch Dataset yielding normalized state observation and action chunk pairs."""

    def __init__(
        self,
        h5_path: str = "datasets/watering_demos_100.h5",
        horizon: int = 16,
        stats: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path).resolve()
        self.horizon = horizon

        self.episodes_obs: List[np.ndarray] = []
        self.episodes_act: List[np.ndarray] = []
        self.indices: List[Tuple[int, int]] = []

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found at: {self.h5_path}")

        self._load_data()

        if stats is None:
            self.stats = self._compute_stats()
        else:
            self.stats = stats

    def _load_data(self) -> None:
        with h5py.File(self.h5_path, "r") as f:
            data_grp = f["data"]
            demo_names = sorted(list(data_grp.keys()), key=lambda x: int(x.split("_")[1]))

            for ep_idx, name in enumerate(demo_names):
                demo = data_grp[name]
                obs_grp = demo["obs"]
                actions = np.array(demo["actions"], dtype=np.float32)
                T = len(actions)

                obs_arrays = []
                for t in range(T):
                    t_obs = {k: np.array(obs_grp[k][t]) for k in OBS_BASE_KEYS}
                    obs_vec = extract_observation_vector(t_obs, step_idx=t)
                    obs_arrays.append(obs_vec)

                ep_obs = np.stack(obs_arrays, axis=0)
                self.episodes_obs.append(ep_obs)
                self.episodes_act.append(actions)

                for t in range(T):
                    self.indices.append((ep_idx, t))

    def _compute_stats(self) -> Dict[str, np.ndarray]:
        all_obs = np.concatenate(self.episodes_obs, axis=0)
        all_act = np.concatenate(self.episodes_act, axis=0)

        obs_mean = np.mean(all_obs, axis=0).astype(np.float32)
        obs_std = np.std(all_obs, axis=0).astype(np.float32)
        obs_std = np.clip(obs_std, a_min=1e-3, a_max=None)

        act_mean = np.mean(all_act, axis=0).astype(np.float32)
        act_std = np.std(all_act, axis=0).astype(np.float32)
        act_std = np.clip(act_std, a_min=1e-3, a_max=None)

        return {
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "act_mean": act_mean,
            "act_std": act_std,
        }

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.stats["obs_mean"]) / self.stats["obs_std"]

    def unnormalize_action(self, act: np.ndarray) -> np.ndarray:
        return act * self.stats["act_std"] + self.stats["act_mean"]

    def normalize_action(self, act: np.ndarray) -> np.ndarray:
        return (act - self.stats["act_mean"]) / self.stats["act_std"]

    def save_stats(self, json_path: str) -> None:
        """Save normalization statistics to JSON file."""
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v.tolist() for k, v in self.stats.items()}
        with open(p, "w") as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load_stats(cls, json_path: str) -> Dict[str, np.ndarray]:
        """Load normalization statistics from JSON file."""
        with open(json_path, "r") as f:
            raw = json.load(f)
        return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ep_idx, t = self.indices[idx]
        obs = self.episodes_obs[ep_idx][t]
        norm_obs = self.normalize_obs(obs)

        ep_act = self.episodes_act[ep_idx]
        chunk = ep_act[t : t + self.horizon]
        if len(chunk) < self.horizon:
            pad_count = self.horizon - len(chunk)
            last_act = ep_act[-1:]
            padding = np.repeat(last_act, pad_count, axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)

        norm_chunk = self.normalize_action(chunk)

        return (
            torch.from_numpy(norm_obs).float(),
            torch.from_numpy(norm_chunk).float(),
        )


class VisionWateringDataset(Dataset):
    """PyTorch Dataset yielding multi-camera image observations and action chunks."""

    def __init__(
        self,
        h5_path: str = "datasets/watering_demos_vision_100.h5",
        horizon: int = 16,
        cameras: Tuple[str, ...] = ("third_person_cam", "overhead_cam"),
        stats: Optional[Dict[str, np.ndarray]] = None,
        preload: bool = True,
        use_progress: bool = True,
        proprio_history_lags: Tuple[int, ...] = (0,),
        trim_stationary: bool = False,
        action_space: str = "joint_abs",
        return_uint8_images: bool = False,
    ) -> None:
        """Initialize Vision Demonstration Dataset.

        Args:
            h5_path: Path to the HDF5 demonstration archive.
            horizon: Prediction action chunk horizon H.
            cameras: Names of camera streams to load.
            stats: Optional precomputed normalization statistics dictionary.
            preload: If True, loads all images and proprioception into RAM for zero-disk-IO training.
            use_progress: If True, appends synthetic [0, 1] step progress clock.
            proprio_history_lags: Causal step lags (e.g. (0, 4, 8, 12)) to stack for temporal proprioception memory.
            trim_stationary: If True, filters out zero-velocity stationary grasp-dwell frames.
            action_space: Action chunk parameterization ('joint_abs', 'joint_delta', or 'eef_se3').
            return_uint8_images: If True, returns raw (H, W, 3) uint8 camera tensors so permutation and
                [0, 1] float normalization can execute in a single vectorized GPU batch kernel.
        """
        super().__init__()
        if action_space not in ("joint_abs", "joint_delta", "eef_se3"):
            raise ValueError(
                f"Unsupported action_space '{action_space}'. Expected 'joint_abs', 'joint_delta', or 'eef_se3'."
            )
        self.h5_path = Path(h5_path).resolve()
        self.horizon = horizon
        self.cameras = cameras
        self.preload = preload
        self.use_progress = use_progress
        self.proprio_history_lags = tuple(int(l) for l in proprio_history_lags)
        self.trim_stationary = trim_stationary
        self.action_space = action_space
        self.return_uint8_images = return_uint8_images
        if action_space == "eef_se3":
            base_dim = 11 if use_progress else 10
            self.act_dim = 10
        else:
            base_dim = 9 if use_progress else 8
            self.act_dim = 8
        self.proprio_dim = base_dim * len(self.proprio_history_lags)

        self.episodes_images: Dict[str, List[np.ndarray]] = {cam: [] for cam in cameras}
        self.episodes_proprio: List[np.ndarray] = []
        self.episodes_pose: List[np.ndarray] = []
        self.episodes_act: List[np.ndarray] = []
        self.indices: List[Tuple[int, int]] = []
        self.has_aux_pose: bool = False

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found at: {self.h5_path}")

        self._load_data()

        if stats is None:
            self.stats = self._compute_stats()
        else:
            self.stats = stats

        self._cached_images: Dict[str, torch.Tensor] = {}
        self._cached_proprio: Optional[torch.Tensor] = None
        self._cached_actions: Optional[torch.Tensor] = None
        self._cached_pose: Optional[torch.Tensor] = None
        if self.preload:
            self._build_tensor_cache()

    def _load_data(self) -> None:
        ik_fk: Optional[IKSolver] = None
        if self.action_space == "eef_se3":
            pkg_root = Path(__file__).resolve().parent.parent.parent
            xml_path = str(pkg_root / "assets" / "scenes" / "desk_scene.xml")
            mj_model = mujoco.MjModel.from_xml_path(xml_path)
            ik_fk = IKSolver(mj_model, site_name="pinch")

        with h5py.File(self.h5_path, "r") as f:
            data_grp = f["data"]
            demo_names = sorted(list(data_grp.keys()), key=lambda x: int(x.split("_")[1]))

            for ep_idx, name in enumerate(demo_names):
                demo = data_grp[name]
                obs_grp = demo["obs"]
                actions = np.array(demo["actions"], dtype=np.float32)
                T = len(actions)

                arm_qpos = np.array(obs_grp["arm_qpos"], dtype=np.float32)
                gripper_width = np.array(obs_grp["gripper_width"], dtype=np.float32)
                if gripper_width.ndim == 1:
                    gripper_width = gripper_width[:, None]

                keep_mask: Optional[np.ndarray] = None
                if self.trim_stationary and T > 2:
                    dq = np.linalg.norm(np.diff(arm_qpos, axis=0, prepend=arm_qpos[:1]), axis=1)
                    dg = np.abs(np.diff(gripper_width[:, 0], prepend=gripper_width[:1, 0]))
                    da = np.linalg.norm(np.diff(actions[:, :7], axis=0, prepend=actions[:1, :7]), axis=1)
                    dead_mask = (np.arange(T) > 0) & (dq < 5e-4) & (dg < 2e-5) & (da < 1e-5)
                    if np.any(~dead_mask):
                        keep_mask = ~dead_mask
                        arm_qpos = arm_qpos[keep_mask]
                        gripper_width = gripper_width[keep_mask]
                        actions = actions[keep_mask]
                        T = len(actions)

                if ik_fk is not None:
                    eef_q = np.zeros((T, 9), dtype=np.float32)
                    eef_a = np.zeros((T, 10), dtype=np.float32)
                    for idx_t in range(T):
                        pq, rq = ik_fk.forward_kinematics(arm_qpos[idx_t])
                        pa, ra = ik_fk.forward_kinematics(actions[idx_t, :7])
                        eef_q[idx_t, :3] = pq
                        eef_q[idx_t, 3:9] = rotmat_to_rot6d(rq)
                        eef_a[idx_t, :3] = pa
                        eef_a[idx_t, 3:9] = rotmat_to_rot6d(ra)
                        eef_a[idx_t, 9] = actions[idx_t, 7]
                    arm_state = eef_q
                    actions = eef_a
                else:
                    arm_state = arm_qpos

                if self.use_progress:
                    progress = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
                    base_proprio = np.concatenate([arm_state, gripper_width, progress], axis=-1)
                else:
                    base_proprio = np.concatenate([arm_state, gripper_width], axis=-1)

                if self.proprio_history_lags == (0,):
                    proprio = base_proprio
                else:
                    lagged_list = []
                    for lag in self.proprio_history_lags:
                        idx = np.clip(np.arange(T) - lag, 0, T - 1)
                        lagged_list.append(base_proprio[idx])
                    proprio = np.concatenate(lagged_list, axis=-1)

                self.episodes_proprio.append(proprio)
                self.episodes_act.append(actions)

                if "grip_pos" in obs_grp and "spout_pos" in obs_grp and "plant_pos" in obs_grp:
                    grip_pos = np.array(obs_grp["grip_pos"], dtype=np.float32)
                    spout_pos = np.array(obs_grp["spout_pos"], dtype=np.float32)
                    plant_pos = np.array(obs_grp["plant_pos"], dtype=np.float32)
                    pose = np.concatenate([grip_pos, spout_pos, plant_pos], axis=-1)
                    if keep_mask is not None:
                        pose = pose[keep_mask]
                    self.episodes_pose.append(pose)
                    self.has_aux_pose = True

                for cam in self.cameras:
                    cam_key = f"rgb_{cam}"
                    img_data = np.array(obs_grp[cam_key], dtype=np.uint8)
                    if keep_mask is not None:
                        img_data = img_data[keep_mask]
                    self.episodes_images[cam].append(img_data)

                for t in range(T):
                    self.indices.append((ep_idx, t))

    def _extract_raw_chunk(self, ep_idx: int, t: int) -> np.ndarray:
        """Extract an (H, act_dim) action chunk at (ep_idx, t) in the configured action_space."""
        ep_act = self.episodes_act[ep_idx]
        chunk = ep_act[t : t + self.horizon]
        if len(chunk) < self.horizon:
            pad_count = self.horizon - len(chunk)
            last_act = ep_act[-1:]
            padding = np.repeat(last_act, pad_count, axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)
        if self.action_space == "joint_delta":
            q_t = self.episodes_proprio[ep_idx][t, :7]
            chunk = chunk.copy()
            chunk[:, :7] = chunk[:, :7] - q_t[None, :]
        return chunk

    def _compute_stats(self) -> Dict[str, np.ndarray]:
        all_proprio = np.concatenate(self.episodes_proprio, axis=0)

        proprio_mean = np.mean(all_proprio, axis=0).astype(np.float32)
        proprio_std = np.std(all_proprio, axis=0).astype(np.float32)
        proprio_std = np.clip(proprio_std, a_min=1e-3, a_max=None)

        if self.action_space == "joint_delta":
            all_chunks = np.stack(
                [self._extract_raw_chunk(ep_idx, t) for ep_idx, t in self.indices],
                axis=0,
            )
            act_mean = np.mean(all_chunks, axis=0).astype(np.float32)
            act_std = np.std(all_chunks, axis=0).astype(np.float32)
            act_std = np.clip(act_std, a_min=1e-3, a_max=None)
        else:
            all_act = np.concatenate(self.episodes_act, axis=0)
            act_mean = np.mean(all_act, axis=0).astype(np.float32)
            act_std = np.std(all_act, axis=0).astype(np.float32)
            act_std = np.clip(act_std, a_min=1e-3, a_max=None)
            if self.action_space == "eef_se3":
                act_std[3:9] = np.clip(act_std[3:9], a_min=0.05, a_max=None)
                base_dim = 11 if self.use_progress else 10
                for lag_i in range(len(self.proprio_history_lags)):
                    offset = lag_i * base_dim
                    proprio_std[offset + 3 : offset + 9] = np.clip(
                        proprio_std[offset + 3 : offset + 9], a_min=0.05, a_max=None
                    )

        stats_dict: Dict[str, np.ndarray] = {
            "proprio_mean": proprio_mean,
            "proprio_std": proprio_std,
            "act_mean": act_mean,
            "act_std": act_std,
        }

        if self.has_aux_pose and len(self.episodes_pose) > 0:
            all_pose = np.concatenate(self.episodes_pose, axis=0)
            pose_mean = np.mean(all_pose, axis=0).astype(np.float32)
            pose_std = np.std(all_pose, axis=0).astype(np.float32)
            pose_std = np.clip(pose_std, a_min=1e-2, a_max=None)
            stats_dict["pose_mean"] = pose_mean
            stats_dict["pose_std"] = pose_std

        return stats_dict

    def _build_tensor_cache(self) -> None:
        """Pre-pack contiguous normalized tensors in RAM so batch loading avoids per-item Python loops."""
        all_proprio = np.concatenate(self.episodes_proprio, axis=0)
        norm_proprio = self.normalize_proprio(all_proprio).astype(np.float32)
        self._cached_proprio = torch.from_numpy(norm_proprio)

        raw_chunks = np.stack(
            [self._extract_raw_chunk(ep_idx, t) for ep_idx, t in self.indices],
            axis=0,
        )
        norm_chunks = self.normalize_action(raw_chunks).astype(np.float32)
        self._cached_actions = torch.from_numpy(norm_chunks)

        if self.has_aux_pose and "pose_mean" in self.stats and len(self.episodes_pose) > 0:
            all_pose = np.concatenate(self.episodes_pose, axis=0)
            norm_pose = ((all_pose - self.stats["pose_mean"]) / self.stats["pose_std"]).astype(np.float32)
            self._cached_pose = torch.from_numpy(norm_pose)

        for cam in self.cameras:
            cat_imgs = np.concatenate(self.episodes_images[cam], axis=0)
            self._cached_images[cam] = torch.from_numpy(cat_imgs)
            # Re-point episodes_images to zero-copy slices of cat_imgs to avoid duplicating RAM
            offset = 0
            for ep_i in range(len(self.episodes_images[cam])):
                ep_len = len(self.episodes_images[cam][ep_i])
                self.episodes_images[cam][ep_i] = cat_imgs[offset : offset + ep_len]
                offset += ep_len

    def get_batch(
        self,
        batch_indices: torch.Tensor,
        device: torch.device,
        shifter: Optional[torch.nn.Module] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Fetch and normalize an entire minibatch directly on the target device in C++."""
        assert self._cached_proprio is not None and self._cached_actions is not None
        dev_obs: Dict[str, torch.Tensor] = {
            "proprio": self._cached_proprio[batch_indices].to(device),
        }
        if self._cached_pose is not None:
            dev_obs["aux_pose"] = self._cached_pose[batch_indices].to(device)

        for cam in self.cameras:
            cam_key = f"rgb_{cam}"
            img_dev = self._cached_images[cam][batch_indices].to(device)
            img_f32 = img_dev.permute(0, 3, 1, 2).float() * (1.0 / 255.0)
            if shifter is not None:
                img_f32 = shifter(img_f32)
            dev_obs[cam_key] = img_f32

        dev_actions = self._cached_actions[batch_indices].to(device)
        return dev_obs, dev_actions

    def normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        return (proprio - self.stats["proprio_mean"]) / self.stats["proprio_std"]

    def normalize_action(self, act: np.ndarray) -> np.ndarray:
        return (act - self.stats["act_mean"]) / self.stats["act_std"]

    def unnormalize_action(self, act: np.ndarray) -> np.ndarray:
        return act * self.stats["act_std"] + self.stats["act_mean"]

    def save_stats(self, json_path: str) -> None:
        """Save normalization statistics to JSON file."""
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v.tolist() for k, v in self.stats.items()}
        with open(p, "w") as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load_stats(cls, json_path: str) -> Dict[str, np.ndarray]:
        """Load normalization statistics from JSON file."""
        with open(json_path, "r") as f:
            raw = json.load(f)
        return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if self._cached_proprio is not None and self._cached_actions is not None:
            obs_dict: Dict[str, torch.Tensor] = {}
            for cam in self.cameras:
                raw_img = self._cached_images[cam][idx]
                if self.return_uint8_images:
                    obs_dict[f"rgb_{cam}"] = raw_img
                else:
                    obs_dict[f"rgb_{cam}"] = raw_img.permute(2, 0, 1).float() / 255.0
            obs_dict["proprio"] = self._cached_proprio[idx]
            if self._cached_pose is not None:
                obs_dict["aux_pose"] = self._cached_pose[idx]
            return obs_dict, self._cached_actions[idx]

        ep_idx, t = self.indices[idx]

        obs_dict = {}
        for cam in self.cameras:
            raw_img_np = self.episodes_images[cam][ep_idx][t]
            if self.return_uint8_images:
                obs_dict[f"rgb_{cam}"] = torch.from_numpy(raw_img_np)
            else:
                img_tensor = torch.from_numpy(raw_img_np).permute(2, 0, 1).float() / 255.0
                obs_dict[f"rgb_{cam}"] = img_tensor

        raw_proprio = self.episodes_proprio[ep_idx][t]
        norm_proprio = self.normalize_proprio(raw_proprio)
        obs_dict["proprio"] = torch.from_numpy(norm_proprio).float()

        if self.has_aux_pose and "pose_mean" in self.stats:
            raw_pose = self.episodes_pose[ep_idx][t]
            norm_pose = (raw_pose - self.stats["pose_mean"]) / self.stats["pose_std"]
            obs_dict["aux_pose"] = torch.from_numpy(norm_pose).float()

        chunk = self._extract_raw_chunk(ep_idx, t)
        norm_chunk = self.normalize_action(chunk)
        act_tensor = torch.from_numpy(norm_chunk).float()

        return obs_dict, act_tensor
