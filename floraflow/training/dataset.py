"""HDF5 State and Multi-Camera Vision Dataset Loaders with Temporal Action Chunking.

Loads demonstration trajectories with state features or synchronized RGB camera frames
and robot proprioception, computing normalization statistics and serving batches for training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

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
    """PyTorch Dataset yielding multi-camera image observations, 8D proprioception, and 8D action chunks."""

    def __init__(
        self,
        h5_path: Union[str, Path, Sequence[Union[str, Path]]] = "datasets/watering_demos_vision_3cam_300.h5",
        horizon: int = 16,
        cameras: Tuple[str, ...] = ("third_person_cam", "overhead_cam", "wrist_cam"),
        stats: Optional[Dict[str, np.ndarray]] = None,
        preload: bool = True,
        return_uint8_images: bool = False,
        sample_stride: Optional[int] = None,
    ) -> None:
        """Initialize Vision Demonstration Dataset.

        Args:
            h5_path: Path or sequence of paths to HDF5 demonstration archives.
            horizon: Prediction action chunk horizon H.
            cameras: Names of camera streams to load.
            stats: Optional precomputed normalization statistics dictionary.
            preload: If True, loads all images and proprioception into RAM for zero-disk-IO training.
            return_uint8_images: If True, returns raw (H, W, 3) uint8 camera tensors so permutation and
                [0, 1] float normalization can execute in a single vectorized GPU batch kernel.
            sample_stride: Optional temporal anchor stride across episodes. If None, defaults to 1 for
                <=350 episodes and 2 for >350 episodes so 600-episode multi-camera archives stay <7 GB in RAM
                while preserving full 20 Hz 16-step action chunks and staggered odd/even phase coverage.
        """
        super().__init__()
        if isinstance(h5_path, (str, Path)):
            raw_str = str(h5_path)
            if "," in raw_str:
                self.h5_paths = [Path(p.strip()).resolve() for p in raw_str.split(",") if p.strip()]
            else:
                self.h5_paths = [Path(h5_path).resolve()]
        else:
            self.h5_paths = [Path(p).resolve() for p in h5_path]
        self.h5_path = self.h5_paths[0]
        self.horizon = horizon
        self.cameras = tuple(cameras)
        self.preload = preload
        self.return_uint8_images = return_uint8_images
        self.act_dim = 8
        self.proprio_dim = 8

        for p in self.h5_paths:
            if not p.exists():
                raise FileNotFoundError(f"HDF5 dataset not found at: {p}")

        if sample_stride is None:
            total_eps = 0
            for p in self.h5_paths:
                with h5py.File(p, "r") as f:
                    total_eps += len(f["data"].keys())
            self.sample_stride = 2 if total_eps > 350 else 1
        else:
            self.sample_stride = max(1, int(sample_stride))

        self.episodes_images: Dict[str, List[np.ndarray]] = {cam: [] for cam in self.cameras}
        self.episodes_proprio: List[np.ndarray] = []
        self.episodes_act: List[np.ndarray] = []
        self.episode_sources: List[int] = []
        self.sample_sources: List[int] = []
        self.indices: List[Tuple[int, int]] = []
        self.local_img_indices: List[int] = []

        self._load_data()

        if stats is None:
            self.stats = self._compute_stats()
        else:
            self.stats = stats

        self._cached_images: Dict[str, torch.Tensor] = {}
        self._cached_proprio: Optional[torch.Tensor] = None
        self._cached_actions: Optional[torch.Tensor] = None
        if self.preload:
            self._build_tensor_cache()

    def _load_data(self) -> None:
        open_files = [h5py.File(path, "r") for path in self.h5_paths]
        try:
            per_source_demos: List[List[Tuple[int, str]]] = []
            for src_idx, f in enumerate(open_files):
                data_grp = f["data"]
                names = sorted(list(data_grp.keys()), key=lambda x: int(x.split("_")[1]))
                per_source_demos.append([(src_idx, name) for name in names])

            # Interleave episodes across sources (e.g. clean_0, dr_0, clean_1, dr_1, ...)
            # so any contiguous memory window has an exact balanced mix of all domains.
            interleaved_demos: List[Tuple[int, str]] = []
            max_len = max((len(lst) for lst in per_source_demos), default=0)
            for i in range(max_len):
                for lst in per_source_demos:
                    if i < len(lst):
                        interleaved_demos.append(lst[i])

            for ep_idx, (src_idx, name) in enumerate(interleaved_demos):
                demo = open_files[src_idx]["data"][name]
                obs_grp = demo["obs"]
                actions = np.array(demo["actions"], dtype=np.float32)
                T = len(actions)

                arm_qpos = np.array(obs_grp["arm_qpos"], dtype=np.float32)
                gripper_width = np.array(obs_grp["gripper_width"], dtype=np.float32)
                if gripper_width.ndim == 1:
                    gripper_width = gripper_width[:, None]

                keep_mask: Optional[np.ndarray] = None
                if T > 2:
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

                proprio = np.concatenate([arm_qpos, gripper_width], axis=-1)
                self.episodes_proprio.append(proprio)
                self.episodes_act.append(actions)
                self.episode_sources.append(src_idx)

                phase_offset = ((ep_idx // max(1, len(open_files))) % self.sample_stride) if T > self.sample_stride else 0
                anchor_ts = np.arange(phase_offset, T, self.sample_stride)

                for cam in self.cameras:
                    cam_key = f"rgb_{cam}"
                    img_data = np.array(obs_grp[cam_key], dtype=np.uint8)
                    if keep_mask is not None:
                        img_data = img_data[keep_mask]
                    if self.sample_stride > 1:
                        img_data = np.ascontiguousarray(img_data[anchor_ts])
                    self.episodes_images[cam].append(img_data)

                for local_i, t in enumerate(anchor_ts):
                    self.indices.append((ep_idx, int(t)))
                    self.local_img_indices.append(local_i)
                    self.sample_sources.append(src_idx)
        finally:
            for f in open_files:
                f.close()

    def _extract_raw_chunk(self, ep_idx: int, t: int) -> np.ndarray:
        """Extract an (H, 8) joint action chunk at (ep_idx, t)."""
        ep_act = self.episodes_act[ep_idx]
        chunk = ep_act[t : t + self.horizon]
        if len(chunk) < self.horizon:
            pad_count = self.horizon - len(chunk)
            last_act = ep_act[-1:]
            padding = np.repeat(last_act, pad_count, axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)
        return chunk

    def _compute_stats(self) -> Dict[str, np.ndarray]:
        all_proprio = np.concatenate(self.episodes_proprio, axis=0)
        proprio_mean = np.mean(all_proprio, axis=0).astype(np.float32)
        proprio_std = np.std(all_proprio, axis=0).astype(np.float32)
        proprio_std = np.clip(proprio_std, a_min=1e-3, a_max=None)

        all_act = np.concatenate(self.episodes_act, axis=0)
        act_mean = np.mean(all_act, axis=0).astype(np.float32)
        act_std = np.std(all_act, axis=0).astype(np.float32)
        act_std = np.clip(act_std, a_min=1e-3, a_max=None)

        return {
            "proprio_mean": proprio_mean,
            "proprio_std": proprio_std,
            "act_mean": act_mean,
            "act_std": act_std,
        }

    def _build_tensor_cache(self) -> None:
        """Pre-pack contiguous normalized tensors in RAM without temporary 2x memory duplication."""
        indexed_proprio = np.stack(
            [self.episodes_proprio[ep_idx][t] for ep_idx, t in self.indices],
            axis=0,
        )
        norm_proprio = self.normalize_proprio(indexed_proprio).astype(np.float32)
        self._cached_proprio = torch.from_numpy(norm_proprio)

        raw_chunks = np.stack(
            [self._extract_raw_chunk(ep_idx, t) for ep_idx, t in self.indices],
            axis=0,
        )
        norm_chunks = self.normalize_action(raw_chunks).astype(np.float32)
        self._cached_actions = torch.from_numpy(norm_chunks)

        num_total = len(self.indices)
        for cam in self.cameras:
            ep_list = self.episodes_images[cam]
            sample_shape = ep_list[0].shape[1:]
            cat_tensor = torch.empty((num_total, *sample_shape), dtype=torch.uint8)
            cat_np = cat_tensor.numpy()
            offset = 0
            for ep_i in range(len(ep_list)):
                ep_arr = ep_list[ep_i]
                ep_len = len(ep_arr)
                cat_np[offset : offset + ep_len] = ep_arr
                ep_list[ep_i] = cat_np[offset : offset + ep_len]
                offset += ep_len
            self._cached_images[cam] = cat_tensor

    def sample_epoch_permutation(self, window_samples: int = 8192) -> torch.Tensor:
        """Generate a cache-friendly rolling-window permutation across the dataset."""
        n = len(self.indices)
        if n <= window_samples or len(self.h5_paths) <= 1:
            return torch.randperm(n)

        shift = int(torch.randint(0, window_samples, (1,)).item())
        base_indices = torch.roll(torch.arange(n, dtype=torch.long), shifts=shift)
        windows = list(torch.split(base_indices, window_samples))
        win_order = torch.randperm(len(windows)).tolist()
        shuffled_windows = [
            windows[w_idx][torch.randperm(len(windows[w_idx]))]
            for w_idx in win_order
        ]
        return torch.cat(shuffled_windows, dim=0)

    def get_batch(
        self,
        batch_indices: torch.Tensor,
        device: torch.device,
        shifter: Optional[torch.nn.Module] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Fetch and normalize an entire minibatch directly on the target device."""
        assert self._cached_proprio is not None and self._cached_actions is not None
        dev_obs: Dict[str, torch.Tensor] = {
            "proprio": self._cached_proprio[batch_indices].to(device),
        }

        for cam in self.cameras:
            cam_key = f"rgb_{cam}"
            img_dev = self._cached_images[cam][batch_indices].to(device)
            img_f32 = img_dev.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
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
            return obs_dict, self._cached_actions[idx]

        ep_idx, t = self.indices[idx]
        local_i = self.local_img_indices[idx]

        obs_dict = {}
        for cam in self.cameras:
            raw_img_np = self.episodes_images[cam][ep_idx][local_i]
            if self.return_uint8_images:
                obs_dict[f"rgb_{cam}"] = torch.from_numpy(raw_img_np)
            else:
                img_tensor = torch.from_numpy(raw_img_np).permute(2, 0, 1).float() / 255.0
                obs_dict[f"rgb_{cam}"] = img_tensor

        raw_proprio = self.episodes_proprio[ep_idx][t]
        norm_proprio = self.normalize_proprio(raw_proprio)
        obs_dict["proprio"] = torch.from_numpy(norm_proprio).float()

        chunk = self._extract_raw_chunk(ep_idx, t)
        norm_chunk = self.normalize_action(chunk)
        act_tensor = torch.from_numpy(norm_chunk).float()

        return obs_dict, act_tensor
