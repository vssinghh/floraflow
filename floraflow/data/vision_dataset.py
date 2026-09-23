"""HDF5 Multi-Camera Vision Dataset Loader with Temporal Action Chunking.

Loads demonstration trajectories with synchronized RGB camera frames and robot proprioception,
computing normalization statistics and serving batches for vision policy training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class VisionWateringDataset(Dataset):
    """PyTorch Dataset yielding multi-camera image observations and action chunks."""

    def __init__(
        self,
        h5_path: str = "data/watering_demos_vision_100.h5",
        horizon: int = 16,
        cameras: Tuple[str, ...] = ("third_person_cam", "overhead_cam"),
        stats: Optional[Dict[str, np.ndarray]] = None,
        preload: bool = True,
    ) -> None:
        """Initialize Vision Demonstration Dataset.

        Args:
            h5_path: Path to the HDF5 demonstration archive.
            horizon: Prediction action chunk horizon H.
            cameras: Names of camera streams to load.
            stats: Optional precomputed normalization statistics dictionary.
            preload: If True, loads all images and proprioception into RAM for zero-disk-IO training.
        """
        super().__init__()
        self.h5_path = Path(h5_path).resolve()
        self.horizon = horizon
        self.cameras = cameras
        self.preload = preload

        self.episodes_images: Dict[str, List[np.ndarray]] = {cam: [] for cam in cameras}
        self.episodes_proprio: List[np.ndarray] = []
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

                # Assemble 9D proprioception (7D arm_qpos + 1D gripper_width + 1D progress)
                arm_qpos = np.array(obs_grp["arm_qpos"], dtype=np.float32)
                gripper_width = np.array(obs_grp["gripper_width"], dtype=np.float32)
                if gripper_width.ndim == 1:
                    gripper_width = gripper_width[:, None]
                progress = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
                proprio = np.concatenate([arm_qpos, gripper_width, progress], axis=-1)

                self.episodes_proprio.append(proprio)
                self.episodes_act.append(actions)

                # Load camera frames
                for cam in self.cameras:
                    cam_key = f"rgb_{cam}"
                    img_data = np.array(obs_grp[cam_key], dtype=np.uint8)
                    self.episodes_images[cam].append(img_data)

                for t in range(T):
                    self.indices.append((ep_idx, t))

    def _compute_stats(self) -> Dict[str, np.ndarray]:
        all_proprio = np.concatenate(self.episodes_proprio, axis=0)
        all_act = np.concatenate(self.episodes_act, axis=0)

        proprio_mean = np.mean(all_proprio, axis=0).astype(np.float32)
        proprio_std = np.std(all_proprio, axis=0).astype(np.float32)
        proprio_std = np.clip(proprio_std, a_min=1e-3, a_max=None)

        act_mean = np.mean(all_act, axis=0).astype(np.float32)
        act_std = np.std(all_act, axis=0).astype(np.float32)
        act_std = np.clip(act_std, a_min=1e-3, a_max=None)

        return {
            "proprio_mean": proprio_mean,
            "proprio_std": proprio_std,
            "act_mean": act_mean,
            "act_std": act_std,
        }

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
        ep_idx, t = self.indices[idx]

        # 1. Camera RGB images: uint8 in (H, W, 3) -> float32 in (3, H, W) normalized to [0, 1]
        obs_dict: Dict[str, torch.Tensor] = {}
        for cam in self.cameras:
            raw_img = self.episodes_images[cam][ep_idx][t]  # (H, W, 3)
            # Permute to (C, H, W) and scale to [0, 1]
            img_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0
            obs_dict[f"rgb_{cam}"] = img_tensor

        # 2. Proprioception: 8D vector normalized with dataset mean and std
        raw_proprio = self.episodes_proprio[ep_idx][t]
        norm_proprio = self.normalize_proprio(raw_proprio)
        obs_dict["proprio"] = torch.from_numpy(norm_proprio).float()

        # 3. Action chunk: slice [t : t + H], padded with final action if needed
        ep_act = self.episodes_act[ep_idx]
        chunk = ep_act[t : t + self.horizon]
        if len(chunk) < self.horizon:
            pad_count = self.horizon - len(chunk)
            last_act = ep_act[-1:]
            padding = np.repeat(last_act, pad_count, axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)

        norm_chunk = self.normalize_action(chunk)
        act_tensor = torch.from_numpy(norm_chunk).float()

        return obs_dict, act_tensor
