"""HDF5 Demonstration Dataset Loader with Rolling Action Chunk Extraction.

Loads synthetic plant watering demonstrations, normalizes observations and
actions using dataset statistics, and constructs temporal action chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    # Explicit relative spatial vectors (9 dimensions)
    vec_ee_to_grip = (grip_pos - ee_pos).astype(np.float32)
    vec_ee_to_plant = (plant_pos - ee_pos).astype(np.float32)
    vec_spout_to_plant = (plant_pos - spout_pos).astype(np.float32)
    parts.extend([vec_ee_to_grip, vec_ee_to_plant, vec_spout_to_plant])

    # Trajectory phase progress in [0, 1] (1 dimension)
    progress = np.array([min(step_idx / TOTAL_DEMO_HORIZON, 1.0)], dtype=np.float32)
    parts.append(progress)

    return np.concatenate(parts, axis=0).astype(np.float32)


class WateringDemonstrationDataset(Dataset):
    """PyTorch Dataset yielding normalized observation and action chunk pairs."""

    def __init__(
        self,
        h5_path: str = "data/watering_demos_100.h5",
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

                # Assemble observations for this episode
                obs_arrays = []
                for t in range(T):
                    t_obs = {k: np.array(obs_grp[k][t]) for k in OBS_BASE_KEYS}
                    obs_vec = extract_observation_vector(t_obs, step_idx=t)
                    obs_arrays.append(obs_vec)

                ep_obs = np.stack(obs_arrays, axis=0)  # (T, 52)
                self.episodes_obs.append(ep_obs)
                self.episodes_act.append(actions)  # (T, 8)

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
        T = len(ep_act)

        # Slice action chunk [t : t + H], padding with final action if needed
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
