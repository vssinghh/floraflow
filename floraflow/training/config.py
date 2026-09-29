"""Single-Source-of-Truth Immutable Configuration for Vision Policy Training.

Defines the frozen VisionTrainConfig dataclass used across the CLI, trainer,
policy architecture, and closed-loop evaluator so hyperparameters cannot drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import torch

RETIRED_CONFIG_KEYS = frozenset({
    "use_progress",
    "proprio_history_lags",
    "trim_stationary",
    "action_space",
    "use_cross_attention",
    "use_aux_pose",
    "aux_pose_weight",
})


@dataclass(frozen=True)
class VisionTrainConfig:
    """Immutable canonical configuration for Multi-Camera Vision Flow Matching."""

    data_path: Tuple[str, ...] = ("datasets/watering_demos_vision_3cam_300.h5",)
    save_dir: str = "checkpoints"
    epochs: int = 20
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    num_flow_samples: int = 4

    # Locked architectural constants
    act_dim: int = 8
    proprio_dim: int = 8
    horizon: int = 16
    vision_feat_dim: int = 32
    proprio_feat_dim: int = 64
    hidden_dim: int = 256
    num_blocks: int = 4
    attn_heads: int = 4
    gripper_weight: float = 2.5
    shift_aug: int = 4
    cameras: Tuple[str, ...] = ("third_person_cam", "overhead_cam", "wrist_cam")
    dropout_cameras: Tuple[str, ...] = ("wrist_cam",)

    # Tunable capacity & regularization
    num_keypoints: int = 32
    dropout: float = 0.05
    keypoint_noise: float = 0.01
    camera_dropout: float = 0.10

    def __post_init__(self) -> None:
        if isinstance(self.data_path, (str, Path)):
            raw_str = str(self.data_path)
            paths = tuple(p.strip() for p in raw_str.split(",") if p.strip())
            object.__setattr__(self, "data_path", paths)
        else:
            object.__setattr__(self, "data_path", tuple(str(p) for p in self.data_path))

        object.__setattr__(self, "cameras", tuple(str(c) for c in self.cameras))
        object.__setattr__(self, "dropout_cameras", tuple(str(c) for c in self.dropout_cameras))

        if self.act_dim != 8 or self.proprio_dim != 8:
            raise ValueError(
                f"VisionTrainConfig requires 8D joint proprioception and 8D joint actions "
                f"(got proprio_dim={self.proprio_dim}, act_dim={self.act_dim})."
            )
        if self.num_flow_samples < 1:
            raise ValueError(f"num_flow_samples must be >= 1, got {self.num_flow_samples}.")

    def with_overrides(self, **kwargs: Any) -> VisionTrainConfig:
        """Return a new frozen VisionTrainConfig with explicit field overrides."""
        valid_names = {f.name for f in fields(self)}
        unknown = set(kwargs.keys()) - valid_names
        if unknown:
            raise ValueError(f"Unknown VisionTrainConfig field(s): {sorted(unknown)}")
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        raw = asdict(self)
        raw["data_path"] = list(self.data_path)
        raw["cameras"] = list(self.cameras)
        raw["dropout_cameras"] = list(self.dropout_cameras)
        return raw

    def diff_from_default(self) -> Dict[str, Tuple[Any, Any]]:
        """Return fields that differ from the canonical VisionTrainConfig() defaults as {key: (default, current)}."""
        default_cfg = VisionTrainConfig()
        diffs: Dict[str, Tuple[Any, Any]] = {}
        for f in fields(self):
            def_val = getattr(default_cfg, f.name)
            cur_val = getattr(self, f.name)
            if def_val != cur_val and f.name not in ("data_path", "save_dir"):
                diffs[f.name] = (def_val, cur_val)
        return diffs

    @classmethod
    def from_dict(cls, data: Dict[str, Any], strict: bool = True) -> VisionTrainConfig:
        """Construct VisionTrainConfig from a dictionary, rejecting unknown keys when strict=True."""
        valid_names = {f.name for f in fields(cls)}
        cleaned: Dict[str, Any] = {}
        unknown = []
        for k, v in data.items():
            if k in valid_names:
                cleaned[k] = v
            elif not strict and k in RETIRED_CONFIG_KEYS:
                continue
            else:
                unknown.append(k)
        if unknown and strict:
            raise ValueError(f"Unrecognized VisionTrainConfig key(s): {sorted(unknown)}")
        return cls(**cleaned)

    @classmethod
    def load(cls, path: Union[str, Path], strict: bool = False) -> VisionTrainConfig:
        """Load VisionTrainConfig from a JSON file or a saved PyTorch (.pt) checkpoint."""
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Config or checkpoint file not found: {p}")
        if p.suffix == ".json":
            with open(p, "r") as f:
                raw = json.load(f)
            return cls.from_dict(raw, strict=True if strict else False)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "config" in ckpt:
            return cls.from_dict(ckpt["config"], strict=strict)
        raise ValueError(f"File {p} does not contain a valid 'config' dictionary.")
