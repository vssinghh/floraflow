"""Training Pipeline for State and Multi-Camera Vision Flow Matching Policies.

Supports accelerated training on Apple Silicon MPS, CUDA GPUs, and CPU, with
GPU-vectorized uint8 image normalization and asynchronous GPU metric accumulation.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import h5py
import torch
from torch.utils.data import DataLoader

from floraflow.training.augmentation import RandomShifter
from floraflow.training.config import VisionTrainConfig
from floraflow.training.dataset import VisionWateringDataset, WateringDemonstrationDataset
from floraflow.training.flow_matching import ConditionalFlowMatcher
from floraflow.training.model import FlowMatchingPolicy
from floraflow.training.vision_model import VisionFlowMatchingPolicy


def resolve_compute_device(device_str: str = "auto") -> torch.device:
    """Resolve PyTorch compute device (mps, cuda, or cpu)."""
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def train_policy(
    data_path: str = "datasets/watering_demos_100.h5",
    save_dir: str = "checkpoints",
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    horizon: int = 16,
    hidden_dim: int = 256,
    num_blocks: int = 4,
    gripper_weight: float = 2.5,
    device_str: str = "auto",
) -> None:
    """Train state-based Flow Matching action chunker policy."""
    device = resolve_compute_device(device_str)
    print(f"Using compute device: {device}")

    save_path = Path(save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    dataset = WateringDemonstrationDataset(data_path, horizon=horizon)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    print(f"Dataset loaded: {len(dataset)} samples across {len(loader)} batches/epoch.")

    stats_file = save_path / "stats.json"
    dataset.save_stats(str(stats_file))
    print(f"Normalization statistics saved to: {stats_file}")

    obs_dim = 52
    model = FlowMatchingPolicy(
        obs_dim=obs_dim,
        act_dim=8,
        horizon=horizon,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Policy network initialized with {total_params:,} trainable parameters.")

    cfm = ConditionalFlowMatcher(sigma_min=1e-4, gripper_weight=gripper_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_loss = float("inf")
    t_start = time.time()

    print("\nStarting Flow Matching training loop:")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_t = torch.zeros((), device=device, dtype=torch.float32)
        epoch_v_norm_t = torch.zeros((), device=device, dtype=torch.float32)
        epoch_u_norm_t = torch.zeros((), device=device, dtype=torch.float32)
        batches = 0

        for norm_obs, norm_act in loader:
            norm_obs = norm_obs.to(device)
            norm_act = norm_act.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = cfm.compute_loss(model, norm_act, norm_obs, async_metrics=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_t += metrics["loss"]
            epoch_v_norm_t += metrics["v_norm"]
            epoch_u_norm_t += metrics["u_norm"]
            batches += 1

        scheduler.step()

        denom = float(max(batches, 1))
        avg_loss = float(epoch_loss_t.item()) / denom
        avg_v_norm = float(epoch_v_norm_t.item()) / denom
        avg_u_norm = float(epoch_u_norm_t.item()) / denom
        current_lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - t_start
            print(
                f"Epoch [{epoch:3d}/{epochs:3d}] | "
                f"Loss: {avg_loss:.5f} | "
                f"v_norm: {avg_v_norm:.2f} | "
                f"u_norm: {avg_u_norm:.2f} | "
                f"LR: {current_lr:.6f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "obs_dim": obs_dim,
                    "act_dim": 8,
                    "horizon": horizon,
                    "hidden_dim": hidden_dim,
                    "num_blocks": num_blocks,
                },
                "loss": avg_loss,
                "stats": dataset.stats,
            }
            torch.save(ckpt_data, save_path / "best_policy.pt")

    final_ckpt = {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": {
            "obs_dim": obs_dim,
            "act_dim": 8,
            "horizon": horizon,
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
        },
        "loss": avg_loss,
        "stats": dataset.stats,
    }
    torch.save(final_ckpt, save_path / "latest_policy.pt")

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time:.1f}s. Best training loss: {best_loss:.5f}.")
    print(f"Checkpoints saved to: {save_path / 'best_policy.pt'}")


def train_vision_policy(
    config: Optional[VisionTrainConfig] = None,
    data_path: Optional[Union[str, Sequence[str]]] = None,
    save_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    num_flow_samples: Optional[int] = None,
    num_keypoints: Optional[int] = None,
    dropout: Optional[float] = None,
    keypoint_noise: Optional[float] = None,
    camera_dropout: Optional[float] = None,
    cameras: Optional[Tuple[str, ...]] = None,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Train Vision Flow Matching action chunker policy using a frozen VisionTrainConfig."""
    base_cfg = config if config is not None else VisionTrainConfig()
    overrides: Dict[str, Any] = {}
    if data_path is not None:
        overrides["data_path"] = data_path
    if save_dir is not None:
        overrides["save_dir"] = save_dir
    if epochs is not None:
        overrides["epochs"] = epochs
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if lr is not None:
        overrides["lr"] = lr
    if num_flow_samples is not None:
        overrides["num_flow_samples"] = num_flow_samples
    if num_keypoints is not None:
        overrides["num_keypoints"] = num_keypoints
    if dropout is not None:
        overrides["dropout"] = dropout
    if keypoint_noise is not None:
        overrides["keypoint_noise"] = keypoint_noise
    if camera_dropout is not None:
        overrides["camera_dropout"] = camera_dropout
    if cameras is not None:
        overrides["cameras"] = cameras

    cfg = base_cfg.with_overrides(**overrides) if overrides else base_cfg

    first_path = cfg.data_path[0]
    with h5py.File(first_path, "r") as f:
        if "cameras" in f.attrs and cameras is None and config is None:
            cfg = cfg.with_overrides(cameras=tuple(f.attrs["cameras"]))

    device = resolve_compute_device(device_str)
    print(f"Using compute device: {device}")

    save_path = Path(cfg.save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    config_file = save_path / "train_config.json"
    with open(config_file, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"Saved frozen training config to: {config_file}")

    diffs = cfg.diff_from_default()
    if diffs:
        diff_str = ", ".join(f"{k}={cur} (default={dflt})" for k, (dflt, cur) in diffs.items())
        print(f"Non-default config overrides: {diff_str}")
    else:
        print("Config check: 100% canonical VisionTrainConfig defaults active.")

    print(f"Active cameras: {cfg.cameras}")
    print(f"Loading visual demonstration dataset from: {cfg.data_path}")
    dataset = VisionWateringDataset(
        h5_path=cfg.data_path,
        horizon=cfg.horizon,
        cameras=cfg.cameras,
        preload=True,
        return_uint8_images=True,
    )
    num_samples = len(dataset)
    num_batches = num_samples // cfg.batch_size
    print(
        f"Dataset loaded: {len(dataset.episodes_act)} episodes, {num_samples} samples across {num_batches} batches/epoch "
        f"(proprio_dim={dataset.proprio_dim}, act_dim={dataset.act_dim}, stride={dataset.sample_stride}, "
        f"num_flow_samples={cfg.num_flow_samples})."
    )

    stats_file = save_path / "vision_stats.json"
    dataset.save_stats(str(stats_file))
    print(f"Vision normalization statistics saved to: {stats_file}")

    model = VisionFlowMatchingPolicy.from_config(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Vision policy initialized with {total_params:,} trainable parameters (fused_dim={model.fused_dim}).")
    if cfg.dropout > 0.0 or cfg.keypoint_noise > 0.0 or cfg.camera_dropout > 0.0:
        print(
            f"Regularization active: dropout={cfg.dropout:.2f}, "
            f"keypoint_noise={cfg.keypoint_noise:.4f}, "
            f"camera_dropout={cfg.camera_dropout:.2f} on {cfg.dropout_cameras}."
        )

    shifter = RandomShifter(max_shift=cfg.shift_aug) if cfg.shift_aug > 0 else None
    if shifter is not None:
        print(f"Visual data augmentation: RandomShifter(max_shift={cfg.shift_aug}) active.")

    cfm = ConditionalFlowMatcher(sigma_min=1e-4, gripper_weight=cfg.gripper_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

    best_loss = float("inf")
    t_start = time.time()

    print("\nStarting Vision Flow Matching training loop:")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss_t = torch.zeros((), device=device, dtype=torch.float32)
        epoch_v_norm_t = torch.zeros((), device=device, dtype=torch.float32)
        epoch_u_norm_t = torch.zeros((), device=device, dtype=torch.float32)
        batches = 0

        perm = dataset.sample_epoch_permutation()
        for b_idx in range(num_batches):
            batch_indices = perm[b_idx * cfg.batch_size : (b_idx + 1) * cfg.batch_size]
            dev_obs, dev_actions = dataset.get_batch(batch_indices, device=device, shifter=shifter)

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = cfm.compute_loss(
                model,
                dev_actions,
                dev_obs,
                async_metrics=True,
                num_flow_samples=cfg.num_flow_samples,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_t += metrics["loss"]
            epoch_v_norm_t += metrics["v_norm"]
            epoch_u_norm_t += metrics["u_norm"]
            batches += 1

        scheduler.step()

        denom = float(max(batches, 1))
        mean_loss = float(epoch_loss_t.item()) / denom
        mean_v_norm = float(epoch_v_norm_t.item()) / denom
        mean_u_norm = float(epoch_u_norm_t.item()) / denom
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start

        if epoch % 5 == 0 or epoch == 1 or epoch == cfg.epochs:
            print(
                f"Epoch [{epoch:3d}/{cfg.epochs:3d}] | "
                f"Loss: {mean_loss:.5f} | "
                f"|v|: {mean_v_norm:.3f} | "
                f"|u|: {mean_u_norm:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        if mean_loss < best_loss:
            best_loss = mean_loss
            ckpt_path = save_path / "best_vision_policy.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": cfg.to_dict(),
                    "stats": dataset.stats,
                },
                ckpt_path,
            )

    total_elapsed = time.time() - t_start
    print(f"\nVision training complete in {total_elapsed:.1f}s. Best Loss: {best_loss:.5f}")
    print(f"Best model saved to: {save_path / 'best_vision_policy.pt'}")
    return {
        "best_loss": best_loss,
        "elapsed_s": total_elapsed,
        "checkpoint_path": str(save_path / "best_vision_policy.pt"),
        "stats_path": str(stats_file),
        "config_path": str(config_file),
    }
