"""Training Pipeline for State and Multi-Camera Vision Flow Matching Policies.

Supports accelerated training on Apple Silicon MPS, CUDA GPUs, and CPU, with
GPU-vectorized uint8 image normalization and asynchronous GPU metric accumulation.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import h5py
import torch
from torch.utils.data import DataLoader

from floraflow.training.augmentation import RandomShifter
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
    data_path: str = "datasets/watering_demos_vision_3cam_300.h5",
    save_dir: str = "checkpoints",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    horizon: int = 16,
    num_keypoints: int = 32,
    vision_feat_dim: int = 64,
    proprio_feat_dim: int = 64,
    hidden_dim: int = 256,
    num_blocks: int = 4,
    gripper_weight: float = 2.5,
    shift_aug: int = 4,
    dropout: float = 0.0,
    keypoint_noise: float = 0.0,
    use_cross_attention: bool = False,
    camera_dropout: float = 0.0,
    dropout_cameras: Tuple[str, ...] = ("wrist_cam",),
    use_aux_pose: bool = False,
    aux_pose_weight: float = 0.5,
    use_progress: bool = True,
    proprio_history_lags: Tuple[int, ...] = (0,),
    trim_stationary: bool = False,
    action_space: str = "joint_abs",
    device_str: str = "auto",
    cameras: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """Train Vision Flow Matching action chunker policy."""
    device = resolve_compute_device(device_str)
    print(f"Using compute device: {device}")

    save_path = Path(save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    if cameras is None:
        with h5py.File(data_path, "r") as f:
            if "cameras" in f.attrs:
                cameras = tuple(f.attrs["cameras"])
            else:
                cameras = ("third_person_cam", "overhead_cam")
    print(f"Active cameras: {cameras}")

    print(f"Loading visual demonstration dataset from: {data_path}")
    dataset = VisionWateringDataset(
        h5_path=data_path,
        horizon=horizon,
        cameras=cameras,
        preload=True,
        use_progress=use_progress,
        proprio_history_lags=proprio_history_lags,
        trim_stationary=trim_stationary,
        action_space=action_space,
        return_uint8_images=True,
    )
    num_samples = len(dataset)
    num_batches = num_samples // batch_size
    print(
        f"Dataset loaded: {num_samples} samples across {num_batches} batches/epoch "
        f"(proprio_dim={dataset.proprio_dim}, action_space={action_space}, use_progress={use_progress}, "
        f"lags={dataset.proprio_history_lags}, trim_stationary={trim_stationary})."
    )

    stats_file = save_path / "vision_stats.json"
    dataset.save_stats(str(stats_file))
    print(f"Vision normalization statistics saved to: {stats_file}")

    model = VisionFlowMatchingPolicy(
        act_dim=dataset.act_dim,
        horizon=horizon,
        proprio_dim=dataset.proprio_dim,
        num_keypoints=num_keypoints,
        vision_feat_dim=vision_feat_dim,
        proprio_feat_dim=proprio_feat_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        keypoint_noise=keypoint_noise,
        cameras=cameras,
        use_cross_attention=use_cross_attention,
        camera_dropout=camera_dropout,
        dropout_cameras=dropout_cameras,
        use_aux_pose=use_aux_pose,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Vision policy initialized with {total_params:,} trainable parameters (fused_dim={model.fused_dim}).")
    if dropout > 0.0 or keypoint_noise > 0.0:
        print(f"Regularization active: dropout={dropout:.2f}, keypoint_noise={keypoint_noise:.4f}.")
    if camera_dropout > 0.0:
        print(f"Modality masking active: Camera dropout p={camera_dropout:.2f} on {dropout_cameras}.")
    if use_aux_pose:
        print(f"3D Ruler Quiz active: Auxiliary 3D Pose Supervision (weight={aux_pose_weight:.2f}).")

    shifter = RandomShifter(max_shift=shift_aug) if shift_aug > 0 else None
    if shifter is not None:
        print(f"Visual data augmentation: RandomShifter(max_shift={shift_aug}) active.")
    else:
        print("Visual data augmentation: RandomShifter disabled (shift_aug=0, exact multi-view geometry).")

    cfm = ConditionalFlowMatcher(sigma_min=1e-4, gripper_weight=gripper_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_loss = float("inf")
    t_start = time.time()

    print("\nStarting Vision Flow Matching training loop:")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_t = torch.zeros((), device=device, dtype=torch.float32)
        epoch_pose_loss_t = torch.zeros((), device=device, dtype=torch.float32)
        epoch_v_norm_t = torch.zeros((), device=device, dtype=torch.float32)
        epoch_u_norm_t = torch.zeros((), device=device, dtype=torch.float32)
        batches = 0

        perm = torch.randperm(num_samples)
        for b_idx in range(num_batches):
            batch_indices = perm[b_idx * batch_size : (b_idx + 1) * batch_size]
            dev_obs, dev_actions = dataset.get_batch(batch_indices, device=device, shifter=shifter)

            optimizer.zero_grad(set_to_none=True)
            cfm_loss, metrics = cfm.compute_loss(model, dev_actions, dev_obs, async_metrics=True)
            if use_aux_pose and "aux_pose" in dev_obs:
                pose_loss = model.compute_aux_pose_loss(dev_obs["aux_pose"])
                loss = cfm_loss + aux_pose_weight * pose_loss
                epoch_pose_loss_t += pose_loss.detach().float()
            else:
                loss = cfm_loss
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
        mean_pose_loss = float(epoch_pose_loss_t.item()) / denom
        mean_v_norm = float(epoch_v_norm_t.item()) / denom
        mean_u_norm = float(epoch_u_norm_t.item()) / denom
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            pose_str = f" | PoseMSE: {mean_pose_loss:.5f}" if use_aux_pose else ""
            print(
                f"Epoch [{epoch:3d}/{epochs:3d}] | "
                f"Loss: {mean_loss:.5f}{pose_str} | "
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
                    "config": {
                        "act_dim": dataset.act_dim,
                        "action_space": action_space,
                        "horizon": horizon,
                        "proprio_dim": dataset.proprio_dim,
                        "use_progress": use_progress,
                        "proprio_history_lags": list(dataset.proprio_history_lags),
                        "trim_stationary": trim_stationary,
                        "num_keypoints": num_keypoints,
                        "vision_feat_dim": vision_feat_dim,
                        "proprio_feat_dim": proprio_feat_dim,
                        "hidden_dim": hidden_dim,
                        "num_blocks": num_blocks,
                        "dropout": dropout,
                        "keypoint_noise": keypoint_noise,
                        "gripper_weight": gripper_weight,
                        "cameras": list(cameras),
                        "use_cross_attention": use_cross_attention,
                        "camera_dropout": camera_dropout,
                        "dropout_cameras": list(dropout_cameras),
                        "use_aux_pose": use_aux_pose,
                        "aux_pose_weight": aux_pose_weight,
                    },
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
    }
