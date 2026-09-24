"""Train Vision-Language-Action Pixel-to-Action Flow Matching Policy.

Trains VisionFlowMatchingPolicy on multi-view camera demonstration frames and robot proprioception
using Optimal Transport Conditional Flow Matching with AdamW and Cosine Annealing scheduling.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from floraflow.data.augmentation import RandomShifter
from floraflow.data.vision_dataset import VisionWateringDataset
from floraflow.policy.flow_matching import ConditionalFlowMatcher
from floraflow.policy.vision_model import VisionFlowMatchingPolicy


def train_vision_policy(
    data_path: str = "data/watering_demos_vision_100.h5",
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
    use_cross_attention: bool = False,
    device_str: str = "auto",
    cameras: Optional[Tuple[str, ...]] = None,
) -> None:
    """Train Vision Flow Matching action chunker policy."""
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"Using compute device: {device}")

    save_path = Path(save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    if cameras is None:
        import h5py
        with h5py.File(data_path, "r") as f:
            if "cameras" in f.attrs:
                cameras = tuple(f.attrs["cameras"])
            else:
                cameras = ("third_person_cam", "overhead_cam")
    print(f"Active cameras: {cameras}")

    # 1. Load Vision Dataset
    print(f"Loading visual demonstration dataset from: {data_path}")
    dataset = VisionWateringDataset(
        h5_path=data_path,
        horizon=horizon,
        cameras=cameras,
        preload=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    print(f"Dataset loaded: {len(dataset)} samples across {len(loader)} batches/epoch.")

    # Save normalization statistics
    stats_file = save_path / "vision_stats.json"
    dataset.save_stats(str(stats_file))
    print(f"Vision normalization statistics saved to: {stats_file}")

    # 2. Instantiate Policy Network and CFM matcher
    model = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=horizon,
        proprio_dim=9,
        num_keypoints=num_keypoints,
        vision_feat_dim=vision_feat_dim,
        proprio_feat_dim=proprio_feat_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        cameras=cameras,
        use_cross_attention=use_cross_attention,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Vision policy initialized with {total_params:,} trainable parameters.")

    shifter = RandomShifter(max_shift=shift_aug) if shift_aug > 0 else None
    if shifter is not None:
        print(f"Visual data augmentation: RandomShifter(max_shift={shift_aug}) active.")

    cfm = ConditionalFlowMatcher(sigma_min=1e-4, gripper_weight=gripper_weight)

    # 3. Optimizer and Learning Rate Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. Training Loop
    best_loss = float("inf")
    t_start = time.time()

    print("\nStarting Vision Flow Matching training loop:")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_v_norm = 0.0
        epoch_u_norm = 0.0
        batches = 0

        for batch_obs, batch_actions in loader:
            # Move observation tensors to device
            dev_obs = {k: v.to(device) for k, v in batch_obs.items()}
            if shifter is not None:
                for cam in cameras:
                    cam_key = f"rgb_{cam}"
                    dev_obs[cam_key] = shifter(dev_obs[cam_key])
            dev_actions = batch_actions.to(device)

            optimizer.zero_grad()
            loss, metrics = cfm.compute_loss(model, dev_actions, dev_obs)
            loss.backward()

            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += metrics["loss"]
            epoch_v_norm += metrics["v_norm"]
            epoch_u_norm += metrics["u_norm"]
            batches += 1

        scheduler.step()

        mean_loss = epoch_loss / batches
        mean_v_norm = epoch_v_norm / batches
        mean_u_norm = epoch_u_norm / batches
        current_lr = scheduler.get_last_lr()[0]

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - t_start
            print(
                f"Epoch [{epoch:3d}/{epochs:3d}] | "
                f"Loss: {mean_loss:.5f} | "
                f"|v|: {mean_v_norm:.3f} | "
                f"|u|: {mean_u_norm:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        # Checkpoint best policy
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
                        "act_dim": 8,
                        "horizon": horizon,
                        "proprio_dim": 9,
                        "num_keypoints": num_keypoints,
                        "vision_feat_dim": vision_feat_dim,
                        "proprio_feat_dim": proprio_feat_dim,
                        "hidden_dim": hidden_dim,
                        "num_blocks": num_blocks,
                        "gripper_weight": gripper_weight,
                        "cameras": list(cameras),
                        "use_cross_attention": use_cross_attention,
                    },
                    "stats": dataset.stats,
                },
                ckpt_path,
            )

    print(f"\nVision training complete in {time.time() - t_start:.1f}s. Best Loss: {best_loss:.5f}")
    print(f"Best model saved to: {save_path / 'best_vision_policy.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Vision Flow Matching Action Chunker Policy")
    parser.add_argument("--data", type=str, default="data/watering_demos_vision_100.h5", help="Path to HDF5 demos")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--horizon", type=int, default=16, help="Action chunk prediction horizon")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension of ResMLP backbone")
    parser.add_argument("--gripper-weight", type=float, default=2.5, help="Gripper loss dimension weight")
    parser.add_argument("--shift-aug", type=int, default=4, help="Maximum random shift pixels for visual data augmentation")
    parser.add_argument("--use-cross-attention", action="store_true", help="Enable Multi-Camera Multi-Head Cross-Attention fusion")
    parser.add_argument("--cameras", nargs="+", default=None, help="Names of cameras to train with (defaults to auto-detect from dataset)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cpu, cuda, or mps")
    args = parser.parse_args()

    train_vision_policy(
        data_path=args.data,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        gripper_weight=args.gripper_weight,
        shift_aug=args.shift_aug,
        use_cross_attention=args.use_cross_attention,
        device_str=args.device,
        cameras=tuple(args.cameras) if args.cameras is not None else None,
    )


if __name__ == "__main__":
    main()
