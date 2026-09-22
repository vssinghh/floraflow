"""Train Cleanroom Flow Matching Action Chunker Policy.

Trains FlowMatchingPolicy on expert demonstration action chunks using Optimal Transport
vector field matching with AdamW and Cosine Annealing learning rate scheduling.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from floraflow.data.dataset import WateringDemonstrationDataset
from floraflow.policy.flow_matching import ConditionalFlowMatcher
from floraflow.policy.model import FlowMatchingPolicy


def train_policy(
    data_path: str = "data/watering_demos_100.h5",
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
    """Train Flow Matching action chunker policy."""
    # Determine compute device
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

    # 1. Load dataset
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

    # Save normalization statistics
    stats_file = save_path / "stats.json"
    dataset.save_stats(str(stats_file))
    print(f"Normalization statistics saved to: {stats_file}")

    # 2. Instantiate Policy Network and CFM matcher
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

    # 3. Optimizer and Learning Rate Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. Training Loop
    best_loss = float("inf")
    t_start = time.time()

    print("\nStarting Flow Matching training loop:")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_v_norm = 0.0
        epoch_u_norm = 0.0
        batches = 0

        for norm_obs, norm_act in loader:
            norm_obs = norm_obs.to(device)
            norm_act = norm_act.to(device)

            optimizer.zero_grad()
            loss, metrics = cfm.compute_loss(model, norm_act, norm_obs)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += metrics["loss"]
            epoch_v_norm += metrics["v_norm"]
            epoch_u_norm += metrics["u_norm"]
            batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(batches, 1)
        avg_v_norm = epoch_v_norm / max(batches, 1)
        avg_u_norm = epoch_u_norm / max(batches, 1)
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

        # Track and save best checkpoint
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

    # Save final checkpoint
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Flow Matching Action Chunker Policy")
    parser.add_argument("--data-path", type=str, default="data/watering_demos_100.h5")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--gripper-weight", type=float, default=2.5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train_policy(
        data_path=args.data_path,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        gripper_weight=args.gripper_weight,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
