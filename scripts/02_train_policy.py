"""CLI Entrypoint: Train State-Based Flow Matching Policy.

Delegates to floraflow.training.trainer.train_policy.
"""

from __future__ import annotations

import argparse

from floraflow.training.trainer import train_policy

__all__ = ["train_policy", "main"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Flow Matching Action Chunker Policy")
    parser.add_argument("--data-path", type=str, default="datasets/watering_demos_100.h5")
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
