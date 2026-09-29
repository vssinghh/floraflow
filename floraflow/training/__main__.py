"""Self-Contained CLI Entrypoint for Policy Training (`python -m floraflow.training`).

Supports both multi-camera Vision Flow Matching training (default) and
low-dimensional state-based Flow Matching training (`--state`).
"""

from __future__ import annotations

import argparse

from floraflow.training.config import VisionTrainConfig
from floraflow.training.trainer import train_policy, train_vision_policy

__all__ = ["train_policy", "train_vision_policy", "main"]


def main() -> None:
    default_cfg = VisionTrainConfig()
    parser = argparse.ArgumentParser(
        prog="python -m floraflow.training",
        description="Train FloraFlow Conditional Flow Matching Policy (vision or state)",
    )
    parser.add_argument(
        "--state",
        action="store_true",
        help="Train low-dimensional state-only policy instead of multi-camera vision policy",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a JSON config file or saved .pt checkpoint to inherit settings from",
    )
    parser.add_argument(
        "--data",
        "--data-path",
        dest="data",
        nargs="+",
        default=None,
        help=f"Path(s) to one or more HDF5 demonstration datasets (default: {list(default_cfg.data_path)})",
    )
    parser.add_argument("--save-dir", type=str, default=None, help=f"Directory to save checkpoints (default: {default_cfg.save_dir})")
    parser.add_argument("--epochs", type=int, default=None, help=f"Number of training epochs (default: {default_cfg.epochs})")
    parser.add_argument("--batch-size", type=int, default=None, help=f"Minibatch size (default: {default_cfg.batch_size})")
    parser.add_argument("--lr", type=float, default=None, help=f"Learning rate (default: {default_cfg.lr})")
    parser.add_argument(
        "--num-flow-samples",
        type=int,
        default=None,
        help=f"Stratified Flow Matching samples amortized per vision pass (default: {default_cfg.num_flow_samples})",
    )
    parser.add_argument("--num-keypoints", type=int, default=None, help=f"2D Spatial Softmax keypoints per camera (default: {default_cfg.num_keypoints})")
    parser.add_argument("--dropout", type=float, default=None, help=f"Dropout probability in obs_proj and ResMLP blocks (default: {default_cfg.dropout})")
    parser.add_argument("--keypoint-noise", type=float, default=None, help=f"Gaussian noise std on 2D keypoints during training (default: {default_cfg.keypoint_noise})")
    parser.add_argument("--camera-dropout", type=float, default=None, help=f"Camera dropout probability during training (default: {default_cfg.camera_dropout})")
    parser.add_argument("--cameras", nargs="+", default=None, help="Names of cameras to train with (defaults to auto-detect from dataset)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cpu, cuda, or mps")
    args = parser.parse_args()

    if args.state:
        data_arg = args.data[0] if args.data else "datasets/watering_demos_100.h5"
        train_policy(
            data_path=str(data_arg),
            save_dir=args.save_dir or "checkpoints",
            epochs=args.epochs or 80,
            batch_size=args.batch_size or 256,
            lr=args.lr or 1e-3,
            device_str=args.device,
        )
    else:
        base_cfg = VisionTrainConfig.load(args.config) if args.config else default_cfg
        train_vision_policy(
            config=base_cfg,
            data_path=list(args.data) if args.data is not None else None,
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_flow_samples=args.num_flow_samples,
            num_keypoints=args.num_keypoints,
            dropout=args.dropout,
            keypoint_noise=args.keypoint_noise,
            camera_dropout=args.camera_dropout,
            cameras=tuple(args.cameras) if args.cameras is not None else None,
            device_str=args.device,
        )


if __name__ == "__main__":
    main()
