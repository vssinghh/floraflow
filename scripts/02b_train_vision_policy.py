"""CLI Entrypoint: Train Multi-Camera Vision Flow Matching Policy.

Delegates to floraflow.training.trainer.train_vision_policy.
"""

from __future__ import annotations

import argparse

from floraflow.training.trainer import train_vision_policy

__all__ = ["train_vision_policy", "main"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Vision Flow Matching Action Chunker Policy")
    parser.add_argument("--data", type=str, default="datasets/watering_demos_vision_3cam_300.h5", help="Path to HDF5 demos")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--horizon", type=int, default=16, help="Action chunk prediction horizon")
    parser.add_argument("--num-keypoints", type=int, default=32, help="Number of 2D Spatial Softmax keypoints per camera")
    parser.add_argument("--vision-feat-dim", type=int, default=64, help="Visual feature projection dimension per camera")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension of ResMLP backbone")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability in obs_proj and ResMLP blocks")
    parser.add_argument("--keypoint-noise", type=float, default=0.0, help="Gaussian noise std injected into 2D keypoints during training")
    parser.add_argument("--gripper-weight", type=float, default=2.5, help="Gripper loss dimension weight")
    parser.add_argument("--shift-aug", type=int, default=4, help="Maximum random shift pixels for visual data augmentation")
    parser.add_argument("--use-cross-attention", action="store_true", help="Enable Multi-Camera Multi-Head Cross-Attention fusion")
    parser.add_argument("--camera-dropout", type=float, default=0.0, help="Camera dropout probability during training")
    parser.add_argument("--dropout-cameras", nargs="+", default=["wrist_cam"], help="Cameras eligible for dropout")
    parser.add_argument("--use-aux-pose", action="store_true", help="Enable 1-Layer 3D Ruler Quiz auxiliary pose supervision")
    parser.add_argument("--aux-pose-weight", type=float, default=0.5, help="Weight for 3D Ruler Quiz auxiliary loss")
    parser.add_argument("--no-progress", action="store_true", help="Exclude synthetic wall-clock progress from proprioception")
    parser.add_argument(
        "--proprio-history-lags",
        nargs="+",
        type=int,
        default=[0],
        help="Causal step lags for temporal proprioception memory (e.g. 0 4 8 12)",
    )
    parser.add_argument("--trim-stationary", action="store_true", help="Filter out zero-velocity stationary grasp-dwell frames")
    parser.add_argument(
        "--action-space",
        type=str,
        default="joint_abs",
        choices=["joint_abs", "joint_delta", "eef_se3"],
        help="Action chunk parameterization ('joint_abs', 'joint_delta', or 'eef_se3')",
    )
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
        num_keypoints=args.num_keypoints,
        vision_feat_dim=args.vision_feat_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        keypoint_noise=args.keypoint_noise,
        gripper_weight=args.gripper_weight,
        shift_aug=args.shift_aug,
        use_cross_attention=args.use_cross_attention,
        camera_dropout=args.camera_dropout,
        dropout_cameras=tuple(args.dropout_cameras),
        use_aux_pose=args.use_aux_pose,
        aux_pose_weight=args.aux_pose_weight,
        use_progress=not args.no_progress,
        proprio_history_lags=tuple(args.proprio_history_lags),
        trim_stationary=args.trim_stationary,
        action_space=args.action_space,
        device_str=args.device,
        cameras=tuple(args.cameras) if args.cameras is not None else None,
    )


if __name__ == "__main__":
    main()
