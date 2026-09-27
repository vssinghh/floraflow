"""Unit tests for vision-based Flow Matching policy and spatial softmax layers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from floraflow.data.vision_dataset import VisionWateringDataset
from floraflow.policy.flow_matching import ConditionalFlowMatcher
from floraflow.policy.spatial_softmax import SpatialSoftmax
from floraflow.policy.vision_model import SpatialSoftmaxConvNet, VisionFlowMatchingPolicy


def test_spatial_softmax_output_shape_and_coords() -> None:
    """Test that SpatialSoftmax correctly maps activation peaks to normalized [-1, 1] coordinates."""
    ssm = SpatialSoftmax(height=8, width=8, num_channels=2, temperature=1.0)
    feat = torch.zeros(1, 2, 8, 8)
    feat[0, 0, 0, 0] = 50.0  # Top-left peak
    feat[0, 1, 7, 7] = 50.0  # Bottom-right peak

    coords = ssm(feat)
    assert coords.shape == (1, 4)
    # Channel 0: (-1.0, -1.0)
    assert abs(coords[0, 0].item() - (-1.0)) < 1e-3
    assert abs(coords[0, 1].item() - (-1.0)) < 1e-3
    # Channel 1: (1.0, 1.0)
    assert abs(coords[0, 2].item() - 1.0) < 1e-3
    assert abs(coords[0, 3].item() - 1.0) < 1e-3


def test_spatial_softmax_convnet_forward() -> None:
    """Test that 4-layer CNN with Spatial Softmax produces expected feature embedding."""
    net = SpatialSoftmaxConvNet(in_channels=3, num_keypoints=16, feature_dim=32)
    img = torch.rand(2, 3, 128, 128)
    out = net(img)
    assert out.shape == (2, 32)


def test_vision_flow_matching_policy_forward_and_sample() -> None:
    """Test forward vector field prediction and Euler ODE integration."""
    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=16,
        proprio_dim=9,
        num_keypoints=16,
        vision_feat_dim=32,
        proprio_feat_dim=32,
        hidden_dim=128,
        num_blocks=2,
    )

    b = 2
    obs = {
        "rgb_third_person_cam": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8),
        "rgb_overhead_cam": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8),
        "proprio": torch.randn(b, 9),
    }
    x_t = torch.randn(b, 16, 8)
    t = torch.rand(b)

    v_pred = policy(x_t, t, obs)
    assert v_pred.shape == (b, 16, 8)

    cfm = ConditionalFlowMatcher(gripper_weight=2.5)
    loss, metrics = cfm.compute_loss(policy, x_t, obs)
    assert loss.item() >= 0.0

    samples = cfm.sample(policy, obs, horizon=16, act_dim=8, num_steps=3)
    assert samples.shape == (b, 16, 8)


def test_vision_dataset_loading() -> None:
    """Test VisionWateringDataset loading from a mock HDF5 archive."""
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        with h5py.File(tmp.name, "w") as f:
            grp = f.create_group("data/demo_0")
            obs_grp = grp.create_group("obs")
            obs_grp.create_dataset("rgb_third_person_cam", data=np.zeros((10, 128, 128, 3), dtype=np.uint8))
            obs_grp.create_dataset("rgb_overhead_cam", data=np.zeros((10, 128, 128, 3), dtype=np.uint8))
            obs_grp.create_dataset("arm_qpos", data=np.zeros((10, 7), dtype=np.float32))
            obs_grp.create_dataset("gripper_width", data=np.zeros((10, 1), dtype=np.float32))
            grp.create_dataset("actions", data=np.zeros((10, 8), dtype=np.float32))

        dataset = VisionWateringDataset(h5_path=tmp.name, horizon=4)
        assert len(dataset) == 10
        item_obs, item_act = dataset[0]
        assert item_obs["rgb_third_person_cam"].shape == (3, 128, 128)
        assert item_obs["rgb_overhead_cam"].shape == (3, 128, 128)
        assert item_obs["proprio"].shape == (9,)
        assert item_act.shape == (4, 8)


def test_random_shifter_preserves_shape() -> None:
    """Test that RandomShifter preserves tensor dimensions and shifts pixels."""
    from floraflow.data.augmentation import RandomShifter
    shifter = RandomShifter(max_shift=4)
    x = torch.rand(4, 3, 128, 128)
    out = shifter(x)
    assert out.shape == (4, 3, 128, 128)
    # Zero shift should return exact tensor
    no_shift = RandomShifter(max_shift=0)
    assert torch.allclose(no_shift(x), x)


def test_tri_camera_vision_policy_and_env() -> None:
    """Test policy and environment with 3 cameras including wrist_cam."""
    from floraflow.env.desk_env import DeskWateringEnv
    cameras = ("third_person_cam", "overhead_cam", "wrist_cam")

    # Verify environment produces all three camera views
    env = DeskWateringEnv(include_rgb=True, rgb_cameras=cameras, rgb_resolution=(64, 64))
    obs = env.reset(seed=42)
    for cam in cameras:
        assert f"rgb_{cam}" in obs
        assert obs[f"rgb_{cam}"].shape == (64, 64, 3)

    # Verify policy creates 3 encoders and runs ODE integration
    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=8,
        proprio_dim=9,
        num_keypoints=16,
        vision_feat_dim=32,
        proprio_feat_dim=32,
        hidden_dim=128,
        num_blocks=2,
        cameras=cameras,
    )
    assert len(policy.encoders) == 3
    assert policy.fused_dim == 3 * 32 + 32  # 3 cameras * 32 + proprio 32 = 128

    b = 2
    batch_obs = {
        f"rgb_{cam}": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8)
        for cam in cameras
    }
    batch_obs["proprio"] = torch.randn(b, 9)

    cfm = ConditionalFlowMatcher()
    actions = cfm.sample(policy, batch_obs, horizon=8, act_dim=8, num_steps=2)
    assert actions.shape == (b, 8, 8)


def test_multi_camera_cross_attention_policy() -> None:
    """Test policy with Multi-Head Cross-Attention across 3 camera streams."""
    cameras = ("third_person_cam", "overhead_cam", "wrist_cam")
    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=8,
        proprio_dim=9,
        num_keypoints=16,
        vision_feat_dim=32,
        proprio_feat_dim=32,
        hidden_dim=128,
        num_blocks=2,
        cameras=cameras,
        use_cross_attention=True,
        attn_heads=4,
    )

    assert policy.use_cross_attention is True
    assert policy.cross_attn is not None
    assert policy.fused_dim == 32 + 3 * 32 + 32  # context (32) + all_vis (96) + proprio (32) = 160

    b = 2
    batch_obs = {
        f"rgb_{cam}": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8)
        for cam in cameras
    }
    batch_obs["proprio"] = torch.randn(b, 9)
    x_t = torch.randn(b, 8, 8)
    t = torch.rand(b)

    # Forward velocity field prediction
    v_pred = policy(x_t, t, batch_obs)
    assert v_pred.shape == (b, 8, 8)

    # Extract attention weights
    weights = policy.get_attention_weights(batch_obs)
    assert weights is not None
    assert weights.shape == (b, 1, 3)
    # Weights should sum to 1.0 across the 3 cameras
    assert torch.allclose(weights.sum(dim=-1), torch.ones(b, 1), atol=1e-4)

    # Test Euler ODE sampling
    cfm = ConditionalFlowMatcher()
    actions = cfm.sample(policy, batch_obs, horizon=8, act_dim=8, num_steps=2)
    assert actions.shape == (b, 8, 8)


def test_camera_dropout_and_modality_masking() -> None:
    """Test that camera dropout masks wrist_cam during training and turns off during evaluation."""
    cameras = ("third_person_cam", "overhead_cam", "wrist_cam")
    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=8,
        proprio_dim=9,
        num_keypoints=16,
        vision_feat_dim=32,
        proprio_feat_dim=32,
        hidden_dim=128,
        num_blocks=2,
        cameras=cameras,
        use_cross_attention=True,
        attn_heads=4,
        camera_dropout=1.0,  # 100% dropout of wrist_cam during training
        dropout_cameras=("wrist_cam",),
    )

    b = 4
    batch_obs = {
        f"rgb_{cam}": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8)
        for cam in cameras
    }
    batch_obs["proprio"] = torch.randn(b, 9)

    # 1. Training mode with 100% dropout on wrist_cam
    policy.train()
    fused_train = policy.extract_obs_features(batch_obs)
    # Check that fused features have expected total dimension:
    # context (32) + 3 * 32 + proprio (32) = 160
    assert fused_train.shape == (b, 160)

    # In all_vis (offset 32 to 32+96), wrist_cam is the third camera (indices 32+64 to 32+96 = 96 to 128)
    wrist_slice = fused_train[:, 96:128]
    assert torch.all(wrist_slice == 0.0)

    # Non-dropped cameras (third_person_cam and overhead_cam) should NOT be all zero
    third_person_slice = fused_train[:, 32:64]
    overhead_slice = fused_train[:, 64:96]
    assert not torch.all(third_person_slice == 0.0)
    assert not torch.all(overhead_slice == 0.0)

    # Forward pass in training mode
    x_t = torch.randn(b, 8, 8)
    t = torch.rand(b)
    v_pred = policy(x_t, t, batch_obs)
    assert v_pred.shape == (b, 8, 8)

    # 2. Evaluation mode: camera dropout must be deactivated
    policy.eval()
    fused_eval = policy.extract_obs_features(batch_obs)
    wrist_slice_eval = fused_eval[:, 96:128]
    # During eval, wrist_cam is active so it must not be all zeros
    assert not torch.all(wrist_slice_eval == 0.0)

    # Check attention weights during eval: wrist_cam has non-zero attention
    weights_eval = policy.get_attention_weights(batch_obs)
    assert weights_eval is not None
    assert torch.all(weights_eval[:, :, 2] > 0.0)


def test_auxiliary_3d_pose_supervision() -> None:
    """Test 1-layer 3D Ruler Quiz auxiliary pose supervision during training and pixel-only inference."""
    cameras = ("third_person_cam", "overhead_cam", "wrist_cam")
    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=8,
        proprio_dim=9,
        num_keypoints=16,
        vision_feat_dim=32,
        proprio_feat_dim=32,
        hidden_dim=128,
        num_blocks=2,
        cameras=cameras,
        use_cross_attention=True,
        attn_heads=4,
        use_aux_pose=True,
        aux_pose_dim=9,
    )

    # context (32) + all_vis (96) + pred_pose (9) + proprio (32) = 169
    assert policy.fused_dim == 169

    b = 4
    batch_obs = {
        f"rgb_{cam}": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8)
        for cam in cameras
    }
    batch_obs["proprio"] = torch.randn(b, 9)
    target_pose = torch.randn(b, 9)

    policy.train()
    cfm = ConditionalFlowMatcher()
    x1 = torch.randn(b, 8, 8)
    cfm_loss, _ = cfm.compute_loss(policy, x1, batch_obs)
    pose_loss = policy.compute_aux_pose_loss(target_pose)
    total_loss = cfm_loss + 0.5 * pose_loss
    total_loss.backward()
    assert pose_loss.item() > 0.0

    # Verify inference works strictly from raw images + proprio without aux_pose
    policy.eval()
    sampled = cfm.sample(policy, batch_obs, horizon=8, act_dim=8, num_steps=2)
    assert sampled.shape == (b, 8, 8)


def test_bottleneck_and_keypoint_noise() -> None:
    """Test compressed 3-camera bottleneck (num_keypoints=16, vision_feat_dim=32, fused_dim=192) and keypoint jitter."""
    cameras = ("third_person_cam", "overhead_cam", "wrist_cam")
    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=16,
        proprio_dim=9,
        num_keypoints=16,
        vision_feat_dim=32,
        proprio_feat_dim=64,
        hidden_dim=256,
        num_blocks=4,
        dropout=0.1,
        keypoint_noise=0.05,
        cameras=cameras,
        use_cross_attention=True,
        attn_heads=4,
    )

    # context (32) + all_vis (3 * 32 = 96) + proprio (64) = 192 (matches Run 7 bottleneck width)
    assert policy.fused_dim == 192
    assert policy.cross_attn is not None
    assert policy.cross_attn.query_proj is not None

    b = 2
    batch_obs = {
        f"rgb_{cam}": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8)
        for cam in cameras
    }
    batch_obs["proprio"] = torch.randn(b, 9)

    # In train mode with keypoint_noise > 0, two forward passes on identical images yield jittered features
    policy.train()
    feat1 = policy.extract_obs_features(batch_obs)
    feat2 = policy.extract_obs_features(batch_obs)
    assert feat1.shape == (b, 192)
    assert not torch.allclose(feat1, feat2)

    # In eval mode, keypoint jitter and dropout are disabled (deterministic output)
    policy.eval()
    eval1 = policy.extract_obs_features(batch_obs)
    eval2 = policy.extract_obs_features(batch_obs)
    assert torch.allclose(eval1, eval2)


def test_clock_free_8d_proprioception_dataset_and_policy() -> None:
    """Test that use_progress=False yields pure 8D physical proprioception without synthetic step clock."""
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        with h5py.File(tmp.name, "w") as f:
            grp = f.create_group("data/demo_0")
            obs_grp = grp.create_group("obs")
            obs_grp.create_dataset("rgb_third_person_cam", data=np.zeros((10, 128, 128, 3), dtype=np.uint8))
            obs_grp.create_dataset("rgb_overhead_cam", data=np.zeros((10, 128, 128, 3), dtype=np.uint8))
            obs_grp.create_dataset("arm_qpos", data=np.ones((10, 7), dtype=np.float32) * 0.5)
            obs_grp.create_dataset("gripper_width", data=np.ones((10, 1), dtype=np.float32) * 0.08)
            grp.create_dataset("actions", data=np.zeros((10, 8), dtype=np.float32))

        dataset = VisionWateringDataset(h5_path=tmp.name, horizon=4, use_progress=False)
        assert dataset.proprio_dim == 8
        assert dataset.use_progress is False
        assert dataset.stats["proprio_mean"].shape == (8,)
        assert dataset.stats["proprio_std"].shape == (8,)

        obs_0, act_0 = dataset[0]
        obs_9, _ = dataset[9]
        assert obs_0["proprio"].shape == (8,)
        assert act_0.shape == (4, 8)
        # Since arm_qpos and gripper_width are identical at t=0 and t=9, clock-free proprio must be identical
        assert torch.allclose(obs_0["proprio"], obs_9["proprio"])

        # Also verify 4-tap causal proprioception history (lags=(0, 4, 8, 12) -> 32D)
        dataset_hist = VisionWateringDataset(
            h5_path=tmp.name,
            horizon=4,
            use_progress=False,
            proprio_history_lags=(0, 4, 8, 12),
        )
        assert dataset_hist.proprio_dim == 32
        obs_h0, _ = dataset_hist[0]
        assert obs_h0["proprio"].shape == (32,)

        # Verify trim_stationary=True drops the 9 stationary frames (t=1..9) and keeps t=0
        dataset_trim = VisionWateringDataset(
            h5_path=tmp.name,
            horizon=4,
            use_progress=False,
            trim_stationary=True,
        )
        assert len(dataset_trim) == 1
        assert dataset_trim.proprio_dim == 8

    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=4,
        proprio_dim=8,
        num_keypoints=16,
        vision_feat_dim=32,
        proprio_feat_dim=64,
        hidden_dim=128,
        num_blocks=2,
        cameras=("third_person_cam", "overhead_cam"),
        use_cross_attention=True,
    )
    b = 2
    batch_obs = {
        "rgb_third_person_cam": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8),
        "rgb_overhead_cam": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8),
        "proprio": torch.randn(b, 8),
    }
    cfm = ConditionalFlowMatcher()
    sampled = cfm.sample(policy, batch_obs, horizon=4, act_dim=8, num_steps=2)
    assert sampled.shape == (b, 4, 8)


def test_vision_dataset_joint_delta_roundtrip() -> None:
    """Test that action_space='joint_delta' computes horizon-wise stats and reconstructs original motor targets."""
    rng = np.random.default_rng(42)
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        qpos = rng.normal(0.0, 0.5, size=(12, 7)).astype(np.float32)
        grip = np.ones((12, 1), dtype=np.float32) * 0.04
        actions = np.concatenate([qpos + rng.normal(0.0, 0.05, size=(12, 7)).astype(np.float32), np.ones((12, 1), dtype=np.float32)], axis=-1)
        with h5py.File(tmp.name, "w") as f:
            grp = f.create_group("data/demo_0")
            obs_grp = grp.create_group("obs")
            obs_grp.create_dataset("rgb_third_person_cam", data=np.zeros((12, 128, 128, 3), dtype=np.uint8))
            obs_grp.create_dataset("rgb_overhead_cam", data=np.zeros((12, 128, 128, 3), dtype=np.uint8))
            obs_grp.create_dataset("arm_qpos", data=qpos)
            obs_grp.create_dataset("gripper_width", data=grip)
            grp.create_dataset("actions", data=actions)

        ds_abs = VisionWateringDataset(h5_path=tmp.name, horizon=4, use_progress=False, action_space="joint_abs")
        ds_delta = VisionWateringDataset(h5_path=tmp.name, horizon=4, use_progress=False, action_space="joint_delta")

        assert ds_delta.stats["act_mean"].shape == (4, 8)
        assert ds_delta.stats["act_std"].shape == (4, 8)

        for t in range(8):
            _, norm_abs = ds_abs[t]
            _, norm_delta = ds_delta[t]
            raw_abs = ds_abs.unnormalize_action(norm_abs.numpy())
            raw_delta = ds_delta.unnormalize_action(norm_delta.numpy())

            reconstructed = raw_delta.copy()
            reconstructed[:, :7] = qpos[t : t + 1, :7] + raw_delta[:, :7]
            assert np.allclose(reconstructed, raw_abs, atol=1e-5)


def test_vision_dataset_eef_se3_and_rot6d() -> None:
    """Test that action_space='eef_se3' yields 10D task-space proprioception and 10D SE(3) action chunks."""
    from floraflow.expert.ik_solver import rot6d_to_rotmat, rotmat_to_rot6d

    # 1. Verify rotmat -> rot6d -> rotmat round-trip on a 50-deg Y-pitch rotation
    theta = np.radians(-50.0)
    c, s = np.cos(theta), np.sin(theta)
    r_orig = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    r6 = rotmat_to_rot6d(r_orig)
    assert r6.shape == (6,)
    r_rec = rot6d_to_rotmat(r6)
    assert np.allclose(r_rec, r_orig, atol=1e-6)
    assert np.allclose(r_rec.T @ r_rec, np.eye(3), atol=1e-6)

    # 2. Verify VisionWateringDataset with action_space="eef_se3"
    rng = np.random.default_rng(7)
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        qpos = rng.normal(0.0, 0.3, size=(10, 7)).astype(np.float32)
        grip = np.ones((10, 1), dtype=np.float32) * 0.04
        actions = np.concatenate([qpos, -np.ones((10, 1), dtype=np.float32)], axis=-1)
        with h5py.File(tmp.name, "w") as f:
            grp = f.create_group("data/demo_0")
            obs_grp = grp.create_group("obs")
            obs_grp.create_dataset("rgb_third_person_cam", data=np.zeros((10, 128, 128, 3), dtype=np.uint8))
            obs_grp.create_dataset("rgb_overhead_cam", data=np.zeros((10, 128, 128, 3), dtype=np.uint8))
            obs_grp.create_dataset("arm_qpos", data=qpos)
            obs_grp.create_dataset("gripper_width", data=grip)
            grp.create_dataset("actions", data=actions)

        ds_se3 = VisionWateringDataset(h5_path=tmp.name, horizon=4, use_progress=False, action_space="eef_se3")
        assert ds_se3.proprio_dim == 10
        assert ds_se3.act_dim == 10
        obs_0, act_0 = ds_se3[0]
        assert obs_0["proprio"].shape == (10,)
        assert act_0.shape == (4, 10)


