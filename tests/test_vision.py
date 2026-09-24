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


