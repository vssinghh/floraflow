"""Unit tests for vision-based Flow Matching policy, frozen VisionTrainConfig, and spatial softmax layers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from floraflow.training.config import VisionTrainConfig
from floraflow.training.dataset import VisionWateringDataset
from floraflow.training.flow_matching import ConditionalFlowMatcher
from floraflow.training.spatial_softmax import SpatialSoftmax
from floraflow.training.vision_model import SpatialSoftmaxConvNet, VisionFlowMatchingPolicy


def test_spatial_softmax_output_shape_and_coords() -> None:
    """Test that SpatialSoftmax correctly maps activation peaks to normalized [-1, 1] coordinates."""
    ssm = SpatialSoftmax(height=8, width=8, num_channels=2, temperature=1.0)
    feat = torch.zeros(1, 2, 8, 8)
    feat[0, 0, 0, 0] = 50.0  # Top-left peak
    feat[0, 1, 7, 7] = 50.0  # Bottom-right peak

    coords = ssm(feat)
    assert coords.shape == (1, 4)
    assert abs(coords[0, 0].item() - (-1.0)) < 1e-3
    assert abs(coords[0, 1].item() - (-1.0)) < 1e-3
    assert abs(coords[0, 2].item() - 1.0) < 1e-3
    assert abs(coords[0, 3].item() - 1.0) < 1e-3


def test_spatial_softmax_convnet_forward() -> None:
    """Test that 4-layer CNN with Spatial Softmax produces expected feature embedding."""
    net = SpatialSoftmaxConvNet(in_channels=3, num_keypoints=16, feature_dim=32)
    img = torch.rand(2, 3, 128, 128)
    out = net(img)
    assert out.shape == (2, 32)


def test_vision_train_config_frozen_and_validation() -> None:
    """Test that VisionTrainConfig is deeply immutable, rejects unknown keys, and builds VisionFlowMatchingPolicy."""
    cfg = VisionTrainConfig()
    assert cfg.act_dim == 8
    assert cfg.proprio_dim == 8
    assert cfg.vision_feat_dim == 32
    assert cfg.num_keypoints == 32
    assert cfg.num_flow_samples == 4
    assert cfg.epochs == 20
    assert isinstance(cfg.cameras, tuple)

    with pytest.raises(FrozenInstanceError):
        cfg.lr = 1e-3  # type: ignore[misc]

    # Reject unknown / misspelled keys in strict mode
    with pytest.raises(ValueError, match="Unrecognized"):
        VisionTrainConfig.from_dict({"num_keypoint": 16}, strict=True)

    # Ignore retired legacy keys when loading historical checkpoints (strict=False)
    legacy_cfg = VisionTrainConfig.from_dict(
        {"num_keypoints": 16, "use_progress": False, "use_aux_pose": False, "use_cross_attention": True},
        strict=False,
    )
    assert legacy_cfg.num_keypoints == 16
    assert legacy_cfg.diff_from_default() == {"num_keypoints": (32, 16)}

    model = VisionFlowMatchingPolicy.from_config(cfg)
    # context (32) + all_vis (3 * 32 = 96) + proprio (64) = 192
    assert model.fused_dim == 192


def test_vision_flow_matching_policy_forward_and_sample() -> None:
    """Test forward vector field prediction and Euler ODE integration."""
    cameras = ("third_person_cam", "overhead_cam")
    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=16,
        proprio_dim=8,
        num_keypoints=16,
        vision_feat_dim=32,
        proprio_feat_dim=32,
        hidden_dim=128,
        num_blocks=2,
        cameras=cameras,
    )

    b = 2
    obs = {
        "rgb_third_person_cam": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8),
        "rgb_overhead_cam": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8),
        "proprio": torch.randn(b, 8),
    }
    x_t = torch.randn(b, 16, 8)
    t = torch.rand(b)

    v_pred = policy(x_t, t, obs)
    assert v_pred.shape == (b, 16, 8)

    cfm = ConditionalFlowMatcher(gripper_weight=2.5)
    loss, _ = cfm.compute_loss(policy, x_t, obs)
    assert loss.item() >= 0.0

    samples = cfm.sample(policy, obs, horizon=16, act_dim=8, num_steps=3)
    assert samples.shape == (b, 16, 8)


def test_vision_dataset_loading_and_stationary_trimming() -> None:
    """Test VisionWateringDataset loading, 8D proprioception, and automatic stationary frame trimming."""
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        qpos = np.linspace(0.0, 0.5, 10, dtype=np.float32)[:, None] * np.ones((10, 7), dtype=np.float32)
        with h5py.File(tmp.name, "w") as f:
            grp = f.create_group("data/demo_0")
            obs_grp = grp.create_group("obs")
            obs_grp.create_dataset("rgb_third_person_cam", data=np.zeros((10, 128, 128, 3), dtype=np.uint8))
            obs_grp.create_dataset("rgb_overhead_cam", data=np.zeros((10, 128, 128, 3), dtype=np.uint8))
            obs_grp.create_dataset("arm_qpos", data=qpos)
            obs_grp.create_dataset("gripper_width", data=np.ones((10, 1), dtype=np.float32) * 0.08)
            grp.create_dataset("actions", data=np.concatenate([qpos, np.ones((10, 1), dtype=np.float32)], axis=-1))

        dataset = VisionWateringDataset(
            h5_path=tmp.name,
            horizon=4,
            cameras=("third_person_cam", "overhead_cam"),
        )
        assert len(dataset) == 10
        assert dataset.proprio_dim == 8
        assert dataset.act_dim == 8
        item_obs, item_act = dataset[0]
        assert item_obs["rgb_third_person_cam"].shape == (3, 128, 128)
        assert item_obs["rgb_overhead_cam"].shape == (3, 128, 128)
        assert item_obs["proprio"].shape == (8,)
        assert item_act.shape == (4, 8)

    # Verify stationary frames (all identical after t=0) are automatically trimmed to 1 active frame
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp_stat:
        with h5py.File(tmp_stat.name, "w") as f:
            grp = f.create_group("data/demo_0")
            obs_grp = grp.create_group("obs")
            obs_grp.create_dataset("rgb_third_person_cam", data=np.zeros((10, 128, 128, 3), dtype=np.uint8))
            obs_grp.create_dataset("rgb_overhead_cam", data=np.zeros((10, 128, 128, 3), dtype=np.uint8))
            obs_grp.create_dataset("arm_qpos", data=np.ones((10, 7), dtype=np.float32) * 0.5)
            obs_grp.create_dataset("gripper_width", data=np.ones((10, 1), dtype=np.float32) * 0.08)
            grp.create_dataset("actions", data=np.zeros((10, 8), dtype=np.float32))

        ds_trim = VisionWateringDataset(
            h5_path=tmp_stat.name,
            horizon=4,
            cameras=("third_person_cam", "overhead_cam"),
        )
        assert len(ds_trim) == 1


def test_random_shifter_preserves_shape() -> None:
    """Test that RandomShifter preserves tensor dimensions and shifts pixels."""
    from floraflow.training.augmentation import RandomShifter
    shifter = RandomShifter(max_shift=4)
    x = torch.rand(4, 3, 128, 128)
    out = shifter(x)
    assert out.shape == (4, 3, 128, 128)
    no_shift = RandomShifter(max_shift=0)
    assert torch.allclose(no_shift(x), x)


def test_tri_camera_cross_attention_policy_and_env() -> None:
    """Test policy and environment with 3 cameras and Multi-Head Cross-Attention."""
    from floraflow.common.env import DeskWateringEnv
    cameras = ("third_person_cam", "overhead_cam", "wrist_cam")

    env = DeskWateringEnv(include_rgb=True, rgb_cameras=cameras, rgb_resolution=(64, 64))
    obs = env.reset(seed=42)
    for cam in cameras:
        assert f"rgb_{cam}" in obs
        assert obs[f"rgb_{cam}"].shape == (64, 64, 3)

    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=8,
        proprio_dim=8,
        num_keypoints=16,
        vision_feat_dim=32,
        proprio_feat_dim=32,
        hidden_dim=128,
        num_blocks=2,
        cameras=cameras,
        attn_heads=4,
    )
    assert len(policy.encoders) == 3
    assert policy.fused_dim == 32 + 3 * 32 + 32  # context (32) + all_vis (96) + proprio (32) = 160

    b = 2
    batch_obs = {
        f"rgb_{cam}": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8)
        for cam in cameras
    }
    batch_obs["proprio"] = torch.randn(b, 8)

    weights = policy.get_attention_weights(batch_obs)
    assert weights.shape == (b, 1, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(b, 1), atol=1e-4)

    cfm = ConditionalFlowMatcher()
    actions = cfm.sample(policy, batch_obs, horizon=8, act_dim=8, num_steps=2)
    assert actions.shape == (b, 8, 8)


def test_camera_dropout_and_modality_masking() -> None:
    """Test that camera dropout masks wrist_cam during training and turns off during evaluation."""
    cameras = ("third_person_cam", "overhead_cam", "wrist_cam")
    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=8,
        proprio_dim=8,
        num_keypoints=16,
        vision_feat_dim=32,
        proprio_feat_dim=32,
        hidden_dim=128,
        num_blocks=2,
        cameras=cameras,
        attn_heads=4,
        camera_dropout=1.0,
        dropout_cameras=("wrist_cam",),
    )

    b = 4
    batch_obs = {
        f"rgb_{cam}": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8)
        for cam in cameras
    }
    batch_obs["proprio"] = torch.randn(b, 8)

    policy.train()
    fused_train = policy.extract_obs_features(batch_obs)
    assert fused_train.shape == (b, 160)

    wrist_slice = fused_train[:, 96:128]
    assert torch.all(wrist_slice == 0.0)

    policy.eval()
    fused_eval = policy.extract_obs_features(batch_obs)
    wrist_slice_eval = fused_eval[:, 96:128]
    assert not torch.all(wrist_slice_eval == 0.0)


def test_bottleneck_and_keypoint_noise() -> None:
    """Test compressed 3-camera bottleneck (fused_dim=192) and keypoint jitter."""
    cameras = ("third_person_cam", "overhead_cam", "wrist_cam")
    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=16,
        proprio_dim=8,
        num_keypoints=32,
        vision_feat_dim=32,
        proprio_feat_dim=64,
        hidden_dim=256,
        num_blocks=4,
        dropout=0.1,
        keypoint_noise=0.05,
        cameras=cameras,
    )

    assert policy.fused_dim == 192
    assert policy.cross_attn.query_proj is not None

    b = 2
    batch_obs = {
        f"rgb_{cam}": torch.randint(0, 256, (b, 128, 128, 3), dtype=torch.uint8)
        for cam in cameras
    }
    batch_obs["proprio"] = torch.randn(b, 8)

    policy.train()
    feat1 = policy.extract_obs_features(batch_obs)
    feat2 = policy.extract_obs_features(batch_obs)
    assert feat1.shape == (b, 192)
    assert not torch.allclose(feat1, feat2)

    policy.eval()
    eval1 = policy.extract_obs_features(batch_obs)
    eval2 = policy.extract_obs_features(batch_obs)
    assert torch.allclose(eval1, eval2)


def test_modular_architecture_and_gpu_batch_optimizations() -> None:
    """Test 4-pillar package imports, return_uint8_images=True batching, and async_metrics=True."""
    from floraflow.common import DeskWateringEnv, IKSolver
    from floraflow.collection import PourExpertPlanner, generate_vision_demonstrations
    from floraflow.training import (
        ConditionalFlowMatcher,
        VisionFlowMatchingPolicy,
        VisionTrainConfig,
        VisionWateringDataset,
        train_vision_policy,
    )
    from floraflow.evaluation import VisionPolicyEvaluator, VisionRolloutVisualizer

    assert DeskWateringEnv is not None and IKSolver is not None
    assert PourExpertPlanner is not None and callable(generate_vision_demonstrations)
    assert VisionPolicyEvaluator is not None and VisionRolloutVisualizer is not None
    assert VisionTrainConfig is not None and callable(train_vision_policy)

    rng = np.random.default_rng(99)
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        imgs = rng.integers(0, 256, size=(8, 128, 128, 3), dtype=np.uint8)
        qpos = np.linspace(0.0, 0.4, 8, dtype=np.float32)[:, None] + rng.normal(0.0, 0.05, size=(8, 7)).astype(np.float32)
        grip = np.ones((8, 1), dtype=np.float32) * 0.04
        actions = np.concatenate([qpos, np.ones((8, 1), dtype=np.float32)], axis=-1)
        with h5py.File(tmp.name, "w") as f:
            grp = f.create_group("data/demo_0")
            obs_grp = grp.create_group("obs")
            obs_grp.create_dataset("rgb_third_person_cam", data=imgs)
            obs_grp.create_dataset("rgb_overhead_cam", data=imgs)
            obs_grp.create_dataset("arm_qpos", data=qpos)
            obs_grp.create_dataset("gripper_width", data=grip)
            grp.create_dataset("actions", data=actions)

        ds_f32 = VisionWateringDataset(
            h5_path=tmp.name,
            horizon=4,
            cameras=("third_person_cam", "overhead_cam"),
            return_uint8_images=False,
        )
        ds_u8 = VisionWateringDataset(
            h5_path=tmp.name,
            horizon=4,
            cameras=("third_person_cam", "overhead_cam"),
            return_uint8_images=True,
        )

        obs_f32, _ = ds_f32[0]
        obs_u8, act_u8 = ds_u8[0]
        assert obs_u8["rgb_third_person_cam"].dtype == torch.uint8
        assert obs_u8["rgb_third_person_cam"].shape == (128, 128, 3)

        converted = obs_u8["rgb_third_person_cam"].unsqueeze(0).permute(0, 3, 1, 2).float().div_(255.0).squeeze(0)
        assert torch.allclose(converted, obs_f32["rgb_third_person_cam"], atol=1e-6)

        model = VisionFlowMatchingPolicy(
            act_dim=8,
            horizon=4,
            proprio_dim=ds_u8.proprio_dim,
            num_keypoints=8,
            vision_feat_dim=16,
            hidden_dim=32,
            num_blocks=1,
            cameras=("third_person_cam", "overhead_cam"),
        )
        cfm = ConditionalFlowMatcher()
        batch_obs = {
            "rgb_third_person_cam": converted.unsqueeze(0),
            "rgb_overhead_cam": converted.unsqueeze(0),
            "proprio": obs_u8["proprio"].unsqueeze(0),
        }
        loss, metrics = cfm.compute_loss(model, act_u8.unsqueeze(0), batch_obs, async_metrics=True)
        assert isinstance(metrics["loss"], torch.Tensor)
        assert metrics["loss"].ndim == 0
        assert not metrics["loss"].requires_grad


def test_vision_dataset_multi_archive_and_rolling_perm() -> None:
    """Test loading multiple HDF5 archives (Clean + DR interleaved), sample_stride, and K=4 stratified loss."""
    rng = np.random.default_rng(123)
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp1, tempfile.NamedTemporaryFile(suffix=".h5") as tmp2:
        for idx, tmp in enumerate((tmp1, tmp2)):
            imgs = rng.integers(0, 256, size=(10, 128, 128, 3), dtype=np.uint8)
            qpos = (np.ones((10, 7), dtype=np.float32) * (idx + 1) * 0.1) + np.linspace(0, 0.2, 10, dtype=np.float32)[:, None]
            grip = np.ones((10, 1), dtype=np.float32) * 0.04
            actions = np.concatenate([qpos, np.ones((10, 1), dtype=np.float32)], axis=-1)
            with h5py.File(tmp.name, "w") as f:
                grp = f.create_group("data/demo_0")
                obs_grp = grp.create_group("obs")
                obs_grp.create_dataset("rgb_third_person_cam", data=imgs)
                obs_grp.create_dataset("rgb_overhead_cam", data=imgs)
                obs_grp.create_dataset("arm_qpos", data=qpos)
                obs_grp.create_dataset("gripper_width", data=grip)
                grp.create_dataset("actions", data=actions)

        ds_multi = VisionWateringDataset(
            h5_path=[tmp1.name, tmp2.name],
            horizon=4,
            cameras=("third_person_cam", "overhead_cam"),
            sample_stride=2,
        )
        assert len(ds_multi) == 10
        perm = ds_multi.sample_epoch_permutation(window_samples=4)
        assert perm.shape == (10,)
        assert set(perm.tolist()) == set(range(10))

        batch_obs, batch_act = ds_multi.get_batch(perm[:4], torch.device("cpu"))
        assert batch_obs["rgb_third_person_cam"].shape == (4, 3, 128, 128)
        assert batch_obs["rgb_third_person_cam"].is_contiguous()
        assert batch_act.shape == (4, 4, 8)

        model = VisionFlowMatchingPolicy(
            act_dim=8,
            horizon=4,
            proprio_dim=ds_multi.proprio_dim,
            num_keypoints=8,
            vision_feat_dim=16,
            hidden_dim=32,
            num_blocks=1,
            cameras=("third_person_cam", "overhead_cam"),
        )
        cfm = ConditionalFlowMatcher()
        loss_k4, metrics_k4 = cfm.compute_loss(
            model, batch_act, batch_obs, async_metrics=True, num_flow_samples=4
        )
        loss_k4.backward()
        assert loss_k4.item() > 0.0
        assert metrics_k4["loss"].ndim == 0
