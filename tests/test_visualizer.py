"""Unit tests for the 4-Layer Multi-Camera VLA Telemetry & Keypoint Visualizer."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from floraflow.common.env import DeskWateringEnv
from floraflow.evaluation.visualizer import (
    VisionRolloutVisualizer,
    project_3d_to_camera_pixels,
)
from floraflow.training.spatial_softmax import SpatialSoftmax
from floraflow.training.vision_model import VisionFlowMatchingPolicy


def test_spatial_softmax_forward_with_confidence() -> None:
    """Verify forward_with_confidence returns expected shapes, bounds, and matches forward()."""
    layer = SpatialSoftmax(height=8, width=8, num_channels=16, temperature=1.0)
    features = torch.randn(2, 16, 8, 8)

    kp_only = layer(features)
    kp_both, conf = layer.forward_with_confidence(features)

    assert kp_both.shape == (2, 32)
    assert conf.shape == (2, 16)
    assert torch.allclose(kp_only, kp_both, atol=1e-6)
    assert torch.all(kp_both >= -1.0) and torch.all(kp_both <= 1.0)
    assert torch.all(conf >= (1.0 / 64.0) - 1e-5) and torch.all(conf <= 1.0 + 1e-5)


def test_vision_policy_extract_visual_telemetry() -> None:
    """Verify VisionFlowMatchingPolicy.extract_visual_telemetry extracts keypoints and attention."""
    cameras = ("third_person_cam", "overhead_cam", "wrist_cam")
    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=16,
        proprio_dim=9,
        num_keypoints=16,
        vision_feat_dim=32,
        proprio_feat_dim=64,
        hidden_dim=128,
        num_blocks=2,
        cameras=cameras,
        use_cross_attention=True,
    )
    policy.eval()

    obs = {
        "rgb_third_person_cam": torch.rand(1, 3, 128, 128),
        "rgb_overhead_cam": torch.rand(1, 3, 128, 128),
        "rgb_wrist_cam": torch.rand(1, 3, 128, 128),
        "proprio": torch.randn(1, 9),
    }

    telemetry = policy.extract_visual_telemetry(obs)
    assert "keypoints" in telemetry
    assert "confidences" in telemetry
    assert "attn_weights" in telemetry

    for cam in cameras:
        assert telemetry["keypoints"][cam].shape == (16, 2)
        assert telemetry["confidences"][cam].shape == (16,)
        assert np.all(telemetry["keypoints"][cam] >= -1.0)
        assert np.all(telemetry["keypoints"][cam] <= 1.0)

    attn = telemetry["attn_weights"]
    assert attn.shape == (3,)
    assert np.isclose(np.sum(attn), 1.0, atol=1e-4)


def test_project_3d_to_camera_pixels() -> None:
    """Verify 3D world coordinates of tabletop sites project inside camera viewports."""
    env = DeskWateringEnv(include_rgb=False)
    obs = env.reset(seed=100)

    pts_3d = np.stack([obs["grip_pos"], obs["spout_pos"], obs["plant_pos"]], axis=0)
    for cam in ("third_person_cam", "overhead_cam"):
        px_2d, valid = project_3d_to_camera_pixels(
            env.model,
            env.data,
            cam,
            pts_3d,
            width=256,
            height=256,
        )
        assert px_2d.shape == (3, 2)
        assert valid.shape == (3,)
        assert np.all(valid), f"Expected all tabletop sites to project inside {cam}"


def test_visualizer_dashboard_rendering_and_export(tmp_path: Path) -> None:
    """Verify VisionRolloutVisualizer records steps and exports GIF and PNG contact sheets."""
    cameras = ("third_person_cam", "overhead_cam", "wrist_cam")
    env = DeskWateringEnv(include_rgb=True, rgb_cameras=cameras, rgb_resolution=(128, 128))
    policy = VisionFlowMatchingPolicy(
        act_dim=8,
        horizon=16,
        proprio_dim=9,
        num_keypoints=16,
        vision_feat_dim=32,
        proprio_feat_dim=64,
        hidden_dim=64,
        num_blocks=2,
        cameras=cameras,
        use_cross_attention=True,
    )
    policy.eval()

    class _DummyEvaluator:
        def __init__(self) -> None:
            self.env = env
            self.model = policy
            self.cameras = cameras

    viz = VisionRolloutVisualizer(
        evaluator=_DummyEvaluator(),
        cam_render_size=128,
        max_steps=6,
    )
    viz.reset(seed=100)
    cb = viz.make_step_callback()

    obs = env.reset(seed=100)
    for step_i in range(4):
        model_obs = {
            f"rgb_{c}": torch.from_numpy(obs[f"rgb_{c}"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            for c in cameras
        }
        model_obs["proprio"] = torch.zeros(1, 9)
        pred_chunk = np.tile(np.append(obs["arm_qpos"], 1.0), (16, 1)).astype(np.float32)

        cb({
            "step_idx": step_i,
            "obs": obs,
            "model_obs": model_obs,
            "pred_chunk": pred_chunk,
            "exec_action": pred_chunk[0],
            "latency_ms": 5.2,
            "spout_dist": 0.25,
            "tilt_deg": 12.0,
            "particles_in_pot": step_i,
            "is_pouring": False,
            "success": step_i >= 2,
        })

    assert len(viz.rendered_frames) == 4
    gif_file = viz.save_gif(tmp_path / "test_rollout.gif", fps=10, frame_stride=1)
    strip_file = viz.save_summary_strip(tmp_path / "test_strip.png", keyframe_steps=(0, 1, 2, 3))
    assert gif_file.exists() and gif_file.stat().st_size > 0
    assert strip_file.exists() and strip_file.stat().st_size > 0
