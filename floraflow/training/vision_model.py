"""Vision-conditioned Flow Matching Action Chunker Policy Architecture.

Integrates Spatial Softmax convolutional vision backbones with proprioception fusion
and continuous Flow Matching vector field generation for pixel-to-action control.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from floraflow.training.config import VisionTrainConfig
from floraflow.training.model import ResMlpBlock, SinusoidalPosEmb
from floraflow.training.spatial_softmax import SpatialSoftmax


class SpatialSoftmaxConvNet(nn.Module):
    """4-layer convolutional encoder with Spatial Softmax keypoint extraction.

    Transforms an RGB image into compact 2D keypoint coordinate representations,
    preserving exact pixel locations of objects without spatial flattening blowup.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_keypoints: int = 32,
        feature_dim: int = 64,
        temperature: float = 1.0,
        keypoint_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.keypoint_noise = keypoint_noise

        # Conv1: 128x128 -> 64x64
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.norm1 = nn.GroupNorm(4, 32)

        # Conv2: 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, 64)

        # Conv3: 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(8, 64)

        # Conv4: 16x16 -> 8x8 with num_keypoints channels
        self.conv4 = nn.Conv2d(64, num_keypoints, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.GroupNorm(4, num_keypoints)

        self.act = nn.SiLU()

        # Spatial Softmax at 8x8 resolution: produces (B, 2 * num_keypoints)
        self.spatial_softmax = SpatialSoftmax(
            height=8,
            width=8,
            num_channels=num_keypoints,
            temperature=temperature,
        )

        # Feature projection
        self.proj = nn.Sequential(
            nn.Linear(2 * num_keypoints, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )

    def extract_keypoints_with_confidence(
        self,
        x: torch.Tensor,
        viz_temperature: float = 0.08,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract projected visual features along with contrast-sharpened 2D keypoints and confidences.

        Args:
            x: RGB image tensor of shape (B, 3, 128, 128) normalized to [0, 1].
            viz_temperature: Sharp softmax temperature for diagnostic keypoint visualization
                to remove post-GroupNorm uniform background dilution.

        Returns:
            features: Visual feature embedding of shape (B, feature_dim).
            keypoints_2d: Spatial coordinates of shape (B, num_keypoints, 2) in [-1, 1].
            confidence: Channel peak probabilities of shape (B, num_keypoints) in [0, 1].
        """
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        h = self.act(self.norm3(self.conv3(h)))
        h = self.act(self.norm4(self.conv4(h)))

        keypoints = self.spatial_softmax(h)
        viz_kp, confidence = self.spatial_softmax.forward_with_confidence(
            h,
            viz_temperature=viz_temperature,
        )
        keypoints_2d = viz_kp.reshape(x.shape[0], self.num_keypoints, 2)
        if self.training and self.keypoint_noise > 0.0:
            keypoints = keypoints + torch.randn_like(keypoints) * self.keypoint_noise
        features = self.proj(keypoints)
        return features, keypoints_2d, confidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting spatial keypoint features.

        Args:
            x: RGB image tensor of shape (B, 3, 128, 128) normalized to [0, 1].

        Returns:
            features: Visual feature embedding of shape (B, feature_dim).
        """
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        h = self.act(self.norm3(self.conv3(h)))
        h = self.act(self.norm4(self.conv4(h)))

        keypoints = self.spatial_softmax(h)
        if self.training and self.keypoint_noise > 0.0:
            keypoints = keypoints + torch.randn_like(keypoints) * self.keypoint_noise
        features = self.proj(keypoints)
        return features


class MultiCameraCrossAttention(nn.Module):
    """Multi-Head Cross-Attention over multi-camera tokens conditioned on robot state."""

    def __init__(
        self,
        feat_dim: int = 64,
        num_heads: int = 4,
        num_cameras: int = 3,
        query_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.num_cameras = num_cameras

        # Optional query projection when proprioception feature dim differs from visual feat_dim
        if query_dim is not None and query_dim != feat_dim:
            self.query_proj: Optional[nn.Linear] = nn.Linear(query_dim, feat_dim)
        else:
            self.query_proj = None

        # Learnable camera ID positional embeddings
        self.camera_emb = nn.Parameter(torch.randn(num_cameras, feat_dim) * 0.02)

        # Multi-Head Attention layer: Proprioception query attends to camera tokens
        self.mha = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(
        self,
        cam_tokens: torch.Tensor,
        proprio_query: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            cam_tokens: Camera tokens of shape (B, num_cameras, feat_dim).
            proprio_query: Query token of shape (B, 1, query_dim).
            key_padding_mask: Optional boolean tensor of shape (B, num_cameras) indicating
                which camera tokens to ignore during cross-attention (True = ignore).

        Returns:
            context: Attended visual context of shape (B, feat_dim).
            attn_weights: Attention distribution of shape (B, 1, num_cameras).
        """
        if self.query_proj is not None:
            proprio_query = self.query_proj(proprio_query)
        tokens = cam_tokens + self.camera_emb.unsqueeze(0)
        attn_out, attn_weights = self.mha(
            query=proprio_query,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
        )
        context = self.norm(attn_out.squeeze(1))
        return context, attn_weights


class VisionFlowMatchingPolicy(nn.Module):
    """Vision-Language-Action Pixel-to-Action Flow Matching Policy.

    Fuses multi-view Spatial Softmax visual keypoints with 8D joint proprioception
    via Multi-Head Cross-Attention and generates continuous action chunks via Flow Matching.
    """

    def __init__(
        self,
        act_dim: int = 8,
        horizon: int = 16,
        proprio_dim: int = 8,
        num_keypoints: int = 32,
        vision_feat_dim: int = 32,
        proprio_feat_dim: int = 64,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.05,
        keypoint_noise: float = 0.01,
        cameras: Tuple[str, ...] = ("third_person_cam", "overhead_cam", "wrist_cam"),
        attn_heads: int = 4,
        camera_dropout: float = 0.10,
        dropout_cameras: Tuple[str, ...] = ("wrist_cam",),
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.horizon = horizon
        self.chunk_dim = horizon * act_dim
        self.cameras = tuple(cameras)
        self.attn_heads = attn_heads
        self.camera_dropout = camera_dropout
        self.dropout_cameras = tuple(dropout_cameras)
        self.dropout_p = dropout
        self.keypoint_noise = keypoint_noise

        # Visual encoders for each camera viewpoint
        self.encoders = nn.ModuleDict({
            cam: SpatialSoftmaxConvNet(
                in_channels=3,
                num_keypoints=num_keypoints,
                feature_dim=vision_feat_dim,
                keypoint_noise=keypoint_noise,
            )
            for cam in self.cameras
        })

        # Proprioception encoder: maps 8D physical state to feature vector
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, proprio_feat_dim),
            nn.LayerNorm(proprio_feat_dim),
            nn.SiLU(),
            nn.Linear(proprio_feat_dim, proprio_feat_dim),
        )

        # Proprioception-queried Multi-Camera Cross-Attention fusion
        total_vision_dim = len(self.cameras) * vision_feat_dim
        self.cross_attn = MultiCameraCrossAttention(
            feat_dim=vision_feat_dim,
            num_heads=attn_heads,
            num_cameras=len(self.cameras),
            query_dim=proprio_feat_dim,
        )
        self.fused_dim = vision_feat_dim + total_vision_dim + proprio_feat_dim

        # Time encoder
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Action and observation projections
        self.act_proj = nn.Linear(self.chunk_dim, hidden_dim)
        self.obs_proj = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.obs_dropout = nn.Dropout(dropout)

        # Vector field fusion layer
        fusion_dim = hidden_dim + hidden_dim + time_dim
        self.in_proj = nn.Linear(fusion_dim, hidden_dim)

        # ResMLP vector field backbone
        self.blocks = nn.ModuleList([
            ResMlpBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])

        # Output projection head predicting velocity field
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.chunk_dim),
        )

    @classmethod
    def from_config(cls, cfg: VisionTrainConfig) -> VisionFlowMatchingPolicy:
        """Instantiate VisionFlowMatchingPolicy directly from a frozen VisionTrainConfig."""
        return cls(
            act_dim=cfg.act_dim,
            horizon=cfg.horizon,
            proprio_dim=cfg.proprio_dim,
            num_keypoints=cfg.num_keypoints,
            vision_feat_dim=cfg.vision_feat_dim,
            proprio_feat_dim=cfg.proprio_feat_dim,
            hidden_dim=cfg.hidden_dim,
            num_blocks=cfg.num_blocks,
            dropout=cfg.dropout,
            keypoint_noise=cfg.keypoint_noise,
            cameras=cfg.cameras,
            attn_heads=cfg.attn_heads,
            camera_dropout=cfg.camera_dropout,
            dropout_cameras=cfg.dropout_cameras,
        )

    def extract_obs_features(
        self,
        obs: Union[Dict[str, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Extract and fuse visual and proprioceptive representations.

        Args:
            obs: Dictionary containing camera images and proprioception,
                 or already fused observation tensor.

        Returns:
            fused_features: Fused representation tensor of shape (B, fused_dim).
        """
        if isinstance(obs, torch.Tensor):
            return obs

        vis_features = []
        for cam in self.cameras:
            img = obs[f"rgb_{cam}"]
            if img.ndim == 4 and img.shape[-1] == 3:
                # Convert (B, H, W, C) to (B, C, H, W)
                img = img.permute(0, 3, 1, 2)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            feat = self.encoders[cam](img)
            vis_features.append(feat)

        proprio = obs["proprio"]
        proprio_feat = self.proprio_encoder(proprio)

        padding_mask = None
        if self.training and self.camera_dropout > 0.0:
            b = proprio.shape[0]
            drop_decisions = torch.zeros((b, len(self.cameras)), dtype=torch.bool, device=proprio.device)
            for idx, cam in enumerate(self.cameras):
                if cam in self.dropout_cameras:
                    drop_decisions[:, idx] = torch.rand(b, device=proprio.device) < self.camera_dropout

            # Ensure at least one camera remains active for every sample
            all_dropped = drop_decisions.all(dim=-1)
            if all_dropped.any():
                drop_decisions[all_dropped, 0] = False

            for idx in range(len(self.cameras)):
                keep_factor = (~drop_decisions[:, idx]).unsqueeze(1).float()
                vis_features[idx] = vis_features[idx] * keep_factor

            padding_mask = drop_decisions

        all_vis = torch.cat(vis_features, dim=-1)
        cam_tokens = torch.stack(vis_features, dim=1)
        proprio_query = proprio_feat.unsqueeze(1)
        context, _ = self.cross_attn(cam_tokens, proprio_query, key_padding_mask=padding_mask)
        return torch.cat([context, all_vis, proprio_feat], dim=-1)

    def get_attention_weights(
        self,
        obs: Dict[str, torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract multi-head cross-attention distribution over cameras.

        Args:
            obs: Observation dictionary with camera images and proprioception.
            key_padding_mask: Optional mask of shape (B, num_cameras) to ignore camera views.

        Returns:
            attn_weights: Tensor of shape (B, 1, num_cameras).
        """
        vis_features = []
        for cam in self.cameras:
            img = obs[f"rgb_{cam}"]
            if img.ndim == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            feat = self.encoders[cam](img)
            vis_features.append(feat)

        cam_tokens = torch.stack(vis_features, dim=1)
        proprio = obs["proprio"]
        proprio_feat = self.proprio_encoder(proprio)
        proprio_query = proprio_feat.unsqueeze(1)
        _, attn_weights = self.cross_attn(cam_tokens, proprio_query, key_padding_mask=key_padding_mask)
        return attn_weights

    @torch.no_grad()
    def extract_visual_telemetry(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Extract 2D keypoints, channel confidences, and cross-attention weights for visualization.

        Args:
            obs: Observation dictionary with camera images and proprioception (batch size 1).

        Returns:
            telemetry: Dictionary containing:
                - keypoints: Dict[str, np.ndarray] of shape (num_keypoints, 2) in [-1, 1] per camera.
                - confidences: Dict[str, np.ndarray] of shape (num_keypoints,) in [0, 1] per camera.
                - attn_weights: np.ndarray of shape (num_cameras,) summing to 1.0.
        """
        vis_features = []
        keypoints_dict = {}
        confidences_dict = {}

        for cam in self.cameras:
            img = obs[f"rgb_{cam}"]
            if img.ndim == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            feat, kp_2d, conf = self.encoders[cam].extract_keypoints_with_confidence(img)
            vis_features.append(feat)
            keypoints_dict[cam] = kp_2d[0].detach().cpu().numpy()
            confidences_dict[cam] = conf[0].detach().cpu().numpy()

        cam_tokens = torch.stack(vis_features, dim=1)
        proprio = obs["proprio"]
        proprio_feat = self.proprio_encoder(proprio)
        proprio_query = proprio_feat.unsqueeze(1)
        _, attn_w = self.cross_attn(cam_tokens, proprio_query)
        attn_np = attn_w[0, 0].detach().cpu().numpy()

        return {
            "keypoints": keypoints_dict,
            "confidences": confidences_dict,
            "attn_weights": attn_np,
        }

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        obs: Union[Dict[str, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Predict flow matching velocity field along optimal transport path.

        Args:
            x_t: Noisy action chunk of shape (B, H, act_dim) or (B, H * act_dim).
            t: Continuous diffusion time in [0, 1] of shape (B,).
            obs: Dictionary of visual observations and proprioception,
                 or precomputed fused feature tensor.

        Returns:
            v_pred: Predicted velocity field of shape (B, H, act_dim).
        """
        b = x_t.shape[0]
        if x_t.ndim == 3:
            x_flat = x_t.reshape(b, self.chunk_dim)
        else:
            x_flat = x_t

        obs_feat = self.extract_obs_features(obs)
        obs_proj = self.obs_dropout(self.obs_proj(obs_feat))
        act_proj = self.act_proj(x_flat)
        t_emb = self.time_encoder(t)

        fused = torch.cat([act_proj, obs_proj, t_emb], dim=-1)
        h = self.in_proj(fused)

        for block in self.blocks:
            h = block(h)

        v_flat = self.out_head(h)
        return v_flat.reshape(b, self.horizon, self.act_dim)
