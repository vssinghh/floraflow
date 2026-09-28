"""Cleanroom Flow Matching Action Chunker Policy Architecture.

Implements an MLP ResNet vector field predictor parameterized over continuous
diffusion time t, conditioning observation features, and noisy action chunks.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for continuous time scalar t in [0, 1]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embedding for input tensor of shape (B,)."""
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class ResMlpBlock(nn.Module):
    """Residual MLP block with LayerNorm, SiLU non-linearity, and skip connection."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FlowMatchingPolicy(nn.Module):
    """Continuous Flow Matching vector field policy network.

    Maps noisy action chunk x_t, continuous time t, and observation c
    to predicted velocity v_theta(x_t, t, c) along the optimal transport path.
    """

    def __init__(
        self,
        obs_dim: int = 52,
        act_dim: int = 8,
        horizon: int = 16,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self.chunk_dim = horizon * act_dim

        # Time encoder
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Observation projection
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Action projection
        self.act_proj = nn.Linear(self.chunk_dim, hidden_dim)

        # Fusion input projection
        fusion_dim = hidden_dim + hidden_dim + time_dim
        self.in_proj = nn.Linear(fusion_dim, hidden_dim)

        # Residual MLP backbone
        self.blocks = nn.ModuleList([
            ResMlpBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])

        # Output projection back to action chunk velocity
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.chunk_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass predicting vector field velocity.

        Args:
            x_t: Noisy action chunk of shape (B, H, act_dim) or (B, H * act_dim).
            t: Continuous time scalar of shape (B,).
            obs: Observation feature tensor of shape (B, obs_dim).

        Returns:
            v_pred: Predicted velocity vector field of shape (B, H, act_dim).
        """
        b = x_t.shape[0]
        if x_t.ndim == 3:
            x_flat = x_t.reshape(b, self.chunk_dim)
        else:
            x_flat = x_t

        t_emb = self.time_encoder(t)
        obs_feat = self.obs_proj(obs)
        act_feat = self.act_proj(x_flat)

        fused = torch.cat([act_feat, obs_feat, t_emb], dim=-1)
        h = self.in_proj(fused)

        for block in self.blocks:
            h = block(h)

        v_flat = self.out_head(h)
        return v_flat.reshape(b, self.horizon, self.act_dim)
