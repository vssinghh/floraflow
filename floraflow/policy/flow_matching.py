"""Cleanroom Conditional Flow Matching (CFM) vector field head.

Implements optimal transport path generation, target velocity calculation,
and an explicit Euler ODE numerical integrator for real-time inference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalFlowMatcher:
    """Optimal Transport Conditional Flow Matcher for action chunk generation."""

    def __init__(self, sigma_min: float = 1e-4, gripper_weight: float = 2.0) -> None:
        self.sigma_min = sigma_min
        self.gripper_weight = gripper_weight

    def compute_loss(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute CFM vector field regression loss.

        Args:
            model: Policy neural network predicting vector field velocity.
            x1: Ground truth target action chunk (B, H, act_dim).
            obs: Observation feature conditioning tensor (B, obs_dim).

        Returns:
            loss: Scalar MSE tensor.
            metrics: Dictionary of diagnostic metric values.
        """
        b = x1.shape[0]
        device = x1.device

        # Sample uniform diffusion time t in [0, 1]
        t = torch.rand(b, device=device, dtype=torch.float32)

        # Sample source Gaussian noise x0
        x0 = torch.randn_like(x1)

        # Optimal transport conditional probability path
        # x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
        t_expanded = t.view(b, 1, 1)
        x_t = (1.0 - (1.0 - self.sigma_min) * t_expanded) * x0 + t_expanded * x1

        # Analytical conditional target velocity along straight-line path
        # u_t = d(x_t)/dt = x1 - (1 - sigma_min) * x0
        u_t = x1 - (1.0 - self.sigma_min) * x0

        # Predict vector field velocity
        v_pred = model(x_t, t, obs)

        # Weighted mean squared error (arm pose: 1.0, gripper final dim: gripper_weight)
        sq_err = (v_pred - u_t) ** 2
        weights = torch.ones(x1.shape[-1], device=device, dtype=torch.float32)
        if weights.shape[0] >= 8:
            weights[-1] = self.gripper_weight
        loss = (sq_err * weights).mean()

        metrics = {
            "loss": float(loss.detach().item()),
            "v_norm": float(v_pred.detach().norm(dim=-1).mean().item()),
            "u_norm": float(u_t.detach().norm(dim=-1).mean().item()),
        }
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        horizon: int = 16,
        act_dim: int = 8,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Sample action chunk via Euler ODE integration from source noise to target.

        Args:
            model: Trained policy network.
            obs: Conditioning observation tensor or dictionary of tensors.
            horizon: Action chunk prediction horizon H.
            act_dim: Action dimension D_act.
            num_steps: Discretization steps for numerical ODE integration.

        Returns:
            action_chunk: Generated action chunk of shape (B, H, act_dim).
        """
        model.eval()

        if isinstance(obs, dict):
            first_val = next(iter(obs.values()))
            b = first_val.shape[0]
            device = first_val.device
            # Precompute visual and proprioceptive features once to avoid redundant conv passes
            obs_cond = model.extract_obs_features(obs) if hasattr(model, "extract_obs_features") else obs
        else:
            b = obs.shape[0]
            device = obs.device
            obs_cond = obs

        # Initial source sample x0 ~ N(0, I) at t = 0
        x = torch.randn(b, horizon, act_dim, device=device, dtype=torch.float32)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((b,), t_val, device=device, dtype=torch.float32)
            v = model(x, t, obs_cond)
            x = x + v * dt

        return x
