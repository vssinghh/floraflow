"""Cleanroom Conditional Flow Matching (CFM) vector field head.

Implements optimal transport path generation, target velocity calculation,
and an explicit Euler ODE numerical integrator for real-time inference.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import torch
import torch.nn as nn


class ConditionalFlowMatcher:
    """Optimal Transport Conditional Flow Matcher for action chunk generation."""

    def __init__(self, sigma_min: float = 1e-4, gripper_weight: float = 2.0) -> None:
        self.sigma_min = sigma_min
        self.gripper_weight = gripper_weight

    def compute_loss(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        async_metrics: bool = False,
        num_flow_samples: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute CFM vector field regression loss.

        Args:
            model: Policy neural network predicting vector field velocity.
            x1: Ground truth target action chunk (B, H, act_dim).
            obs: Observation feature conditioning tensor or dict of tensors.
            async_metrics: If True, returns detached GPU scalar tensors in metrics
                instead of calling .item() mid-batch, avoiding GPU/TPU pipeline stalls.
            num_flow_samples: Number of stratified flow matching (t, x0) samples K evaluated
                per observation feature extraction pass. When K > 1, visual features are
                extracted once (B, D_obs) and amortized across K stratified time intervals.

        Returns:
            loss: Scalar MSE tensor.
            metrics: Dictionary of diagnostic metric values (floats or detached scalar tensors).
        """
        b = x1.shape[0]
        device = x1.device
        k = max(1, int(num_flow_samples))

        if k > 1:
            obs_cond = model.extract_obs_features(obs) if hasattr(model, "extract_obs_features") else obs
            if isinstance(obs_cond, torch.Tensor):
                obs_input: Union[torch.Tensor, Dict[str, torch.Tensor]] = obs_cond.repeat_interleave(k, dim=0)
            else:
                obs_input = {key: val.repeat_interleave(k, dim=0) for key, val in obs_cond.items()}
            x1_target = x1.repeat_interleave(k, dim=0)
            strata = torch.arange(k, device=device, dtype=torch.float32).unsqueeze(0)
            u = torch.rand(b, k, device=device, dtype=torch.float32)
            t = ((strata + u) / float(k)).reshape(b * k)
        else:
            obs_input = obs
            x1_target = x1
            t = torch.rand(b, device=device, dtype=torch.float32)

        bk = x1_target.shape[0]

        # Sample source Gaussian noise x0
        x0 = torch.randn_like(x1_target)

        # Optimal transport conditional probability path
        # x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
        t_expanded = t.view(bk, 1, 1)
        x_t = (1.0 - (1.0 - self.sigma_min) * t_expanded) * x0 + t_expanded * x1_target

        # Analytical conditional target velocity along straight-line path
        # u_t = d(x_t)/dt = x1 - (1 - sigma_min) * x0
        u_t = x1_target - (1.0 - self.sigma_min) * x0

        # Predict vector field velocity
        v_pred = model(x_t, t, obs_input)

        # Weighted mean squared error (arm pose: 1.0, gripper final dim: gripper_weight)
        sq_err = (v_pred.float() - u_t.float()) ** 2
        weights = torch.ones(x1.shape[-1], device=device, dtype=torch.float32)
        if weights.shape[0] >= 8:
            weights[-1] = self.gripper_weight
        loss = (sq_err * weights).mean()

        loss_det = loss.detach()
        v_norm_det = v_pred.detach().float().norm(dim=-1).mean()
        u_norm_det = u_t.detach().float().norm(dim=-1).mean()

        if async_metrics:
            metrics: Dict[str, Any] = {
                "loss": loss_det,
                "v_norm": v_norm_det,
                "u_norm": u_norm_det,
            }
        else:
            metrics = {
                "loss": float(loss_det.item()),
                "v_norm": float(v_norm_det.item()),
                "u_norm": float(u_norm_det.item()),
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
