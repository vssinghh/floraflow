"""Unit tests for Flow Matching Policy and Observation Vectorizer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from floraflow.common.env import DeskWateringEnv
from floraflow.training.dataset import extract_observation_vector
from floraflow.training.flow_matching import ConditionalFlowMatcher
from floraflow.training.model import FlowMatchingPolicy, SinusoidalPosEmb


def test_sinusoidal_positional_embedding() -> None:
    """Verify continuous time embedding dimensionality and properties."""
    time_dim = 64
    emb_module = SinusoidalPosEmb(time_dim)
    t = torch.tensor([0.0, 0.5, 1.0])
    emb = emb_module(t)
    assert emb.shape == (3, time_dim)
    assert not torch.isnan(emb).any()


def test_policy_forward_shape() -> None:
    """Verify FlowMatchingPolicy processes batch inputs and produces matching velocity chunks."""
    b, horizon, act_dim, obs_dim = 4, 16, 8, 52
    policy = FlowMatchingPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        horizon=horizon,
        hidden_dim=128,
        num_blocks=2,
    )

    x_t = torch.randn(b, horizon, act_dim)
    t = torch.rand(b)
    obs = torch.randn(b, obs_dim)

    v_pred = policy(x_t, t, obs)
    assert v_pred.shape == (b, horizon, act_dim)
    assert not torch.isnan(v_pred).any()


def test_conditional_flow_matcher_loss_and_sample() -> None:
    """Verify CFM vector field loss computation and ODE sampling."""
    b, horizon, act_dim, obs_dim = 2, 16, 8, 52
    policy = FlowMatchingPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        horizon=horizon,
        hidden_dim=64,
        num_blocks=2,
    )
    cfm = ConditionalFlowMatcher(sigma_min=1e-4, gripper_weight=2.0)

    x1 = torch.randn(b, horizon, act_dim)
    obs = torch.randn(b, obs_dim)

    loss, metrics = cfm.compute_loss(policy, x1, obs)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert "loss" in metrics
    assert "v_norm" in metrics

    # Test Euler sampling
    sampled_chunk = cfm.sample(policy, obs, horizon=horizon, act_dim=act_dim, num_steps=5)
    assert sampled_chunk.shape == (b, horizon, act_dim)


def test_observation_vectorizer_dimension() -> None:
    """Verify observation vectorizer extracts exactly 52 features."""
    env = DeskWateringEnv()
    obs = env.reset(seed=0)
    vec = extract_observation_vector(obs, step_idx=10)
    assert vec.shape == (52,)
    assert vec.dtype == np.float32
