"""Closed-Loop Policy Evaluator and Benchmark Harness.

Executes trained Flow Matching action chunker policies in MuJoCo simulation,
benchmarking physical task success rates, pour metrics, and inference latencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from floraflow.common.env import DeskWateringEnv
from floraflow.training.dataset import extract_observation_vector
from floraflow.training.flow_matching import ConditionalFlowMatcher
from floraflow.training.model import FlowMatchingPolicy


@dataclass
class EpisodeResult:
    """Diagnostic outcome metrics for an individual evaluation episode."""

    seed: int
    success: bool
    is_pouring: bool
    particles_in_pot: int
    min_spout_dist: float
    max_tilt_deg: float
    total_steps: int
    mean_inference_latency_ms: float
    can_xy: Tuple[float, float]
    plant_xy: Tuple[float, float]


@dataclass
class BenchmarkScorecard:
    """Summary statistics across evaluation trial episodes."""

    total_episodes: int
    successful_episodes: int
    success_rate_pct: float
    mean_min_spout_dist: float
    mean_max_tilt_deg: float
    mean_particles_in_pot: float
    mean_episode_steps: float
    mean_latency_ms: float
    episode_results: List[EpisodeResult]


class PolicyEvaluator:
    """Evaluates Flow Matching policies in closed-loop MuJoCo simulation."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_policy.pt",
        device_str: str = "auto",
        control_hz: int = 20,
    ) -> None:
        if device_str == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)

        ckpt_file = Path(checkpoint_path).resolve()
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

        ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        config = ckpt["config"]
        self.stats = ckpt["stats"]

        self.obs_mean = torch.tensor(self.stats["obs_mean"], device=self.device, dtype=torch.float32)
        self.obs_std = torch.tensor(self.stats["obs_std"], device=self.device, dtype=torch.float32)
        self.act_mean = self.stats["act_mean"]
        self.act_std = self.stats["act_std"]

        self.horizon = config["horizon"]
        self.act_dim = config["act_dim"]

        self.model = FlowMatchingPolicy(
            obs_dim=config["obs_dim"],
            act_dim=config["act_dim"],
            horizon=config["horizon"],
            hidden_dim=config["hidden_dim"],
            num_blocks=config["num_blocks"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.cfm = ConditionalFlowMatcher(sigma_min=1e-4)
        self.env = DeskWateringEnv(control_hz=control_hz)

    def run_episode(
        self,
        seed: int,
        can_xy: Optional[Tuple[float, float]] = None,
        plant_xy: Optional[Tuple[float, float]] = None,
        max_steps: int = 200,
        exec_horizon: int = 8,
        num_ode_steps: int = 10,
        temporal_ensemble: bool = False,
        ensemble_decay: float = 0.05,
    ) -> EpisodeResult:
        """Execute one complete closed-loop evaluation episode."""
        torch.manual_seed(seed)
        obs = self.env.reset(seed=seed, can_xy=can_xy, plant_xy=plant_xy)
        actual_can_xy = (float(obs["can_pos"][0]), float(obs["can_pos"][1]))
        actual_plant_xy = (float(obs["plant_pos"][0]), float(obs["plant_pos"][1]))

        latencies = []
        min_spout_dist = float("inf")
        max_tilt_deg = 0.0
        success = False
        is_pouring = False
        step_count = 0

        if temporal_ensemble:
            # Temporal Ensembling with exponential decay weighting
            buf_len = max_steps + self.horizon + 1
            action_buffer = np.zeros((buf_len, self.act_dim), dtype=np.float32)
            weight_buffer = np.zeros((buf_len,), dtype=np.float32)
            ensemble_weights = np.exp(-ensemble_decay * np.arange(self.horizon)).astype(np.float32)

            while step_count < max_steps:
                # 1. Feature extraction and normalization
                obs_vec = extract_observation_vector(obs, step_idx=step_count)
                norm_obs = (torch.tensor(obs_vec, device=self.device, dtype=torch.float32) - self.obs_mean) / self.obs_std
                norm_obs_batch = norm_obs.unsqueeze(0)

                # 2. Flow Matching ODE inference
                t_infer_start = time.perf_counter()
                with torch.no_grad():
                    pred_chunk_norm = self.cfm.sample(
                        self.model,
                        norm_obs_batch,
                        horizon=self.horizon,
                        act_dim=self.act_dim,
                        num_steps=num_ode_steps,
                    )
                latency_ms = (time.perf_counter() - t_infer_start) * 1000.0
                latencies.append(latency_ms)

                # 3. Unnormalize predicted action chunk (H, act_dim)
                pred_chunk = pred_chunk_norm.squeeze(0).cpu().numpy()
                pred_chunk = pred_chunk * self.act_std + self.act_mean

                # 4. Accumulate into future buffer slots
                for h in range(self.horizon):
                    idx = step_count + h
                    action_buffer[idx] += ensemble_weights[h] * pred_chunk[h]
                    weight_buffer[idx] += ensemble_weights[h]

                # 5. Execute blended action for current timestep
                blended_act = action_buffer[step_count] / max(weight_buffer[step_count], 1e-6)
                obs, _, _, info = self.env.step(blended_act)
                step_count += 1

                min_spout_dist = min(min_spout_dist, info["spout_dist"])
                max_tilt_deg = max(max_tilt_deg, info["tilt_deg"])
                if info["success"]:
                    success = True
                if info["is_pouring"]:
                    is_pouring = True
        else:
            # Standard sub-horizon open-loop execution
            while step_count < max_steps:
                # 1. Feature extraction and normalization
                obs_vec = extract_observation_vector(obs, step_idx=step_count)
                norm_obs = (torch.tensor(obs_vec, device=self.device, dtype=torch.float32) - self.obs_mean) / self.obs_std
                norm_obs_batch = norm_obs.unsqueeze(0)

                # 2. Flow Matching ODE inference
                t_infer_start = time.perf_counter()
                with torch.no_grad():
                    pred_chunk_norm = self.cfm.sample(
                        self.model,
                        norm_obs_batch,
                        horizon=self.horizon,
                        act_dim=self.act_dim,
                        num_steps=num_ode_steps,
                    )
                latency_ms = (time.perf_counter() - t_infer_start) * 1000.0
                latencies.append(latency_ms)

                # 3. Unnormalize predicted action chunk
                pred_chunk = pred_chunk_norm.squeeze(0).cpu().numpy()
                pred_chunk = pred_chunk * self.act_std + self.act_mean

                # 4. Execute sub-horizon actions in environment
                steps_to_exec = min(exec_horizon, max_steps - step_count)
                for i in range(steps_to_exec):
                    action = pred_chunk[i]
                    obs, _, _, info = self.env.step(action)
                    step_count += 1

                    min_spout_dist = min(min_spout_dist, info["spout_dist"])
                    max_tilt_deg = max(max_tilt_deg, info["tilt_deg"])
                    if info["success"]:
                        success = True
                    if info["is_pouring"]:
                        is_pouring = True

                    if step_count >= max_steps:
                        break

        particles = int(obs["particles_in_pot"][0])
        mean_latency = float(np.mean(latencies)) if latencies else 0.0

        return EpisodeResult(
            seed=seed,
            success=success,
            is_pouring=is_pouring,
            particles_in_pot=particles,
            min_spout_dist=min_spout_dist,
            max_tilt_deg=max_tilt_deg,
            total_steps=step_count,
            mean_inference_latency_ms=mean_latency,
            can_xy=actual_can_xy,
            plant_xy=actual_plant_xy,
        )

    def evaluate_benchmark(
        self,
        seeds: List[int],
        can_xy_list: Optional[List[Tuple[float, float]]] = None,
        plant_xy_list: Optional[List[Tuple[float, float]]] = None,
        max_steps: int = 200,
        exec_horizon: int = 8,
        num_ode_steps: int = 10,
        temporal_ensemble: bool = False,
        ensemble_decay: float = 0.05,
    ) -> BenchmarkScorecard:
        """Run evaluation benchmark across a collection of seeds."""
        results: List[EpisodeResult] = []
        n = len(seeds)

        for i, seed in enumerate(seeds):
            can_xy = can_xy_list[i] if can_xy_list is not None else None
            plant_xy = plant_xy_list[i] if plant_xy_list is not None else None

            res = self.run_episode(
                seed=seed,
                can_xy=can_xy,
                plant_xy=plant_xy,
                max_steps=max_steps,
                exec_horizon=exec_horizon,
                num_ode_steps=num_ode_steps,
                temporal_ensemble=temporal_ensemble,
                ensemble_decay=ensemble_decay,
            )
            results.append(res)

        succ_count = sum(1 for r in results if r.success)
        succ_rate = (succ_count / n) * 100.0 if n > 0 else 0.0

        return BenchmarkScorecard(
            total_episodes=n,
            successful_episodes=succ_count,
            success_rate_pct=succ_rate,
            mean_min_spout_dist=float(np.mean([r.min_spout_dist for r in results])),
            mean_max_tilt_deg=float(np.mean([r.max_tilt_deg for r in results])),
            mean_particles_in_pot=float(np.mean([r.particles_in_pot for r in results])),
            mean_episode_steps=float(np.mean([r.total_steps for r in results])),
            mean_latency_ms=float(np.mean([r.mean_inference_latency_ms for r in results])),
            episode_results=results,
        )


def generate_ood_configurations(
    num_episodes: int,
    base_seed: int = 200,
    difficulty: str = "hard",
) -> Tuple[List[int], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Generate out-of-distribution can and plant configurations outside training bounds.

    Enforces physical spawn clearance via DeskWateringEnv.is_valid_spawn so that
    neither object spawns colliding with the robot's initial home fingers or with each other.
    """
    rng = np.random.default_rng(base_seed)
    seeds = [base_seed + i for i in range(num_episodes)]
    can_xys = []
    plant_xys = []

    for _ in range(num_episodes):
        while True:
            if difficulty == "mild":
                if rng.random() > 0.5:
                    cx = rng.uniform(0.45, 0.47) if rng.random() > 0.5 else rng.uniform(0.57, 0.60)
                    cy = rng.uniform(0.13, 0.21)
                else:
                    cx = rng.uniform(0.49, 0.55)
                    cy = rng.uniform(0.225, 0.24) if rng.random() > 0.5 else rng.uniform(0.23, 0.25)

                if rng.random() > 0.5:
                    px = rng.uniform(0.35, 0.37) if rng.random() > 0.5 else rng.uniform(0.47, 0.50)
                    py = rng.uniform(-0.25, -0.19)
                else:
                    px = rng.uniform(0.39, 0.45)
                    py = rng.uniform(-0.29, -0.27) if rng.random() > 0.5 else rng.uniform(-0.178, -0.170)

            elif difficulty == "hard":
                if rng.random() > 0.5:
                    cx = rng.uniform(0.41, 0.45) if rng.random() > 0.5 else rng.uniform(0.59, 0.64)
                    cy = rng.uniform(0.125, 0.24)
                else:
                    cx = rng.uniform(0.46, 0.58)
                    cy = rng.uniform(0.23, 0.26) if rng.random() > 0.5 else rng.uniform(0.25, 0.31)

                if rng.random() > 0.5:
                    px = rng.uniform(0.31, 0.35) if rng.random() > 0.5 else rng.uniform(0.49, 0.54)
                    py = rng.uniform(-0.27, -0.175)
                else:
                    px = rng.uniform(0.36, 0.48)
                    py = rng.uniform(-0.35, -0.29) if rng.random() > 0.5 else rng.uniform(-0.178, -0.170)

            elif difficulty == "extreme":
                if rng.random() > 0.5:
                    cx = rng.uniform(0.37, 0.41) if rng.random() > 0.5 else rng.uniform(0.64, 0.69)
                    cy = rng.uniform(0.125, 0.25)
                else:
                    cx = rng.uniform(0.44, 0.60)
                    cy = rng.uniform(0.26, 0.30) if rng.random() > 0.5 else rng.uniform(0.30, 0.36)

                if rng.random() > 0.5:
                    px = rng.uniform(0.27, 0.32) if rng.random() > 0.5 else rng.uniform(0.53, 0.58)
                    py = rng.uniform(-0.28, -0.175)
                else:
                    px = rng.uniform(0.34, 0.50)
                    py = rng.uniform(-0.40, -0.32) if rng.random() > 0.5 else rng.uniform(-0.178, -0.170)
            else:
                raise ValueError(f"Unknown difficulty: {difficulty}")

            if DeskWateringEnv.is_valid_spawn((float(cx), float(cy)), (float(px), float(py))):
                break

        can_xys.append((float(cx), float(cy)))
        plant_xys.append((float(px), float(py)))

    return seeds, can_xys, plant_xys


def print_scorecard(title: str, card: BenchmarkScorecard) -> None:
    """Print formatted evaluation scorecard."""
    border = "=" * 68
    sub_border = "-" * 68
    print(f"\n{border}")
    print(f" {title.upper()} SCORECARD")
    print(f"{border}")
    print(f" Total Evaluation Episodes:       {card.total_episodes}")
    print(f" Successful Episodes:             {card.successful_episodes} / {card.total_episodes}")
    print(f" Task Success Rate:               {card.success_rate_pct:.1f}%")
    print(f" Mean Spout Alignment Distance:   {card.mean_min_spout_dist * 100:.1f} cm")
    print(f" Mean Maximum Tilt Angle:         {card.mean_max_tilt_deg:.1f} deg")
    print(f" Mean Particles in Pot:           {card.mean_particles_in_pot:.2f}")
    print(f" Mean Episode Horizon Steps:      {card.mean_episode_steps:.1f} steps")
    print(f" Mean Inference Latency:          {card.mean_latency_ms:.2f} ms")
    print(f"{sub_border}")
    meets_realtime = card.mean_latency_ms < 50.0
    print(f" Real-time 20 Hz Feasibility:     {'PASS (<50ms budget)' if meets_realtime else 'FAIL'}")
    print(f"{border}\n")
