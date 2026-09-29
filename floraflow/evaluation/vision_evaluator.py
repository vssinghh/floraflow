"""Closed-Loop Multi-Camera Vision Policy Evaluator.

Executes trained VisionFlowMatchingPolicy directly from raw camera pixels and robot proprioception
in MuJoCo simulation, strictly without simulator state cheat codes (can_pos, plant_pos).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from floraflow.common.env import DeskWateringEnv
from floraflow.evaluation.evaluator import BenchmarkScorecard, EpisodeResult
from floraflow.training.config import VisionTrainConfig
from floraflow.training.flow_matching import ConditionalFlowMatcher
from floraflow.training.vision_model import VisionFlowMatchingPolicy


class VisionPolicyEvaluator:
    """Evaluates Vision-Language-Action Pixel-to-Action policies in closed-loop MuJoCo simulation."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_vision_policy.pt",
        device_str: str = "auto",
        control_hz: int = 20,
        resolution: Tuple[int, int] = (128, 128),
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
            raise FileNotFoundError(f"Vision checkpoint file not found: {ckpt_file}")

        ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        self.cfg = VisionTrainConfig.from_dict(ckpt["config"], strict=False)
        self.stats = ckpt["stats"]

        self.proprio_mean = torch.tensor(self.stats["proprio_mean"], device=self.device, dtype=torch.float32)
        self.proprio_std = torch.tensor(self.stats["proprio_std"], device=self.device, dtype=torch.float32)
        self.act_mean = np.array(self.stats["act_mean"], dtype=np.float32)
        self.act_std = np.array(self.stats["act_std"], dtype=np.float32)

        self.horizon = self.cfg.horizon
        self.act_dim = self.cfg.act_dim
        self.proprio_dim = self.cfg.proprio_dim
        self.cameras = self.cfg.cameras

        self.model = VisionFlowMatchingPolicy.from_config(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.cfm = ConditionalFlowMatcher(sigma_min=1e-4)
        self.resolution = resolution
        self.env = DeskWateringEnv(
            control_hz=control_hz,
            include_rgb=True,
            rgb_cameras=self.cameras,
            rgb_resolution=resolution,
        )

    def prepare_vision_obs(
        self,
        env_obs: Dict[str, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """Convert environment observations into normalized PyTorch model inputs.

        Strictly extracts ONLY raw camera frames and 8D physical proprioception (arm_qpos + gripper_width).
        Does NOT access can_pos, plant_pos, spout_pos, or grip_pos.
        """
        model_obs: Dict[str, torch.Tensor] = {}

        for cam in self.cameras:
            raw_img = env_obs[f"rgb_{cam}"]  # (H, W, 3) uint8
            img_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            model_obs[f"rgb_{cam}"] = img_tensor.to(self.device)

        arm_qpos = env_obs["arm_qpos"]
        gripper_w = env_obs["gripper_width"]
        if np.ndim(gripper_w) == 0:
            gripper_w = np.array([gripper_w], dtype=np.float32)
        proprio = np.concatenate([arm_qpos, gripper_w], axis=-1).astype(np.float32)

        proprio_tensor = torch.from_numpy(proprio).to(self.device).unsqueeze(0)
        model_obs["proprio"] = (proprio_tensor - self.proprio_mean) / self.proprio_std

        return model_obs

    def run_episode(
        self,
        seed: int,
        can_xy: Optional[Tuple[float, float]] = None,
        plant_xy: Optional[Tuple[float, float]] = None,
        max_steps: int = 200,
        exec_horizon: int = 8,
        num_ode_steps: int = 10,
        temporal_ensemble: bool = True,
        ensemble_decay: float = 0.05,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        domain_params: Optional[Dict[str, Any]] = None,
    ) -> EpisodeResult:
        """Execute one complete closed-loop evaluation episode using raw vision."""
        torch.manual_seed(seed)
        obs = self.env.reset(
            seed=seed,
            can_xy=can_xy,
            plant_xy=plant_xy,
            domain_params=domain_params,
        )
        actual_can_xy = (float(obs["can_pos"][0]), float(obs["can_pos"][1]))
        actual_plant_xy = (float(obs["plant_pos"][0]), float(obs["plant_pos"][1]))

        latencies = []
        min_spout_dist = float("inf")
        max_tilt_deg = 0.0
        success = False
        is_pouring = False
        step_count = 0

        if temporal_ensemble:
            buf_len = max_steps + self.horizon + 1
            action_buffer = np.zeros((buf_len, self.act_dim), dtype=np.float32)
            weight_buffer = np.zeros((buf_len,), dtype=np.float32)
            ensemble_weights = np.exp(-ensemble_decay * np.arange(self.horizon)).astype(np.float32)

            while step_count < max_steps:
                model_obs = self.prepare_vision_obs(obs)

                t_infer_start = time.perf_counter()
                with torch.no_grad():
                    pred_chunk_norm = self.cfm.sample(
                        self.model,
                        model_obs,
                        horizon=self.horizon,
                        act_dim=self.act_dim,
                        num_steps=num_ode_steps,
                    )
                latency_ms = (time.perf_counter() - t_infer_start) * 1000.0
                latencies.append(latency_ms)

                pred_chunk = pred_chunk_norm.squeeze(0).cpu().numpy()
                pred_chunk = pred_chunk * self.act_std + self.act_mean

                for h in range(self.horizon):
                    idx = step_count + h
                    action_buffer[idx] += ensemble_weights[h] * pred_chunk[h]
                    weight_buffer[idx] += ensemble_weights[h]

                env_act = action_buffer[step_count] / max(weight_buffer[step_count], 1e-6)

                if step_callback is not None:
                    spout_d = float(np.linalg.norm(obs["spout_pos"] - obs["plant_pos"]))
                    tilt_d = float(obs["tilt_angle_deg"][0])
                    parts = int(obs["particles_in_pot"][0])
                    is_pour = (tilt_d >= 40.0) and (spout_d <= 0.14)
                    is_succ = success or (parts >= 2) or (is_pour and spout_d <= 0.14)
                    step_callback({
                        "step_idx": step_count,
                        "obs": obs,
                        "model_obs": model_obs,
                        "pred_chunk": pred_chunk,
                        "exec_action": env_act,
                        "latency_ms": latency_ms,
                        "spout_dist": spout_d,
                        "tilt_deg": tilt_d,
                        "particles_in_pot": parts,
                        "is_pouring": is_pour,
                        "success": is_succ,
                    })

                obs, _, _, info = self.env.step(env_act)
                step_count += 1

                min_spout_dist = min(min_spout_dist, info["spout_dist"])
                max_tilt_deg = max(max_tilt_deg, info["tilt_deg"])
                if info["success"]:
                    success = True
                if info["is_pouring"]:
                    is_pouring = True
        else:
            while step_count < max_steps:
                model_obs = self.prepare_vision_obs(obs)

                t_infer_start = time.perf_counter()
                with torch.no_grad():
                    pred_chunk_norm = self.cfm.sample(
                        self.model,
                        model_obs,
                        horizon=self.horizon,
                        act_dim=self.act_dim,
                        num_steps=num_ode_steps,
                    )
                latency_ms = (time.perf_counter() - t_infer_start) * 1000.0
                latencies.append(latency_ms)

                pred_chunk = pred_chunk_norm.squeeze(0).cpu().numpy()
                pred_chunk = pred_chunk * self.act_std + self.act_mean

                steps_to_exec = min(exec_horizon, max_steps - step_count)
                for i in range(steps_to_exec):
                    env_act = pred_chunk[i]
                    if step_callback is not None:
                        spout_d = float(np.linalg.norm(obs["spout_pos"] - obs["plant_pos"]))
                        tilt_d = float(obs["tilt_angle_deg"][0])
                        parts = int(obs["particles_in_pot"][0])
                        is_pour = (tilt_d >= 40.0) and (spout_d <= 0.14)
                        is_succ = success or (parts >= 2) or (is_pour and spout_d <= 0.14)
                        step_callback({
                            "step_idx": step_count,
                            "obs": obs,
                            "model_obs": model_obs,
                            "pred_chunk": pred_chunk[i:],
                            "exec_action": env_act,
                            "latency_ms": latency_ms,
                            "spout_dist": spout_d,
                            "tilt_deg": tilt_d,
                            "particles_in_pot": parts,
                            "is_pouring": is_pour,
                            "success": is_succ,
                        })
                    obs, _, _, info = self.env.step(env_act)
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
        return EpisodeResult(
            seed=seed,
            success=success,
            is_pouring=is_pouring,
            particles_in_pot=particles,
            min_spout_dist=min_spout_dist,
            max_tilt_deg=max_tilt_deg,
            total_steps=step_count,
            mean_inference_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            can_xy=actual_can_xy,
            plant_xy=actual_plant_xy,
        )

    def evaluate_benchmark(
        self,
        seeds: List[int],
        can_xys: Optional[List[Tuple[float, float]]] = None,
        plant_xys: Optional[List[Tuple[float, float]]] = None,
        max_steps: int = 200,
        exec_horizon: int = 8,
        num_ode_steps: int = 10,
        temporal_ensemble: bool = True,
        ensemble_decay: float = 0.05,
        domain_params_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> BenchmarkScorecard:
        """Run benchmark across list of seeds, object locations, and optional domain randomization dicts."""
        results: List[EpisodeResult] = []
        n = len(seeds)

        for i, seed in enumerate(seeds):
            c_xy = can_xys[i] if can_xys is not None else None
            p_xy = plant_xys[i] if plant_xys is not None else None
            d_params = domain_params_list[i] if domain_params_list is not None else None

            res = self.run_episode(
                seed=seed,
                can_xy=c_xy,
                plant_xy=p_xy,
                max_steps=max_steps,
                exec_horizon=exec_horizon,
                num_ode_steps=num_ode_steps,
                temporal_ensemble=temporal_ensemble,
                ensemble_decay=ensemble_decay,
                domain_params=d_params,
            )
            results.append(res)
            status_char = "PASS" if res.success else "FAIL"
            print(
                f"[{i + 1:2d}/{n:2d}] Seed {seed:4d}: {status_char} | "
                f"Dist: {res.min_spout_dist * 100:.1f} cm | "
                f"Tilt: {res.max_tilt_deg:.1f} deg | "
                f"Particles: {res.particles_in_pot} | "
                f"Latency: {res.mean_inference_latency_ms:.2f} ms"
            )

        successes = sum(1 for r in results if r.success)
        return BenchmarkScorecard(
            total_episodes=n,
            successful_episodes=successes,
            success_rate_pct=(successes / n) * 100.0 if n > 0 else 0.0,
            mean_min_spout_dist=float(np.mean([r.min_spout_dist for r in results])),
            mean_max_tilt_deg=float(np.mean([r.max_tilt_deg for r in results])),
            mean_particles_in_pot=float(np.mean([r.particles_in_pot for r in results])),
            mean_episode_steps=float(np.mean([r.total_steps for r in results])),
            mean_latency_ms=float(np.mean([r.mean_inference_latency_ms for r in results])),
            episode_results=results,
        )
