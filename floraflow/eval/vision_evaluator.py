"""Closed-Loop Multi-Camera Vision Policy Evaluator.

Executes trained VisionFlowMatchingPolicy directly from raw camera pixels and robot proprioception
in MuJoCo simulation, strictly without simulator state cheat codes (can_pos, plant_pos).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from floraflow.env.desk_env import DeskWateringEnv
from floraflow.eval.evaluator import BenchmarkScorecard, EpisodeResult
from floraflow.policy.flow_matching import ConditionalFlowMatcher
from floraflow.policy.vision_model import VisionFlowMatchingPolicy


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
        config = ckpt["config"]
        self.stats = ckpt["stats"]

        self.proprio_mean = torch.tensor(self.stats["proprio_mean"], device=self.device, dtype=torch.float32)
        self.proprio_std = torch.tensor(self.stats["proprio_std"], device=self.device, dtype=torch.float32)
        self.act_mean = np.array(self.stats["act_mean"], dtype=np.float32)
        self.act_std = np.array(self.stats["act_std"], dtype=np.float32)

        self.horizon = config["horizon"]
        self.act_dim = config["act_dim"]
        self.cameras = tuple(config.get("cameras", ["third_person_cam", "overhead_cam"]))

        self.model = VisionFlowMatchingPolicy(
            act_dim=self.act_dim,
            horizon=self.horizon,
            proprio_dim=config["proprio_dim"],
            num_keypoints=config["num_keypoints"],
            vision_feat_dim=config["vision_feat_dim"],
            proprio_feat_dim=config["proprio_feat_dim"],
            hidden_dim=config["hidden_dim"],
            num_blocks=config["num_blocks"],
            cameras=self.cameras,
            use_cross_attention=config.get("use_cross_attention", False),
            attn_heads=config.get("attn_heads", 4),
            camera_dropout=config.get("camera_dropout", 0.0),
            dropout_cameras=tuple(config.get("dropout_cameras", ["wrist_cam"])),
            use_aux_pose=config.get("use_aux_pose", False),
        ).to(self.device)
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
        step_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Convert environment observations into normalized PyTorch model inputs.

        Strictly extracts ONLY raw camera frames and 9D proprioception (7 qpos + 1 gripper + 1 progress).
        Does NOT access can_pos, plant_pos, spout_pos, or grip_pos.
        """
        model_obs: Dict[str, torch.Tensor] = {}

        for cam in self.cameras:
            raw_img = env_obs[f"rgb_{cam}"]  # (H, W, 3) uint8
            # Permute to (1, C, H, W) and scale to [0, 1]
            img_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            model_obs[f"rgb_{cam}"] = img_tensor.to(self.device)

        # Assemble 9D proprioception: (7D arm_qpos + 1D gripper_width + 1D progress)
        arm_qpos = env_obs["arm_qpos"]
        gripper_w = env_obs["gripper_width"]
        if np.ndim(gripper_w) == 0:
            gripper_w = np.array([gripper_w], dtype=np.float32)
        progress = np.array([min(step_idx / 174.0, 1.0)], dtype=np.float32)
        proprio = np.concatenate([arm_qpos, gripper_w, progress], axis=-1).astype(np.float32)

        proprio_tensor = torch.from_numpy(proprio).to(self.device).unsqueeze(0)
        norm_proprio = (proprio_tensor - self.proprio_mean) / self.proprio_std
        model_obs["proprio"] = norm_proprio

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
    ) -> EpisodeResult:
        """Execute one complete closed-loop evaluation episode using raw vision."""
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
            buf_len = max_steps + self.horizon + 1
            action_buffer = np.zeros((buf_len, self.act_dim), dtype=np.float32)
            weight_buffer = np.zeros((buf_len,), dtype=np.float32)
            ensemble_weights = np.exp(-ensemble_decay * np.arange(self.horizon)).astype(np.float32)

            while step_count < max_steps:
                model_obs = self.prepare_vision_obs(obs, step_idx=step_count)

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
            while step_count < max_steps:
                model_obs = self.prepare_vision_obs(obs, step_idx=step_count)

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
    ) -> BenchmarkScorecard:
        """Run benchmark across list of seeds and object locations."""
        results: List[EpisodeResult] = []
        n = len(seeds)

        for i, seed in enumerate(seeds):
            c_xy = can_xys[i] if can_xys is not None else None
            p_xy = plant_xys[i] if plant_xys is not None else None

            res = self.run_episode(
                seed=seed,
                can_xy=c_xy,
                plant_xy=p_xy,
                max_steps=max_steps,
                exec_horizon=exec_horizon,
                num_ode_steps=num_ode_steps,
                temporal_ensemble=temporal_ensemble,
                ensemble_decay=ensemble_decay,
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
