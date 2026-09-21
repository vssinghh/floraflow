"""Evaluate Flow Matching Policy on In-Distribution and Out-of-Distribution Tasks.

Benchmarks closed-loop task success rate, kinematic accuracy, pour angle,
and inference latency across held-out evaluation scenarios.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple
import numpy as np

from floraflow.eval.evaluator import BenchmarkScorecard, PolicyEvaluator


def generate_ood_configurations(num_episodes: int, base_seed: int = 200) -> Tuple[List[int], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Generate out-of-distribution can and plant configurations outside training bounds."""
    rng = np.random.default_rng(base_seed)
    seeds = [base_seed + i for i in range(num_episodes)]
    can_xys = []
    plant_xys = []

    for _ in range(num_episodes):
        # Training can range: X in [0.48, 0.56], Y in [0.12, 0.22]
        # OOD: Perturb X into [0.45, 0.47] or [0.57, 0.60], Y into [0.09, 0.11] or [0.23, 0.26]
        if rng.random() > 0.5:
            cx = rng.uniform(0.45, 0.47) if rng.random() > 0.5 else rng.uniform(0.57, 0.60)
            cy = rng.uniform(0.13, 0.21)
        else:
            cx = rng.uniform(0.49, 0.55)
            cy = rng.uniform(0.09, 0.11) if rng.random() > 0.5 else rng.uniform(0.23, 0.25)
        can_xys.append((float(cx), float(cy)))

        # Training plant range: X in [0.38, 0.46], Y in [-0.26, -0.18]
        # OOD: Perturb into shifted regions
        if rng.random() > 0.5:
            px = rng.uniform(0.35, 0.37) if rng.random() > 0.5 else rng.uniform(0.47, 0.50)
            py = rng.uniform(-0.25, -0.19)
        else:
            px = rng.uniform(0.39, 0.45)
            py = rng.uniform(-0.29, -0.27) if rng.random() > 0.5 else rng.uniform(-0.16, -0.14)
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
    meets_realtime = card.mean_latency_ms < 50.0  # 20 Hz cycle requires < 50ms
    print(f" Real-time 20 Hz Feasibility:     {'PASS (<50ms budget)' if meets_realtime else 'FAIL'}")
    print(f"{border}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Flow Matching Policy")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_policy.pt")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--mode", type=str, choices=["in_dist", "ood", "both"], default="both")
    parser.add_argument("--exec-horizon", type=int, default=12)
    parser.add_argument("--num-ode-steps", type=int, default=10)
    args = parser.parse_args()

    evaluator = PolicyEvaluator(checkpoint_path=args.checkpoint)

    # 1. In-Distribution Evaluation (held-out seeds)
    if args.mode in ["in_dist", "both"]:
        id_seeds = [100 + i for i in range(args.num_episodes)]
        print(f"Evaluating {args.num_episodes} In-Distribution episodes (held-out seeds)...")
        id_card = evaluator.evaluate_benchmark(
            seeds=id_seeds,
            exec_horizon=args.exec_horizon,
            num_ode_steps=args.num_ode_steps,
        )
        print_scorecard("In-Distribution Evaluation", id_card)

    # 2. Out-of-Distribution Evaluation
    if args.mode in ["ood", "both"]:
        ood_seeds, ood_cans, ood_plants = generate_ood_configurations(args.num_episodes, base_seed=200)
        print(f"Evaluating {args.num_episodes} Out-of-Distribution episodes (perturbed spatial configs)...")
        ood_card = evaluator.evaluate_benchmark(
            seeds=ood_seeds,
            can_xy_list=ood_cans,
            plant_xy_list=ood_plants,
            exec_horizon=args.exec_horizon,
            num_ode_steps=args.num_ode_steps,
        )
        print_scorecard("Out-of-Distribution Generalization", ood_card)


if __name__ == "__main__":
    main()
