"""Evaluate Vision Flow Matching Policy on In-Distribution and Out-of-Distribution Tasks.

Benchmarks closed-loop task success rate, kinematic accuracy, pour angle,
and inference latency directly from raw camera observations without simulator cheats.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple
import numpy as np

from floraflow.evaluation.evaluator import BenchmarkScorecard, generate_ood_configurations, print_scorecard
from floraflow.evaluation.vision_evaluator import VisionPolicyEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Vision Flow Matching Policy")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_vision_policy.pt", help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--mode", type=str, choices=["id", "ood", "both"], default="both", help="Evaluation scenario")
    parser.add_argument("--ood-difficulty", type=str, choices=["mild", "hard", "extreme"], default="hard", help="OOD difficulty level")
    parser.add_argument("--ode-steps", type=int, default=10, help="Numerical Euler ODE integration steps")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable temporal ensembling")
    parser.add_argument("--ensemble-decay", type=float, default=0.05, help="Exponential decay rate for ensembling")
    parser.add_argument("--exec-horizon", type=int, default=8, help="Execution sub-horizon when ensembling is disabled")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cpu, cuda, or mps")
    args = parser.parse_args()

    evaluator = VisionPolicyEvaluator(
        checkpoint_path=args.checkpoint,
        device_str=args.device,
    )

    ensemble_active = not args.no_ensemble
    print(f"\nEvaluating Vision Policy: {args.checkpoint}")
    print(f"Temporal Ensembling: {'ENABLED (decay=' + str(args.ensemble_decay) + ')' if ensemble_active else 'DISABLED'}")
    print(f"ODE Steps: {args.ode_steps}")

    # In-Distribution Benchmark
    if args.mode in ("id", "both"):
        id_seeds = [100 + i for i in range(args.episodes)]
        print(f"\n>>> Running In-Distribution (ID) Vision Benchmark ({args.episodes} episodes) <<<")
        id_card = evaluator.evaluate_benchmark(
            seeds=id_seeds,
            temporal_ensemble=ensemble_active,
            ensemble_decay=args.ensemble_decay,
            exec_horizon=args.exec_horizon,
            num_ode_steps=args.ode_steps,
        )
        print_scorecard("Vision In-Distribution (ID)", id_card)

    # Out-of-Distribution Benchmark
    if args.mode in ("ood", "both"):
        ood_episodes = 50 if args.mode == "both" else args.episodes
        ood_seeds, can_xys, plant_xys = generate_ood_configurations(
            num_episodes=ood_episodes,
            base_seed=200,
            difficulty=args.ood_difficulty,
        )
        print(f"\n>>> Running Out-of-Distribution (OOD, {args.ood_difficulty.upper()}) Vision Benchmark ({ood_episodes} episodes) <<<")
        ood_card = evaluator.evaluate_benchmark(
            seeds=ood_seeds,
            can_xys=can_xys,
            plant_xys=plant_xys,
            temporal_ensemble=ensemble_active,
            ensemble_decay=args.ensemble_decay,
            exec_horizon=args.exec_horizon,
            num_ode_steps=args.ode_steps,
        )
        print_scorecard(f"Vision Out-of-Distribution ({args.ood_difficulty.upper()})", ood_card)


if __name__ == "__main__":
    main()
