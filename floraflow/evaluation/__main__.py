"""Self-Contained CLI Entrypoint for Policy Evaluation (`python -m floraflow.evaluation`).

Benchmarks closed-loop task success rate, kinematic accuracy, pour angle,
and inference latency across In-Distribution (ID) and Out-of-Distribution (OOD) tiers.
"""

from __future__ import annotations

import argparse

from floraflow.evaluation.evaluator import (
    PolicyEvaluator,
    generate_ood_configurations,
    print_scorecard,
)
from floraflow.evaluation.vision_evaluator import VisionPolicyEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m floraflow.evaluation",
        description="Evaluate FloraFlow Flow Matching Policy (vision or state)",
    )
    parser.add_argument(
        "--state",
        action="store_true",
        help="Evaluate low-dimensional state-only policy instead of multi-camera vision policy",
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_vision_policy.pt", help="Path to checkpoint")
    parser.add_argument("--episodes", "--num-episodes", dest="episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--mode", type=str, choices=["id", "in_dist", "ood", "both"], default="both", help="Evaluation scenario")
    parser.add_argument(
        "--ood-difficulty",
        "--difficulty",
        dest="ood_difficulty",
        type=str,
        choices=["mild", "hard", "extreme"],
        default="hard",
        help="OOD difficulty level",
    )
    parser.add_argument("--ode-steps", "--num-ode-steps", dest="ode_steps", type=int, default=10, help="Numerical Euler ODE integration steps")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable temporal ensembling")
    parser.add_argument("--ensemble-decay", type=float, default=0.05, help="Exponential decay rate for ensembling")
    parser.add_argument("--exec-horizon", type=int, default=8, help="Execution sub-horizon when ensembling is disabled")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cpu, cuda, or mps")
    args = parser.parse_args()

    ensemble_active = not args.no_ensemble

    if args.state:
        evaluator = PolicyEvaluator(checkpoint_path=args.checkpoint)
        if args.mode in ("id", "in_dist", "both"):
            id_seeds = [100 + i for i in range(args.episodes)]
            print(f"Evaluating {args.episodes} In-Distribution state episodes...")
            id_card = evaluator.evaluate_benchmark(
                seeds=id_seeds,
                exec_horizon=args.exec_horizon,
                num_ode_steps=args.ode_steps,
                temporal_ensemble=ensemble_active,
                ensemble_decay=args.ensemble_decay,
            )
            print_scorecard("In-Distribution Evaluation", id_card)

        if args.mode in ("ood", "both"):
            ood_seeds, ood_cans, ood_plants = generate_ood_configurations(
                args.episodes,
                base_seed=200,
                difficulty=args.ood_difficulty,
            )
            mode_label = f"{args.ood_difficulty.upper()} + ENSEMBLE" if ensemble_active else args.ood_difficulty.upper()
            print(f"Evaluating {args.episodes} Out-of-Distribution state episodes ({mode_label} tier)...")
            ood_card = evaluator.evaluate_benchmark(
                seeds=ood_seeds,
                can_xy_list=ood_cans,
                plant_xy_list=ood_plants,
                exec_horizon=args.exec_horizon,
                num_ode_steps=args.ode_steps,
                temporal_ensemble=ensemble_active,
                ensemble_decay=args.ensemble_decay,
            )
            print_scorecard(f"Out-of-Distribution Generalization ({mode_label})", ood_card)
    else:
        evaluator_vis = VisionPolicyEvaluator(
            checkpoint_path=args.checkpoint,
            device_str=args.device,
        )
        print(f"\nEvaluating Vision Policy: {args.checkpoint}")
        print(f"Temporal Ensembling: {'ENABLED (decay=' + str(args.ensemble_decay) + ')' if ensemble_active else 'DISABLED'}")
        print(f"ODE Steps: {args.ode_steps}")

        if args.mode in ("id", "in_dist", "both"):
            id_seeds = [100 + i for i in range(args.episodes)]
            print(f"\n>>> Running In-Distribution (ID) Vision Benchmark ({args.episodes} episodes) <<<")
            id_card = evaluator_vis.evaluate_benchmark(
                seeds=id_seeds,
                temporal_ensemble=ensemble_active,
                ensemble_decay=args.ensemble_decay,
                exec_horizon=args.exec_horizon,
                num_ode_steps=args.ode_steps,
            )
            print_scorecard("Vision In-Distribution (ID)", id_card)

        if args.mode in ("ood", "both"):
            ood_episodes = 50 if args.mode == "both" else args.episodes
            ood_seeds, can_xys, plant_xys = generate_ood_configurations(
                num_episodes=ood_episodes,
                base_seed=200,
                difficulty=args.ood_difficulty,
            )
            print(f"\n>>> Running Out-of-Distribution (OOD, {args.ood_difficulty.upper()}) Vision Benchmark ({ood_episodes} episodes) <<<")
            ood_card = evaluator_vis.evaluate_benchmark(
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
