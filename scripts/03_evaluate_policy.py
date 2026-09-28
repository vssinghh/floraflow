"""CLI Entrypoint: Evaluate State-Based Flow Matching Policy.

Delegates to floraflow.evaluation.evaluator.
"""

from __future__ import annotations

import argparse

from floraflow.evaluation.evaluator import (
    BenchmarkScorecard,
    PolicyEvaluator,
    generate_ood_configurations,
    print_scorecard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Flow Matching Policy")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_policy.pt")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--mode", type=str, choices=["in_dist", "ood", "both"], default="both")
    parser.add_argument("--difficulty", type=str, choices=["mild", "hard", "extreme"], default="hard")
    parser.add_argument("--exec-horizon", type=int, default=12)
    parser.add_argument("--num-ode-steps", type=int, default=10)
    parser.add_argument("--no-ensemble", action="store_true", help="Disable temporal ensembling")
    parser.add_argument("--ensemble-decay", type=float, default=0.05, help="Decay factor for temporal ensembling weights")
    args = parser.parse_args()

    use_ensemble = not args.no_ensemble
    evaluator = PolicyEvaluator(checkpoint_path=args.checkpoint)

    if args.mode in ["in_dist", "both"]:
        id_seeds = [100 + i for i in range(args.num_episodes)]
        print(f"Evaluating {args.num_episodes} In-Distribution episodes (held-out seeds)...")
        id_card = evaluator.evaluate_benchmark(
            seeds=id_seeds,
            exec_horizon=args.exec_horizon,
            num_ode_steps=args.num_ode_steps,
            temporal_ensemble=use_ensemble,
            ensemble_decay=args.ensemble_decay,
        )
        print_scorecard("In-Distribution Evaluation", id_card)

    if args.mode in ["ood", "both"]:
        ood_seeds, ood_cans, ood_plants = generate_ood_configurations(
            args.num_episodes,
            base_seed=200,
            difficulty=args.difficulty,
        )
        mode_label = f"{args.difficulty.upper()} + ENSEMBLE" if use_ensemble else args.difficulty.upper()
        print(f"Evaluating {args.num_episodes} Out-of-Distribution episodes ({mode_label} tier)...")
        ood_card = evaluator.evaluate_benchmark(
            seeds=ood_seeds,
            can_xy_list=ood_cans,
            plant_xy_list=ood_plants,
            exec_horizon=args.exec_horizon,
            num_ode_steps=args.num_ode_steps,
            temporal_ensemble=use_ensemble,
            ensemble_decay=args.ensemble_decay,
        )
        print_scorecard(f"Out-of-Distribution Generalization ({mode_label})", ood_card)


if __name__ == "__main__":
    main()
