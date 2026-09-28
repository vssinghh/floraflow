"""CLI Entrypoint for 4-Layer Multi-Camera VLA Telemetry Rollout Visualization (`python -m floraflow.evaluation.visualize`)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from floraflow.evaluation.evaluator import generate_ood_configurations
from floraflow.evaluation.vision_evaluator import VisionPolicyEvaluator
from floraflow.evaluation.visualizer import VisionRolloutVisualizer, save_comparison_gif


def _resolve_spawn_coords(
    mode: str,
    seed: int,
    ood_difficulty: str,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Resolve deterministic (can_xy, plant_xy) spawn coordinates matching benchmark seeds."""
    if mode == "id":
        return None, None

    offset = max(0, seed - 200)
    num_needed = offset + 1
    _, can_xys, plant_xys = generate_ood_configurations(
        num_episodes=num_needed,
        base_seed=200,
        difficulty=ood_difficulty,
    )
    return can_xys[offset], plant_xys[offset]


def _default_label(ckpt_path: str) -> str:
    """Infer a concise human-readable model label from checkpoint path."""
    p = Path(ckpt_path)
    parent = p.parent.name
    if parent and parent != "checkpoints":
        return parent
    return p.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m floraflow.evaluation.visualize",
        description="Visualize VLA Multi-Camera Keypoint & Telemetry Rollouts",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/run14_clean_data_3cam/best_vision_policy.pt",
        help="Primary vision policy checkpoint path",
    )
    parser.add_argument("--label", type=str, default=None, help="Subtitle label for primary checkpoint")
    parser.add_argument(
        "--compare-checkpoint",
        type=str,
        default=None,
        help="Optional second checkpoint for head-to-head synchronized comparison GIF",
    )
    parser.add_argument("--compare-label", type=str, default=None, help="Subtitle label for comparison checkpoint")
    parser.add_argument("--mode", type=str, choices=["id", "ood"], default="ood", help="Spawn distribution mode")
    parser.add_argument(
        "--ood-difficulty",
        type=str,
        choices=["mild", "hard", "extreme"],
        default="hard",
        help="Out-of-distribution difficulty tier",
    )
    parser.add_argument("--seed", type=int, default=204, help="Evaluation episode seed (e.g. 100 for ID, 204 for OOD)")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum control steps per episode")
    parser.add_argument("--ode-steps", type=int, default=10, help="Euler ODE integration steps")
    parser.add_argument("--cam-size", type=int, default=256, help="Per-camera render resolution (pixels)")
    parser.add_argument("--fps", type=int, default=20, help="Output animation playback frame rate")
    parser.add_argument("--frame-stride", type=int, default=2, help="Frame subsampling stride for compact GIF size")
    parser.add_argument(
        "--output",
        type=str,
        default="assets/media/vla_telemetry_rollout.gif",
        help="Output .gif file path",
    )
    parser.add_argument("--no-strip", action="store_true", help="Skip saving companion 6-keyframe .png strip")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cpu, cuda, or mps")
    args = parser.parse_args()

    can_xy, plant_xy = _resolve_spawn_coords(args.mode, args.seed, args.ood_difficulty)
    scenario_tag = "In-Distribution" if args.mode == "id" else f"OOD ({args.ood_difficulty.upper()})"

    primary_label = args.label or _default_label(args.checkpoint)
    subtitle_a = f"{primary_label}  |  {scenario_tag}"

    print(f"\n[1/2] Running & recording primary policy: {args.checkpoint}")
    print(f"      Scenario: {scenario_tag} | Seed: {args.seed}")
    eval_a = VisionPolicyEvaluator(checkpoint_path=args.checkpoint, device_str=args.device)
    viz_a = VisionRolloutVisualizer(
        evaluator=eval_a,
        title="FLORAFLOW VLA TELEMETRY",
        subtitle=subtitle_a,
        cam_render_size=args.cam_size,
        max_steps=args.max_steps,
    )
    viz_a.reset(seed=args.seed, subtitle=subtitle_a)

    res_a = eval_a.run_episode(
        seed=args.seed,
        can_xy=can_xy,
        plant_xy=plant_xy,
        max_steps=args.max_steps,
        num_ode_steps=args.ode_steps,
        temporal_ensemble=True,
        step_callback=viz_a.make_step_callback(),
    )
    status_a = "PASS" if res_a.success else "FAIL"
    print(
        f"      Result: {status_a} | Min Spout Dist: {res_a.min_spout_dist * 100:.1f} cm | "
        f"Max Tilt: {res_a.max_tilt_deg:.1f} deg | Particles: {res_a.particles_in_pot}/8"
    )

    out_path = Path(args.output)
    if args.compare_checkpoint:
        comp_label = args.compare_label or _default_label(args.compare_checkpoint)
        subtitle_b = f"{comp_label}  |  {scenario_tag}"
        print(f"\n[2/2] Running & recording comparison policy: {args.compare_checkpoint}")
        eval_b = VisionPolicyEvaluator(checkpoint_path=args.compare_checkpoint, device_str=args.device)
        viz_b = VisionRolloutVisualizer(
            evaluator=eval_b,
            title="FLORAFLOW VLA TELEMETRY (COMPARISON)",
            subtitle=subtitle_b,
            cam_render_size=args.cam_size,
            max_steps=args.max_steps,
        )
        viz_b.reset(seed=args.seed, subtitle=subtitle_b)
        res_b = eval_b.run_episode(
            seed=args.seed,
            can_xy=can_xy,
            plant_xy=plant_xy,
            max_steps=args.max_steps,
            num_ode_steps=args.ode_steps,
            temporal_ensemble=True,
            step_callback=viz_b.make_step_callback(),
        )
        status_b = "PASS" if res_b.success else "FAIL"
        print(
            f"      Result: {status_b} | Min Spout Dist: {res_b.min_spout_dist * 100:.1f} cm | "
            f"Max Tilt: {res_b.max_tilt_deg:.1f} deg | Particles: {res_b.particles_in_pot}/8"
        )
        saved_gif = save_comparison_gif(
            viz_a,
            viz_b,
            output_path=out_path,
            fps=args.fps,
            frame_stride=args.frame_stride,
        )
        print(f"\nSaved head-to-head comparison GIF: {saved_gif}")
    else:
        saved_gif = viz_a.save_gif(output_path=out_path, fps=args.fps, frame_stride=args.frame_stride)
        print(f"\nSaved telemetry rollout GIF: {saved_gif}")

    if not args.no_strip:
        strip_path = out_path.with_name(f"{out_path.stem}_strip.png")
        saved_strip = viz_a.save_summary_strip(output_path=strip_path)
        print(f"Saved 6-phase keyframe contact sheet: {saved_strip}")


if __name__ == "__main__":
    main()
