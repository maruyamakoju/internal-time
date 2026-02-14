from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONDITION_SUITES: dict[str, dict[str, list[str]]] = {
    "stage1_core": {
        "standard_gru_base": [
            "model.transition_mode=standard",
            "env.reward_delay=0",
            "env.flicker_prob=0.0",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "fixed_tau_1_base": [
            "model.transition_mode=fixed",
            "model.fixed_tau=1.0",
            "env.reward_delay=0",
            "env.flicker_prob=0.0",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "fixed_tau_3_base": [
            "model.transition_mode=fixed",
            "model.fixed_tau=3.0",
            "env.reward_delay=0",
            "env.flicker_prob=0.0",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "fixed_tau_10_base": [
            "model.transition_mode=fixed",
            "model.fixed_tau=10.0",
            "env.reward_delay=0",
            "env.flicker_prob=0.0",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "learned_tau_base": [
            "model.transition_mode=learned",
            "env.reward_delay=0",
            "env.flicker_prob=0.0",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
    },
    "stage1_full": {
        "standard_gru_base": [
            "model.transition_mode=standard",
            "env.reward_delay=0",
            "env.flicker_prob=0.0",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "fixed_tau_10_base": [
            "model.transition_mode=fixed",
            "model.fixed_tau=10.0",
            "env.reward_delay=0",
            "env.flicker_prob=0.0",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "learned_tau_base": [
            "model.transition_mode=learned",
            "env.reward_delay=0",
            "env.flicker_prob=0.0",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "standard_gru_delay10": [
            "model.transition_mode=standard",
            "env.reward_delay=10",
            "env.flicker_prob=0.0",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "fixed_tau_10_delay10": [
            "model.transition_mode=fixed",
            "model.fixed_tau=10.0",
            "env.reward_delay=10",
            "env.flicker_prob=0.0",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "learned_tau_delay10": [
            "model.transition_mode=learned",
            "env.reward_delay=10",
            "env.flicker_prob=0.0",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "standard_gru_flicker03": [
            "model.transition_mode=standard",
            "env.reward_delay=0",
            "env.flicker_prob=0.3",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "fixed_tau_10_flicker03": [
            "model.transition_mode=fixed",
            "model.fixed_tau=10.0",
            "env.reward_delay=0",
            "env.flicker_prob=0.3",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "learned_tau_flicker03": [
            "model.transition_mode=learned",
            "env.reward_delay=0",
            "env.flicker_prob=0.3",
            "env.variable_speed=false",
            "train.discount_mode=fixed",
        ],
        "standard_gru_speed": [
            "model.transition_mode=standard",
            "env.reward_delay=0",
            "env.flicker_prob=0.0",
            "env.variable_speed=true",
            "env.min_repeat=1",
            "env.max_repeat=4",
            "train.discount_mode=env_dt",
        ],
        "fixed_tau_10_speed": [
            "model.transition_mode=fixed",
            "model.fixed_tau=10.0",
            "env.reward_delay=0",
            "env.flicker_prob=0.0",
            "env.variable_speed=true",
            "env.min_repeat=1",
            "env.max_repeat=4",
            "train.discount_mode=env_dt",
        ],
        "learned_tau_speed_objdt": [
            "model.transition_mode=learned",
            "env.reward_delay=0",
            "env.flicker_prob=0.0",
            "env.variable_speed=true",
            "env.min_repeat=1",
            "env.max_repeat=4",
            "train.discount_mode=env_dt",
        ],
        "learned_tau_speed_subjdt": [
            "model.transition_mode=learned",
            "env.reward_delay=0",
            "env.flicker_prob=0.0",
            "env.variable_speed=true",
            "env.min_repeat=1",
            "env.max_repeat=4",
            "train.discount_mode=internal_tau",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed stage-1 sweeps.")
    parser.add_argument(
        "--suite",
        type=str,
        default="stage1_core",
        choices=sorted(CONDITION_SUITES.keys()),
        help="Condition suite to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="List of seeds.",
    )
    parser.add_argument("--timesteps", type=int, default=200000, help="Total timesteps per run.")
    parser.add_argument(
        "--base-run-dir",
        type=str,
        default="runs/sweeps/stage1",
        help="Base output directory.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used to launch train.py.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if metrics.csv already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of condition names to run.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides applied to every run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = CONDITION_SUITES[args.suite]
    if args.conditions:
        selected: dict[str, list[str]] = {}
        for name in args.conditions:
            if name not in suite:
                raise SystemExit(
                    f"Unknown condition '{name}' for suite '{args.suite}'. "
                    f"Available: {', '.join(suite.keys())}"
                )
            selected[name] = suite[name]
        suite = selected
    base_dir = Path(args.base_run_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(suite) * len(args.seeds)
    run_idx = 0

    for condition_name, condition_overrides in suite.items():
        for seed in args.seeds:
            run_idx += 1
            run_dir = base_dir / condition_name / f"seed{seed}"
            metrics_path = run_dir / "metrics.csv"
            if args.skip_existing and metrics_path.exists():
                print(f"[{run_idx}/{total_runs}] skip: {metrics_path}")
                continue

            cmd = [
                args.python_exe,
                "train.py",
                f"seed={seed}",
                f"train.total_timesteps={args.timesteps}",
                f"logging.run_dir={run_dir.as_posix()}",
                *condition_overrides,
                *args.overrides,
            ]

            print(f"[{run_idx}/{total_runs}] run: {condition_name} seed={seed}")
            print("  " + " ".join(cmd))

            if args.dry_run:
                continue

            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                raise SystemExit(
                    f"Run failed (code={result.returncode}): condition={condition_name}, seed={seed}"
                )


if __name__ == "__main__":
    main()
