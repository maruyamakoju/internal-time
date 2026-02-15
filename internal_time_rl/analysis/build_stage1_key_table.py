from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed Stage-1 key result table.")
    parser.add_argument("--runs-root", type=str, default="runs/sweeps", help="Root directory of sweep outputs.")
    parser.add_argument("--out", type=str, default="docs/stage1_key_table.csv", help="Output CSV path.")
    return parser.parse_args()


def load_summary(summary_path: Path) -> dict[str, float]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    df = pd.read_csv(summary_path)
    if "condition" not in df.columns or "final_mean" not in df.columns:
        raise ValueError(f"Unexpected columns in: {summary_path}")
    return {str(row["condition"]): float(row["final_mean"]) for _, row in df.iterrows()}


def round3(value: float) -> float:
    return float(f"{value:.3f}")


def best_gamma(rows: list[tuple[str, float]]) -> tuple[str, float]:
    gamma, value = max(rows, key=lambda item: item[1])
    return gamma, round3(value)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cartpole_main = load_summary(
        runs_root / "stage1_full_lm0_lv1e-2" / "aggregate" / "condition_summary.csv"
    )
    pendulum_speed = load_summary(
        runs_root / "pendulum_speed_stage1_lm0_lv1e-2" / "aggregate" / "condition_summary.csv"
    )
    acrobot_speed = load_summary(
        runs_root / "acrobot_speed_stage1_lm0_lv1e-2" / "aggregate" / "condition_summary.csv"
    )
    learned_gamma097 = load_summary(
        runs_root / "controls" / "cartpole_speed_learned_gamma097" / "aggregate" / "condition_summary.csv"
    )
    fixed_internal_gamma097 = load_summary(
        runs_root
        / "controls"
        / "fixed_tau10_internal_tau_gamma097"
        / "aggregate"
        / "condition_summary.csv"
    )
    fixed_internal_gamma099 = load_summary(
        runs_root
        / "controls"
        / "fixed_tau10_internal_tau_gamma099"
        / "aggregate"
        / "condition_summary.csv"
    )

    std_gamma_rows: list[tuple[str, float]] = []
    fixed_gamma_rows: list[tuple[str, float]] = []
    for gamma, suffix in [("0.97", "097"), ("0.99", "099"), ("0.995", "0995"), ("0.999", "0999")]:
        gamma_summary = load_summary(
            runs_root / "controls" / f"cartpole_speed_gamma{suffix}" / "aggregate" / "condition_summary.csv"
        )
        std_gamma_rows.append((gamma, gamma_summary["standard_gru_speed"]))
        fixed_gamma_rows.append((gamma, gamma_summary["fixed_tau_10_speed"]))

    best_std_gamma, best_std = best_gamma(std_gamma_rows)
    best_fixed_gamma, best_fixed = best_gamma(fixed_gamma_rows)

    const_rows: list[tuple[str, float]] = []
    for c in ["1", "2", "3"]:
        const_summary = load_summary(
            runs_root / "controls" / f"const_tau_c{c}" / "aggregate" / "condition_summary.csv"
        )
        const_rows.append((c, const_summary["standard_gru_speed"]))
    best_const_c, best_const = max(const_rows, key=lambda item: item[1])
    best_const = round3(best_const)

    cartpole_subjdt_099 = round3(cartpole_main["learned_tau_speed_subjdt"])
    cartpole_objdt_099 = round3(cartpole_main["learned_tau_speed_objdt"])
    cartpole_subjdt_097 = round3(learned_gamma097["learned_tau_speed_subjdt"])
    cartpole_objdt_097 = round3(learned_gamma097["learned_tau_speed_objdt"])
    fixed_internal_097 = round3(fixed_internal_gamma097["fixed_tau_10_speed"])
    fixed_internal_099 = round3(fixed_internal_gamma099["fixed_tau_10_speed"])

    rows = [
        {
            "task": "cartpole_speed",
            "standard_gru_speed": round3(cartpole_main["standard_gru_speed"]),
            "fixed_tau_10_speed": round3(cartpole_main["fixed_tau_10_speed"]),
            "learned_tau_speed_objdt": cartpole_objdt_099,
            "learned_tau_speed_subjdt": cartpole_subjdt_099,
            "learned_tau_speed_objdt_gamma097": cartpole_objdt_097,
            "learned_tau_speed_subjdt_gamma097": cartpole_subjdt_097,
            "best_standard_gamma": best_std_gamma,
            "best_standard_gamma_value": best_std,
            "best_fixed_tau10_gamma": best_fixed_gamma,
            "best_fixed_tau10_value": best_fixed,
            "best_const_tau_gamma099_c": best_const_c,
            "best_const_tau_gamma099_value": best_const,
            "fixed_tau10_internal_tau_gamma097": fixed_internal_097,
            "fixed_tau10_internal_tau_gamma099": fixed_internal_099,
            "delta_subjdt099_vs_best_standard": round3(cartpole_subjdt_099 - best_std),
            "delta_subjdt099_vs_best_fixed_envdt": round3(cartpole_subjdt_099 - best_fixed),
            "delta_subjdt099_vs_best_const_tau": round3(cartpole_subjdt_099 - best_const),
            "delta_subjdt099_vs_fixed_internal_tau099": round3(cartpole_subjdt_099 - fixed_internal_099),
        },
        {
            "task": "pendulum_speed",
            "standard_gru_speed": round3(pendulum_speed["standard_gru_speed"]),
            "fixed_tau_10_speed": round3(pendulum_speed["fixed_tau_10_speed"]),
            "learned_tau_speed_objdt": round3(pendulum_speed["learned_tau_speed_objdt"]),
            "learned_tau_speed_subjdt": round3(pendulum_speed["learned_tau_speed_subjdt"]),
        },
        {
            "task": "acrobot_speed",
            "standard_gru_speed": round3(acrobot_speed["standard_gru_speed"]),
            "fixed_tau_10_speed": round3(acrobot_speed["fixed_tau_10_speed"]),
            "learned_tau_speed_objdt": round3(acrobot_speed["learned_tau_speed_objdt"]),
            "learned_tau_speed_subjdt": round3(acrobot_speed["learned_tau_speed_subjdt"]),
        },
        {
            "task": "cartpole_base",
            "standard_gru_speed": round3(cartpole_main["standard_gru_base"]),
            "fixed_tau_10_speed": round3(cartpole_main["fixed_tau_10_base"]),
            "learned_tau_speed_objdt": round3(cartpole_main["learned_tau_base"]),
        },
        {
            "task": "cartpole_flicker03",
            "standard_gru_speed": round3(cartpole_main["standard_gru_flicker03"]),
            "fixed_tau_10_speed": round3(cartpole_main["fixed_tau_10_flicker03"]),
            "learned_tau_speed_objdt": round3(cartpole_main["learned_tau_flicker03"]),
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
