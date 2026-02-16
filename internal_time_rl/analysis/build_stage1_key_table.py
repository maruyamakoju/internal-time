from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CondStats:
    final_mean: float
    final_std: float
    auc_mean: float
    auc_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready Stage-1 key tables.")
    parser.add_argument("--runs-root", type=str, default="runs/sweeps", help="Root directory of sweep outputs.")
    parser.add_argument("--out", type=str, default="docs/stage1_key_table.csv", help="Output CSV path.")
    parser.add_argument(
        "--tex-out",
        type=str,
        default="docs/stage1_key_table.tex",
        help="Output LaTeX table path.",
    )
    return parser.parse_args()


def load_summary(summary_path: Path) -> dict[str, CondStats]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    df = pd.read_csv(summary_path)
    required = {"condition", "final_mean", "final_std", "auc_mean", "auc_std"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected columns in: {summary_path}")
    out: dict[str, CondStats] = {}
    for _, row in df.iterrows():
        out[str(row["condition"])] = CondStats(
            final_mean=float(row["final_mean"]),
            final_std=float(row["final_std"]),
            auc_mean=float(row["auc_mean"]),
            auc_std=float(row["auc_std"]),
        )
    return out


def round3(value: float) -> float:
    return float(f"{value:.3f}")


def add_row(
    rows: list[dict[str, float | str]],
    task: str,
    condition: str,
    setting: str,
    stats: CondStats,
) -> None:
    rows.append(
        {
            "task": task,
            "condition": condition,
            "setting": setting,
            "final_mean": round3(stats.final_mean),
            "final_std": round3(stats.final_std),
            "auc_mean": round3(stats.auc_mean),
            "auc_std": round3(stats.auc_std),
        }
    )


def best_by_final(rows: list[tuple[str, CondStats]]) -> tuple[str, CondStats]:
    return max(rows, key=lambda item: item[1].final_mean)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_path = Path(args.out)
    tex_out_path = Path(args.tex_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tex_out_path.parent.mkdir(parents=True, exist_ok=True)

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

    std_gamma_rows: list[tuple[str, CondStats]] = []
    fixed_gamma_rows: list[tuple[str, CondStats]] = []
    for gamma, suffix in [("0.97", "097"), ("0.99", "099"), ("0.995", "0995"), ("0.999", "0999")]:
        gamma_summary = load_summary(
            runs_root / "controls" / f"cartpole_speed_gamma{suffix}" / "aggregate" / "condition_summary.csv"
        )
        std_gamma_rows.append((gamma, gamma_summary["standard_gru_speed"]))
        fixed_gamma_rows.append((gamma, gamma_summary["fixed_tau_10_speed"]))
    best_std_gamma, best_std = best_by_final(std_gamma_rows)
    best_fixed_gamma, best_fixed = best_by_final(fixed_gamma_rows)

    const_rows: list[tuple[str, CondStats]] = []
    for c in ["1", "2", "3"]:
        const_summary = load_summary(
            runs_root / "controls" / f"const_tau_c{c}" / "aggregate" / "condition_summary.csv"
        )
        const_rows.append((c, const_summary["standard_gru_speed"]))
    best_const_c, best_const = best_by_final(const_rows)

    rows: list[dict[str, float | str]] = []

    # CartPole speed: main and key controls.
    add_row(rows, "cartpole_speed", "standard_gru_speed", "main_lm0_lv1e-2", cartpole_main["standard_gru_speed"])
    add_row(rows, "cartpole_speed", "fixed_tau_10_speed", "main_lm0_lv1e-2", cartpole_main["fixed_tau_10_speed"])
    add_row(
        rows,
        "cartpole_speed",
        "learned_tau_speed_objdt",
        "main_lm0_lv1e-2_gamma0.99",
        cartpole_main["learned_tau_speed_objdt"],
    )
    add_row(
        rows,
        "cartpole_speed",
        "learned_tau_speed_subjdt",
        "main_lm0_lv1e-2_gamma0.99",
        cartpole_main["learned_tau_speed_subjdt"],
    )
    add_row(
        rows,
        "cartpole_speed",
        "learned_tau_speed_objdt",
        "control_gamma0.97",
        learned_gamma097["learned_tau_speed_objdt"],
    )
    add_row(
        rows,
        "cartpole_speed",
        "learned_tau_speed_subjdt",
        "control_gamma0.97",
        learned_gamma097["learned_tau_speed_subjdt"],
    )
    add_row(
        rows,
        "cartpole_speed",
        "standard_gru_speed",
        f"best_gamma_envdt={best_std_gamma}",
        best_std,
    )
    add_row(
        rows,
        "cartpole_speed",
        "fixed_tau_10_speed",
        f"best_gamma_envdt={best_fixed_gamma}",
        best_fixed,
    )
    add_row(
        rows,
        "cartpole_speed",
        "standard_gru_speed_internal_tau_const",
        f"best_const_tau_gamma0.99_c={best_const_c}",
        best_const,
    )
    add_row(
        rows,
        "cartpole_speed",
        "fixed_tau_10_speed_internal_tau",
        "control_gamma0.97",
        fixed_internal_gamma097["fixed_tau_10_speed"],
    )
    add_row(
        rows,
        "cartpole_speed",
        "fixed_tau_10_speed_internal_tau",
        "control_gamma0.99",
        fixed_internal_gamma099["fixed_tau_10_speed"],
    )

    # Cross-environment speed rows.
    for task, summary in [
        ("pendulum_speed", pendulum_speed),
        ("acrobot_speed", acrobot_speed),
    ]:
        add_row(rows, task, "standard_gru_speed", "main_lm0_lv1e-2", summary["standard_gru_speed"])
        add_row(rows, task, "fixed_tau_10_speed", "main_lm0_lv1e-2", summary["fixed_tau_10_speed"])
        add_row(rows, task, "learned_tau_speed_objdt", "main_lm0_lv1e-2", summary["learned_tau_speed_objdt"])
        add_row(rows, task, "learned_tau_speed_subjdt", "main_lm0_lv1e-2", summary["learned_tau_speed_subjdt"])

    # Optional supporting rows for CartPole base/flicker.
    for task, standard_key, fixed_key, learned_key in [
        ("cartpole_base", "standard_gru_base", "fixed_tau_10_base", "learned_tau_base"),
        ("cartpole_flicker03", "standard_gru_flicker03", "fixed_tau_10_flicker03", "learned_tau_flicker03"),
    ]:
        add_row(rows, task, standard_key, "main_lm0_lv1e-2", cartpole_main[standard_key])
        add_row(rows, task, fixed_key, "main_lm0_lv1e-2", cartpole_main[fixed_key])
        add_row(rows, task, learned_key, "main_lm0_lv1e-2", cartpole_main[learned_key])

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    df.to_latex(tex_out_path, index=False, float_format="%.3f")

    print(f"Saved: {out_path}")
    print(f"Saved: {tex_out_path}")


if __name__ == "__main__":
    main()
