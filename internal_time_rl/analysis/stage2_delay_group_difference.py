from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute high-delay vs low-delay paired group difference from delay sweep runs."
    )
    parser.add_argument("--base-root", type=str, required=True, help="Base root containing delay*/aggregate.")
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 5, 10, 20], help="All delay values.")
    parser.add_argument("--low-delays", type=int, nargs="+", default=[0, 5], help="Low-delay group.")
    parser.add_argument("--high-delays", type=int, nargs="+", default=[10, 20], help="High-delay group.")
    parser.add_argument(
        "--baseline-condition",
        type=str,
        default="learned_tau_delay10",
        help="Baseline condition in per_run_summary.",
    )
    parser.add_argument(
        "--selfmodel-condition",
        type=str,
        default="learned_tau_delay10_selfmodel",
        help="Self-model condition in per_run_summary.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["final_mean", "auc"],
        help="Metrics to evaluate from per_run_summary.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--out-tex", type=str, default=None)
    parser.add_argument("--out-per-seed", type=str, default=None)
    return parser.parse_args()


def bootstrap_ci_mean(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(samples, n))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_delay_delta_table(
    base_root: Path,
    delays: list[int],
    baseline_condition: str,
    selfmodel_condition: str,
    metric: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for delay in delays:
        per_run_path = base_root / f"delay{delay}" / "aggregate" / "per_run_summary.csv"
        if not per_run_path.exists():
            raise FileNotFoundError(f"Missing file: {per_run_path}")
        per_df = pd.read_csv(per_run_path)
        required = {"condition", "seed", metric}
        missing = [c for c in required if c not in per_df.columns]
        if missing:
            raise ValueError(f"{per_run_path} missing columns: {missing}")

        base = per_df[per_df["condition"] == baseline_condition][["seed", metric]].rename(
            columns={metric: "baseline"}
        )
        selfm = per_df[per_df["condition"] == selfmodel_condition][["seed", metric]].rename(
            columns={metric: "selfmodel"}
        )
        merged = base.merge(selfm, on="seed", how="inner")
        for _, row in merged.iterrows():
            rows.append(
                {
                    "delay": int(delay),
                    "seed": int(row["seed"]),
                    "delta": float(row["selfmodel"] - row["baseline"]),
                }
            )
    out = pd.DataFrame(rows).sort_values(["seed", "delay"]).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    base_root = Path(args.base_root)
    delays = [int(d) for d in args.delays]
    low_delays = [int(d) for d in args.low_delays]
    high_delays = [int(d) for d in args.high_delays]

    if not set(low_delays).issubset(set(delays)):
        raise SystemExit("--low-delays must be a subset of --delays")
    if not set(high_delays).issubset(set(delays)):
        raise SystemExit("--high-delays must be a subset of --delays")

    summary_rows: list[dict[str, float | int | str]] = []
    per_seed_rows: list[dict[str, float | int | str]] = []

    for metric in args.metrics:
        delta_df = load_delay_delta_table(
            base_root=base_root,
            delays=delays,
            baseline_condition=args.baseline_condition,
            selfmodel_condition=args.selfmodel_condition,
            metric=metric,
        )
        wide = delta_df.pivot(index="seed", columns="delay", values="delta")
        needed_cols = sorted(set(low_delays + high_delays))
        wide = wide.dropna(subset=needed_cols)
        if wide.empty:
            raise SystemExit(f"No seed has complete delay coverage for metric={metric}.")

        low_vals = wide[low_delays].mean(axis=1)
        high_vals = wide[high_delays].mean(axis=1)
        diff_vals = high_vals - low_vals

        arr_low = low_vals.to_numpy(dtype=np.float64)
        arr_high = high_vals.to_numpy(dtype=np.float64)
        arr_diff = diff_vals.to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci_mean(
            arr_diff,
            samples=int(args.bootstrap_samples),
            seed=int(args.bootstrap_seed),
        )

        summary_rows.append(
            {
                "metric": metric,
                "n_seeds": int(arr_diff.size),
                "low_delays": ",".join(str(d) for d in low_delays),
                "high_delays": ",".join(str(d) for d in high_delays),
                "low_delta_mean": float(np.mean(arr_low)),
                "low_delta_std": float(np.std(arr_low, ddof=0)),
                "high_delta_mean": float(np.mean(arr_high)),
                "high_delta_std": float(np.std(arr_high, ddof=0)),
                "group_diff_mean": float(np.mean(arr_diff)),
                "group_diff_std": float(np.std(arr_diff, ddof=0)),
                "group_diff_ci95_low": ci_low,
                "group_diff_ci95_high": ci_high,
                "n_positive": int(np.sum(arr_diff > 0)),
                "n_negative": int(np.sum(arr_diff < 0)),
                "n_zero": int(np.sum(arr_diff == 0)),
            }
        )

        for seed, low_v, high_v, diff_v in zip(wide.index.to_numpy(), arr_low, arr_high, arr_diff):
            per_seed_rows.append(
                {
                    "metric": metric,
                    "seed": int(seed),
                    "low_group_delta": float(low_v),
                    "high_group_delta": float(high_v),
                    "group_diff": float(diff_v),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    if args.out_per_seed:
        per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["metric", "seed"])
        out_per_seed = Path(args.out_per_seed)
        out_per_seed.parent.mkdir(parents=True, exist_ok=True)
        per_seed_df.to_csv(out_per_seed, index=False)
        print(f"Saved: {out_per_seed}")

    if args.out_tex:
        out_tex = Path(args.out_tex)
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        tex_df = summary_df.copy()
        numeric_cols = [
            "low_delta_mean",
            "low_delta_std",
            "high_delta_mean",
            "high_delta_std",
            "group_diff_mean",
            "group_diff_std",
            "group_diff_ci95_low",
            "group_diff_ci95_high",
        ]
        for col in numeric_cols:
            tex_df[col] = tex_df[col].map(lambda x: f"{x:.3f}")
        tex_df.to_latex(out_tex, index=False, escape=False)
        print(f"Saved: {out_tex}")


if __name__ == "__main__":
    main()

