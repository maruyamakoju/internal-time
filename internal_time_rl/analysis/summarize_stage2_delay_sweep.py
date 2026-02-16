from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Stage-2 delay sweep with paired effects."
    )
    parser.add_argument(
        "--base-root",
        type=str,
        required=True,
        help="Base root that contains delay{N}/aggregate/*.csv",
    )
    parser.add_argument(
        "--delays",
        type=int,
        nargs="+",
        default=[0, 5, 10, 20],
        help="Delay values to summarize.",
    )
    parser.add_argument(
        "--baseline-condition",
        type=str,
        default="learned_tau_delay10",
        help="Baseline condition name in condition_summary/per_run_summary.",
    )
    parser.add_argument(
        "--selfmodel-condition",
        type=str,
        default="learned_tau_delay10_selfmodel",
        help="Self-model condition name in condition_summary/per_run_summary.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Bootstrap resamples for paired CI.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Bootstrap RNG seed.",
    )
    parser.add_argument("--out-csv", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--out-tex", type=str, default=None, help="Optional LaTeX output path.")
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


def pick_condition_row(df: pd.DataFrame, condition: str) -> pd.Series:
    sub = df[df["condition"] == condition]
    if sub.empty:
        raise ValueError(f"Condition not found: {condition}")
    return sub.iloc[0]


def paired_delta(
    per_run_df: pd.DataFrame,
    baseline_condition: str,
    selfmodel_condition: str,
    metric: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    need_cols = {"condition", "seed", metric}
    missing = [c for c in need_cols if c not in per_run_df.columns]
    if missing:
        raise ValueError(f"per_run_summary is missing columns: {missing}")

    base = per_run_df[per_run_df["condition"] == baseline_condition][["seed", metric]].rename(
        columns={metric: "baseline"}
    )
    selfm = per_run_df[per_run_df["condition"] == selfmodel_condition][["seed", metric]].rename(
        columns={metric: "selfmodel"}
    )
    merged = base.merge(selfm, on="seed", how="inner").sort_values("seed")
    if merged.empty:
        return {
            "n_pairs": 0,
            "delta_mean": float("nan"),
            "delta_std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
        }

    delta = (merged["selfmodel"] - merged["baseline"]).to_numpy(dtype=np.float64)
    ci_low, ci_high = bootstrap_ci_mean(delta, bootstrap_samples, bootstrap_seed)
    return {
        "n_pairs": int(delta.size),
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta, ddof=0)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "n_positive": int(np.sum(delta > 0)),
        "n_negative": int(np.sum(delta < 0)),
        "n_zero": int(np.sum(delta == 0)),
    }


def main() -> None:
    args = parse_args()
    base_root = Path(args.base_root)
    rows: list[dict[str, float | int | str]] = []

    for delay in args.delays:
        root = base_root / f"delay{delay}"
        cond_path = root / "aggregate" / "condition_summary.csv"
        per_path = root / "aggregate" / "per_run_summary.csv"
        if not cond_path.exists() or not per_path.exists():
            raise SystemExit(
                f"Missing aggregate files for delay={delay}: "
                f"{cond_path} and/or {per_path}"
            )

        cond_df = pd.read_csv(cond_path)
        per_df = pd.read_csv(per_path)

        row_base = pick_condition_row(cond_df, args.baseline_condition)
        row_self = pick_condition_row(cond_df, args.selfmodel_condition)

        delta_final = paired_delta(
            per_df,
            baseline_condition=args.baseline_condition,
            selfmodel_condition=args.selfmodel_condition,
            metric="final_mean",
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        delta_auc = paired_delta(
            per_df,
            baseline_condition=args.baseline_condition,
            selfmodel_condition=args.selfmodel_condition,
            metric="auc",
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=int(args.bootstrap_seed),
        )

        rows.append(
            {
                "delay": int(delay),
                "root": root.as_posix(),
                "baseline_condition": args.baseline_condition,
                "selfmodel_condition": args.selfmodel_condition,
                "baseline_final_mean": float(row_base["final_mean"]),
                "baseline_final_std": float(row_base["final_std"]),
                "selfmodel_final_mean": float(row_self["final_mean"]),
                "selfmodel_final_std": float(row_self["final_std"]),
                "delta_final_mean": float(delta_final["delta_mean"]),
                "delta_final_std": float(delta_final["delta_std"]),
                "delta_final_ci95_low": float(delta_final["ci95_low"]),
                "delta_final_ci95_high": float(delta_final["ci95_high"]),
                "delta_final_n_pairs": int(delta_final["n_pairs"]),
                "delta_final_n_positive": int(delta_final["n_positive"]),
                "delta_final_n_negative": int(delta_final["n_negative"]),
                "baseline_auc_mean": float(row_base["auc_mean"]),
                "baseline_auc_std": float(row_base["auc_std"]),
                "selfmodel_auc_mean": float(row_self["auc_mean"]),
                "selfmodel_auc_std": float(row_self["auc_std"]),
                "delta_auc_mean": float(delta_auc["delta_mean"]),
                "delta_auc_std": float(delta_auc["delta_std"]),
                "delta_auc_ci95_low": float(delta_auc["ci95_low"]),
                "delta_auc_ci95_high": float(delta_auc["ci95_high"]),
                "delta_auc_n_pairs": int(delta_auc["n_pairs"]),
                "delta_auc_n_positive": int(delta_auc["n_positive"]),
                "delta_auc_n_negative": int(delta_auc["n_negative"]),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows).sort_values("delay").reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    if args.out_tex:
        out_tex = Path(args.out_tex)
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        tex_df = out_df.copy()
        for col in tex_df.columns:
            if col in {"delay", "root", "baseline_condition", "selfmodel_condition"}:
                continue
            tex_df[col] = pd.to_numeric(tex_df[col], errors="coerce").map(
                lambda x: f"{x:.3f}" if pd.notna(x) else "nan"
            )
        tex_df.to_latex(out_tex, index=False, escape=False)
        print(f"Saved: {out_tex}")


if __name__ == "__main__":
    main()

