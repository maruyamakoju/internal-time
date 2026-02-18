from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired seed-wise effect sizes from per_run_summary.csv."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to aggregate/per_run_summary.csv",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        required=True,
        help="Anchor condition name (delta = anchor - comparator).",
    )
    parser.add_argument(
        "--comparators",
        type=str,
        nargs="+",
        required=True,
        help="Comparator condition names.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["final_mean", "auc"],
        help="Metric columns in per_run_summary.csv.",
    )
    parser.add_argument(
        "--join-key",
        type=str,
        default="seed",
        help="Join key for pairing (default: seed).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Bootstrap resamples for 95%% CI.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Bootstrap RNG seed.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV for summary stats.",
    )
    parser.add_argument(
        "--out-per-seed",
        type=str,
        default=None,
        help="Optional output CSV for seed-wise paired deltas.",
    )
    parser.add_argument(
        "--out-tex",
        type=str,
        default=None,
        help="Optional LaTeX table output.",
    )
    parser.add_argument(
        "--min-n-pairs",
        type=int,
        default=None,
        help=(
            "Optional minimum required paired samples for every "
            "(comparator, metric). Raises an error if unmet."
        ),
    )
    return parser.parse_args()


def bootstrap_ci_mean(
    values: np.ndarray,
    samples: int,
    seed: int,
    ci_low: float = 2.5,
    ci_high: float = 97.5,
) -> tuple[float, float]:
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
    return float(np.percentile(means, ci_low)), float(np.percentile(means, ci_high))


def ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    df = pd.read_csv(in_path)
    ensure_columns(df, ["condition", args.join_key, *args.metrics])

    anchor_df = df[df["condition"] == args.anchor][[args.join_key, *args.metrics]].copy()
    if anchor_df.empty:
        raise SystemExit(f"Anchor condition not found: {args.anchor}")
    anchor_df = anchor_df.rename(
        columns={metric: f"{metric}_anchor" for metric in args.metrics}
    )

    summary_rows: list[dict[str, float | int | str]] = []
    per_seed_rows: list[dict[str, float | int | str]] = []

    for comp in args.comparators:
        comp_df = df[df["condition"] == comp][[args.join_key, *args.metrics]].copy()
        if comp_df.empty:
            raise SystemExit(f"Comparator condition not found: {comp}")
        comp_df = comp_df.rename(columns={metric: f"{metric}_comp" for metric in args.metrics})

        merged = anchor_df.merge(comp_df, on=args.join_key, how="inner")
        merged = merged.sort_values(args.join_key).reset_index(drop=True)
        if merged.empty:
            raise SystemExit(
                f"No paired rows for anchor='{args.anchor}' and comparator='{comp}' "
                f"using join key '{args.join_key}'."
            )

        for metric in args.metrics:
            anchor_col = f"{metric}_anchor"
            comp_col = f"{metric}_comp"
            delta = (
                pd.to_numeric(merged[anchor_col], errors="coerce")
                - pd.to_numeric(merged[comp_col], errors="coerce")
            )
            valid = delta.dropna().to_numpy(dtype=np.float64)
            n = int(valid.size)
            if n == 0:
                row = {
                    "anchor": args.anchor,
                    "comparator": comp,
                    "metric": metric,
                    "n_pairs": 0,
                    "delta_mean": float("nan"),
                    "delta_std": float("nan"),
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                    "delta_min": float("nan"),
                    "delta_max": float("nan"),
                    "n_positive": 0,
                    "n_negative": 0,
                    "n_zero": 0,
                }
            else:
                ci_low, ci_high = bootstrap_ci_mean(
                    valid,
                    samples=int(args.bootstrap_samples),
                    seed=int(args.bootstrap_seed),
                )
                row = {
                    "anchor": args.anchor,
                    "comparator": comp,
                    "metric": metric,
                    "n_pairs": n,
                    "delta_mean": float(np.mean(valid)),
                    "delta_std": float(np.std(valid, ddof=0)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "delta_min": float(np.min(valid)),
                    "delta_max": float(np.max(valid)),
                    "n_positive": int(np.sum(valid > 0)),
                    "n_negative": int(np.sum(valid < 0)),
                    "n_zero": int(np.sum(valid == 0)),
                }

            summary_rows.append(row)

            for _, m_row in merged.iterrows():
                anchor_val = m_row[anchor_col]
                comp_val = m_row[comp_col]
                if pd.isna(anchor_val) or pd.isna(comp_val):
                    continue
                per_seed_rows.append(
                    {
                        "anchor": args.anchor,
                        "comparator": comp,
                        "metric": metric,
                        args.join_key: int(m_row[args.join_key]),
                        "anchor_value": float(anchor_val),
                        "comparator_value": float(comp_val),
                        "delta": float(anchor_val - comp_val),
                    }
                )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)

    if args.min_n_pairs is not None:
        min_pairs = int(args.min_n_pairs)
        missing = summary_df[summary_df["n_pairs"] < min_pairs]
        if not missing.empty:
            subset_cols = ["comparator", "metric", "n_pairs"]
            details = ", ".join(
                [
                    f"{r['comparator']}:{r['metric']}={int(r['n_pairs'])}"
                    for _, r in missing[subset_cols].iterrows()
                ]
            )
            raise SystemExit(
                f"n_pairs check failed (required >= {min_pairs}): {details}"
            )

    summary_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    if args.out_per_seed:
        out_per_seed = Path(args.out_per_seed)
        out_per_seed.parent.mkdir(parents=True, exist_ok=True)
        per_seed_df = pd.DataFrame(per_seed_rows).sort_values(
            ["metric", "comparator", args.join_key]
        )
        per_seed_df.to_csv(out_per_seed, index=False)
        print(f"Saved: {out_per_seed}")

    if args.out_tex:
        out_tex = Path(args.out_tex)
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        summary_fmt = summary_df.copy()
        for col in ["delta_mean", "delta_std", "ci95_low", "ci95_high", "delta_min", "delta_max"]:
            if col in summary_fmt.columns:
                summary_fmt[col] = summary_fmt[col].map(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "nan"
                )
        summary_fmt.to_latex(out_tex, index=False, escape=False)
        print(f"Saved: {out_tex}")


if __name__ == "__main__":
    main()
