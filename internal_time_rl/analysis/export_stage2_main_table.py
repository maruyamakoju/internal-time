from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from internal_time_rl.analysis.common import require_path, save_tex_with_numeric_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a compact Stage-2 main results table from existing CSV artifacts."
    )
    parser.add_argument(
        "--warmup-paired",
        type=str,
        default="docs/stage2_delay_warmup_both20k_paired_2026-02-16.csv",
        help="Paired effects CSV for warmup delay=10 root.",
    )
    parser.add_argument(
        "--warmup-groupdiff",
        type=str,
        default="docs/stage2_delay_warmup_groupdiff_0_vs_10_20_2026-02-16.csv",
        help="Group-difference CSV for warmup high={10,20} vs low={0}.",
    )
    parser.add_argument(
        "--nonwarmup-paired",
        type=str,
        default="docs/stage2_delay_paired_effects_2026-02-16.csv",
        help="Paired effects CSV for non-warmup noerr causal evidence.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="docs/stage2_main_table_2026-02-16.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-tex",
        type=str,
        default="docs/stage2_main_table_2026-02-16.tex",
        help="Output TeX path.",
    )
    return parser.parse_args()


def classify(delta_mean: float, ci_low: float, ci_high: float) -> str:
    if pd.notna(ci_low) and ci_low > 0:
        return "fixed_positive"
    if pd.notna(ci_high) and ci_high < 0:
        return "fixed_negative"
    if pd.notna(delta_mean):
        if delta_mean > 0:
            return "trend_positive"
        if delta_mean < 0:
            return "trend_negative"
    return "inconclusive"


def load_paired_row(
    csv_path: Path,
    *,
    comparator: str,
    metric: str,
    anchor: str | None = None,
) -> pd.Series:
    df = pd.read_csv(csv_path)
    query = (df["comparator"] == comparator) & (df["metric"] == metric)
    if anchor is not None and "anchor" in df.columns:
        query = query & (df["anchor"] == anchor)
    rows = df.loc[query]
    if rows.empty:
        raise SystemExit(
            f"Row not found in {csv_path}: comparator={comparator}, metric={metric}, anchor={anchor}"
        )
    return rows.iloc[0]


def load_groupdiff_row(csv_path: Path, metric: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    rows = df.loc[df["metric"] == metric]
    if rows.empty:
        raise SystemExit(f"Metric row not found in {csv_path}: metric={metric}")
    return rows.iloc[0]


def main() -> None:
    args = parse_args()
    warmup_paired = require_path(args.warmup_paired)
    warmup_groupdiff = require_path(args.warmup_groupdiff)
    nonwarmup_paired = require_path(args.nonwarmup_paired)

    records: list[dict[str, str | int | float]] = []

    for metric in ("final_mean", "auc"):
        row = load_paired_row(
            warmup_paired,
            comparator="learned_tau_delay10",
            metric=metric,
            anchor="learned_tau_delay10_selfmodel",
        )
        records.append(
            {
                "claim": "warmup_delay10_selfmodel_minus_learned",
                "metric": metric,
                "n_obs": int(row["n_pairs"]),
                "delta_mean": float(row["delta_mean"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "status": classify(
                    float(row["delta_mean"]),
                    float(row["ci95_low"]),
                    float(row["ci95_high"]),
                ),
                "source": str(warmup_paired),
            }
        )

    for metric in ("final_mean", "auc"):
        row = load_groupdiff_row(warmup_groupdiff, metric=metric)
        records.append(
            {
                "claim": "warmup_groupdiff_high_10_20_minus_low_0",
                "metric": metric,
                "n_obs": int(row["n_seeds"]),
                "delta_mean": float(row["group_diff_mean"]),
                "ci95_low": float(row["group_diff_ci95_low"]),
                "ci95_high": float(row["group_diff_ci95_high"]),
                "status": classify(
                    float(row["group_diff_mean"]),
                    float(row["group_diff_ci95_low"]),
                    float(row["group_diff_ci95_high"]),
                ),
                "source": str(warmup_groupdiff),
            }
        )

    for metric in ("final_mean", "auc"):
        row = load_paired_row(
            nonwarmup_paired,
            comparator="learned_tau_delay10_selfmodel_noerr",
            metric=metric,
            anchor="learned_tau_delay10_selfmodel",
        )
        records.append(
            {
                "claim": "nonwarmup_causal_selfmodel_minus_noerr",
                "metric": metric,
                "n_obs": int(row["n_pairs"]),
                "delta_mean": float(row["delta_mean"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "status": classify(
                    float(row["delta_mean"]),
                    float(row["ci95_low"]),
                    float(row["ci95_high"]),
                ),
                "source": str(nonwarmup_paired),
            }
        )

    out_df = pd.DataFrame.from_records(records)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    out_tex = save_tex_with_numeric_format(
        out_df,
        args.out_tex,
        non_numeric_cols=["claim", "metric", "status", "source"],
        digits=3,
    )
    print(f"Saved: {out_tex}")


if __name__ == "__main__":
    main()
