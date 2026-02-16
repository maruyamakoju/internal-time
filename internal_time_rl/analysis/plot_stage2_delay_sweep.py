from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Stage-2 delay sweep summary from CSV."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to delay sweep summary CSV.")
    parser.add_argument(
        "--out-final",
        type=str,
        required=True,
        help="Output figure path for final mean curves.",
    )
    parser.add_argument(
        "--out-delta",
        type=str,
        required=True,
        help="Output figure path for paired deltas with CI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input).sort_values("delay")
    if df.empty:
        raise SystemExit(f"No rows found in {args.input}")

    delays = df["delay"].to_numpy(dtype=np.float64)

    # Figure 1: final performance means with std error bars.
    plt.figure(figsize=(8, 5))
    base_mean = df["baseline_final_mean"].to_numpy(dtype=np.float64)
    base_std = df["baseline_final_std"].to_numpy(dtype=np.float64)
    self_mean = df["selfmodel_final_mean"].to_numpy(dtype=np.float64)
    self_std = df["selfmodel_final_std"].to_numpy(dtype=np.float64)

    plt.errorbar(delays, base_mean, yerr=base_std, marker="o", capsize=3, label="learned")
    plt.errorbar(delays, self_mean, yerr=self_std, marker="o", capsize=3, label="selfmodel")
    plt.xlabel("Reward Delay")
    plt.ylabel("Final Return (mean +/- std)")
    plt.title("Stage-2 Delay Sweep: Final Performance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_final = Path(args.out_final)
    out_final.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_final, dpi=170)
    plt.close()
    print(f"Saved: {out_final}")

    # Figure 2: paired deltas with bootstrap CI for final and AUC.
    plt.figure(figsize=(8, 5))
    delta_final = df["delta_final_mean"].to_numpy(dtype=np.float64)
    delta_final_low = df["delta_final_ci95_low"].to_numpy(dtype=np.float64)
    delta_final_high = df["delta_final_ci95_high"].to_numpy(dtype=np.float64)
    err_low = delta_final - delta_final_low
    err_high = delta_final_high - delta_final
    yerr = np.vstack([err_low, err_high])

    plt.errorbar(
        delays,
        delta_final,
        yerr=yerr,
        marker="o",
        capsize=3,
        label="Delta Final (selfmodel - learned)",
    )
    plt.plot(delays, df["delta_auc_mean"].to_numpy(dtype=np.float64), marker="s", label="Delta AUC")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    plt.xlabel("Reward Delay")
    plt.ylabel("Paired Delta")
    plt.title("Stage-2 Delay Sweep: Paired Effect Sizes")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_delta = Path(args.out_delta)
    out_delta.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_delta, dpi=170)
    plt.close()
    print(f"Saved: {out_delta}")


if __name__ == "__main__":
    main()

