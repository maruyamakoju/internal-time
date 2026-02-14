from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot internal-time metrics from metrics.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to metrics.csv")
    parser.add_argument("--out", type=str, required=True, help="Output image path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "global_step" not in df.columns:
        raise ValueError("metrics.csv must include global_step")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    if "tau/mean" in df.columns:
        axes[0].plot(df["global_step"], df["tau/mean"], label="tau/mean")
    if "tau/rollout_mean" in df.columns:
        axes[0].plot(df["global_step"], df["tau/rollout_mean"], label="tau/rollout_mean")
    axes[0].set_ylabel("Internal Time")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if "episode/return_mean_20" in df.columns:
        axes[1].plot(df["global_step"], df["episode/return_mean_20"], label="episode/return_mean_20")
    axes[1].set_xlabel("Global Step")
    axes[1].set_ylabel("Return")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

