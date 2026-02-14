from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED_DIR_PATTERN = re.compile(r"^seed(\d+)$")


@dataclass
class RunRecord:
    condition: str
    seed: int
    metrics_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed run metrics.")
    parser.add_argument("--root", type=str, required=True, help="Root dir containing condition/seed*/metrics.csv")
    parser.add_argument("--metric", type=str, default="episode/return_mean_20", help="Metric column for curve/final/AUC")
    parser.add_argument("--step-col", type=str, default="global_step", help="Step column name")
    parser.add_argument("--last-n", type=int, default=10, help="Use last N points to estimate final score")
    parser.add_argument("--points", type=int, default=200, help="Interpolation points for mean curve")
    parser.add_argument("--out-dir", type=str, default=None, help="Output dir (default: <root>/aggregate)")
    return parser.parse_args()


def discover_runs(root: Path) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for metrics_path in root.rglob("metrics.csv"):
        rel = metrics_path.relative_to(root)
        parts = rel.parts
        seed_idx = None
        seed = None
        for i, part in enumerate(parts):
            m = SEED_DIR_PATTERN.match(part)
            if m is not None:
                seed_idx = i
                seed = int(m.group(1))
                break
        if seed_idx is None or seed is None:
            continue
        condition_parts = parts[:seed_idx]
        condition = "/".join(condition_parts) if condition_parts else "default"
        runs.append(RunRecord(condition=condition, seed=seed, metrics_path=metrics_path))
    return sorted(runs, key=lambda r: (r.condition, r.seed, str(r.metrics_path)))


def safe_auc(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    denom = x[-1] - x[0]
    if denom <= 0:
        return float("nan")
    return float(np.trapz(y, x) / denom)


def aggregate_curve(
    frames: list[pd.DataFrame], step_col: str, metric_col: str, points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    cleaned: list[tuple[np.ndarray, np.ndarray]] = []
    for df in frames:
        if step_col not in df.columns or metric_col not in df.columns:
            continue
        tmp = df[[step_col, metric_col]].dropna().sort_values(step_col)
        if tmp.empty:
            continue
        x = tmp[step_col].to_numpy(dtype=np.float64)
        y = tmp[metric_col].to_numpy(dtype=np.float64)
        cleaned.append((x, y))

    if not cleaned:
        return None

    x_min = max(x[0] for x, _ in cleaned)
    x_max = min(x[-1] for x, _ in cleaned)
    if x_max <= x_min:
        return None

    grid = np.linspace(x_min, x_max, points, dtype=np.float64)
    ys = np.stack([np.interp(grid, x, y) for x, y in cleaned], axis=0)
    mean = np.nanmean(ys, axis=0)
    std = np.nanstd(ys, axis=0)
    return grid, mean, std


def mean_std(series: pd.Series) -> tuple[float, float]:
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(root)
    if not runs:
        raise SystemExit(f"No runs found under: {root}")

    per_run_rows: list[dict[str, float | str | int]] = []
    curves_rows: list[dict[str, float | str | int]] = []
    raw_frames_by_condition: dict[str, list[pd.DataFrame]] = {}

    diagnostic_cols = [
        "tau/mean",
        "tau/obs_zero_mean",
        "tau/obs_nonzero_mean",
        "corr/tau_abs_adv",
        "corr/tau_abs_td",
        "corr/tau_action_repeat",
    ]

    for run in runs:
        df = pd.read_csv(run.metrics_path)
        raw_frames_by_condition.setdefault(run.condition, []).append(df)
        if args.step_col not in df.columns or args.metric not in df.columns:
            continue

        tmp = df[[args.step_col, args.metric]].dropna().sort_values(args.step_col)
        if tmp.empty:
            continue

        x = tmp[args.step_col].to_numpy(dtype=np.float64)
        y = tmp[args.metric].to_numpy(dtype=np.float64)
        final_mean = float(np.mean(y[-args.last_n :])) if y.size >= 1 else float("nan")
        auc = safe_auc(x, y)
        row: dict[str, float | str | int] = {
            "condition": run.condition,
            "seed": run.seed,
            "metrics_path": run.metrics_path.as_posix(),
            "final_mean": final_mean,
            "auc": auc,
            "steps_last": float(x[-1]),
        }

        for col in diagnostic_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
                row[f"{col}_lastn_mean"] = float(np.mean(vals[-args.last_n :])) if vals.size else float("nan")

        per_run_rows.append(row)

    per_run_df = pd.DataFrame(per_run_rows)
    per_run_df.to_csv(out_dir / "per_run_summary.csv", index=False)

    summary_rows: list[dict[str, float | str | int]] = []
    if not per_run_df.empty:
        for condition, grp in per_run_df.groupby("condition"):
            final_mean, final_std = mean_std(grp["final_mean"])
            auc_mean, auc_std = mean_std(grp["auc"])
            row: dict[str, float | str | int] = {
                "condition": condition,
                "n_seeds": int(len(grp)),
                "final_mean": final_mean,
                "final_std": final_std,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
            }
            for col in diagnostic_cols:
                key = f"{col}_lastn_mean"
                if key in grp.columns:
                    m, s = mean_std(grp[key])
                    row[f"{col}_mean"] = m
                    row[f"{col}_std"] = s
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("condition")
    summary_df.to_csv(out_dir / "condition_summary.csv", index=False)

    plt.figure(figsize=(11, 7))
    for condition, frames in sorted(raw_frames_by_condition.items()):
        out = aggregate_curve(frames, step_col=args.step_col, metric_col=args.metric, points=args.points)
        if out is None:
            continue
        grid, mean, std = out
        for x_i, mean_i, std_i in zip(grid, mean, std):
            curves_rows.append(
                {
                    "condition": condition,
                    args.step_col: float(x_i),
                    f"{args.metric}_mean": float(mean_i),
                    f"{args.metric}_std": float(std_i),
                }
            )
        plt.plot(grid, mean, label=condition)
        plt.fill_between(grid, mean - std, mean + std, alpha=0.2)

    plt.xlabel(args.step_col)
    plt.ylabel(args.metric)
    plt.title("Learning Curves (mean ± std across seeds)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curves.png", dpi=170)
    plt.close()

    curves_df = pd.DataFrame(curves_rows)
    curves_df.to_csv(out_dir / "learning_curves.csv", index=False)

    if not summary_df.empty:
        summary_df_sorted = summary_df.sort_values("final_mean", ascending=False).reset_index(drop=True)
        plt.figure(figsize=(11, 6))
        x = np.arange(len(summary_df_sorted))
        y = summary_df_sorted["final_mean"].to_numpy(dtype=np.float64)
        yerr = summary_df_sorted["final_std"].to_numpy(dtype=np.float64)
        plt.bar(x, y, yerr=yerr, capsize=3)
        plt.xticks(x, summary_df_sorted["condition"], rotation=45, ha="right")
        plt.ylabel(f"{args.metric} (final mean ± std)")
        plt.title("Final Performance by Condition")
        plt.tight_layout()
        plt.savefig(out_dir / "final_performance.png", dpi=170)
        plt.close()

    print(f"Saved: {out_dir / 'per_run_summary.csv'}")
    print(f"Saved: {out_dir / 'condition_summary.csv'}")
    print(f"Saved: {out_dir / 'learning_curves.csv'}")
    print(f"Saved: {out_dir / 'learning_curves.png'}")
    if not summary_df.empty:
        print(f"Saved: {out_dir / 'final_performance.png'}")


if __name__ == "__main__":
    main()

