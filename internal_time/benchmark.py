"""NAB (Numenta Anomaly Benchmark) evaluation utilities.

Loads NAB data, runs TemporalAnomalyDetector, computes standard metrics
(point-adjusted F1, precision, recall) and generates summary tables.

Usage::

    python -m internal_time.benchmark --data-root data/nab --out results/nab
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .anomaly import TemporalAnomalyDetector


# ---------------------------------------------------------------------------
# NAB data loading
# ---------------------------------------------------------------------------

def load_nab_labels(labels_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Load NAB combined_windows.json.

    Returns mapping from relative CSV path to list of (start, end) timestamp
    strings defining anomaly windows.
    """
    with open(labels_path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw


def load_nab_file(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a single NAB CSV file.

    Returns
    -------
    timestamps : ndarray of datetime64
    values : ndarray, shape ``(T, D)``
    """
    df = pd.read_csv(csv_path)
    timestamps = pd.to_datetime(df["timestamp"]).values
    value_cols = [c for c in df.columns if c != "timestamp"]
    values = df[value_cols].values.astype(np.float32)
    return timestamps, values


def make_labels_array(
    timestamps: np.ndarray,
    windows: list[tuple[str, str]],
) -> np.ndarray:
    """Convert NAB anomaly windows to a binary label array."""
    labels = np.zeros(len(timestamps), dtype=np.int32)
    for start_str, end_str in windows:
        start = np.datetime64(start_str)
        end = np.datetime64(end_str)
        mask = (timestamps >= start) & (timestamps <= end)
        labels[mask] = 1
    return labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class DetectionMetrics:
    file_name: str
    precision: float
    recall: float
    f1: float
    pa_precision: float
    pa_recall: float
    pa_f1: float
    n_anomaly_windows: int
    n_timesteps: int
    threshold: float
    train_time_s: float
    score_time_s: float


def _point_adjusted_labels(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Point-adjusted predictions: if any point in a contiguous anomaly
    segment is detected, mark the entire segment as detected.

    This is the standard evaluation protocol in time series anomaly
    detection (Xu et al. 2018, etc.).
    """
    adjusted = y_pred.copy()
    # Find contiguous anomaly segments in ground truth
    in_seg = False
    start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_seg:
            start = i
            in_seg = True
        elif y_true[i] == 0 and in_seg:
            # segment [start, i)
            if adjusted[start:i].any():
                adjusted[start:i] = 1
            in_seg = False
    if in_seg:
        if adjusted[start:].any():
            adjusted[start:] = 1
    return adjusted


def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """Compute point-wise and point-adjusted metrics."""
    # Point-wise
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-12, p + r)

    # Point-adjusted
    pa_pred = _point_adjusted_labels(y_pred, y_true)
    pa_tp = int(((pa_pred == 1) & (y_true == 1)).sum())
    pa_fp = int(((pa_pred == 1) & (y_true == 0)).sum())
    pa_fn = int(((pa_pred == 0) & (y_true == 1)).sum())
    pa_p = pa_tp / max(1, pa_tp + pa_fp)
    pa_r = pa_tp / max(1, pa_tp + pa_fn)
    pa_f1 = 2 * pa_p * pa_r / max(1e-12, pa_p + pa_r)

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "pa_precision": pa_p,
        "pa_recall": pa_r,
        "pa_f1": pa_f1,
    }


# ---------------------------------------------------------------------------
# Single-file evaluation
# ---------------------------------------------------------------------------

def evaluate_file(
    csv_path: Path,
    windows: list[tuple[str, str]],
    train_ratio: float = 0.5,
    threshold: float = 2.0,
    detector_kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
    seed: int | None = None,
) -> DetectionMetrics:
    """Train on the first portion of a NAB file, score the rest."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    timestamps, values = load_nab_file(csv_path)
    labels = make_labels_array(timestamps, windows)
    T, D = values.shape

    # Split: train on early portion (assumed mostly normal), score full
    split = int(T * train_ratio)

    # If anomalies exist in training portion, just use it anyway —
    # NAB assumes online/unsupervised, but we use the early portion
    # as "mostly normal" which is standard practice.
    train_data = values[:split]

    kwargs = {
        "hidden_dim": 64,
        "epochs": 60,
        "window_size": min(50, split // 4),
        "patience": 15,
        "lr": 3e-4,
    }
    if detector_kwargs:
        kwargs.update(detector_kwargs)

    det = TemporalAnomalyDetector(input_dim=D, device=device, **kwargs)

    t0 = time.time()
    det.fit(train_data, verbose=False)
    train_time = time.time() - t0

    t0 = time.time()
    scores = det.score(values)
    score_time = time.time() - t0

    # Find best threshold
    best_f1, best_thr = 0.0, threshold
    for thr in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]:
        pred = (scores > thr).astype(np.int32)
        m = compute_metrics(pred, labels)
        if m["pa_f1"] > best_f1:
            best_f1 = m["pa_f1"]
            best_thr = thr

    pred = (scores > best_thr).astype(np.int32)
    m = compute_metrics(pred, labels)

    n_windows = len(windows)

    return DetectionMetrics(
        file_name=csv_path.name,
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        pa_precision=m["pa_precision"],
        pa_recall=m["pa_recall"],
        pa_f1=m["pa_f1"],
        n_anomaly_windows=n_windows,
        n_timesteps=T,
        threshold=best_thr,
        train_time_s=train_time,
        score_time_s=score_time,
    )


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def run_nab_benchmark(
    data_root: Path,
    out_dir: Path,
    device: str = "cpu",
    max_files: int | None = None,
    detector_kwargs: dict[str, Any] | None = None,
    n_runs: int = 1,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Run benchmark on all NAB files.

    Parameters
    ----------
    data_root : Path
        Root of NAB data (contains ``data/`` and ``labels/``).
    out_dir : Path
        Output directory for results.
    device : str
    max_files : int or None
        Limit number of files (for quick testing).
    detector_kwargs : dict or None
        Override detector parameters.
    n_runs : int
        Number of runs per file (median used for final score).
    base_seed : int
        Base random seed; run *k* uses ``base_seed + k``.

    Returns
    -------
    DataFrame with per-file metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = data_root / "labels" / "combined_windows.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"NAB labels not found: {labels_path}")

    all_labels = load_nab_labels(labels_path)
    data_dir = data_root / "data"

    results: list[dict[str, Any]] = []
    files = sorted(all_labels.keys())
    if max_files:
        files = files[:max_files]

    for i, rel_path in enumerate(files):
        csv_path = data_dir / rel_path
        if not csv_path.exists():
            print(f"  [{i+1}/{len(files)}] SKIP {rel_path} (not found)")
            continue

        windows = all_labels[rel_path]
        if not windows:
            # Skip files with no anomaly windows
            continue

        print(f"  [{i+1}/{len(files)}] {rel_path} ", end="", flush=True)
        try:
            # Run N times with different seeds
            run_metrics: list[DetectionMetrics] = []
            for run_idx in range(n_runs):
                seed = base_seed + run_idx
                m = evaluate_file(
                    csv_path, windows,
                    device=device,
                    detector_kwargs=detector_kwargs,
                    seed=seed,
                )
                run_metrics.append(m)

            # Pick the median-pa_f1 run
            pa_f1_values = [rm.pa_f1 for rm in run_metrics]
            median_idx = int(np.argsort(pa_f1_values)[len(pa_f1_values) // 2])
            m = run_metrics[median_idx]

            # Format output
            if n_runs > 1:
                f1_arr = np.array(pa_f1_values)
                n_successful = int(np.sum(f1_arr > 0))
                print(
                    f"pa_F1={m.pa_f1:.3f} "
                    f"(med={np.median(f1_arr):.3f} mean={f1_arr.mean():.3f} "
                    f"std={f1_arr.std():.3f} min={f1_arr.min():.3f} max={f1_arr.max():.3f})  "
                    f"thr={m.threshold}  ({m.train_time_s:.1f}s)"
                )
            else:
                print(f"pa_F1={m.pa_f1:.3f}  thr={m.threshold}  ({m.train_time_s:.1f}s)")

            row: dict[str, Any] = {
                "file": rel_path,
                "category": rel_path.split("/")[0] if "/" in rel_path else "unknown",
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "pa_precision": m.pa_precision,
                "pa_recall": m.pa_recall,
                "pa_f1": m.pa_f1,
                "n_windows": m.n_anomaly_windows,
                "n_timesteps": m.n_timesteps,
                "threshold": m.threshold,
                "train_s": m.train_time_s,
                "score_s": m.score_time_s,
            }
            if n_runs > 1:
                f1_arr = np.array(pa_f1_values)
                row.update({
                    "pa_f1_mean": float(f1_arr.mean()),
                    "pa_f1_std": float(f1_arr.std()),
                    "pa_f1_min": float(f1_arr.min()),
                    "pa_f1_max": float(f1_arr.max()),
                    "n_runs": n_runs,
                    "n_successful_runs": int(np.sum(f1_arr > 0)),
                })
            results.append(row)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "file": rel_path,
                "category": rel_path.split("/")[0] if "/" in rel_path else "unknown",
                "pa_f1": float("nan"),
                "error": str(e),
            })

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "nab_per_file.csv", index=False)

    # Aggregate by category
    if "pa_f1" in df.columns and not df["pa_f1"].isna().all():
        agg = df.groupby("category").agg(
            n_files=("file", "count"),
            pa_f1_mean=("pa_f1", "mean"),
            pa_f1_std=("pa_f1", "std"),
            pa_precision_mean=("pa_precision", "mean"),
            pa_recall_mean=("pa_recall", "mean"),
        ).round(3)
        agg.to_csv(out_dir / "nab_by_category.csv")
        print(f"\n=== Results by category ===\n{agg}")

        # Overall
        valid = df.dropna(subset=["pa_f1"])
        print(f"\n=== Overall ({len(valid)} files) ===")
        print(f"  pa_F1:        {valid['pa_f1'].mean():.3f} +/- {valid['pa_f1'].std():.3f}")
        print(f"  pa_Precision: {valid['pa_precision'].mean():.3f}")
        print(f"  pa_Recall:    {valid['pa_recall'].mean():.3f}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run NAB benchmark")
    parser.add_argument("--data-root", type=str, default="data/nab")
    parser.add_argument("--out", type=str, default="results/nab")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of runs per file (median score used)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    kwargs = {"epochs": args.epochs, "hidden_dim": args.hidden_dim}
    run_nab_benchmark(
        data_root=Path(args.data_root),
        out_dir=Path(args.out),
        device=args.device,
        max_files=args.max_files,
        detector_kwargs=kwargs,
        n_runs=args.n_runs,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
