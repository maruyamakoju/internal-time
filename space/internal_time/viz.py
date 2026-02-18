"""Visualization utilities for internal-time anomaly detection.

All functions return matplotlib Figure objects so they can be displayed
inline (Jupyter) or saved to files.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_mpl() -> None:
    if not HAS_MPL:
        raise ImportError("matplotlib is required for visualization.  pip install matplotlib")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_anomaly_scores(
    data: np.ndarray,
    scores: np.ndarray,
    threshold: float = 2.0,
    labels_true: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    title: str = "Temporal Anomaly Detection",
    figsize: tuple[float, float] = (14, 6),
) -> "Figure":
    """Plot time series with anomaly score overlay.

    Parameters
    ----------
    data : ndarray, shape ``(T,)`` or ``(T, D)``
        Raw input time series.
    scores : ndarray, shape ``(T,)``
        Anomaly scores (z-normalised).
    threshold : float
        Horizontal line on the score axis.
    labels_true : ndarray or None
        Ground-truth binary labels for shading anomaly regions.
    feature_names : list of str or None
        Names for each feature column.
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    _require_mpl()

    if data.ndim == 1:
        data = data[:, None]
    T, D = data.shape

    n_panels = D + 1  # one per feature + score panel
    fig, axes = plt.subplots(n_panels, 1, figsize=(figsize[0], figsize[1] * n_panels / 2),
                             sharex=True, squeeze=False)
    axes = axes.flatten()

    time = np.arange(T)

    # Shade true anomaly regions
    def shade_anomalies(ax: Any) -> None:
        if labels_true is not None:
            lbl = np.asarray(labels_true, dtype=bool)
            in_anomaly = False
            start = 0
            for i in range(len(lbl)):
                if lbl[i] and not in_anomaly:
                    start = i
                    in_anomaly = True
                elif not lbl[i] and in_anomaly:
                    ax.axvspan(start, i, alpha=0.15, color="red")
                    in_anomaly = False
            if in_anomaly:
                ax.axvspan(start, len(lbl), alpha=0.15, color="red")

    # Feature panels
    for d in range(D):
        ax = axes[d]
        name = feature_names[d] if feature_names and d < len(feature_names) else f"feature {d}"
        ax.plot(time, data[:, d], linewidth=0.8, color="#2563eb")
        ax.set_ylabel(name, fontsize=9)
        shade_anomalies(ax)
        ax.grid(True, alpha=0.3)

    # Score panel
    ax_score = axes[-1]
    color_map = np.where(scores > threshold, "#ef4444", "#6b7280")
    ax_score.bar(time, scores, width=1.0, color=color_map, alpha=0.7)
    ax_score.axhline(threshold, color="#ef4444", linestyle="--", linewidth=1, label=f"threshold={threshold}")
    ax_score.set_ylabel("anomaly score", fontsize=9)
    ax_score.set_xlabel("timestep")
    shade_anomalies(ax_score)
    ax_score.legend(fontsize=8)
    ax_score.grid(True, alpha=0.3)

    axes[0].set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_detail(
    detail: dict[str, np.ndarray],
    data: np.ndarray | None = None,
    title: str = "Internal Time Signals",
    figsize: tuple[float, float] = (14, 8),
) -> "Figure":
    """Plot all internal signals: delta_tau, pred_error, alpha, score.

    Parameters
    ----------
    detail : dict
        Output of :meth:`TemporalAnomalyDetector.score_detail`.
    data : ndarray or None
        Original time series (plotted at top if provided).
    """
    _require_mpl()

    signals = ["delta_tau", "pred_error", "alpha", "score"]
    available = [s for s in signals if s in detail]
    n_panels = len(available) + (1 if data is not None else 0)

    fig, axes = plt.subplots(n_panels, 1, figsize=(figsize[0], 2.5 * n_panels),
                             sharex=True, squeeze=False)
    axes = axes.flatten()

    idx = 0
    if data is not None:
        ax = axes[idx]
        if data.ndim == 1:
            ax.plot(data, linewidth=0.8, color="#2563eb")
        else:
            for d in range(min(data.shape[1], 5)):
                ax.plot(data[:, d], linewidth=0.8, alpha=0.8)
        ax.set_ylabel("input", fontsize=9)
        ax.grid(True, alpha=0.3)
        idx += 1

    colors = {"delta_tau": "#f59e0b", "pred_error": "#ef4444", "alpha": "#8b5cf6", "score": "#10b981"}

    for s in available:
        ax = axes[idx]
        c = colors.get(s, "#6b7280")
        ax.plot(detail[s], linewidth=0.8, color=c)
        ax.set_ylabel(s, fontsize=9)
        ax.grid(True, alpha=0.3)
        idx += 1

    axes[0].set_title(title, fontsize=12, fontweight="bold")
    axes[-1].set_xlabel("timestep")
    fig.tight_layout()
    return fig


def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training Loss",
    figsize: tuple[float, float] = (8, 4),
) -> "Figure":
    """Plot training and validation loss curves."""
    _require_mpl()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="train", color="#2563eb")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val", color="#f59e0b")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
