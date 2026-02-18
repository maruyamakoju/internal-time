"""
Gradio demo for internal-time anomaly detector.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr

from internal_time import TemporalAnomalyDetector
from internal_time.viz import plot_anomaly_scores, plot_detail


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _base_signal(T: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T)
    x = np.sin(t) + 0.5 * np.sin(2.3 * t) + 0.15 * rng.standard_normal(T)
    return x.astype(np.float32)


def make_synthetic(anomaly_type: str, T: int = 600, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (values [T,1], labels [T]) for the chosen anomaly type."""
    rng = np.random.default_rng(seed)
    x = _base_signal(T, seed)
    labels = np.zeros(T, dtype=np.int32)

    if anomaly_type == "Spike":
        positions = rng.integers(T // 2, T, size=3)
        for p in positions:
            x[p] += rng.choice([-1, 1]) * rng.uniform(3, 5)
            labels[max(0, p - 2) : p + 3] = 1

    elif anomaly_type == "Step (mean shift)":
        start, end = T // 2, T // 2 + 80
        x[start:end] += 2.5
        labels[start:end] = 1

    elif anomaly_type == "Oscillation (freq change)":
        start, end = T // 2, T // 2 + 100
        t2 = np.linspace(0, 12 * np.pi, end - start)
        x[start:end] += 1.2 * np.sin(t2)
        labels[start:end] = 1

    elif anomaly_type == "Variance burst":
        start, end = T // 2, T // 2 + 80
        x[start:end] += rng.standard_normal(end - start) * 1.8
        labels[start:end] = 1

    elif anomaly_type == "Mixed":
        p = T // 2 + 50
        x[p] += 4.0
        labels[max(0, p - 2) : p + 3] = 1
        s, e = T // 2 + 150, T // 2 + 230
        x[s:e] += 2.2
        labels[s:e] = 1

    return x.reshape(-1, 1), labels


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def run_detection(
    values: np.ndarray,
    labels: np.ndarray | None,
    epochs: int,
    threshold: float,
    train_ratio: float = 0.5,
) -> tuple[plt.Figure, plt.Figure, str]:
    T = len(values)
    split = int(T * train_ratio)
    train_data = values[:split]

    det = TemporalAnomalyDetector(
        input_dim=values.shape[1],
        hidden_dim=48,
        epochs=epochs,
        window_size=min(50, max(10, split // 5)),
        patience=12,
        lr=3e-4,
    )
    det.fit(train_data, verbose=False)

    scores = det.score(values)
    detail = det.score_detail(values)
    preds = (scores > threshold).astype(np.int32)

    fig1 = plot_anomaly_scores(
        data=values,
        scores=scores,
        threshold=threshold,
        labels_true=labels,
        title="Anomaly Detection — internal-time",
        figsize=(12, 5),
    )

    fig2 = plot_detail(
        detail=detail,
        data=values,
        title="Internal Signals (\u0394\u03c4, pred_error, \u03b1)",
        figsize=(12, 5),
    )

    lines = []
    if labels is not None and labels.sum() > 0:
        from internal_time.benchmark import compute_metrics
        m = compute_metrics(preds, labels)
        lines += [
            f"**Point-wise F1**: {m['f1']:.3f}  (precision {m['precision']:.3f}, recall {m['recall']:.3f})",
            f"**Point-adjusted F1**: {m['pa_f1']:.3f}  (precision {m['pa_precision']:.3f}, recall {m['pa_recall']:.3f})",
        ]
    lines += [
        f"**Score range**: [{scores.min():.2f}, {scores.max():.2f}]",
        f"**Flagged timesteps**: {int(preds.sum())} / {T}",
    ]
    stats = "\n\n".join(lines)
    return fig1, fig2, stats


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def demo_callback(
    anomaly_type: str,
    epochs: int,
    threshold: float,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    progress(0.0, desc="Generating synthetic data…")
    values, labels = make_synthetic(anomaly_type)
    progress(0.1, desc="Training detector…")
    fig1, fig2, stats = run_detection(values, labels, epochs=int(epochs), threshold=float(threshold))
    progress(1.0, desc="Done")
    return fig1, fig2, stats


def upload_callback(
    file,
    epochs: int,
    threshold: float,
    label_col: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    if file is None:
        return None, None, "Please upload a CSV file."

    progress(0.0, desc="Reading CSV…")
    try:
        path = file.name if hasattr(file, "name") else file
        df = pd.read_csv(path)
    except Exception as e:
        return None, None, f"Error reading CSV: {e}"

    label_col = (label_col or "").strip()
    labels = None
    skip = {"timestamp"}
    if label_col and label_col in df.columns:
        labels = df[label_col].values.astype(np.int32)
        skip.add(label_col)

    value_cols = [c for c in df.columns if c not in skip]
    if not value_cols:
        return None, None, "No value columns found."

    values = df[value_cols].values.astype(np.float32)

    progress(0.1, desc="Training detector…")
    try:
        fig1, fig2, stats = run_detection(
            values, labels, epochs=int(epochs), threshold=float(threshold)
        )
    except Exception as e:
        return None, None, f"Detection error: {e}"

    progress(1.0, desc="Done")
    return fig1, fig2, stats


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

DESCRIPTION = """
# Internal Time — Temporal Anomaly Detector

An AI agent learns its own **internal clock**. When something unexpected happens,
the clock reacts — giving a natural anomaly signal that captures *temporal surprise*,
not just static outliers.

- **pa-F1 0.775** overall on the [NAB benchmark](https://github.com/numenta/NAB) (52 real-world files)
- Works on any univariate or multivariate time series
- No labels required — fully self-supervised
"""

ANOMALY_TYPES = [
    "Spike",
    "Step (mean shift)",
    "Oscillation (freq change)",
    "Variance burst",
    "Mixed",
]

with gr.Blocks(theme=gr.themes.Soft(), title="Internal Time") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ---- Tab 1: Quick Demo ----------------------------------------
        with gr.Tab("Quick Demo"):
            gr.Markdown(
                "Select an anomaly type and click **Run**. "
                "The model trains on the first 50% (normal) then scores the full sequence."
            )
            with gr.Row():
                atype = gr.Dropdown(ANOMALY_TYPES, value="Spike", label="Anomaly type")
                epochs_s = gr.Slider(10, 100, value=40, step=5, label="Training epochs")
                thr_s = gr.Slider(0.5, 4.0, value=2.0, step=0.1, label="Threshold (\u03c3)")
                run_btn = gr.Button("Run", variant="primary")

            stats_out = gr.Markdown()
            fig_main = gr.Plot(label="Time series + anomaly scores")
            fig_detail = gr.Plot(label="Internal signals (\u0394\u03c4, pred_error, \u03b1)")

            run_btn.click(
                fn=demo_callback,
                inputs=[atype, epochs_s, thr_s],
                outputs=[fig_main, fig_detail, stats_out],
            )

        # ---- Tab 2: Upload CSV ----------------------------------------
        with gr.Tab("Upload Your CSV"):
            gr.Markdown(
                "Upload any CSV with numeric columns. "
                "Optional `timestamp` column is ignored. "
                "Optional binary label column (0/1) enables metric display."
            )
            with gr.Row():
                csv_file = gr.File(label="CSV file", file_types=[".csv"])
                label_col_in = gr.Textbox(
                    label="Label column name (optional)", placeholder="e.g. label"
                )
            with gr.Row():
                epochs_u = gr.Slider(10, 100, value=40, step=5, label="Training epochs")
                thr_u = gr.Slider(0.5, 4.0, value=2.0, step=0.1, label="Threshold (\u03c3)")
                run_btn_u = gr.Button("Run", variant="primary")

            stats_out_u = gr.Markdown()
            fig_main_u = gr.Plot(label="Time series + anomaly scores")
            fig_detail_u = gr.Plot(label="Internal signals")

            run_btn_u.click(
                fn=upload_callback,
                inputs=[csv_file, epochs_u, thr_u, label_col_in],
                outputs=[fig_main_u, fig_detail_u, stats_out_u],
            )

    gr.Markdown(
        "---\n"
        "**How it works**: A GRU processes the sequence step by step. "
        "A *self-model* predicts each next hidden state. "
        "When the prediction error is high, the agent's internal clock Δτ reacts — "
        "flagging *temporal surprise*. Trained on normal data only; no labels needed.\n\n"
        "MIT License"
    )


if __name__ == "__main__":
    demo.launch(show_error=True)
