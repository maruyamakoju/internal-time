"""Synthetic anomaly detection demo.

Generates a time series with normal behaviour, injects several anomaly
types, trains a TemporalAnomalyDetector on the normal portion, then
scores the full series.  Produces a figure showing that delta_tau spikes
precisely at anomaly locations.

Anomaly types injected:
  1. Level shift
  2. Spike burst
  3. Frequency change
  4. Variance explosion
  5. Trend reversal
"""

from pathlib import Path

import numpy as np

from internal_time import TemporalAnomalyDetector
from internal_time.viz import plot_anomaly_scores, plot_detail, plot_training_history


# ======================================================================
# 1. Generate synthetic data
# ======================================================================

def generate_normal(T: int, D: int = 3, seed: int = 42) -> np.ndarray:
    """Smooth multi-channel signal with sinusoidal components."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 8 * np.pi, T)
    data = np.zeros((T, D), dtype=np.float32)
    for d in range(D):
        freq = 1.0 + d * 0.5
        phase = rng.uniform(0, 2 * np.pi)
        data[:, d] = np.sin(freq * t + phase) + 0.1 * rng.randn(T)
    return data


def inject_anomalies(data: np.ndarray, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    """Inject anomalies into a copy of the data.  Returns (data, labels)."""
    rng = np.random.RandomState(seed)
    out = data.copy()
    T, D = out.shape
    labels = np.zeros(T, dtype=np.int32)

    # 1. Level shift
    a, b = 1200, 1250
    out[a:b, :] += 3.0
    labels[a:b] = 1

    # 2. Spike burst
    a, b = 1500, 1515
    out[a:b, 0] += rng.uniform(4, 8, size=b - a)
    labels[a:b] = 1

    # 3. Frequency change
    a, b = 1800, 1870
    t_seg = np.linspace(0, 6 * np.pi, b - a)
    for d in range(D):
        out[a:b, d] = 3.0 * np.sin(5.0 * t_seg) + 0.1 * rng.randn(b - a)
    labels[a:b] = 1

    # 4. Variance explosion
    a, b = 2100, 2150
    out[a:b, :] += rng.randn(b - a, D) * 4.0
    labels[a:b] = 1

    # 5. Trend reversal
    a, b = 2500, 2550
    ramp = np.linspace(0, 5, b - a)[:, None]
    out[a:b, :] += ramp
    labels[a:b] = 1

    return out, labels


# ======================================================================
# 2. Run detection pipeline
# ======================================================================

def main() -> None:
    print("=== Temporal Anomaly Detection Demo ===\n")

    # Generate data
    T_train = 2000  # normal-only training data (more = better model)
    T_total = 3000  # full series with anomalies
    D = 3

    normal_data = generate_normal(T_train, D, seed=42)
    full_normal = generate_normal(T_total, D, seed=42)
    test_data, true_labels = inject_anomalies(full_normal, seed=123)

    print(f"Training data:  {normal_data.shape}  (normal only)")
    print(f"Test data:      {test_data.shape}  (with {true_labels.sum()} anomaly steps)")

    # Create detector
    detector = TemporalAnomalyDetector(
        input_dim=D,
        hidden_dim=48,
        epochs=100,
        window_size=50,
        lr=3e-4,
        patience=20,
    )
    print(f"\n{detector}\n")

    # Train
    print("Training on normal data...")
    history = detector.fit(normal_data, verbose=True)

    # Score
    print("\nScoring test data...")
    scores = detector.score(test_data)
    detail = detector.score_detail(test_data)
    labels_pred = detector.detect(test_data, threshold=2.0)

    # Try multiple thresholds, pick best F1
    best_f1, best_thr = 0.0, 2.0
    for thr in [1.0, 1.5, 2.0, 2.5, 3.0]:
        lp = detector.detect(test_data, threshold=thr)
        tp = int(((lp == 1) & (true_labels == 1)).sum())
        fp = int(((lp == 1) & (true_labels == 0)).sum())
        fn = int(((lp == 0) & (true_labels == 1)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-8, p + r)
        print(f"  threshold={thr:.1f}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")
        if f > best_f1:
            best_f1, best_thr = f, thr

    labels_pred = detector.detect(test_data, threshold=best_thr)
    tp = int(((labels_pred == 1) & (true_labels == 1)).sum())
    fp = int(((labels_pred == 1) & (true_labels == 0)).sum())
    fn = int(((labels_pred == 0) & (true_labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    print(f"\nBest detection (threshold={best_thr}):")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")

    # Save model
    out_dir = Path("outputs/anomaly_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    detector.save(out_dir / "detector.pt")
    print(f"\nModel saved to {out_dir / 'detector.pt'}")

    # Figures
    fig1 = plot_anomaly_scores(
        test_data, scores,
        threshold=best_thr,
        labels_true=true_labels,
        feature_names=["sensor_0", "sensor_1", "sensor_2"],
        title="Temporal Anomaly Detection — Synthetic Demo",
    )
    fig1.savefig(out_dir / "anomaly_scores.png", dpi=150)
    print(f"Figure saved to {out_dir / 'anomaly_scores.png'}")

    fig2 = plot_detail(detail, data=test_data, title="Internal Time Signals")
    fig2.savefig(out_dir / "internal_signals.png", dpi=150)
    print(f"Figure saved to {out_dir / 'internal_signals.png'}")

    fig3 = plot_training_history(history)
    fig3.savefig(out_dir / "training_loss.png", dpi=150)
    print(f"Figure saved to {out_dir / 'training_loss.png'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
