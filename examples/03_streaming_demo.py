"""Streaming / online anomaly scoring demo.

Demonstrates how to use the StreamingScorer for real-time, one-step-at-a-time
anomaly detection.  The demo:

  1. Trains a TemporalAnomalyDetector on normal synthetic data.
  2. Creates a StreamingScorer from the fitted detector.
  3. Feeds test data (with injected anomalies) one observation at a time.
  4. Prints alerts when anomalies are detected.
  5. Verifies that streaming scores are consistent with batch scores.
"""

import numpy as np

from internal_time import TemporalAnomalyDetector, StreamingScorer


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

    # Level shift at steps 300-320
    out[300:320, :] += 4.0
    labels[300:320] = 1

    # Spike burst at steps 500-510
    out[500:510, 0] += rng.uniform(5, 9, size=10)
    labels[500:510] = 1

    # Variance explosion at steps 700-730
    out[700:730, :] += rng.randn(30, D) * 5.0
    labels[700:730] = 1

    return out, labels


# ======================================================================
# 2. Main demo
# ======================================================================

def main() -> None:
    print("=== Streaming Anomaly Detection Demo ===\n")

    D = 3
    T_train = 1000
    T_test = 800

    # Generate data
    train_data = generate_normal(T_train, D, seed=42)
    test_normal = generate_normal(T_test, D, seed=99)
    test_data, true_labels = inject_anomalies(test_normal, seed=123)

    print(f"Training data:  {train_data.shape}  (normal only)")
    print(f"Test data:      {test_data.shape}  ({true_labels.sum()} anomaly steps)\n")

    # ------------------------------------------------------------------
    # Train detector
    # ------------------------------------------------------------------
    detector = TemporalAnomalyDetector(
        input_dim=D,
        hidden_dim=48,
        epochs=60,
        window_size=40,
        lr=3e-4,
        patience=15,
        score_mode="pred_error",
    )
    print("Training on normal data...")
    detector.fit(train_data, verbose=False)
    print("Training complete.\n")

    # ------------------------------------------------------------------
    # Batch scoring (reference)
    # ------------------------------------------------------------------
    batch_scores = detector.score(test_data)

    # ------------------------------------------------------------------
    # Streaming scoring
    # ------------------------------------------------------------------
    scorer = detector.create_streaming_scorer(threshold=2.0)
    print(f"Created: {scorer}\n")

    streaming_scores = []
    alerts = []

    for t in range(T_test):
        result = scorer.step(test_data[t])
        streaming_scores.append(result["score"])

        if result["is_anomaly"]:
            alerts.append(t)
            # Only print a subset to keep output manageable
            if len(alerts) <= 20 or len(alerts) % 10 == 0:
                print(
                    f"  ALERT  t={t:>4d}  score={result['score']:+.3f}  "
                    f"delta_tau={result['delta_tau']:.4f}  "
                    f"pred_error={result['pred_error']:.4f}"
                )

    streaming_scores = np.array(streaming_scores)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n--- Summary ---")
    print(f"Total steps processed:  {T_test}")
    print(f"Anomaly alerts raised:  {len(alerts)}")
    print(f"True anomaly steps:     {int(true_labels.sum())}")

    # Detection quality on streaming alerts
    pred_labels = np.zeros(T_test, dtype=np.int32)
    for t in alerts:
        pred_labels[t] = 1

    tp = int(((pred_labels == 1) & (true_labels == 1)).sum())
    fp = int(((pred_labels == 1) & (true_labels == 0)).sum())
    fn = int(((pred_labels == 0) & (true_labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    print(f"\nStreaming detection quality:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")

    # ------------------------------------------------------------------
    # Compare streaming vs batch scores
    # ------------------------------------------------------------------
    # Note: streaming and batch scores will NOT be identical because:
    #   - Batch mode uses forward_sequence (processes all steps, then
    #     computes score_mode-specific normalization over the full series).
    #   - Streaming mode processes one step at a time and uses pred_error
    #     as the raw signal (no sequence-level normalization possible).
    #
    # However, both should agree on *where* anomalies are: the correlation
    # between the two score series should be positive.
    correlation = float(np.corrcoef(streaming_scores, batch_scores)[0, 1])
    print(f"\nStreaming vs batch score correlation: {correlation:.3f}")
    if correlation > 0.3:
        print("  Scores are positively correlated — streaming and batch agree on anomaly locations.")
    else:
        print("  Correlation is low — this can happen with short series or 'combined' score mode.")

    # ------------------------------------------------------------------
    # Demonstrate reset
    # ------------------------------------------------------------------
    print("\n--- Reset demo ---")
    scorer.reset()
    print(f"After reset: {scorer}")

    # Re-process first 5 steps
    for t in range(5):
        result = scorer.step(test_data[t])
        print(
            f"  t={t}  score={result['score']:+.3f}  "
            f"is_anomaly={result['is_anomaly']}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
