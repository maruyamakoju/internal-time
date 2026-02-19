# internal-time

**Temporal anomaly detection via learned internal time.**

An AI agent learns its own internal clock. When something unexpected happens, the clock reacts — giving you a natural anomaly signal that captures *temporal surprise*, not just static outliers.

## Quick Start

```bash
pip install -e .
```

```python
from internal_time import TemporalAnomalyDetector

detector = TemporalAnomalyDetector(input_dim=3)
detector.fit(normal_data)           # train on normal time series
scores = detector.score(test_data)  # anomaly scores per timestep
```

## NAB Benchmark Results

Evaluated on all 52 labeled files from the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) (point-adjusted F1):

| Category | Files | pa-F1 (mean) | pa-Precision | pa-Recall |
|---|---|---|---|---|
| artificialWithAnomaly | 6 | **0.947** | 0.899 | 1.000 |
| realTweets | 10 | **0.887** | 0.875 | 0.924 |
| realKnownCause | 7 | **0.844** | 0.780 | 0.929 |
| realAWSCloudwatch | 16 | **0.836** | 0.797 | 0.917 |
| realTraffic | 7 | 0.722 | 0.708 | 0.845 |
| realAdExchange | 6 | 0.574 | 0.544 | 0.667 |
| **Overall** | **52** | **0.814** | **0.780** | **0.891** |

v1.1 improvements: RevIN (per-window instance normalization) + IQR-based robust calibration.
AWS +14.7%, AdExchange +19.9%, artificial +1.1%. Overall +3.9% vs v1.0 (0.775).

## How It Works

A GRU processes the time series step by step. At each step:

1. A **self-model** predicts the next hidden state from the current one
2. The **prediction error** measures how surprising the actual transition was
3. An **internal time head** outputs `delta_tau` — how fast the agent's internal clock should tick
4. The hidden state is mixed: `h_next = (1-alpha) * h_prev + alpha * h_candidate`

After training on normal data, the prediction error and delta_tau naturally spike at anomalous transitions — the model has learned what "normal temporal dynamics" look like, and anything else triggers surprise.

## Streaming Mode

Process data one observation at a time for real-time monitoring:

```python
scorer = detector.create_streaming_scorer(threshold=2.0)

for obs in live_data_stream:
    result = scorer.step(obs)
    if result["is_anomaly"]:
        alert(f"Anomaly detected! score={result['score']:.2f}")
```

## Core Module (for integration)

Use `TemporalGRUCell` directly in any PyTorch pipeline:

```python
from internal_time import TemporalGRUCell

cell = TemporalGRUCell(input_dim=10, hidden_dim=64, use_self_model=True)
h = cell.init_hidden(batch_size=32)

for t in range(seq_len):
    h, info = cell(x[:, t, :], h)
    # info.delta_tau  — internal time step
    # info.pred_error — self-model prediction error
```

Or process a full sequence:

```python
out = cell.forward_sequence(x_seq)  # (batch, seq_len, input_dim)
# out.delta_taus — (batch, seq_len)
# out.pred_errors — (batch, seq_len)
# out.self_model_loss — scalar, for training
```

## Examples

```bash
python examples/01_core_basics.py       # TemporalGRUCell usage
python examples/02_anomaly_synthetic.py  # Synthetic anomaly detection
python examples/03_streaming_demo.py     # Real-time streaming mode
```

## NAB Benchmark

```bash
python -m internal_time.benchmark --data-root data/nab --out results/nab --device cuda
```

## Package Structure

```
internal_time/          # Product package
  core.py               # TemporalGRUCell, InternalTimeHead, SelfModel
  anomaly.py            # TemporalAnomalyDetector, StreamingScorer
  viz.py                # Visualization utilities
  benchmark.py          # NAB evaluation

internal_time_rl/       # Research package (RL experiments)
  models/               # Policy, time module, encoder
  algorithms/           # PPO + internal time
  envs/                 # Gymnasium wrappers
  analysis/             # Experiment analysis tools
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.1
- NumPy >= 1.26
- matplotlib >= 3.8 (for visualization)
- pandas (for benchmark)

## Research Background

This project originated from research on adaptive temporal reparameterization for RL agents. The core insight — that a learned internal clock produces useful temporal surprise signals — turned out to be directly applicable to anomaly detection without requiring RL rewards at all. See `docs/research_plan.md` for the research context.
