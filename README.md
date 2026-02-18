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
| artificialWithAnomaly | 6 | **0.936** | 0.886 | 1.000 |
| realTweets | 10 | **0.926** | 0.887 | 0.972 |
| realTraffic | 7 | **0.893** | 0.849 | 0.964 |
| realKnownCause | 7 | **0.846** | 0.891 | 0.857 |
| realAWSCloudwatch | 16 | 0.689 | 0.678 | 0.781 |
| realAdExchange | 6 | 0.375 | 0.446 | 0.417 |
| **Overall** | **52** | **0.775** | **0.767** | **0.836** |

Highlights: nyc_taxi 0.932, Twitter volume 0.926 avg, traffic 0.893 avg.

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
