"""internal_time — Adaptive temporal anomaly detection via learned internal time.

Core modules::

    from internal_time import TemporalGRUCell, StepOutput, SequenceOutput

Anomaly detection::

    from internal_time import TemporalAnomalyDetector

Visualization::

    from internal_time.viz import plot_anomaly_scores, plot_detail
"""

from .core import (
    InternalTimeHead,
    InputEncoder,
    SelfModel,
    SequenceOutput,
    StepOutput,
    TemporalGRUCell,
    TimeRegConfig,
    compute_time_reg,
)
from .anomaly import (
    DetectorConfig,
    StreamingScorer,
    TemporalAnomalyDetector,
)

__all__ = [
    # core
    "TemporalGRUCell",
    "InternalTimeHead",
    "SelfModel",
    "InputEncoder",
    "StepOutput",
    "SequenceOutput",
    "TimeRegConfig",
    "compute_time_reg",
    # anomaly
    "TemporalAnomalyDetector",
    "DetectorConfig",
    "StreamingScorer",
]
