"""Temporal anomaly detection powered by learned internal time.

The key insight: a self-model learns to predict normal temporal dynamics.
When it encounters unexpected transitions, prediction error rises and the
internal time head produces high ``delta_tau`` — a natural anomaly score
that captures *temporal surprise*, not just static outliers.

Usage::

    import numpy as np
    from internal_time import TemporalAnomalyDetector

    # Train on normal data
    detector = TemporalAnomalyDetector(input_dim=3)
    detector.fit(normal_data)  # shape (T, 3) or (N, T, 3)

    # Score new data
    scores = detector.score(test_data)

    # Detect anomalies
    labels = detector.detect(test_data, threshold=2.0)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .core import TemporalGRUCell, TimeRegConfig, compute_time_reg


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """All knobs for :class:`TemporalAnomalyDetector`."""

    hidden_dim: int = 64
    embed_dim: int | None = None
    encoder_hidden_dims: tuple[int, ...] = (64, 64)
    min_tau: float = 1e-4

    # training
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-5
    window_size: int = 50
    stride: int | None = None  # defaults to window_size // 4
    batch_size: int = 32
    val_ratio: float = 0.1
    patience: int = 20  # early stopping patience (0 = off)
    jitter_std: float = 0.01  # Gaussian noise augmentation during training

    # normalization
    revin: bool = True
    """Per-window (instance) normalization.

    When True, each sliding window is independently normalised to zero mean
    and unit variance *before* being fed to the model.  This makes the model
    robust to non-stationary data (e.g. financial, cloud metrics) where the
    global mean/std of the training set may differ from the test period.
    """

    # regularization (light defaults — for anomaly detection we *want*
    # delta_tau to vary, so time_reg should be weaker than RL defaults)
    time_reg: TimeRegConfig = field(
        default_factory=lambda: TimeRegConfig(lambda_var=1e-4, lambda_mean=0.0)
    )
    lambda_self: float = 1.0  # weight for self-model loss

    # scoring
    score_mode: str = "combined"
    """One of ``'delta_tau'``, ``'pred_error'``, ``'combined'``.

    * ``'pred_error'``: self-model prediction error (high = unusual dynamics).
    * ``'delta_tau'``: absolute deviation of internal time step from its mean
      (works in both directions — unusually fast or slow processing).
    * ``'combined'``: mean of z-normalised pred_error and |delta_tau deviation|.
    """

    score_smoothing: int = 5
    """Rolling-average window for output scores.  1 = no smoothing."""


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def _to_windows(
    data: np.ndarray,
    window_size: int,
    stride: int,
    revin: bool = False,
) -> np.ndarray:
    """Slide a window over a 2-D array and return ``(N, W, D)``.

    Parameters
    ----------
    revin : bool
        If True, normalise each window by its own mean and std.
    """
    T, D = data.shape
    starts = list(range(0, T - window_size + 1, stride))
    if not starts:
        starts = [0]
    windows = np.stack([data[s : s + window_size] for s in starts])  # (N, W, D)

    if revin:
        # Per-window normalisation: each window to zero-mean, unit-variance
        wmean = windows.mean(axis=1, keepdims=True)       # (N, 1, D)
        wstd  = windows.std(axis=1, keepdims=True) + 1e-8
        windows = (windows - wmean) / wstd

    return windows


def _prepare_data(
    raw: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Coerce input to ``(T, D)`` float32 numpy array."""
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().numpy()
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim == 3:
        # (N, T, D) → concatenate along time for simplicity
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2:
        raise ValueError(f"Expected 1-D, 2-D, or 3-D input, got shape {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TemporalAnomalyDetector:
    """Anomaly detector driven by learned internal time.

    Parameters
    ----------
    input_dim : int
        Number of features per timestep.
    config : DetectorConfig or None
        Full configuration.  Individual kwargs override config fields.
    device : str
        PyTorch device string.
    **kwargs
        Overrides for :class:`DetectorConfig` fields.
    """

    def __init__(
        self,
        input_dim: int,
        config: DetectorConfig | None = None,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.input_dim = input_dim
        self.cfg = config or DetectorConfig()
        # apply overrides
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

        self.device = torch.device(device)
        self.model = TemporalGRUCell(
            input_dim=input_dim,
            hidden_dim=self.cfg.hidden_dim,
            embed_dim=self.cfg.embed_dim,
            encoder_hidden_dims=self.cfg.encoder_hidden_dims,
            use_self_model=True,
            min_tau=self.cfg.min_tau,
        ).to(self.device)

        # Fitted state
        self._fitted = False
        self._train_mean: np.ndarray | None = None
        self._train_std: np.ndarray | None = None
        self._score_mean: float = 0.0
        self._score_std: float = 1.0
        self._train_tau_mean: float | None = None   # global tau ref for Fix B
        self._history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        if self._train_mean is not None and self._train_std is not None:
            return (data - self._train_mean) / (self._train_std + 1e-8)
        return data

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray | torch.Tensor,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train on data assumed to be *normal* (non-anomalous).

        Parameters
        ----------
        data : array-like, shape ``(T, D)`` or ``(T,)``
            Training time series.
        verbose : bool
            Print progress.

        Returns
        -------
        dict with ``'train_loss'`` and ``'val_loss'`` histories.
        """
        arr = _prepare_data(data)
        if arr.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {arr.shape[-1]}"
            )

        # Fit global normalisation (robust: median + IQR-sigma)
        self._train_mean = np.median(arr, axis=0).astype(np.float32)
        iqr = (np.percentile(arr, 75, axis=0) - np.percentile(arr, 25, axis=0)).astype(np.float32)
        self._train_std = (iqr / 1.3489 + 1e-8).astype(np.float32)
        arr = self._normalize(arr).astype(np.float32)

        # Window (with optional per-window RevIN)
        stride = self.cfg.stride or max(1, self.cfg.window_size // 4)
        windows = _to_windows(arr, self.cfg.window_size, stride, revin=self.cfg.revin)
        n_windows = len(windows)

        # Train / val split
        n_val = max(1, int(n_windows * self.cfg.val_ratio))
        n_train = n_windows - n_val
        perm = np.random.permutation(n_windows)
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        train_windows = torch.tensor(windows[train_idx], device=self.device)
        val_windows = torch.tensor(windows[val_idx], device=self.device)

        self._history = {"train_loss": [], "val_loss": []}

        # ---- Phase 1: stable self-model training ----
        phase1_epochs = max(1, int(self.cfg.epochs * 0.7))
        phase2_epochs = self.cfg.epochs - phase1_epochs

        if verbose:
            print(f"  phase 1 ({phase1_epochs} ep): learning stable representations")

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        best_val = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, phase1_epochs + 1):
            avg_train = self._train_epoch(
                train_windows, n_train, optimizer, detach_temporal_mix=True
            )
            val_loss = self._val_epoch(val_windows, detach_temporal_mix=True)
            self._history["train_loss"].append(avg_train)
            self._history["val_loss"].append(val_loss)

            if verbose and (epoch <= 3 or epoch % max(1, phase1_epochs // 10) == 0):
                print(f"  [{epoch:>4d}/{phase1_epochs}]  train={avg_train:.6f}  val={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if self.cfg.patience > 0 and patience_counter >= self.cfg.patience:
                    if verbose:
                        print(f"  early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # ---- Phase 2: fine-tune with time modulation active ----
        if phase2_epochs > 0:
            if verbose:
                print(f"  phase 2 ({phase2_epochs} ep): fine-tuning temporal modulation")

            optimizer_ft = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.lr * 0.3,
                weight_decay=self.cfg.weight_decay,
            )
            best_val_p2 = float("inf")
            best_state_p2 = {k: v.clone() for k, v in self.model.state_dict().items()}
            patience_counter = 0

            for epoch in range(1, phase2_epochs + 1):
                avg_train = self._train_epoch(
                    train_windows, n_train, optimizer_ft, detach_temporal_mix=False
                )
                val_loss = self._val_epoch(val_windows, detach_temporal_mix=False)
                self._history["train_loss"].append(avg_train)
                self._history["val_loss"].append(val_loss)

                if verbose and (epoch <= 2 or epoch % max(1, phase2_epochs // 5) == 0):
                    print(f"  [{epoch:>4d}/{phase2_epochs}]  train={avg_train:.6f}  val={val_loss:.6f}  (ft)")

                if val_loss < best_val_p2:
                    best_val_p2 = val_loss
                    best_state_p2 = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if self.cfg.patience > 0 and patience_counter >= self.cfg.patience:
                        if verbose:
                            print(f"  early stopping at epoch {epoch} (ft)")
                        break

            self.model.load_state_dict(best_state_p2)

        # Calibrate score distribution from training data
        self._calibrate_scores(arr)
        self._fitted = True

        return self._history

    def _train_epoch(
        self,
        train_windows: torch.Tensor,
        n_train: int,
        optimizer: torch.optim.Optimizer,
        detach_temporal_mix: bool,
    ) -> float:
        self.model.train()
        order = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, self.cfg.batch_size):
            idx = order[start : start + self.cfg.batch_size]
            batch = train_windows[idx]
            if self.cfg.jitter_std > 0:
                batch = batch + torch.randn_like(batch) * self.cfg.jitter_std
            out = self.model.forward_sequence(batch, detach_temporal_mix=detach_temporal_mix)
            loss = (
                self.cfg.lambda_self * out.self_model_loss
                + compute_time_reg(out.delta_taus, self.cfg.time_reg)
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        return epoch_loss / max(1, n_batches)

    def _val_epoch(
        self,
        val_windows: torch.Tensor,
        detach_temporal_mix: bool,
    ) -> float:
        self.model.eval()
        with torch.no_grad():
            out = self.model.forward_sequence(val_windows, detach_temporal_mix=detach_temporal_mix)
            return (
                self.cfg.lambda_self * out.self_model_loss
                + compute_time_reg(out.delta_taus, self.cfg.time_reg)
            ).item()

    def _calibrate_scores(self, arr_normed: np.ndarray) -> None:
        """Calibrate score statistics from training data using robust IQR-based stats."""
        raw_scores = self._raw_score(arr_normed)
        q25, q50, q75 = np.percentile(raw_scores, [25, 50, 75])
        iqr = q75 - q25
        self._score_mean = float(q50)
        # IQR / 1.3489 ≈ σ for a normal distribution — robust to outliers
        self._score_std = float(max(iqr / 1.3489, 1e-8))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _window_scores(
        self,
        chunk: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the model on a single chunk and return (tau, pred_error, alpha)."""
        self.model.eval()
        x = torch.tensor(chunk[None], device=self.device)
        with torch.no_grad():
            out = self.model.forward_sequence(x)
        tau = out.delta_taus.squeeze(0).cpu().numpy()
        pe  = out.pred_errors.squeeze(0).cpu().numpy()
        alp = out.alphas.squeeze(0).cpu().numpy()
        return tau, pe, alp

    def _combine_signals(
        self,
        tau: np.ndarray,
        pe: np.ndarray,
        tau_ref: float | None = None,
    ) -> np.ndarray:
        """Combine tau deviation and pred_error into a single raw score.

        Parameters
        ----------
        tau_ref : float or None
            When provided, use as the reference centre for tau deviation
            (Fix B — global tau mean from training).  When None, falls back
            to the within-window mean.
        """
        center = tau_ref if tau_ref is not None else tau.mean()
        if self.cfg.score_mode == "pred_error":
            return pe
        elif self.cfg.score_mode == "delta_tau":
            tau_dev = np.abs(tau - center)
            return tau_dev
        elif self.cfg.score_mode == "combined":
            tau_dev = np.abs(tau - center)
            tau_z = tau_dev / (tau_dev.std() + 1e-8)
            pe_z  = pe / (pe.std() + 1e-8)
            return (tau_z + pe_z) / 2.0
        else:
            raise ValueError(f"Unknown score_mode: {self.cfg.score_mode}")

    def _raw_score(self, arr_normed: np.ndarray) -> np.ndarray:
        """Per-timestep raw anomaly score on globally-normalised data.

        When ``revin=True``, uses overlapping windowed scoring where each
        window is independently normalised (RevIN).  This handles non-
        stationary data where the test period may differ from training.

        When ``revin=False``, runs the full sequence in one pass (original
        behaviour, faster but sensitive to distribution shift).
        """
        if self.cfg.revin:
            return self._raw_score_windowed(arr_normed)
        return self._raw_score_full(arr_normed)

    def _raw_score_full(self, arr_normed: np.ndarray) -> np.ndarray:
        """Original single-pass full-sequence scoring."""
        tau, pe, _ = self._window_scores(arr_normed)
        raw = self._combine_signals(tau, pe)
        if self.cfg.score_smoothing > 1:
            kernel = np.ones(self.cfg.score_smoothing) / self.cfg.score_smoothing
            raw = np.convolve(raw, kernel, mode="same")
        return raw

    def _run_windowed_pass(
        self,
        arr_normed: np.ndarray,
        window_size: int,
        tau_ref: float | None,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """Single windowed scoring pass at a given scale.

        Parameters
        ----------
        arr_normed : ndarray, shape (T, D)
        window_size : int
        tau_ref : float or None
            Passed through to ``_combine_signals`` for Fix B.

        Returns
        -------
        score_max : ndarray (T,)   — raw max-aggregated score, un-smoothed
        tau_global_mean : float    — mean tau across all windows (for calibration)
        last_alpha : ndarray (T,)
        last_tau   : ndarray (T,)
        last_pe    : ndarray (T,)
        """
        T = len(arr_normed)
        stride = max(1, window_size // 4)

        score_max = np.full(T, -1e9, dtype=np.float32)
        alpha_sum = np.zeros(T, dtype=np.float32)
        alpha_cnt = np.zeros(T, dtype=np.int32)
        tau_sum   = np.zeros(T, dtype=np.float32)
        pe_sum    = np.zeros(T, dtype=np.float32)

        # Build window start positions that cover every timestep
        starts = list(range(0, T - window_size + 1, stride))
        if not starts or starts[-1] + window_size < T:
            starts.append(max(0, T - window_size))
        starts = sorted(set(starts))

        # Batch windows for efficient GPU/CPU use
        batch_size = 64
        window_list = []
        for s in starts:
            end = s + window_size
            chunk = arr_normed[s:end]
            if len(chunk) < window_size:
                pad = np.zeros((window_size - len(chunk), arr_normed.shape[1]), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)
            # RevIN: per-window normalisation
            c_mean = chunk.mean(axis=0, keepdims=True)
            c_std  = chunk.std(axis=0, keepdims=True) + 1e-8
            window_list.append((s, (chunk - c_mean) / c_std))

        all_tau_vals: list[np.ndarray] = []

        self.model.eval()
        with torch.no_grad():
            for b in range(0, len(window_list), batch_size):
                batch_starts  = [w[0] for w in window_list[b : b + batch_size]]
                batch_windows = np.stack([w[1] for w in window_list[b : b + batch_size]])
                x = torch.tensor(batch_windows, device=self.device)
                out = self.model.forward_sequence(x)

                taus  = out.delta_taus.cpu().numpy()   # (B, W)
                pes   = out.pred_errors.cpu().numpy()  # (B, W)
                alps  = out.alphas.cpu().numpy()       # (B, W)

                for j, s in enumerate(batch_starts):
                    end = min(s + window_size, T)
                    L   = end - s
                    tau_w = taus[j, :L]
                    pe_w  = pes[j, :L]
                    alp_w = alps[j, :L]
                    all_tau_vals.append(tau_w)
                    local = self._combine_signals(tau_w, pe_w, tau_ref=tau_ref)
                    score_max[s:end] = np.maximum(score_max[s:end], local)
                    alpha_sum[s:end] += alp_w
                    alpha_cnt[s:end] += 1
                    tau_sum[s:end]   += tau_w
                    pe_sum[s:end]    += pe_w

        score_max[score_max == -1e9] = 0.0  # guard against uncovered positions

        # Global tau mean across all windows for Fix B calibration
        tau_global_mean = float(np.concatenate(all_tau_vals).mean()) if all_tau_vals else 0.0

        cnt = np.maximum(alpha_cnt, 1)
        last_alpha = alpha_sum / cnt
        last_tau   = tau_sum   / cnt
        last_pe    = pe_sum    / cnt

        return score_max, tau_global_mean, last_alpha, last_tau, last_pe

    def _raw_score_windowed(self, arr_normed: np.ndarray) -> np.ndarray:
        """Windowed scoring with per-window RevIN normalisation.

        Runs at two scales (Fix A) and uses a global tau reference (Fix B):
        - Small-window pass at ``cfg.window_size``
        - Large-window pass at ``4 × cfg.window_size`` (if long enough)
        - Per-timestep score = max(small, large)
        - Smoothing applied once after fusion
        """
        T = len(arr_normed)
        small_ws = self.cfg.window_size
        tau_ref = self._train_tau_mean   # None during first (calibration) call

        # Fix A: large-window pass — always use within-window tau mean (tau_ref=None).
        # Using the small-window tau_ref here would cause systematic score inflation
        # because large-window tau values live on a different scale.
        large_ws = min(small_ws * 4, T // 3)
        if large_ws >= small_ws * 2:
            score_large, _, _, _, _ = self._run_windowed_pass(arr_normed, large_ws, tau_ref=None)
        else:
            score_large = None

        # Small-window pass (also sets the per-timestep detail signals)
        score_small, tau_global_mean, last_alpha, last_tau, last_pe = \
            self._run_windowed_pass(arr_normed, small_ws, tau_ref)

        # Fix B: cache training tau mean after the first (calibration) call
        if self._train_tau_mean is None:
            self._train_tau_mean = tau_global_mean

        # Fuse scales
        raw = np.maximum(score_small, score_large) if score_large is not None else score_small

        # Smoothing — applied once after fusion
        if self.cfg.score_smoothing > 1:
            kernel = np.ones(self.cfg.score_smoothing) / self.cfg.score_smoothing
            raw = np.convolve(raw, kernel, mode="same")

        # Store fine-grained signals for score_detail
        self._last_alpha = last_alpha
        self._last_tau   = last_tau
        self._last_pe    = last_pe

        return raw

    def score(self, data: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return z-normalised anomaly scores for each timestep.

        Scores are calibrated so that ``score ~ 0`` means normal
        (relative to training data) and high positive values indicate
        anomalies.

        Parameters
        ----------
        data : array-like, shape ``(T, D)`` or ``(T,)``

        Returns
        -------
        scores : ndarray, shape ``(T,)``
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .score()")
        arr = _prepare_data(data)
        arr = self._normalize(arr)
        raw = self._raw_score(arr)
        return (raw - self._score_mean) / self._score_std

    def score_detail(self, data: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        """Return detailed per-step signals.

        Returns
        -------
        dict with keys ``'score'``, ``'delta_tau'``, ``'pred_error'``,
        ``'alpha'``.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .score_detail()")
        arr = _prepare_data(data)
        arr = self._normalize(arr)

        # _raw_score populates _last_{alpha,tau,pe} as a side-effect
        scores = self.score(data)

        if self.cfg.revin:
            # Populated by _raw_score_windowed
            delta_tau  = getattr(self, "_last_tau",   np.zeros(len(arr)))
            pred_error = getattr(self, "_last_pe",    np.zeros(len(arr)))
            alpha      = getattr(self, "_last_alpha", np.zeros(len(arr)))
        else:
            # Re-run full sequence to get signals
            x = torch.tensor(arr, device=self.device).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                out = self.model.forward_sequence(x)
            delta_tau  = out.delta_taus.squeeze(0).cpu().numpy()
            pred_error = out.pred_errors.squeeze(0).cpu().numpy()
            alpha      = out.alphas.squeeze(0).cpu().numpy()

        return {
            "score":       scores,
            "delta_tau":   delta_tau,
            "pred_error":  pred_error,
            "alpha":       alpha,
        }

    def detect(
        self,
        data: np.ndarray | torch.Tensor,
        threshold: float = 2.0,
    ) -> np.ndarray:
        """Return binary anomaly labels (1 = anomaly).

        Parameters
        ----------
        data : array-like
        threshold : float
            Z-score threshold.  Default 2.0 means scores > 2 std above
            training mean are flagged.

        Returns
        -------
        labels : ndarray of int, shape ``(T,)``
        """
        scores = self.score(data)
        return (scores > threshold).astype(np.int32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save detector state to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "input_dim": self.input_dim,
            "cfg": self.cfg,
            "model_state": self.model.state_dict(),
            "train_mean": self._train_mean,
            "train_std": self._train_std,
            "score_mean": self._score_mean,
            "score_std": self._score_std,
            "train_tau_mean": self._train_tau_mean,
            "fitted": self._fitted,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TemporalAnomalyDetector":
        """Load a saved detector."""
        state = torch.load(path, map_location=device, weights_only=False)
        det = cls(
            input_dim=state["input_dim"],
            config=state["cfg"],
            device=device,
        )
        det.model.load_state_dict(state["model_state"])
        det._train_mean      = state["train_mean"]
        det._train_std       = state["train_std"]
        det._score_mean      = state["score_mean"]
        det._score_std       = state["score_std"]
        det._train_tau_mean  = state.get("train_tau_mean", None)
        det._fitted          = state["fitted"]
        return det

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def create_streaming_scorer(self, threshold: float = 2.0) -> "StreamingScorer":
        """Create a streaming scorer for real-time use."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .create_streaming_scorer()")
        return StreamingScorer(detector=self, threshold=threshold)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"TemporalAnomalyDetector(input_dim={self.input_dim}, "
            f"hidden_dim={self.cfg.hidden_dim}, "
            f"score_mode='{self.cfg.score_mode}', {status})"
        )


# ---------------------------------------------------------------------------
# Streaming scorer
# ---------------------------------------------------------------------------

class StreamingScorer:
    """Online / streaming anomaly scorer.

    Wraps a fitted :class:`TemporalAnomalyDetector` and maintains hidden
    state internally so that callers can feed one observation at a time.

    Do not instantiate directly — use
    :meth:`TemporalAnomalyDetector.create_streaming_scorer`.
    """

    def __init__(self, detector: TemporalAnomalyDetector, threshold: float = 2.0) -> None:
        if not detector._fitted:
            raise RuntimeError("Detector must be fitted before creating a StreamingScorer")
        self._detector   = detector
        self._model      = detector.model
        self._device     = detector.device
        self._threshold  = threshold
        self._train_mean = detector._train_mean
        self._train_std  = detector._train_std
        self._score_mean = detector._score_mean
        self._score_std  = detector._score_std
        self._hidden: torch.Tensor | None = None

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def reset(self) -> None:
        """Clear the hidden state, restarting the sequence."""
        self._hidden = None

    def step(self, obs: np.ndarray) -> dict[str, float | bool]:
        """Process a single observation and return anomaly information."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 0:
            obs = obs.reshape(1)
        if obs.ndim != 1:
            raise ValueError(f"Expected 1-D observation, got shape {obs.shape}")

        if self._train_mean is not None and self._train_std is not None:
            obs_normed = (obs - self._train_mean) / (self._train_std + 1e-8)
        else:
            obs_normed = obs

        x = torch.tensor(obs_normed, device=self._device).unsqueeze(0)

        if self._hidden is None:
            self._hidden = self._model.init_hidden(batch_size=1, device=self._device)

        self._model.eval()
        with torch.no_grad():
            step_out = self._model(x, self._hidden)

        self._hidden  = step_out.hidden
        delta_tau     = float(step_out.delta_tau.item())
        pred_error    = float(step_out.pred_error.item())
        raw_score     = self._compute_raw_score(delta_tau, pred_error)
        score         = (raw_score - self._score_mean) / self._score_std

        return {
            "score":      score,
            "delta_tau":  delta_tau,
            "pred_error": pred_error,
            "is_anomaly": bool(score > self._threshold),
        }

    def _compute_raw_score(self, delta_tau: float, pred_error: float) -> float:
        score_mode = self._detector.cfg.score_mode
        if score_mode == "pred_error":
            return pred_error
        elif score_mode == "delta_tau":
            return delta_tau
        elif score_mode == "combined":
            return (delta_tau + pred_error) / 2.0
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

    def __repr__(self) -> str:
        state = "active" if self._hidden is not None else "reset"
        return (
            f"StreamingScorer(threshold={self._threshold}, "
            f"input_dim={self._detector.input_dim}, {state})"
        )
