"""Composable temporal modules for learning internal time.

Core building blocks that can be used standalone or integrated into any
PyTorch pipeline.  No dependency on Gymnasium, PPO, or any RL framework.

Typical usage::

    from internal_time import TemporalGRUCell

    cell = TemporalGRUCell(input_dim=10, hidden_dim=64)
    h = cell.init_hidden(batch_size=32)
    for t in range(seq_len):
        h, info = cell(x[:, t, :], h)
        print(info.delta_tau)  # internal time step
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

class StepOutput(NamedTuple):
    """Output of a single TemporalGRUCell step."""

    hidden: torch.Tensor
    """New hidden state, shape ``(batch, hidden_dim)``."""

    delta_tau: torch.Tensor
    """Internal time step, shape ``(batch,)``.  Always positive."""

    alpha: torch.Tensor
    """Temporal mixing coefficient ``1 - exp(-delta_tau)``, shape ``(batch,)``."""

    pred_error: torch.Tensor
    """Self-model prediction error (RMS over hidden dim), shape ``(batch,)``."""

    self_model_loss: torch.Tensor
    """Scalar MSE loss for self-model training."""


class SequenceOutput(NamedTuple):
    """Output of :meth:`TemporalGRUCell.forward_sequence`."""

    hiddens: torch.Tensor
    """All hidden states, shape ``(batch, seq_len, hidden_dim)``."""

    delta_taus: torch.Tensor
    """Internal time steps, shape ``(batch, seq_len)``."""

    alphas: torch.Tensor
    """Mixing coefficients, shape ``(batch, seq_len)``."""

    pred_errors: torch.Tensor
    """Prediction errors, shape ``(batch, seq_len)``."""

    self_model_loss: torch.Tensor
    """Mean self-model loss over the sequence (scalar)."""

    final_hidden: torch.Tensor
    """Last hidden state, shape ``(batch, hidden_dim)``."""


# ---------------------------------------------------------------------------
# Time regularization
# ---------------------------------------------------------------------------

@dataclass
class TimeRegConfig:
    """Configuration for internal-time regularization."""

    lambda_var: float = 0.01
    """Penalise variance of delta_tau within a batch/sequence."""

    lambda_mean: float = 0.0
    """Penalise mean delta_tau (encourages smaller time steps)."""

    lambda_energy: float = 0.0
    """Penalise E[delta_tau^2]."""


def compute_time_reg(
    delta_tau: torch.Tensor,
    cfg: TimeRegConfig,
) -> torch.Tensor:
    """Return scalar regularization loss for delta_tau."""
    loss = torch.zeros((), device=delta_tau.device, dtype=delta_tau.dtype)
    if cfg.lambda_var > 0:
        loss = loss + cfg.lambda_var * delta_tau.var()
    if cfg.lambda_mean > 0:
        loss = loss + cfg.lambda_mean * delta_tau.mean()
    if cfg.lambda_energy > 0:
        loss = loss + cfg.lambda_energy * delta_tau.square().mean()
    return loss


# ---------------------------------------------------------------------------
# Internal Time Head
# ---------------------------------------------------------------------------

class InternalTimeHead(nn.Module):
    """Produces a positive scalar delta_tau from hidden state + observation.

    Parameters
    ----------
    in_dim : int
        Concatenated dimension of ``[h, x]`` (plus optional extra features).
    hidden_dim : int
        Width of the internal MLP.
    min_tau : float
        Floor value added after softplus to prevent collapse to zero.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, min_tau: float = 1e-4) -> None:
        super().__init__()
        self.min_tau = min_tau
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        extra: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        delta_tau : Tensor, shape ``(batch,)``
        raw : Tensor, shape ``(batch,)`` — pre-softplus logit
        """
        parts = [h, x]
        if extra is not None:
            if extra.dim() == 1:
                extra = extra.unsqueeze(-1)
            parts.append(extra)
        z = torch.cat(parts, dim=-1)
        raw = self.net(z).squeeze(-1)
        delta_tau = F.softplus(raw) + self.min_tau
        return delta_tau, raw

    @staticmethod
    def alpha_from_tau(delta_tau: torch.Tensor) -> torch.Tensor:
        """Convert delta_tau to mixing coefficient ``1 - exp(-delta_tau)``."""
        return 1.0 - torch.exp(-delta_tau)


# ---------------------------------------------------------------------------
# Self-Model
# ---------------------------------------------------------------------------

class SelfModel(nn.Module):
    """Predicts the next hidden state from the current one.

    The prediction error serves two purposes:

    1. Training signal for the self-model itself (MSE loss).
    2. Extra input feature to :class:`InternalTimeHead` so that
       ``delta_tau`` can respond to surprise.
    """

    def __init__(self, hidden_dim: int, mlp_dim: int | None = None) -> None:
        super().__init__()
        mlp_dim = mlp_dim or hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.Tanh(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, h_prev: torch.Tensor) -> torch.Tensor:
        """Return predicted next hidden state ``h_hat``."""
        return self.net(h_prev)

    @staticmethod
    def prediction_error(
        h_actual: torch.Tensor,
        h_predicted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RMS prediction error and MSE loss.

        Returns
        -------
        rms_error : shape ``(batch,)``
        mse_loss : scalar
        """
        diff = h_actual.detach() - h_predicted
        sq = diff.square().mean(dim=-1)
        rms = torch.sqrt(sq + 1e-8)
        mse = sq.mean()
        return rms, mse


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class InputEncoder(nn.Module):
    """Simple MLP encoder for raw observations / features.

    Reduces arbitrary input dimensionality to a fixed embedding size
    suitable for the GRU.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        hidden_dims: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev, hd))
            layers.append(activation())
            prev = hd
        layers.append(nn.Linear(prev, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x.view(x.shape[0], -1))


# ---------------------------------------------------------------------------
# TemporalGRUCell — the main composable unit
# ---------------------------------------------------------------------------

class TemporalGRUCell(nn.Module):
    """GRU cell with learnable internal time modulation.

    This is the core building block.  It wraps:

    * An :class:`InputEncoder` (MLP) to embed raw inputs.
    * A standard ``GRUCell`` to produce a candidate hidden state.
    * An :class:`InternalTimeHead` to compute ``delta_tau``.
    * Temporal mixing: ``h_next = (1 - alpha) * h_prev + alpha * h_candidate``.
    * Optionally a :class:`SelfModel` whose prediction error feeds into the
      time head, enabling surprise-driven time modulation.

    Parameters
    ----------
    input_dim : int
        Dimensionality of raw input features per timestep.
    hidden_dim : int
        Hidden state size for the GRU and all downstream heads.
    embed_dim : int or None
        Encoder output dimension.  Defaults to ``hidden_dim``.
    encoder_hidden_dims : tuple of int
        Layer widths for the input encoder MLP.
    use_self_model : bool
        If True, attach a self-model and feed prediction error to the time head.
    min_tau : float
        Minimum delta_tau (floor after softplus).
    time_head_hidden : int or None
        Hidden width of the time head MLP.  Defaults to ``hidden_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embed_dim: int | None = None,
        encoder_hidden_dims: tuple[int, ...] = (64, 64),
        use_self_model: bool = True,
        min_tau: float = 1e-4,
        time_head_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        embed_dim = embed_dim or hidden_dim
        time_head_hidden = time_head_hidden or hidden_dim
        self.use_self_model = use_self_model

        self.encoder = InputEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dims=encoder_hidden_dims,
        )
        self.gru = nn.GRUCell(embed_dim, hidden_dim)

        # Time head input = h_prev + x_embed + (optionally pred_error scalar)
        time_in_dim = hidden_dim + embed_dim + (1 if use_self_model else 0)
        self.time_head = InternalTimeHead(
            in_dim=time_in_dim,
            hidden_dim=time_head_hidden,
            min_tau=min_tau,
        )

        self.self_model: SelfModel | None = None
        if use_self_model:
            self.self_model = SelfModel(hidden_dim)

    # -- helpers --

    def init_hidden(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return zero-initialised hidden state."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    # -- single-step forward --

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        tau_scale: float = 1.0,
        detach_temporal_mix: bool = False,
    ) -> StepOutput:
        """Process one timestep.

        Parameters
        ----------
        x : Tensor, shape ``(batch, input_dim)``
        h_prev : Tensor, shape ``(batch, hidden_dim)``
        tau_scale : float
            Multiplier on the prediction-error extra feature (useful for warmup).
        detach_temporal_mix : bool
            If True, detach ``alpha`` from the computation graph before mixing.
            This breaks the feedback loop between time-head and hidden-state
            dynamics, stabilising self-model training.  The time head still
            produces delta_tau (and receives its own regularization gradient),
            but cannot destabilise the GRU trajectory.

        Returns
        -------
        StepOutput
        """
        x_embed = self.encoder(x)
        h_candidate = self.gru(x_embed, h_prev)

        # Self-model
        pred_error = torch.zeros(h_prev.shape[0], device=h_prev.device)
        self_model_loss = torch.zeros((), device=h_prev.device)
        extra: torch.Tensor | None = None

        if self.use_self_model and self.self_model is not None:
            h_hat = self.self_model(h_prev)
            pred_error, self_model_loss = SelfModel.prediction_error(h_candidate, h_hat)
            extra = (pred_error.detach() * max(0.0, tau_scale)).unsqueeze(-1)

        # Time head
        delta_tau, _raw = self.time_head(h_prev, x_embed, extra=extra)
        alpha = InternalTimeHead.alpha_from_tau(delta_tau)

        # Temporal mixing
        alpha_mix = alpha.detach() if detach_temporal_mix else alpha
        h_next = (1.0 - alpha_mix.unsqueeze(-1)) * h_prev + alpha_mix.unsqueeze(-1) * h_candidate

        return StepOutput(
            hidden=h_next,
            delta_tau=delta_tau,
            alpha=alpha,
            pred_error=pred_error,
            self_model_loss=self_model_loss,
        )

    # -- sequence forward --

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        h0: torch.Tensor | None = None,
        tau_scale: float = 1.0,
        detach_temporal_mix: bool = False,
    ) -> SequenceOutput:
        """Process a full sequence.

        Parameters
        ----------
        x_seq : Tensor, shape ``(batch, seq_len, input_dim)``
        h0 : Tensor or None
            Initial hidden state.  If None, uses zeros.
        tau_scale : float
            Warmup multiplier for prediction-error feature.
        detach_temporal_mix : bool
            If True, break feedback loop (see :meth:`forward`).

        Returns
        -------
        SequenceOutput
        """
        batch, seq_len, _ = x_seq.shape
        if h0 is None:
            h0 = self.init_hidden(batch, device=x_seq.device)

        all_h = []
        all_tau = []
        all_alpha = []
        all_pe = []
        total_sm_loss = torch.zeros((), device=x_seq.device)

        h = h0
        for t in range(seq_len):
            step = self.forward(
                x_seq[:, t, :], h,
                tau_scale=tau_scale,
                detach_temporal_mix=detach_temporal_mix,
            )
            h = step.hidden
            all_h.append(h)
            all_tau.append(step.delta_tau)
            all_alpha.append(step.alpha)
            all_pe.append(step.pred_error)
            total_sm_loss = total_sm_loss + step.self_model_loss

        return SequenceOutput(
            hiddens=torch.stack(all_h, dim=1),
            delta_taus=torch.stack(all_tau, dim=1),
            alphas=torch.stack(all_alpha, dim=1),
            pred_errors=torch.stack(all_pe, dim=1),
            self_model_loss=total_sm_loss / seq_len,
            final_hidden=h,
        )
