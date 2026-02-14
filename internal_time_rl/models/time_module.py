from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TimeRegConfig:
    lambda_var: float = 0.01
    lambda_mean: float = 0.001
    lambda_energy: float = 0.0
    kl_beta: float = 0.0
    prior_log_mean: float = -0.5
    prior_log_std: float = 0.75


class InternalTimeHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, min_tau: float = 1e-4) -> None:
        super().__init__()
        self.min_tau = min_tau
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([h, x], dim=-1)
        raw = self.net(z).squeeze(-1)
        delta_tau = F.softplus(raw) + self.min_tau
        return delta_tau, raw

    @staticmethod
    def time_alpha(delta_tau: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.exp(-delta_tau)

    @staticmethod
    def compute_time_regularization(
        delta_tau: torch.Tensor,
        config: TimeRegConfig,
        raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mean_tau = delta_tau.mean()
        var_tau = delta_tau.var(unbiased=False)
        energy = (delta_tau.square()).mean()

        loss = (
            config.lambda_var * var_tau
            + config.lambda_mean * mean_tau
            + config.lambda_energy * energy
        )

        metrics = {
            "tau_mean": float(mean_tau.detach().cpu()),
            "tau_var": float(var_tau.detach().cpu()),
            "tau_energy": float(energy.detach().cpu()),
            "tau_reg_loss": float(loss.detach().cpu()),
        }

        if config.kl_beta > 0.0:
            if raw is None:
                raw = torch.log(delta_tau + 1e-8)
            prior_mean = config.prior_log_mean
            prior_std = config.prior_log_std
            prior_var = prior_std**2
            centered = raw - prior_mean
            kl = 0.5 * ((centered.square() / prior_var) + math.log(prior_var)).mean()
            loss = loss + config.kl_beta * kl
            metrics["tau_kl"] = float(kl.detach().cpu())
        else:
            metrics["tau_kl"] = 0.0

        return loss, metrics

