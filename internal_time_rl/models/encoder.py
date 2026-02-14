from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def build_mlp(
    in_dim: int,
    hidden_dims: Iterable[int],
    out_dim: int,
    activation: type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    dims = [in_dim, *hidden_dims, out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class ObsEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        embed_dim: int,
        hidden_dims: Iterable[int] = (128, 128),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, embed_dim, activation=activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

