from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .encoder import ObsEncoder
from .time_module import InternalTimeHead, TimeRegConfig


@dataclass
class PolicyConfig:
    obs_embed_dim: int = 128
    hidden_dim: int = 128
    encoder_hidden_dim: int = 128
    activation: str = "tanh"
    init_log_std: float = -0.5
    min_tau: float = 1e-4
    time_head_hidden_dim: int = 128
    transition_mode: str = "learned"
    fixed_tau: float = 1.0
    standard_tau_proxy: float = 10.0
    use_self_model: bool = False
    use_pred_error_for_tau: bool = True
    self_model_hidden_dim: int = 128
    time_reg: TimeRegConfig = field(default_factory=TimeRegConfig)


def _get_activation(name: str) -> type[nn.Module]:
    table = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
    }
    return table.get(name.lower(), nn.Tanh)


class InternalTimeActorCritic(nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.discrete_action = isinstance(action_space, gym.spaces.Discrete)
        self.transition_mode = str(cfg.transition_mode).lower()
        if self.transition_mode not in {"standard", "fixed", "learned"}:
            raise ValueError(
                f"Unsupported transition_mode='{cfg.transition_mode}'. "
                "Choose one of: standard, fixed, learned."
            )

        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError("Only Box observation spaces are currently supported.")

        obs_dim = int(np.prod(observation_space.shape))
        activation = _get_activation(cfg.activation)

        self.encoder = ObsEncoder(
            obs_dim=obs_dim,
            embed_dim=cfg.obs_embed_dim,
            hidden_dims=(cfg.encoder_hidden_dim, cfg.encoder_hidden_dim),
            activation=activation,
        )
        self.gru_cell = nn.GRUCell(cfg.obs_embed_dim, cfg.hidden_dim)
        self.use_self_model = bool(cfg.use_self_model)
        self.self_model: nn.Module | None = None
        if self.use_self_model:
            self.self_model = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.self_model_hidden_dim),
                activation(),
                nn.Linear(cfg.self_model_hidden_dim, cfg.hidden_dim),
            )
        self.time_head: InternalTimeHead | None = None
        if self.transition_mode == "learned":
            self.time_head = InternalTimeHead(
                in_dim=cfg.hidden_dim + cfg.obs_embed_dim + (1 if self.use_self_model else 0),
                hidden_dim=cfg.time_head_hidden_dim,
                min_tau=cfg.min_tau,
            )

        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            activation(),
            nn.Linear(cfg.hidden_dim, 1),
        )

        if self.discrete_action:
            n_actions = action_space.n
            self.actor_head = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                activation(),
                nn.Linear(cfg.hidden_dim, n_actions),
            )
            self.log_std = None
            self.action_dim = 1
        else:
            if not isinstance(action_space, gym.spaces.Box):
                raise ValueError("Only Discrete or Box action spaces are currently supported.")
            action_dim = int(np.prod(action_space.shape))
            self.actor_head = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                activation(),
                nn.Linear(cfg.hidden_dim, action_dim),
            )
            self.log_std = nn.Parameter(torch.ones(action_dim) * cfg.init_log_std)
            self.action_dim = action_dim

    def get_initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.cfg.hidden_dim, device=device)

    @staticmethod
    def _flatten_obs(obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs.view(obs.shape[0], -1)

    def _forward_transition(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
        tau_extra_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = self._flatten_obs(obs)
        x = self.encoder(obs)
        h_candidate = self.gru_cell(x, h_prev)
        pred_error = torch.zeros((obs.shape[0],), dtype=obs.dtype, device=obs.device)
        self_model_loss = torch.zeros((), dtype=obs.dtype, device=obs.device)
        if self.use_self_model:
            if self.self_model is None:
                raise RuntimeError("self_model is not initialized.")
            h_hat = self.self_model(h_prev)
            diff = h_candidate.detach() - h_hat
            pred_error_sq = diff.square().mean(dim=-1)
            pred_error = torch.sqrt(pred_error_sq + 1e-8)
            self_model_loss = pred_error_sq.mean()

        if self.transition_mode == "standard":
            # Pure GRU transition (no temporal mixing) for a fair standard baseline.
            h_next = h_candidate
            delta_tau = torch.full(
                (obs.shape[0],),
                float(self.cfg.standard_tau_proxy),
                dtype=obs.dtype,
                device=obs.device,
            )
            raw_tau = torch.log(delta_tau + 1e-8)
            return h_next, delta_tau, raw_tau, pred_error, self_model_loss

        if self.transition_mode == "fixed":
            delta_tau = torch.full(
                (obs.shape[0],),
                float(self.cfg.fixed_tau),
                dtype=obs.dtype,
                device=obs.device,
            )
            raw_tau = torch.log(delta_tau + 1e-8)
        else:
            if self.time_head is None:
                raise RuntimeError("time_head is not initialized in learned mode.")
            tau_extra = None
            if self.use_self_model:
                tau_scale = max(0.0, float(tau_extra_scale))
                if self.cfg.use_pred_error_for_tau:
                    tau_extra = (pred_error.detach() * tau_scale).unsqueeze(-1)
                else:
                    # Keep input dimensionality/parameter count unchanged for fair ablation.
                    tau_extra = torch.zeros_like(pred_error).unsqueeze(-1)
            delta_tau, raw_tau = self.time_head(h_prev, x, extra=tau_extra)

        alpha = InternalTimeHead.time_alpha(delta_tau).unsqueeze(-1)
        h_next = (1.0 - alpha) * h_prev + alpha * h_candidate
        return h_next, delta_tau, raw_tau, pred_error, self_model_loss

    def _distribution(self, latent: torch.Tensor):
        if self.discrete_action:
            logits = self.actor_head(latent)
            return Categorical(logits=logits)
        mean = self.actor_head(latent)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
        deterministic: bool = False,
        tau_extra_scale: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        h_next, delta_tau, raw_tau, pred_error, self_model_loss = self._forward_transition(
            obs, h_prev, tau_extra_scale=tau_extra_scale
        )
        dist = self._distribution(h_next)

        if deterministic:
            if self.discrete_action:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.mean
        else:
            action = dist.sample()

        if self.discrete_action:
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        value = self.value_head(h_next).squeeze(-1)
        return action, log_prob, value, entropy, h_next, delta_tau, raw_tau, pred_error, self_model_loss

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        h_prev: torch.Tensor,
        tau_extra_scale: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        h_next, delta_tau, raw_tau, pred_error, self_model_loss = self._forward_transition(
            obs, h_prev, tau_extra_scale=tau_extra_scale
        )
        dist = self._distribution(h_next)
        if self.discrete_action:
            log_prob = dist.log_prob(actions.long())
            entropy = dist.entropy()
        else:
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        value = self.value_head(h_next).squeeze(-1)
        return log_prob, entropy, value, delta_tau, raw_tau, pred_error, self_model_loss
