from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from internal_time_rl.envs.wrappers import (
    DelayedRewardWrapper,
    FlickeringObservationWrapper,
    VariableSpeedWrapper,
)
from internal_time_rl.models.policy import InternalTimeActorCritic, PolicyConfig
from internal_time_rl.models.time_module import InternalTimeHead, TimeRegConfig


@dataclass
class PPOConfig:
    total_timesteps: int = 500_000
    num_envs: int = 8
    rollout_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    update_epochs: int = 8
    mini_batch_size: int = 512
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    normalize_advantage: bool = True
    anneal_lr: bool = True
    discount_mode: str = "fixed"
    max_discount_exponent: float = 10.0


class CSVLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        hidden_dim: int,
        device: torch.device,
        discrete_action: bool,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        self.discrete_action = discrete_action

        self.obs = torch.zeros((rollout_steps, num_envs, *obs_shape), dtype=torch.float32, device=device)
        if discrete_action:
            self.actions = torch.zeros((rollout_steps, num_envs), dtype=torch.long, device=device)
        else:
            self.actions = torch.zeros((rollout_steps, num_envs, *action_shape), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.h_prev = torch.zeros((rollout_steps, num_envs, hidden_dim), dtype=torch.float32, device=device)
        self.delta_tau = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.td_error = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.obs_norm = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.action_repeat = torch.ones((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.discount_exponent = torch.ones((rollout_steps, num_envs), dtype=torch.float32, device=device)

        self.advantages = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)

    def add(
        self,
        step: int,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        h_prev: torch.Tensor,
        delta_tau: torch.Tensor,
        obs_norm: torch.Tensor,
        action_repeat: torch.Tensor,
        discount_exponent: torch.Tensor,
    ) -> None:
        self.obs[step].copy_(obs)
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.rewards[step].copy_(reward)
        self.dones[step].copy_(done)
        self.values[step].copy_(value)
        self.h_prev[step].copy_(h_prev)
        self.delta_tau[step].copy_(delta_tau)
        self.obs_norm[step].copy_(obs_norm)
        self.action_repeat[step].copy_(action_repeat)
        self.discount_exponent[step].copy_(discount_exponent)

    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,
        next_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        lastgaelam = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for t in reversed(range(self.rollout_steps)):
            gamma_t = gamma ** self.discount_exponent[t]
            if t == self.rollout_steps - 1:
                next_nonterminal = 1.0 - next_done
                next_values = next_value
            else:
                next_nonterminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + gamma_t * next_values * next_nonterminal - self.values[t]
            self.td_error[t] = delta
            lastgaelam = delta + gamma_t * gae_lambda * next_nonterminal * lastgaelam
            self.advantages[t] = lastgaelam
        self.returns = self.advantages + self.values

    def flatten(self) -> dict[str, torch.Tensor]:
        flat = {
            "obs": self.obs.reshape((-1,) + self.obs.shape[2:]),
            "actions": self.actions.reshape((-1,) + self.actions.shape[2:]) if not self.discrete_action else self.actions.reshape(-1),
            "log_probs": self.log_probs.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
            "values": self.values.reshape(-1),
            "h_prev": self.h_prev.reshape((-1, self.h_prev.shape[-1])),
            "delta_tau": self.delta_tau.reshape(-1),
            "td_error": self.td_error.reshape(-1),
            "obs_norm": self.obs_norm.reshape(-1),
            "action_repeat": self.action_repeat.reshape(-1),
            "discount_exponent": self.discount_exponent.reshape(-1),
        }
        return flat


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(env_cfg: DictConfig, seed: int, index: int):
    def thunk():
        env = gym.make(env_cfg.id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if int(env_cfg.reward_delay) > 0:
            env = DelayedRewardWrapper(env, delay_steps=int(env_cfg.reward_delay))
        if float(env_cfg.flicker_prob) > 0.0:
            env = FlickeringObservationWrapper(env, flicker_prob=float(env_cfg.flicker_prob))
        if bool(env_cfg.variable_speed):
            env = VariableSpeedWrapper(
                env,
                min_repeat=int(env_cfg.min_repeat),
                max_repeat=int(env_cfg.max_repeat),
            )
        env.action_space.seed(seed + index)
        env.observation_space.seed(seed + index)
        return env

    return thunk


def _to_numpy_action(action: torch.Tensor, discrete: bool) -> np.ndarray:
    if discrete:
        return action.detach().cpu().numpy().astype(np.int64)
    return action.detach().cpu().numpy().astype(np.float32)


def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y == 0:
        return np.nan
    return 1 - np.var(y_true - y_pred) / var_y


def _extract_episodes(info: dict[str, Any]) -> list[dict[str, float]]:
    episodes = []
    if "episode" in info:
        ep = info["episode"]
        mask = info.get("_episode")
        if isinstance(ep, dict) and mask is not None:
            rewards = np.asarray(ep["r"])
            lengths = np.asarray(ep["l"])
            mask_arr = np.asarray(mask, dtype=bool)
            for r, l, m in zip(rewards, lengths, mask_arr):
                if m:
                    episodes.append({"r": float(r), "l": float(l)})
    if "final_info" in info:
        final_infos = info["final_info"]
        mask = info.get("_final_info")
        if mask is None:
            mask = [True] * len(final_infos)
        for final_info, m in zip(final_infos, mask):
            if not m or final_info is None:
                continue
            ep = final_info.get("episode")
            if ep is not None:
                episodes.append({"r": float(ep["r"]), "l": float(ep["l"])})
    return episodes


def _build_policy_cfg(cfg: DictConfig) -> PolicyConfig:
    time_reg = TimeRegConfig(
        lambda_var=float(cfg.time_reg.lambda_var),
        lambda_mean=float(cfg.time_reg.lambda_mean),
        lambda_energy=float(cfg.time_reg.lambda_energy),
        kl_beta=float(cfg.time_reg.kl_beta),
        prior_log_mean=float(cfg.time_reg.prior_log_mean),
        prior_log_std=float(cfg.time_reg.prior_log_std),
    )
    transition_mode = str(cfg.model.get("transition_mode", ""))
    if not transition_mode:
        transition_mode = "learned" if bool(cfg.model.get("learn_internal_time", True)) else "fixed"

    return PolicyConfig(
        obs_embed_dim=int(cfg.model.obs_embed_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        encoder_hidden_dim=int(cfg.model.encoder_hidden_dim),
        activation=str(cfg.model.activation),
        init_log_std=float(cfg.model.init_log_std),
        min_tau=float(cfg.model.min_tau),
        time_head_hidden_dim=int(cfg.model.time_head_hidden_dim),
        transition_mode=transition_mode,
        fixed_tau=float(cfg.model.fixed_tau),
        standard_tau_proxy=float(cfg.model.get("standard_tau_proxy", 10.0)),
        time_reg=time_reg,
    )


def _build_ppo_cfg(cfg: DictConfig) -> PPOConfig:
    return PPOConfig(
        total_timesteps=int(cfg.train.total_timesteps),
        num_envs=int(cfg.train.num_envs),
        rollout_steps=int(cfg.train.rollout_steps),
        gamma=float(cfg.train.gamma),
        gae_lambda=float(cfg.train.gae_lambda),
        learning_rate=float(cfg.train.learning_rate),
        update_epochs=int(cfg.train.update_epochs),
        mini_batch_size=int(cfg.train.mini_batch_size),
        clip_coef=float(cfg.train.clip_coef),
        clip_vloss=bool(cfg.train.clip_vloss),
        ent_coef=float(cfg.train.ent_coef),
        vf_coef=float(cfg.train.vf_coef),
        max_grad_norm=float(cfg.train.max_grad_norm),
        target_kl=float(cfg.train.target_kl),
        normalize_advantage=bool(cfg.train.normalize_advantage),
        anneal_lr=bool(cfg.train.anneal_lr),
        discount_mode=str(cfg.train.get("discount_mode", "fixed")),
        max_discount_exponent=float(cfg.train.get("max_discount_exponent", 10.0)),
    )


def _extract_info_array(
    info: dict[str, Any], key: str, num_envs: int, default_value: float = 1.0
) -> np.ndarray:
    if key not in info:
        return np.full((num_envs,), default_value, dtype=np.float32)

    values = info[key]
    arr = np.asarray(values)
    if arr.shape == ():
        arr = np.full((num_envs,), float(arr), dtype=np.float32)
    arr = arr.astype(np.float32).reshape(-1)
    if arr.shape[0] != num_envs:
        out = np.full((num_envs,), default_value, dtype=np.float32)
        out[: min(num_envs, arr.shape[0])] = arr[: min(num_envs, arr.shape[0])]
        arr = out

    mask_key = f"_{key}"
    mask = np.asarray(info.get(mask_key, np.ones((num_envs,), dtype=bool))).astype(bool).reshape(-1)
    if mask.shape[0] != num_envs:
        fixed_mask = np.zeros((num_envs,), dtype=bool)
        fixed_mask[: min(num_envs, mask.shape[0])] = mask[: min(num_envs, mask.shape[0])]
        mask = fixed_mask

    default_arr = np.full((num_envs,), default_value, dtype=np.float32)
    return np.where(mask, arr, default_arr).astype(np.float32)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def train_ppo_with_internal_time(cfg: DictConfig) -> dict[str, Any]:
    seed = int(cfg.seed)
    set_global_seed(seed)

    device_name = str(cfg.device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA device requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    ppo_cfg = _build_ppo_cfg(cfg)
    policy_cfg = _build_policy_cfg(cfg)
    use_time_regularization = policy_cfg.transition_mode == "learned"

    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env, seed=seed, index=i) for i in range(ppo_cfg.num_envs)]
    )
    obs_shape = envs.single_observation_space.shape
    discrete_action = isinstance(envs.single_action_space, gym.spaces.Discrete)
    action_shape = () if discrete_action else envs.single_action_space.shape

    if not isinstance(envs.single_observation_space, gym.spaces.Box):
        raise ValueError("Only Box observation spaces are currently supported.")

    policy = InternalTimeActorCritic(
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        cfg=policy_cfg,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)

    run_dir = Path(str(cfg.logging.run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_logger = CSVLogger(run_dir / "metrics.csv")

    use_wandb = bool(cfg.logging.use_wandb)
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(cfg.logging.wandb_project),
                name=str(cfg.logging.wandb_name),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as exc:  # pragma: no cover
            print(f"[warn] wandb init failed: {exc}")
            wandb_run = None

    global_step = 0
    num_updates = ppo_cfg.total_timesteps // (ppo_cfg.rollout_steps * ppo_cfg.num_envs)
    if num_updates < 1:
        raise ValueError(
            "num_updates became 0. Increase train.total_timesteps or reduce "
            "train.rollout_steps/train.num_envs so that "
            "train.total_timesteps >= train.rollout_steps * train.num_envs."
        )
    start_time = time.time()

    obs, _ = envs.reset(seed=seed)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    done_t = torch.zeros(ppo_cfg.num_envs, dtype=torch.float32, device=device)
    hidden_t = policy.get_initial_state(ppo_cfg.num_envs, device=device)

    episode_returns: list[float] = []
    episode_lengths: list[float] = []

    for update in range(1, num_updates + 1):
        if ppo_cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / max(1, num_updates)
            optimizer.param_groups[0]["lr"] = ppo_cfg.learning_rate * frac

        buffer = RolloutBuffer(
            rollout_steps=ppo_cfg.rollout_steps,
            num_envs=ppo_cfg.num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            hidden_dim=policy_cfg.hidden_dim,
            device=device,
            discrete_action=discrete_action,
        )

        for step in range(ppo_cfg.rollout_steps):
            global_step += ppo_cfg.num_envs
            h_prev_step = hidden_t.clone()
            obs_norm_t = torch.linalg.vector_norm(obs_t.view(obs_t.shape[0], -1), dim=-1)

            with torch.no_grad():
                action, log_prob, value, _, h_next, delta_tau, _ = policy.act(
                    obs_t, hidden_t, deterministic=False
                )

            np_action = _to_numpy_action(action, discrete=discrete_action)
            next_obs, reward, terminated, truncated, infos = envs.step(np_action)
            done = np.logical_or(terminated, truncated)

            reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
            done_t_step = torch.tensor(done, dtype=torch.float32, device=device)
            action_repeat_np = _extract_info_array(
                infos,
                key="action_repeat",
                num_envs=ppo_cfg.num_envs,
                default_value=1.0,
            )
            action_repeat_t = torch.tensor(action_repeat_np, dtype=torch.float32, device=device)
            if ppo_cfg.discount_mode == "fixed":
                discount_exponent_t = torch.ones_like(action_repeat_t)
            elif ppo_cfg.discount_mode == "env_dt":
                discount_exponent_t = action_repeat_t
            elif ppo_cfg.discount_mode == "internal_tau":
                discount_exponent_t = delta_tau.detach()
            else:
                raise ValueError(
                    f"Unsupported discount_mode='{ppo_cfg.discount_mode}'. "
                    "Choose one of: fixed, env_dt, internal_tau."
                )
            discount_exponent_t = torch.clamp(
                discount_exponent_t,
                min=1e-4,
                max=ppo_cfg.max_discount_exponent,
            )

            buffer.add(
                step=step,
                obs=obs_t,
                action=action if not discrete_action else action.long(),
                log_prob=log_prob,
                reward=reward_t,
                done=done_t_step,
                value=value,
                h_prev=h_prev_step,
                delta_tau=delta_tau,
                obs_norm=obs_norm_t,
                action_repeat=action_repeat_t,
                discount_exponent=discount_exponent_t,
            )

            hidden_t = h_next
            if done.any():
                done_idx = torch.as_tensor(done, dtype=torch.bool, device=device)
                hidden_t[done_idx] = 0.0

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            done_t = done_t_step

            episodes = _extract_episodes(infos)
            for ep in episodes:
                episode_returns.append(ep["r"])
                episode_lengths.append(ep["l"])

        with torch.no_grad():
            _, _, next_value, _, _, _, _ = policy.act(obs_t, hidden_t, deterministic=True)

        buffer.compute_returns_and_advantages(
            next_value=next_value,
            next_done=done_t,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )

        batch = buffer.flatten()
        b_obs = batch["obs"]
        b_actions = batch["actions"]
        b_log_probs = batch["log_probs"]
        b_advantages = batch["advantages"]
        b_returns = batch["returns"]
        b_values = batch["values"]
        b_h_prev = batch["h_prev"]

        if ppo_cfg.normalize_advantage:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = b_obs.shape[0]
        mini_batch_size = min(ppo_cfg.mini_batch_size, batch_size)
        indices = np.arange(batch_size)

        clipfracs: list[float] = []
        epoch_policy_losses: list[float] = []
        epoch_value_losses: list[float] = []
        epoch_entropy: list[float] = []
        epoch_time_losses: list[float] = []
        epoch_approx_kl: list[float] = []
        epoch_tau_means: list[float] = []
        epoch_tau_vars: list[float] = []

        for _epoch in range(ppo_cfg.update_epochs):
            np.random.shuffle(indices)
            early_stop = False
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_log_probs = b_log_probs[mb_idx]
                mb_advantages = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_old_values = b_values[mb_idx]
                mb_h_prev = b_h_prev[mb_idx]

                new_log_prob, entropy, new_value, delta_tau, raw_tau = policy.evaluate_actions(
                    mb_obs, mb_actions, mb_h_prev
                )

                log_ratio = new_log_prob - mb_old_log_probs
                ratio = log_ratio.exp()

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - ppo_cfg.clip_coef, 1.0 + ppo_cfg.clip_coef
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                if ppo_cfg.clip_vloss:
                    value_unclipped = (new_value - mb_returns).square()
                    value_clipped = mb_old_values + torch.clamp(
                        new_value - mb_old_values, -ppo_cfg.clip_coef, ppo_cfg.clip_coef
                    )
                    value_clipped_loss = (value_clipped - mb_returns).square()
                    value_loss = 0.5 * torch.max(value_unclipped, value_clipped_loss).mean()
                else:
                    value_loss = 0.5 * (new_value - mb_returns).square().mean()

                entropy_loss = entropy.mean()
                if use_time_regularization:
                    time_loss, time_metrics = InternalTimeHead.compute_time_regularization(
                        delta_tau=delta_tau,
                        config=policy_cfg.time_reg,
                        raw=raw_tau,
                    )
                else:
                    time_loss = torch.zeros((), dtype=torch.float32, device=device)
                    mean_tau = delta_tau.mean()
                    var_tau = delta_tau.var(unbiased=False)
                    time_metrics = {
                        "tau_mean": float(mean_tau.detach().cpu()),
                        "tau_var": float(var_tau.detach().cpu()),
                        "tau_energy": float((delta_tau.square().mean()).detach().cpu()),
                        "tau_reg_loss": 0.0,
                        "tau_kl": 0.0,
                    }

                loss = (
                    policy_loss
                    + ppo_cfg.vf_coef * value_loss
                    - ppo_cfg.ent_coef * entropy_loss
                    + time_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > ppo_cfg.clip_coef).float().mean().item()

                clipfracs.append(clipfrac)
                epoch_policy_losses.append(float(policy_loss.detach().cpu()))
                epoch_value_losses.append(float(value_loss.detach().cpu()))
                epoch_entropy.append(float(entropy_loss.detach().cpu()))
                epoch_time_losses.append(float(time_loss.detach().cpu()))
                epoch_approx_kl.append(float(approx_kl))
                epoch_tau_means.append(time_metrics["tau_mean"])
                epoch_tau_vars.append(time_metrics["tau_var"])

                if ppo_cfg.target_kl > 0 and approx_kl > ppo_cfg.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        explained_var = _explained_variance(y_pred, y_true)

        flat_tau = batch["delta_tau"].detach().cpu().numpy()
        flat_adv_abs = batch["advantages"].abs().detach().cpu().numpy()
        flat_td_abs = batch["td_error"].abs().detach().cpu().numpy()
        flat_obs_norm = batch["obs_norm"].detach().cpu().numpy()
        flat_action_repeat = batch["action_repeat"].detach().cpu().numpy()
        flat_discount_exp = batch["discount_exponent"].detach().cpu().numpy()

        corr_tau_adv = _safe_corr(flat_tau, flat_adv_abs)
        corr_tau_td = _safe_corr(flat_tau, flat_td_abs)
        corr_tau_repeat = _safe_corr(flat_tau, flat_action_repeat)

        obs_zero_threshold = float(cfg.logging.get("obs_zero_threshold", 1e-8))
        zero_mask = flat_obs_norm <= obs_zero_threshold
        nonzero_mask = ~zero_mask
        tau_obs_zero_mean = float(np.mean(flat_tau[zero_mask])) if zero_mask.any() else np.nan
        tau_obs_nonzero_mean = float(np.mean(flat_tau[nonzero_mask])) if nonzero_mask.any() else np.nan

        sps = int(global_step / max(1e-6, (time.time() - start_time)))
        mean_ep_ret = float(np.mean(episode_returns[-20:])) if episode_returns else np.nan
        mean_ep_len = float(np.mean(episode_lengths[-20:])) if episode_lengths else np.nan

        metrics = {
            "update": update,
            "global_step": global_step,
            "model/transition_mode": policy_cfg.transition_mode,
            "train/discount_mode": ppo_cfg.discount_mode,
            "sps": sps,
            "lr": optimizer.param_groups[0]["lr"],
            "loss/policy": float(np.mean(epoch_policy_losses)),
            "loss/value": float(np.mean(epoch_value_losses)),
            "loss/entropy": float(np.mean(epoch_entropy)),
            "loss/time_reg": float(np.mean(epoch_time_losses)),
            "approx_kl": float(np.mean(epoch_approx_kl)),
            "clipfrac": float(np.mean(clipfracs)),
            "explained_variance": float(explained_var),
            "tau/mean": float(np.mean(epoch_tau_means)),
            "tau/var": float(np.mean(epoch_tau_vars)),
            "tau/rollout_mean": float(buffer.delta_tau.mean().detach().cpu()),
            "tau/rollout_var": float(buffer.delta_tau.var(unbiased=False).detach().cpu()),
            "tau/obs_zero_mean": tau_obs_zero_mean,
            "tau/obs_nonzero_mean": tau_obs_nonzero_mean,
            "corr/tau_abs_adv": corr_tau_adv,
            "corr/tau_abs_td": corr_tau_td,
            "corr/tau_action_repeat": corr_tau_repeat,
            "adv/abs_mean": float(np.mean(flat_adv_abs)),
            "td/abs_mean": float(np.mean(flat_td_abs)),
            "repeat/mean": float(np.mean(flat_action_repeat)),
            "discount/exponent_mean": float(np.mean(flat_discount_exp)),
            "episode/return_mean_20": mean_ep_ret,
            "episode/length_mean_20": mean_ep_len,
        }

        metrics_logger.log(metrics)
        if wandb_run is not None:
            wandb_run.log(metrics, step=global_step)

        if update % int(cfg.logging.print_every_updates) == 0:
            print(
                f"[update {update}/{num_updates}] "
                f"step={global_step} "
                f"mode={policy_cfg.transition_mode} "
                f"ep_ret20={mean_ep_ret:.2f} "
                f"tau_mean={metrics['tau/mean']:.4f} "
                f"kl={metrics['approx_kl']:.5f}"
            )

    ckpt_path = run_dir / "policy_final.pt"
    torch.save(policy.state_dict(), ckpt_path)
    config_path = run_dir / "resolved_config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    if wandb_run is not None:
        wandb_run.finish()

    envs.close()

    return {
        "checkpoint": str(ckpt_path),
        "metrics_csv": str(run_dir / "metrics.csv"),
        "run_dir": str(run_dir),
    }
