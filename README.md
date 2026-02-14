# Learning Internal Time RL

`Adaptive Temporal Reparameterization` を強化学習エージェントへ実装するための研究用コードベースです。

## 実装済みの中核

- `internal_time_rl/models/time_module.py`
  - 学習可能な内部時間 `Δτ_t = g(h_t, x_t)`（`softplus` で正値制約）
  - 時間正則化（平均・分散・エネルギー）
  - オプションの KL 制約（対数時間をガウス事前分布へ）
- `internal_time_rl/models/policy.py`
  - `transition_mode = standard / fixed / learned`
  - `standard`: `h_{t+1} = GRU(x_t, h_t)`（time-scalingなし）
  - `fixed|learned`: `h_{t+1} = (1 - α_t) h_t + α_t \tilde{h}_{t+1}`, `α_t = 1 - exp(-Δτ_t)`
  - 離散/連続 action space に対応
- `internal_time_rl/algorithms/ppo_time.py`
  - PPO + GAE
  - 内部時間正則化を PPO 損失へ統合
  - `discount_mode = fixed / env_dt / internal_tau`
  - `corr(Δτ, |Adv|)`, `corr(Δτ, |TD error|)`, `corr(Δτ, action_repeat)` をログ出力
- `internal_time_rl/envs/wrappers.py`
  - Delayed Reward
  - Flickering Observation
  - Variable Speed (action repeat)
- `analysis/plot_internal_time.py`
  - 学習ログから `Δτ` の推移を可視化

## セットアップ

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## 学習実行

```bash
python train.py
```

注意:
- `train.total_timesteps >= train.num_envs * train.rollout_steps` にしてください（0 update回避）。

Hydra でオーバーライド可能:

```bash
python train.py env.id=Pendulum-v1 train.total_timesteps=300000 env.reward_delay=10
python train.py env.id=CartPole-v1 env.flicker_prob=0.35
python train.py env.id=Pendulum-v1 env.variable_speed=true env.min_repeat=1 env.max_repeat=5
python train.py model.transition_mode=standard
python train.py model.transition_mode=fixed model.fixed_tau=10.0
python train.py model.transition_mode=learned
python train.py env.variable_speed=true env.min_repeat=1 env.max_repeat=4 train.discount_mode=env_dt
```

## ログ解析

```bash
python -m internal_time_rl.analysis.plot_internal_time --csv runs/latest/metrics.csv --out runs/latest/internal_time.png
python -m internal_time_rl.analysis.aggregate_runs --root runs/sweeps/stage1 --metric episode/return_mean_20
```

## 追加スクリプト

- `scripts/smoke_test.cmd`: 短時間の起動確認
- `scripts/run_ablation.cmd`: `standard/fixed(1,3,10)/learned` の単発比較
- `scripts/run_sweep.py`: `condition x seed` の一括実行（論文向け）
- `scripts/run_sweep.cmd`: `stage1_full x 5 seeds` 実行
- `scripts/aggregate_runs.cmd`: multi-seed 集計と図生成
- `docs/stage1_protocol.md`: Stage-1 論文化用プロトコル
- `docs/research_plan.md`: 数理拡張を含む研究計画メモ

## 推奨ロードマップ

1. Phase-1: Baseline PPO と Internal-Time PPO の比較（遅延報酬・POMDP・可変速度）
2. Phase-2: `Self Model` を導入して `Δτ_t = g(h_t, x_t, \hat{h}_{t+1})` へ拡張
3. 理論: 収束性/安定性の補題をタスク条件ごとに整理
