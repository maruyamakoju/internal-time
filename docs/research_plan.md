# Research Plan: Learning Internal Time

## Core Hypothesis

内部時間 `Δτ_t` を外部ステップから分離して学習させることで、遅延報酬・部分観測・時間スケール変動下での方策最適化を改善する。

## Implemented Stage-1 Formulation

- Transition:
  - `h~_{t+1} = GRU(x_t, h_t)`
  - `Δτ_t = g(h_t, x_t), Δτ_t > 0`
  - `α_t = 1 - exp(-Δτ_t)`
  - `h_{t+1} = (1 - α_t) h_t + α_t h~_{t+1}`
- Objective:
  - `L = L_PPO + L_time`
  - `L_time = λ_var Var(Δτ) + λ_mean E[Δτ] + λ_energy E[Δτ^2] + β KL(log Δτ || N(μ0, σ0^2))`

## Experimental Matrix

1. Baseline PPO (`learn_internal_time=false`) vs Internal-Time PPO
2. Delayed Reward (`reward_delay > 0`)
3. POMDP (`flicker_prob > 0`)
4. Variable Speed (`variable_speed=true`)

## Quantitative Targets

1. Sample efficiency: fixed wall-clock step budgetでの平均リターン改善
2. Robustness: 環境速度変動時の性能分散低下
3. Temporal adaptation: `Δτ` と TD error / policy entropy / episode phase の相関

## Stage-2 (Self Model + Time)

- Self model: `h^_{t+1} = f_self(h_t)`
- Prediction error: `e_t = ||h_{t+1} - h^_{t+1}||`
- Extension: `Δτ_t = g(h_t, x_t, h^_{t+1}, e_t)`

## Theory Targets

1. Time reparameterization equivalence 条件
2. Lyapunov 的安定性条件
3. Delayed reward 条件下での収束速度改善境界

