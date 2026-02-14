# Stage-1 Paper-Ready Protocol

## Goal

Stage-1 (`Internal Time + PPO`) が、フェアな比較で再現的に有効であることを示す。

## P0: Fair Baselines

必須条件:

1. `standard_gru`: `model.transition_mode=standard`
2. `fixed_tau`: `model.transition_mode=fixed`, `model.fixed_tau in {1,3,10}`
3. `learned_tau`: `model.transition_mode=learned`

これで「内部時間の効果」と「単なる更新スムージング効果」を分離する。

## P0: Multi-Seed

5 seeds (`0..4`) を固定。  
出力先: `runs/sweeps/stage1/<condition>/seed<seed>/`.

注意:
- `train.total_timesteps` は `train.num_envs * train.rollout_steps` 以上にする（`num_updates >= 1` 必須）。

実行:

```bash
python scripts/run_sweep.py --suite stage1_full --seeds 0 1 2 3 4 --timesteps 200000 --skip-existing
```

## P0: Paper Figures

集計:

```bash
python -m internal_time_rl.analysis.aggregate_runs --root runs/sweeps/stage1 --metric episode/return_mean_20
```

生成物:

1. `aggregate/learning_curves.png` (mean ± std)
2. `aggregate/final_performance.png` (final mean ± std)
3. `aggregate/condition_summary.csv` (Final/AUC table)

## P0: Temporal Dynamics Evidence

`metrics.csv` に以下を保存:

1. `corr/tau_abs_adv`
2. `corr/tau_abs_td`
3. `corr/tau_action_repeat`
4. `tau/obs_zero_mean`, `tau/obs_nonzero_mean`

これで「Δτが何に反応しているか」を定量化する。

注記:
- `corr/tau_abs_td` は `|r_t + gamma_t * V_{t+1} - V_t|`（one-step TD error）を使用する。

## P1: Variable-Speed Discounting

`train.discount_mode`:

1. `fixed`: 通常PPO（対照）
2. `env_dt`: `gamma^action_repeat`（客観時間）
3. `internal_tau`: `gamma^Δτ`（主観時間）

`stage1_full` には `learned_tau_speed_objdt` と `learned_tau_speed_subjdt` を含む。
