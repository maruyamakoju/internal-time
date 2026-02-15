# Stage-1 Results Snapshot (2026-02-15)

This file fixes the current Stage-1 status after protocol freeze and multi-seed sweeps.

## Protocol freeze

- Commit: `1c69995`
- Tag: `stage1_spec_v1`
- Key fixes included:
  - `corr/tau_abs_td` uses true one-step TD error (`r + gamma * V(next) - V(now)`)
  - `num_updates < 1` now hard-fails early

## Main CartPole run (stage1_full, lambda_mean=0, lambda_var=1e-2)

- Root: `runs/sweeps/stage1_full_lm0_lv1e-2`
- Summary: `runs/sweeps/stage1_full_lm0_lv1e-2/aggregate/condition_summary.csv`
- Figures:
  - `runs/sweeps/stage1_full_lm0_lv1e-2/aggregate/learning_curves.png`
  - `runs/sweeps/stage1_full_lm0_lv1e-2/aggregate/final_performance.png`

Key final means (`episode/return_mean_20`, 5 seeds):

- `standard_gru_base`: 223.385
- `fixed_tau_10_base`: 221.373
- `learned_tau_base`: 277.668
- `standard_gru_delay10`: 260.382
- `fixed_tau_10_delay10`: 213.121
- `learned_tau_delay10`: 182.478
- `standard_gru_flicker03`: 242.702
- `fixed_tau_10_flicker03`: 251.349
- `learned_tau_flicker03`: 249.542
- `standard_gru_speed`: 318.210
- `fixed_tau_10_speed`: 288.718
- `learned_tau_speed_objdt`: 300.463
- `learned_tau_speed_subjdt`: 353.998

## Delay-focused regularization check

- Roots:
  - `runs/sweeps/delay_tune/lm0_lv0`
  - `runs/sweeps/delay_tune/lm1e-4_lv0`
  - `runs/sweeps/delay_tune/lm1e-3_lv1e-2`

`learned_tau_delay10` final means (5 seeds):

- `lm0_lv0`: 160.101
- `lm1e-4_lv0`: 187.510
- `lm1e-3_lv1e-2`: 232.915

## Tradeoff check for learned conditions only (stage1_full learned-only)

- Root: `runs/sweeps/stage1_full_lm1e-3_lv1e-2_learned_only`
- Summary: `runs/sweeps/stage1_full_lm1e-3_lv1e-2_learned_only/aggregate/condition_summary.csv`

Compared to `lm0_lv1e-2`, `lm1e-3_lv1e-2`:

- improves `learned_tau_delay10` (182.478 -> 232.915)
- slightly improves `learned_tau_flicker03` (249.542 -> 252.140)
- hurts `learned_tau_base` (277.668 -> 209.623)
- hurts speed variants (`objdt`: 300.463 -> 287.274, `subjdt`: 353.998 -> 334.205)

## Environment extension

### Acrobot (stage1_core, lambda_mean=0, lambda_var=1e-2)

- Root: `runs/sweeps/acrobot_stage1_core_lm0_lv1e-2`
- Summary: `runs/sweeps/acrobot_stage1_core_lm0_lv1e-2/aggregate/condition_summary.csv`

Final means:

- `standard_gru_base`: -377.524 (best among tested)
- `learned_tau_base`: -425.173

### Pendulum base-only (stage1_core, lambda_mean=0, lambda_var=1e-2)

- Root: `runs/sweeps/pendulum_stage1_core_lm0_lv1e-2`
- Summary: `runs/sweeps/pendulum_stage1_core_lm0_lv1e-2/aggregate/condition_summary.csv`

Final means:

- `standard_gru_base`: -1246.964
- `learned_tau_base`: -1291.050

### Pendulum variable-speed focus (4 conditions, 5 seeds)

- Root: `runs/sweeps/pendulum_speed_stage1_lm0_lv1e-2`
- Summary: `runs/sweeps/pendulum_speed_stage1_lm0_lv1e-2/aggregate/condition_summary.csv`
- Figures:
  - `runs/sweeps/pendulum_speed_stage1_lm0_lv1e-2/aggregate/learning_curves.png`
  - `runs/sweeps/pendulum_speed_stage1_lm0_lv1e-2/aggregate/final_performance.png`

Final means:

- `standard_gru_speed`: -1194.068
- `fixed_tau_10_speed`: -1115.918
- `learned_tau_speed_objdt`: -1139.003
- `learned_tau_speed_subjdt`: -1120.663

Interpretation:

- learned internal time helps vs standard in variable-speed Pendulum
- subjective discounting (`internal_tau`) improves over objective variant (`env_dt`)
- fixed high tau is still a strong control baseline

## Current decision

- Keep `lambda_mean=0`, `lambda_var=1e-2` as the primary global Stage-1 setting.
- Keep `lambda_mean=1e-3`, `lambda_var=1e-2` as a delay-biased alternative (appendix ablation).
- For the next paper plot set, prioritize:
  - CartPole `stage1_full` (main figure set)
  - Pendulum variable-speed subset (cross-environment support for temporal reparameterization claim)
