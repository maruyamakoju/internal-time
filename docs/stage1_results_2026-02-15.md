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

## Defense controls (added 2026-02-15)

### 1) Gamma-tuning control on CartPole speed

Purpose: test whether speed gains are only from selecting a better fixed `gamma`.

Roots:

- `runs/sweeps/controls/cartpole_speed_gamma097`
- `runs/sweeps/controls/cartpole_speed_gamma099`
- `runs/sweeps/controls/cartpole_speed_gamma0995`
- `runs/sweeps/controls/cartpole_speed_gamma0999`

Final means (`episode/return_mean_20`, 5 seeds):

- gamma 0.97:
  - `standard_gru_speed`: 333.649
  - `fixed_tau_10_speed`: 369.006
- gamma 0.99:
  - `standard_gru_speed`: 318.210
  - `fixed_tau_10_speed`: 288.718
- gamma 0.995:
  - `standard_gru_speed`: 297.135
  - `fixed_tau_10_speed`: 277.740
- gamma 0.999:
  - `standard_gru_speed`: 272.545
  - `fixed_tau_10_speed`: 303.819

Reference from main run (`runs/sweeps/stage1_full_lm0_lv1e-2`):

- `learned_tau_speed_subjdt`: 353.998

Result:

- `learned_tau_speed_subjdt` beats best gamma-tuned `standard_gru_speed` (353.998 vs 333.649).
- best gamma-tuned `fixed_tau_10_speed` remains stronger (369.006).

### 2) Constant subjective-discount control (non-adaptive)

Purpose: test whether adaptivity is necessary vs fixed subjective discount.

Roots:

- `runs/sweeps/controls/const_tau_c1`
- `runs/sweeps/controls/const_tau_c2`
- `runs/sweeps/controls/const_tau_c3`

Setup: `standard_gru_speed` with `train.discount_mode=internal_tau` and fixed `model.standard_tau_proxy=c`.

Final means:

- `c=1`: 315.075
- `c=2`: 309.552
- `c=3`: 288.438

Reference:

- `learned_tau_speed_subjdt`: 353.998

Result:

- adaptive subjective time beats all tested constant subjective-discount controls.

### 3) Cross-env speed extension (Acrobot)

Root:

- `runs/sweeps/acrobot_speed_stage1_lm0_lv1e-2`

Final means:

- `standard_gru_speed`: -125.209
- `fixed_tau_10_speed`: -134.833
- `learned_tau_speed_objdt`: -185.355
- `learned_tau_speed_subjdt`: -199.451

Result:

- speed benefit of learned internal time does not transfer to Acrobot at this budget/config.

### 4) Fairness closure: same-gamma controls for learned/const/fixed

Purpose: close the last fairness gap by evaluating learned/const/fixed controls under the same `gamma=0.97`.

Roots:

- `runs/sweeps/controls/cartpole_speed_learned_gamma097`
- `runs/sweeps/controls/const_tau_c1_gamma097`
- `runs/sweeps/controls/fixed_tau10_internal_tau_gamma097`

Final means:

- learned (`gamma=0.97`):
  - `learned_tau_speed_objdt`: 352.650
  - `learned_tau_speed_subjdt`: 288.242
- constant subjective discount (`gamma=0.97`, `c=1`):
  - `standard_gru_speed`: 274.278
- fixed tau with subjective discount (`gamma=0.97`):
  - `fixed_tau_10_speed`: 252.442

Result:

- At `gamma=0.97`, learned `objdt` remains strong and stays above gamma-tuned `standard_gru_speed` (`352.650` vs `333.649`).
- Learned `subjdt` is gamma-sensitive (`353.998 @ gamma=0.99` to `288.242 @ gamma=0.97`).
- Even at `gamma=0.97`, adaptive learned `subjdt` is above non-adaptive subjective controls (`288.242` vs `274.278` and `252.442`).

### Story lock after controls

- Keep the main positive claim focused on variable-speed settings (CartPole, supported by Pendulum).
- Separate two effects explicitly:
  - adaptive internal-time dynamics (strong under `objdt`, including `gamma=0.97`)
  - adaptive subjective discounting (`subjdt`), which helps at default gamma but is gamma-sensitive
- Keep the non-adaptive control claim:
  - learned `subjdt` > constant subjective discount controls
  - learned `subjdt` > fixed tau + subjective discount control
- Do not claim universal superiority over all fixed controls, since gamma-tuned `fixed_tau_10_speed` (`env_dt`) can be stronger on CartPole speed.
- Treat Acrobot speed as a negative/generalization-limit result.

### Figure bundle (fixed names)

Copied into `docs/figures`:

- `docs/figures/fig_cartpole_stage1_full_learning_curves.png`
- `docs/figures/fig_cartpole_stage1_full_final.png`
- `docs/figures/fig_cartpole_speed_gamma097_final.png`
- `docs/figures/fig_cartpole_speed_gamma099_final.png`
- `docs/figures/fig_cartpole_speed_gamma0995_final.png`
- `docs/figures/fig_cartpole_speed_gamma0999_final.png`
- `docs/figures/fig_cartpole_speed_const_tau_c1_final.png`
- `docs/figures/fig_cartpole_speed_const_tau_c1_gamma097_final.png`
- `docs/figures/fig_cartpole_speed_const_tau_c2_final.png`
- `docs/figures/fig_cartpole_speed_const_tau_c3_final.png`
- `docs/figures/fig_cartpole_speed_fixed_tau10_internal_tau_gamma097_final.png`
- `docs/figures/fig_cartpole_speed_learned_gamma097_final.png`
- `docs/figures/fig_pendulum_speed_final.png`
- `docs/figures/fig_acrobot_speed_final.png`

Machine-readable summary table:

- `docs/stage1_controls_summary_2026-02-15.csv`
