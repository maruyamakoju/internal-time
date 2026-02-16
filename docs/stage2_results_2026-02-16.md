# Stage-2 Delay Results (2026-02-16)

This file records the first Stage-2 run after Stage-1 freeze (`stage1_defense_v2`).

## Protocol

- Suite: `stage2_delay`
- Root: `runs/sweeps/stage2_delay_lm0_lv1e-2`
- Budget: 5 seeds, 200k timesteps/seed
- Shared overrides:
  - `time_reg.lambda_mean=0`
  - `time_reg.lambda_var=1e-2`
- Stage-2 condition:
  - `learned_tau_delay10_selfmodel`
  - `model.use_self_model=true`
  - `train.lambda_self=0.1`

## Final Performance (`episode/return_mean_20`)

From `runs/sweeps/stage2_delay_lm0_lv1e-2/aggregate/condition_summary.csv`:

- `standard_gru_delay10`: `260.382 +/- 140.283`
- `fixed_tau_10_delay10`: `213.121 +/- 91.623`
- `learned_tau_delay10`: `182.478 +/- 31.951`
- `learned_tau_delay10_selfmodel`: `163.119 +/- 20.858`

## Stage-2 Logging Signals

For `learned_tau_delay10_selfmodel`:

- `corr/tau_pred_error_mean`: `0.579 +/- 0.162`
- `pred_error/mean_mean`: `0.070 +/- 0.032`
- `loss/self_model_mean`: `0.011 +/- 0.012`

Interpretation:

- The new Stage-2 signal is active and non-degenerate (`corr/tau_pred_error` is finite and positive).
- At current settings, Stage-2 does **not** improve delay return over Stage-1 learned baseline.

## Immediate Next Tuning Axis

Keep architecture fixed and sweep only `train.lambda_self` first, e.g.:

- `0.01`
- `0.03`
- `0.1` (current)

Then rerun `stage2_delay` and compare:

- `learned_tau_delay10_selfmodel` vs `learned_tau_delay10`
- `corr/tau_pred_error`
- `pred_error/mean` trend

## Figures

- `runs/sweeps/stage2_delay_lm0_lv1e-2/aggregate/learning_curves.png`
- `runs/sweeps/stage2_delay_lm0_lv1e-2/aggregate/final_performance.png`
- `docs/figures/fig_stage2_delay_learning_curves.png`
- `docs/figures/fig_stage2_delay_final.png`
