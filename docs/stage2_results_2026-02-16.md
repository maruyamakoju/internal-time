# Stage-2 Delay Results (2026-02-16)

This file records Stage-2 delay experiments after Stage-1 freeze (`stage1_defense_v2`).

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

## Lambda_self Sweep (Architecture Fixed)

We ran the self-model condition only, keeping all other settings fixed:

- `runs/sweeps/stage2_delay_ls1e-2_lm0_lv1e-2` (`lambda_self=0.01`)
- `runs/sweeps/stage2_delay_ls3e-2_lm0_lv1e-2` (`lambda_self=0.03`)
- `runs/sweeps/stage2_delay_ls3e-1_lm0_lv1e-2` (`lambda_self=0.3`)

Final means (`learned_tau_delay10_selfmodel`, 5 seeds):

- `lambda_self=0.01`: `201.725 +/- 75.494`
- `lambda_self=0.03`: `185.421 +/- 67.385`
- `lambda_self=0.3`: `250.933 +/- 53.486`

Signal columns:

- `lambda_self=0.01`: `corr/tau_pred_error=0.363`, `pred_error/mean=0.119`
- `lambda_self=0.03`: `corr/tau_pred_error=0.602`, `pred_error/mean=0.113`
- `lambda_self=0.3`: `corr/tau_pred_error=-0.164`, `pred_error/mean=0.067`

Decision:

- Adopt `lambda_self=0.3` as the best setting by final return.

Machine-readable sweep summary:

- `docs/stage2_lambda_sweep_2026-02-16.csv`

## Delay-Biased Time Reg Check (BEST lambda_self)

Using `lambda_self=0.3`, we compared `learned` vs `selfmodel` under delay-biased time regularization:

- Root: `runs/sweeps/stage2_delay_delaybiased_ls3e-1`
- Overrides:
  - `time_reg.lambda_mean=1e-3`
  - `time_reg.lambda_var=1e-2`

From `runs/sweeps/stage2_delay_delaybiased_ls3e-1/aggregate/condition_summary.csv`:

- `learned_tau_delay10`: `232.915 +/- 99.640`
- `learned_tau_delay10_selfmodel`: `294.247 +/- 105.984`

Stage-2 signal at best setting:

- `corr/tau_pred_error_mean`: `0.591 +/- 0.109`
- `pred_error/mean_mean`: `0.090 +/- 0.015`
- `loss/self_model_mean`: `0.011 +/- 0.004`

Interpretation:

- With delay-biased time regularization and tuned `lambda_self`, Stage-2 improves over Stage-1 learned delay baseline.
- Final performance improves but AUC is lower (`171.216` vs `198.547`), so sample efficiency remains a tuning target.

## Ablation: pred_error -> tau OFF (One-Bit)

To isolate causal contribution of the error-to-time pathway, we added:

- `model.use_pred_error_for_tau=false`
- Implementation detail for fairness:
  - Keep the same time-head input dimensionality in self-model mode.
  - Pass `zeros_like(pred_error)` as the extra feature when OFF.

Run:

- Added condition: `learned_tau_delay10_selfmodel_noerr`
- Root: `runs/sweeps/stage2_delay_delaybiased_ls3e-1`
- Overrides:
  - `time_reg.lambda_mean=1e-3`
  - `time_reg.lambda_var=1e-2`
  - `train.lambda_self=0.3`

From `runs/sweeps/stage2_delay_delaybiased_ls3e-1/aggregate/condition_summary.csv`:

- `learned_tau_delay10`: final `232.915 +/- 99.640`, AUC `198.547 +/- 34.286`
- `learned_tau_delay10_selfmodel`: final `294.247 +/- 105.984`, AUC `171.216 +/- 27.657`
- `learned_tau_delay10_selfmodel_noerr`: final `184.063 +/- 57.756`, AUC `175.358 +/- 46.799`

Signal columns:

- `learned_tau_delay10_selfmodel`: `corr/tau_pred_error=0.591 +/- 0.109`
- `learned_tau_delay10_selfmodel_noerr`: `corr/tau_pred_error=0.686 +/- 0.081`

Interpretation:

- Turning OFF the `pred_error -> tau` pathway removes the Stage-2 gain (`294.247 -> 184.063`).
- With self-model training still enabled, performance drops below the learned Stage-1 delay baseline in this setting.
- `corr/tau_pred_error` can stay non-zero as an observational correlation from shared latent dynamics; the intervention result above is the causal evidence.
- This supports the causal claim that Stage-2 improvement in delay-biased setup comes from using prediction error to modulate internal time.

## Figures

- `runs/sweeps/stage2_delay_lm0_lv1e-2/aggregate/learning_curves.png`
- `runs/sweeps/stage2_delay_lm0_lv1e-2/aggregate/final_performance.png`
- `docs/figures/fig_stage2_delay_learning_curves.png`
- `docs/figures/fig_stage2_delay_final.png`
- `runs/sweeps/stage2_delay_ls3e-1_lm0_lv1e-2/aggregate/learning_curves.png`
- `runs/sweeps/stage2_delay_ls3e-1_lm0_lv1e-2/aggregate/final_performance.png`
- `docs/figures/fig_stage2_delay_lambda_sweep_learning_curves.png`
- `docs/figures/fig_stage2_delay_lambda_sweep_final.png`
- `runs/sweeps/stage2_delay_delaybiased_ls3e-1/aggregate/learning_curves.png`
- `runs/sweeps/stage2_delay_delaybiased_ls3e-1/aggregate/final_performance.png`
- `docs/figures/fig_stage2_delay_delaybiased_best_learning_curves.png`
- `docs/figures/fig_stage2_delay_delaybiased_best_final.png`
- `docs/figures/fig_stage2_delay_delaybiased_ablation_noerr_learning_curves.png`
- `docs/figures/fig_stage2_delay_delaybiased_ablation_noerr_final.png`
