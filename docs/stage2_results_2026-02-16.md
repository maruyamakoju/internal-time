# Stage-2 Delay Results (2026-02-16)

This file records Stage-2 delay experiments after Stage-1 freeze (`stage1_defense_v2`).

## Release Summary (Fixed / Trend / Limitation)

- **Fixed:** non-flicker `delay=10` + warmup(both20k) gives `selfmodel - learned` **final > 0** with **CI95 low > 0** (`n_pairs=30`).
- **Trend:** warmup delay dependence by group contrast **`{10,20} - 0`** is **not significant** even with `n=30` (CI crosses zero).
- **Limitation:** with `env.flicker_prob=0.1`, warmup `delay=10` **does not hold** at `n=5`; **AUC is significantly negative**.

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

- `learned_tau_delay10`: `214.513 +/- 85.890`
- `learned_tau_delay10_selfmodel`: `229.079 +/- 90.789`

Stage-2 signal at best setting:

- `corr/tau_pred_error_mean`: `0.666 +/- 0.115`
- `pred_error/mean_mean`: `0.086 +/- 0.021`
- `loss/self_model_mean`: `0.011 +/- 0.005`

Interpretation:

- With delay-biased time regularization and tuned `lambda_self`, Stage-2 improves over Stage-1 learned delay baseline.
- Final performance improves but AUC is lower (`160.669` vs `179.016`), so sample efficiency remains a tuning target.

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

- `learned_tau_delay10`: final `214.513 +/- 85.890`, AUC `179.016 +/- 34.679`
- `learned_tau_delay10_selfmodel`: final `229.079 +/- 90.789`, AUC `160.669 +/- 31.596`
- `learned_tau_delay10_selfmodel_noerr`: final `166.977 +/- 52.895`, AUC `150.569 +/- 39.343`

Signal columns:

- `learned_tau_delay10_selfmodel`: `corr/tau_pred_error=0.666 +/- 0.115`
- `learned_tau_delay10_selfmodel_noerr`: `corr/tau_pred_error=0.706 +/- 0.106`

Interpretation:

- Turning OFF the `pred_error -> tau` pathway reduces final performance (`229.079 -> 166.977`).
- With self-model training still enabled, performance drops below the learned Stage-1 delay baseline in this setting.
- `corr/tau_pred_error` can stay non-zero as an observational correlation from shared latent dynamics; the intervention result above is the causal evidence.
- This remains directionally consistent with the causal claim that Stage-2 improvement is carried by prediction-error-based time modulation.

## Paired Effects (Seed-Matched, No Re-Training)

We computed paired seed-wise deltas from:

- `runs/sweeps/stage2_delay_delaybiased_ls3e-1/aggregate/per_run_summary.csv`
- Script: `python -m internal_time_rl.analysis.paired_effects ...`

Artifacts:

- `docs/stage2_delay_paired_effects_2026-02-16.csv`
- `docs/stage2_delay_paired_effects_seeds_2026-02-16.csv`
- `docs/stage2_delay_paired_effects_2026-02-16.tex`

Main paired results (`delta = selfmodel - comparator`, bootstrap 95% CI):

- vs `learned_tau_delay10`:
  - `final_mean`: `+14.567` (CI `[-49.764, 85.432]`)
  - `AUC`: `-18.348` (CI `[-36.348, 1.274]`)
- vs `learned_tau_delay10_selfmodel_noerr`:
  - `final_mean`: `+62.102` (CI `[7.224, 124.360]`)
  - `AUC`: `+10.099` (CI `[-10.801, 31.636]`)

Interpretation:

- `selfmodel - noerr` final delta is now strictly positive at 95% CI lower bound (`+7.224`), which strengthens the causal ablation claim.
- `selfmodel - learned` remains direction-positive but not significant at current variance.
- AUC differences are unstable across comparators, so sample-efficiency claims are not yet fixed.

## Delay-Length Sweep (0 / 5 / 10 / 20)

Protocol:

- Root: `runs/sweeps/stage2_delay_sweep`
- Conditions: `learned_tau_delay10` vs `learned_tau_delay10_selfmodel`
- Shared overrides:
  - `time_reg.lambda_mean=1e-3`
  - `time_reg.lambda_var=1e-2`
  - `train.lambda_self=0.3`
- Budget: 10 seeds, 200k per seed, per delay

Summary artifacts:

- `docs/stage2_delay_sweep_summary_2026-02-16.csv`
- `docs/stage2_delay_sweep_summary_2026-02-16.tex`
- `docs/figures/fig_stage2_delay_sweep_final.png`
- `docs/figures/fig_stage2_delay_sweep_paired_delta.png`

Paired final deltas (`selfmodel - learned`):

- `delay=0`: `-23.765` (CI `[-80.421, 28.852]`)
- `delay=5`: `-6.342` (CI `[-64.759, 55.565]`)
- `delay=10`: `+16.513` (CI `[-75.805, 120.497]`)
- `delay=20`: `+24.762` (CI `[-19.969, 78.132]`)

Paired AUC deltas (`selfmodel - learned`):

- `delay=0`: `-6.594`
- `delay=5`: `+5.239`
- `delay=10`: `-14.630`
- `delay=20`: `-20.454`

Interpretation:

- Direction is still delay-dependent but non-monotonic, and per-delay CIs still cross zero at this budget.
- `delay=0/5` stay near zero or negative; `delay=10/20` are positive in mean but not yet conclusive.
- Warmup/schedule is still the lowest-cost next step if we need stronger AUC and tighter separation.

## Delay Group Difference (Low vs High, No Re-Training)

We computed seed-matched group contrast from delay sweep deltas:

- Low-delay group: `delay={0,5}`
- High-delay group: `delay={10,20}`
- Delta definition: `selfmodel - learned`

Artifacts:

- `docs/stage2_delay_group_diff_2026-02-16.csv`
- `docs/stage2_delay_group_diff_per_seed_2026-02-16.csv`
- `docs/stage2_delay_group_diff_2026-02-16.tex`

Results (bootstrap 95% CI):

- `final_mean`:
  - Low-group delta mean: `-15.054`
  - High-group delta mean: `+20.638`
  - Group diff (`high - low`): `+35.692` (CI `[-44.409, 113.127]`)
- `AUC`:
  - Low-group delta mean: `-0.677`
  - High-group delta mean: `-17.542`
  - Group diff (`high - low`): `-16.864` (CI `[-46.469, 13.513]`)

Interpretation:

- The trend supports stronger final-score gains at higher delays, but CI still crosses zero.
- Additional power, or a lower-variance training schedule (warmup), is required for a hard significance claim on delay-group contrast.

## Warmup Search (Performance-Focused)

Main-table artifacts:

- `docs/stage2_main_table_2026-02-16.csv`
- `docs/stage2_main_table_2026-02-16.tex`

We added two schedule controls:

- `train.pred_error_tau_warmup_timesteps`
- `train.lambda_self_warmup_timesteps`

and evaluated delay-biased `delay=10` with `learned_tau_delay10` vs `learned_tau_delay10_selfmodel`.

Settings compared:

- `both50k`: both warmups = `50000`
- `tauonly50k`: `pred_error_tau_warmup=50000`, `lambda_self_warmup=0`
- `lambdaonly50k`: `pred_error_tau_warmup=0`, `lambda_self_warmup=50000`
- `both20k`: both warmups = `20000`

Machine-readable summary:

- `docs/stage2_warmup_search_2026-02-16.csv`
- `docs/stage2_delay_warmup_both20k_paired_2026-02-16.csv`
- `docs/stage2_delay_warmup_both20k_noerr_paired_2026-02-16.csv`
- `docs/stage2_delay_warmup_both20k_noerr_vs_learned_paired_2026-02-16.csv`

Seed-30 confirm (`both20k`, root `runs/sweeps/stage2_delay_delaybiased_warmup_both20k_v1`):

- Paired final delta (`selfmodel - learned`, `n_pairs=30`): `+44.040` (CI `[3.169, 84.658]`)
- Paired AUC delta (`selfmodel - learned`, `n_pairs=30`): `+13.615` (CI `[-11.065, 40.145]`)
- Paired final delta (`selfmodel - noerr`, `n_pairs=30`): `+19.698` (CI `[-28.252, 65.353]`)
- Paired AUC delta (`selfmodel - noerr`, `n_pairs=30`): `+4.307` (CI `[-23.792, 32.134]`)
- Paired final delta (`noerr - learned`, `n_pairs=30`): `+24.342` (CI `[-10.996, 64.059]`)
- Paired AUC delta (`noerr - learned`, `n_pairs=30`): `+9.309` (CI `[-7.395, 26.280]`)

Interpretation:

- With statistical power increased to 30 paired seeds, `selfmodel - learned` final performance is now positive with CI lower bound above zero.
- This fixes the main performance claim for warmup on `delay=10` under the current protocol.
- AUC remains direction-positive but not conclusive at 95% CI.
- Under warmup, `selfmodel - noerr` is direction-positive but not conclusive at 95% CI, so causal claims for warmup remain weaker than the non-warmup noerr ablation.
- `noerr - learned` is also direction-positive but inconclusive, which suggests warmup can partially lift noerr and reduce separation between `selfmodel` and `noerr`.
- `tauonly50k` and `lambdaonly50k` both underperformed, indicating that balanced short warmup is the stable direction rather than one-sided long warmup.

## Warmup Delay Sweep Refresh (0 / 10 / 20)

To place the strong `delay=10` result in the same sweep context, we refreshed warmup delay analysis:

- Base root: `runs/sweeps/stage2_delay_sweep_warmup_both20k_v1`
- Conditions: `learned_tau_delay10` vs `learned_tau_delay10_selfmodel`
- Warmup schedule: `pred_error_tau_warmup_timesteps=20000`, `lambda_self_warmup_timesteps=20000`
- Time regularization: `time_reg.lambda_mean=1e-3`, `time_reg.lambda_var=1e-2`
- Delay budgets:
  - `delay=0`: 30 seeds
  - `delay=10`: 30 seeds (imported aggregate from `runs/sweeps/stage2_delay_delaybiased_warmup_both20k_v1`)
  - `delay=20`: 30 seeds

Artifacts:

- `docs/stage2_delay_warmup_delay0_paired_2026-02-16.csv`
- `docs/stage2_delay_warmup_delay20_paired_2026-02-16.csv`
- `docs/stage2_delay_sweep_warmup_both20k_summary_2026-02-16.csv`
- `docs/stage2_delay_sweep_warmup_both20k_summary_2026-02-16.tex`
- `docs/figures/fig_stage2_delay_sweep_warmup_both20k_final.png`
- `docs/figures/fig_stage2_delay_sweep_warmup_both20k_paired_delta.png`

Paired final deltas (`selfmodel - learned`):

- `delay=0` (`n=30`): `+16.910` (CI `[-34.583, 73.464]`)
- `delay=10` (`n=30`): `+44.040` (CI `[3.169, 84.658]`)
- `delay=20` (`n=30`): `+14.472` (CI `[-10.750, 39.046]`)

Paired AUC deltas (`selfmodel - learned`):

- `delay=0` (`n=30`): `+7.073` (CI `[-18.177, 34.160]`)
- `delay=10` (`n=30`): `+13.615` (CI `[-11.065, 40.145]`)
- `delay=20` (`n=30`): `-9.176` (CI `[-26.908, 7.256]`)

Interpretation:

- The strongest and only fixed positive final-effect point remains `delay=10`.
- `delay=0` and `delay=20` remain inconclusive at current confidence.

## Warmup Delay Group Difference (Primary: {10,20} vs 0)

We computed warmup group contrast with matched seeds:

- Low-delay group: `delay={0}`
- High-delay group: `delay={10,20}`
- Delta definition: `selfmodel - learned`

Artifacts:

- `docs/stage2_delay_warmup_groupdiff_0_vs_10_20_2026-02-16.csv`
- `docs/stage2_delay_warmup_groupdiff_0_vs_10_20_per_seed_2026-02-16.csv`
- `docs/stage2_delay_warmup_groupdiff_0_vs_10_20_2026-02-16.tex`

Results (bootstrap 95% CI):

- `final_mean`: group diff (`{10,20} - 0`) = `+12.346` (CI `[-50.539, 70.687]`)
- `AUC`: group diff (`{10,20} - 0`) = `-4.853` (CI `[-36.271, 25.431]`)

Interpretation:

- Even after balancing `delay=20` to 30 seeds and averaging high-delay points, group-level CI still crosses zero.
- Warmup delay dependence remains a trend-level observation; fixed claim remains the per-delay `delay=10` result.

Reference group contrasts:

- `10 - 0` final: `+27.130` (CI `[-42.314, 90.577]`), AUC: `+6.542` (CI `[-30.845, 44.310]`)
- `20 - 0` final: `-2.439` (CI `[-62.867, 56.197]`), AUC: `-16.249` (CI `[-46.497, 12.539]`)
- Artifacts:
  - `docs/stage2_delay_warmup_groupdiff_0_vs_10_2026-02-16.csv`
  - `docs/stage2_delay_warmup_groupdiff_0_vs_20_2026-02-16.csv`

## Robustness Check (Flicker = 0.1, Warmup both20k, delay=10)

To test whether the fixed `delay=10` warmup gain survives observation corruption, we ran:

- Root: `runs/sweeps/stage2_delay_delaybiased_warmup_both20k_flicker01_v1`
- Conditions: `learned_tau_delay10` vs `learned_tau_delay10_selfmodel`
- Budget: `5` seeds, `200k` timesteps/seed
- Overrides:
  - `env.reward_delay=10`
  - `env.flicker_prob=0.1`
  - `time_reg.lambda_mean=1e-3`
  - `time_reg.lambda_var=1e-2`
  - `train.lambda_self=0.3`
  - `train.pred_error_tau_warmup_timesteps=20000`
  - `train.lambda_self_warmup_timesteps=20000`

Artifacts:

- `docs/stage2_delay_warmup_flicker01_paired_2026-02-16.csv`
- `docs/stage2_delay_warmup_flicker01_paired_seeds_2026-02-16.csv`
- `docs/stage2_delay_warmup_flicker01_paired_2026-02-16.tex`
- `docs/figures/fig_stage2_delay_warmup_flicker01_learning_curves.png`
- `docs/figures/fig_stage2_delay_warmup_flicker01_final.png`

Paired deltas (`selfmodel - learned`):

- `final_mean` (`n=5`): `-45.002` (CI `[-115.582, 23.081]`)
- `AUC` (`n=5`): `-21.048` (CI `[-36.080, -6.538]`)

Interpretation:

- Under `flicker_prob=0.1`, the warmup `delay=10` advantage does not hold at this budget.
- Final-score delta is inconclusive and direction-negative in mean; AUC is significantly negative.
- For release claims, keep the fixed result scoped to the non-flicker `delay=10` setting; treat flicker robustness as a current limitation.

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
- `runs/sweeps/stage2_delay_sweep/delay0/aggregate/final_performance.png`
- `runs/sweeps/stage2_delay_sweep/delay5/aggregate/final_performance.png`
- `runs/sweeps/stage2_delay_sweep/delay10/aggregate/final_performance.png`
- `runs/sweeps/stage2_delay_sweep/delay20/aggregate/final_performance.png`
- `docs/figures/fig_stage2_delay_sweep_final.png`
- `docs/figures/fig_stage2_delay_sweep_paired_delta.png`
- `runs/sweeps/stage2_delay_delaybiased_warmup_both20k_v1/aggregate/learning_curves.png`
- `runs/sweeps/stage2_delay_delaybiased_warmup_both20k_v1/aggregate/final_performance.png`
- `docs/figures/fig_stage2_delay_warmup_both20k_learning_curves.png`
- `docs/figures/fig_stage2_delay_warmup_both20k_final.png`
- `docs/figures/fig_stage2_delay_sweep_warmup_both20k_final.png`
- `docs/figures/fig_stage2_delay_sweep_warmup_both20k_paired_delta.png`
