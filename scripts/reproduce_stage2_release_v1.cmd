@echo off
setlocal enabledelayedexpansion
pushd "%~dp0.."

set WARMUP_DELAY10_ROOT=runs\sweeps\stage2_delay_delaybiased_warmup_both20k_v1
set WARMUP_SWEEP_ROOT=runs\sweeps\stage2_delay_sweep_warmup_both20k_v1
set NONWARMUP_ROOT=runs\sweeps\stage2_delay_delaybiased_ls3e-1
set FLICKER_ROOT=runs\sweeps\stage2_delay_delaybiased_warmup_both20k_flicker01_v1

python -V || exit /b 1
python -m compileall internal_time_rl >NUL || exit /b 1

REM ---- Non-warmup causal evidence (seed15) ----
python -m internal_time_rl.analysis.aggregate_runs --root "%NONWARMUP_ROOT%" --metric episode/return_mean_20 --last-n 10 --points 200 || exit /b 1
python -m internal_time_rl.analysis.paired_effects --input "%NONWARMUP_ROOT%\aggregate\per_run_summary.csv" --anchor learned_tau_delay10_selfmodel --comparators learned_tau_delay10 learned_tau_delay10_selfmodel_noerr --metrics final_mean auc --min-n-pairs 15 --out-csv docs\stage2_delay_paired_effects_2026-02-16.csv --out-per-seed docs\stage2_delay_paired_effects_seeds_2026-02-16.csv --out-tex docs\stage2_delay_paired_effects_2026-02-16.tex || exit /b 1

REM ---- Warmup delay=10 main result (seed30) ----
python -m internal_time_rl.analysis.aggregate_runs --root "%WARMUP_DELAY10_ROOT%" --metric episode/return_mean_20 --last-n 10 --points 200 || exit /b 1
python -m internal_time_rl.analysis.paired_effects --input "%WARMUP_DELAY10_ROOT%\aggregate\per_run_summary.csv" --anchor learned_tau_delay10_selfmodel --comparators learned_tau_delay10 learned_tau_delay10_selfmodel_noerr --metrics final_mean auc --min-n-pairs 30 --out-csv docs\stage2_delay_warmup_both20k_paired_2026-02-16.csv --out-per-seed docs\stage2_delay_warmup_both20k_paired_seeds_2026-02-16.csv --out-tex docs\stage2_delay_warmup_both20k_paired_2026-02-16.tex || exit /b 1

REM ---- Ensure sweep root has delay10 aggregate (import from delay10 warmup root) ----
if not exist "%WARMUP_SWEEP_ROOT%\delay10\aggregate" mkdir "%WARMUP_SWEEP_ROOT%\delay10\aggregate"
copy /Y "%WARMUP_DELAY10_ROOT%\aggregate\condition_summary.csv" "%WARMUP_SWEEP_ROOT%\delay10\aggregate\" >NUL
copy /Y "%WARMUP_DELAY10_ROOT%\aggregate\per_run_summary.csv" "%WARMUP_SWEEP_ROOT%\delay10\aggregate\" >NUL

REM ---- Warmup sweep summary/plots (0/10/20) ----
python -m internal_time_rl.analysis.summarize_stage2_delay_sweep --base-root "%WARMUP_SWEEP_ROOT%" --delays 0 10 20 --baseline-condition learned_tau_delay10 --selfmodel-condition learned_tau_delay10_selfmodel --out-csv docs\stage2_delay_sweep_warmup_both20k_summary_2026-02-16.csv --out-tex docs\stage2_delay_sweep_warmup_both20k_summary_2026-02-16.tex || exit /b 1
python -m internal_time_rl.analysis.plot_stage2_delay_sweep --input docs\stage2_delay_sweep_warmup_both20k_summary_2026-02-16.csv --out-final docs\figures\fig_stage2_delay_sweep_warmup_both20k_final.png --out-delta docs\figures\fig_stage2_delay_sweep_warmup_both20k_paired_delta.png || exit /b 1

REM ---- Groupdiff high={10,20} vs low={0} ----
python -m internal_time_rl.analysis.stage2_delay_group_difference --base-root "%WARMUP_SWEEP_ROOT%" --delays 0 10 20 --low-delays 0 --high-delays 10 20 --baseline-condition learned_tau_delay10 --selfmodel-condition learned_tau_delay10_selfmodel --metrics final_mean auc --out-csv docs\stage2_delay_warmup_groupdiff_0_vs_10_20_2026-02-16.csv --out-tex docs\stage2_delay_warmup_groupdiff_0_vs_10_20_2026-02-16.tex --out-per-seed docs\stage2_delay_warmup_groupdiff_0_vs_10_20_per_seed_2026-02-16.csv || exit /b 1

REM ---- Flicker robustness (analysis only) ----
if exist "%FLICKER_ROOT%" (
  python -m internal_time_rl.analysis.aggregate_runs --root "%FLICKER_ROOT%" --metric episode/return_mean_20 --last-n 10 --points 200 || exit /b 1
  python -m internal_time_rl.analysis.paired_effects --input "%FLICKER_ROOT%\aggregate\per_run_summary.csv" --anchor learned_tau_delay10_selfmodel --comparators learned_tau_delay10 --metrics final_mean auc --min-n-pairs 5 --out-csv docs\stage2_delay_warmup_flicker01_paired_2026-02-16.csv --out-per-seed docs\stage2_delay_warmup_flicker01_paired_seeds_2026-02-16.csv --out-tex docs\stage2_delay_warmup_flicker01_paired_2026-02-16.tex || exit /b 1
  if not exist "docs\figures" mkdir "docs\figures"
  copy /Y "%FLICKER_ROOT%\aggregate\learning_curves.png" "docs\figures\fig_stage2_delay_warmup_flicker01_learning_curves.png" >NUL
  copy /Y "%FLICKER_ROOT%\aggregate\final_performance.png" "docs\figures\fig_stage2_delay_warmup_flicker01_final.png" >NUL
) else (
  echo NOTE: FLICKER_ROOT not found, skipping flicker aggregation: %FLICKER_ROOT%
)

REM ---- Main table ----
python -m internal_time_rl.analysis.export_stage2_main_table --out-csv docs\stage2_main_table_2026-02-16.csv --out-tex docs\stage2_main_table_2026-02-16.tex || exit /b 1

echo.
echo Reproduce completed.
popd
exit /b 0
