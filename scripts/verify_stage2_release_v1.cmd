@echo off
setlocal
pushd "%~dp0.."

python -V || exit /b 1
python -m compileall internal_time_rl >NUL || exit /b 1

REM ---- Check Release Summary is in the document HEAD ----
python -c "from pathlib import Path; lines=Path(r'docs/stage2_results_2026-02-16.md').read_text(encoding='utf-8').splitlines(); head='\n'.join(lines[:60]); assert 'Fixed:' in head and 'Trend:' in head and 'Limitation:' in head, 'Missing Fixed/Trend/Limitation in document head'" || exit /b 1

REM ---- Warmup delay10 fixed claim (n_pairs>=30, CI low > 0) ----
python -c "import pandas as pd; d=pd.read_csv(r'docs/stage2_delay_warmup_both20k_paired_2026-02-16.csv'); r=d[(d.comparator=='learned_tau_delay10')&(d.metric=='final_mean')].iloc[0]; assert int(r.n_pairs)>=30; assert float(r.ci95_low)>0, r.to_dict()" || exit /b 1

REM ---- Non-warmup causal claim (n_pairs>=15, CI low > 0) ----
python -c "import pandas as pd; d=pd.read_csv(r'docs/stage2_delay_paired_effects_2026-02-16.csv'); r=d[(d.comparator=='learned_tau_delay10_selfmodel_noerr')&(d.metric=='final_mean')].iloc[0]; assert int(r.n_pairs)>=15; assert float(r.ci95_low)>0, r.to_dict()" || exit /b 1

REM ---- Groupdiff (n_seeds>=30) ----
python -c "import pandas as pd; d=pd.read_csv(r'docs/stage2_delay_warmup_groupdiff_0_vs_10_20_2026-02-16.csv'); r=d[d.metric=='final_mean'].iloc[0]; assert int(r.n_seeds)>=30, r.to_dict()" || exit /b 1

REM ---- Flicker limitation (n_pairs>=5, AUC CI high < 0) ----
python -c "import pandas as pd; d=pd.read_csv(r'docs/stage2_delay_warmup_flicker01_paired_2026-02-16.csv'); r=d[(d.comparator=='learned_tau_delay10')&(d.metric=='auc')].iloc[0]; assert int(r.n_pairs)>=5; assert float(r.ci95_high)<0, r.to_dict()" || exit /b 1

echo Verify OK.
popd
exit /b 0
