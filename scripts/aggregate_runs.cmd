@echo off
set ROOT=runs/sweeps/stage1

python -m internal_time_rl.analysis.aggregate_runs --root %ROOT% --metric episode/return_mean_20 --last-n 10 --points 200

