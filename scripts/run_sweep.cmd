@echo off
set SUITE=stage1_full
set TIMESTEPS=200000

python scripts\run_sweep.py --suite %SUITE% --timesteps %TIMESTEPS% --seeds 0 1 2 3 4 --skip-existing

