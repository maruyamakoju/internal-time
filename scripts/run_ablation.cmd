@echo off
set TIMESTEPS=200000

python train.py logging.run_dir=runs/standard_gru model.transition_mode=standard train.total_timesteps=%TIMESTEPS%
python train.py logging.run_dir=runs/fixed_tau_1 model.transition_mode=fixed model.fixed_tau=1.0 train.total_timesteps=%TIMESTEPS%
python train.py logging.run_dir=runs/fixed_tau_3 model.transition_mode=fixed model.fixed_tau=3.0 train.total_timesteps=%TIMESTEPS%
python train.py logging.run_dir=runs/fixed_tau_10 model.transition_mode=fixed model.fixed_tau=10.0 train.total_timesteps=%TIMESTEPS%
python train.py logging.run_dir=runs/learned_tau model.transition_mode=learned train.total_timesteps=%TIMESTEPS%
