@echo off
python train.py train.total_timesteps=4096 train.num_envs=4 train.rollout_steps=128 logging.print_every_updates=1

