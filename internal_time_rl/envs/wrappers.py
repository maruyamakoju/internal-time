from __future__ import annotations

import random

import gymnasium as gym
import numpy as np


class DelayedRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, delay_steps: int = 10) -> None:
        super().__init__(env)
        if delay_steps < 1:
            raise ValueError("delay_steps must be >= 1")
        self.delay_steps = delay_steps
        self._acc_reward = 0.0
        self._counter = 0

    def reset(self, **kwargs):
        self._acc_reward = 0.0
        self._counter = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._acc_reward += float(reward)
        self._counter += 1

        emit_reward = 0.0
        if terminated or truncated or self._counter >= self.delay_steps:
            emit_reward = self._acc_reward
            self._acc_reward = 0.0
            self._counter = 0
        return obs, emit_reward, terminated, truncated, info


class FlickeringObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, flicker_prob: float = 0.25) -> None:
        super().__init__(env)
        if not (0.0 <= flicker_prob <= 1.0):
            raise ValueError("flicker_prob must be in [0, 1]")
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("FlickeringObservationWrapper supports only Box observation spaces")
        self.flicker_prob = flicker_prob

    def observation(self, observation):
        if random.random() < self.flicker_prob:
            return np.zeros_like(observation)
        return observation


class VariableSpeedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, min_repeat: int = 1, max_repeat: int = 4) -> None:
        super().__init__(env)
        if min_repeat < 1 or max_repeat < min_repeat:
            raise ValueError("Require 1 <= min_repeat <= max_repeat")
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat

    def step(self, action):
        repeat = random.randint(self.min_repeat, self.max_repeat)
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None

        for _ in range(repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        info = dict(info)
        info["action_repeat"] = repeat
        return obs, total_reward, terminated, truncated, info

