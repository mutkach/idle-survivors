#!/usr/bin/env python
"""Run the VampireWorld environment with random actions for visual testing."""

import gymnasium
import gymnasium_env  # noqa: F401
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = gymnasium.make(
    "gymnasium_env/VampireWorld-v0",
    window_size=512,
    render_mode="human",
    size=3,
    movement="wasd",
)

env = DummyVecEnv([lambda: env])
env = VecNormalize(env, training=False)

observation = env.reset()

for _ in range(1200):
    action = [env.action_space.sample(mask=np.array([0, 0, 0, 0, 1], dtype=np.int8))]
    observation, reward, terminated, info = env.step(action)
    print(observation["enemy_sensing"].round(2))

    if terminated:
        observation = env.reset()
    env.render()
