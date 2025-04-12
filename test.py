#!env python
import gymnasium
import gymnasium_env
import time
import numpy as np

env = gymnasium.make("gymnasium_env/VampireWorld-v0", render_mode="human")


env.reset()

for _ in range(1200):
    action = env.action_space.sample(np.array([0, 1, 0, 0], dtype=np.int8))

    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
    env.render()
