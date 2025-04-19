#!env python
import gymnasium
import gymnasium_env
import time
import numpy as np
from stable_baselines3 import A2C

env = gymnasium.make(
    "gymnasium_env/VampireWorld-v0",
    window_size=512,
    render_mode="human",
    size=5,
    movement="stick",
)


observation, _ = env.reset()
# model = A2C.load("./models/vampire-ppo-v1")

for _ in range(1200):
    # action, _ = model.predict(observation)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # print(
    #     f"Current_reward: {reward}"
    #     if not terminated
    #     else f"TERMINATED WITH REWARD {reward}"
    # )
    # print(info["agent_health"])
    if terminated or truncated:
        observation, info = env.reset()
    env.render()
