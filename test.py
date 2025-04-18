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
model = A2C.load(
    "./models/vampire-a2c-5enemies-nsteps2048-normalized-entcoef0.05-vf0.25-sde"
)

for _ in range(1200):
    action, _ = model.predict(observation)
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)

    print(
        f"Current_reward: {reward}"
        if not terminated
        else f"TERMINATED WITH REWARD {reward}"
    )
    # print(info["agent_health"])
    if terminated or truncated:
        observation, info = env.reset()
    env.render()
