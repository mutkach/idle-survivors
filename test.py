#!env python
import gymnasium
import gymnasium_env
import time
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = gymnasium.make(
    "gymnasium_env/VampireWorld-v0",
    window_size=512,
    render_mode="human",
    size=5,
    movement="stick",
)

env = DummyVecEnv([lambda: env])
env = VecNormalize(env)

#model = A2C.load("./models/best_model")
observation = env.reset()

for _ in range(1200):
    action = [env.action_space.sample()]
    #action, _ = model.predict(observation)
    print(action)
    observation, reward, terminated, info = env.step(action)

    print(
        f"Current_reward: {reward}"
        if not terminated
        else f"TERMINATED WITH REWARD {reward}"
    )
    # print(info["agent_health"])
    if terminated:
        observation = env.reset()
    env.render()
