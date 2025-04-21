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
    size=3,
    movement="wasd",
)

env = DummyVecEnv([lambda: env])
env = VecNormalize(env, training=False)

# model = A2C.load("./models/best_model", env=env)
observation = env.reset()

for _ in range(1200):
    action = [env.action_space.sample(mask=np.array([0, 0, 0, 0, 1], dtype=np.int8))]
    # action, _ = model.predict(observation)
    # print(action)
    observation, reward, terminated, info = env.step(action)
    print(observation["enemy_sensing"].round(2))

    # print(
    #    f"Current_reward: {reward}"
    #    if not terminated
    #    else f"TERMINATED WITH REWARD {reward}"
    # )
    # print(info["agent_health"])
    if terminated:
        observation = env.reset()
    env.render()
