from gymnasium.envs.registration import register

register(
    id="gymnasium_env/VampireWorld-v0",
    entry_point="gymnasium_env.envs:VampireWorldEnv",
)
