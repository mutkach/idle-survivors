from gymnasium.envs.registration import register

register(
    id="idle_vampires/VampireWorld-v0",
    entry_point="idle_vampires.envs:VampireWorldEnv",
)
