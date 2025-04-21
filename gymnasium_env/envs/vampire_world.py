from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from scipy import stats
from collections import deque
import pathlib
from itertools import product
from omegaconf import OmegaConf


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    nothing = 4


DEFAULT_SIZE = 512


class VampireWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        render_mode=None,
        size=5,
        window_size=512,
        movement: str = "wasd",
        config: pathlib.Path = None,
    ):
        self.size = size  # The size of the square grid
        self.window_size = window_size  # The size of the PyGame window
        self.movement = movement
        self.render_mode = render_mode
        self.n_steps = 0
        if not config:
            self.config = OmegaConf.load("configs/base_vampire.yaml")
        else:
            self.config = OmegaConf.load(config)

        self.speed = self.window_size * 0.01
        self.agent_radius = self.window_size * 0.02

        if self.render_mode == "rgb_array":
            self.observation_space = spaces.Dict(
                {
                    "agent_location": spaces.Box(
                        0, self.window_size, shape=(2,), dtype=float
                    ),
                    "target_location": spaces.Box(
                        0, self.window_size, shape=(2,), dtype=float
                    ),
                    "screen": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.window_size, self.window_size, 3),
                        dtype=np.uint8,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_location": spaces.Box(
                        0, self.window_size, shape=(2,), dtype=float
                    ),
                    "target_location": spaces.Box(
                        0, self.window_size, shape=(2,), dtype=float
                    ),
                    "enemy_location": spaces.Box(
                        0, self.window_size, shape=(2,), dtype=float
                    ),
                }
            )
        if movement == "wasd":
            self.action_space = spaces.Discrete(5)
        elif movement == "stick":
            self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=float)

        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
            Actions.nothing.value: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        if self.render_mode == "rgb_array":
            return {
                "agent_location": self._agent_location,
                "enemy_location": self._enemy_location,
                "target_location": self._target_location,
                "screen" : self._render_frame()
            }
        else:
            return {
                "agent_location": self._agent_location,
                "enemy_location": self._enemy_location,
                "target_location": self._target_location,
            }

    def _get_info(self):
        return {}

    def reward_for_place(self, location: np.ndarray, n_steps=100):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = np.random.randint(0, self.window_size, size=(2,))

        self._target_location = np.random.randint(0, self.window_size, size=(2,))
        while (
            np.linalg.norm(self._target_location - self._agent_location, ord=1)
            < self.window_size // 2
        ):
            self._target_location = np.random.randint(0, self.window_size, size=(2,))

        self._enemy_location = np.random.randint(0, self.window_size, size=(2,))
        while (
            np.linalg.norm(self._enemy_location - self._agent_location, ord=1)
            < self.window_size * 0.1
        ):
            self._enemy_location = np.random.randint(0, self.window_size, size=(2,))

        self._target_distance = np.linalg.norm(
            self._target_location - self._agent_location, ord=2
        )
        self._enemy_distance = np.linalg.norm(
            self._enemy_location - self._agent_location, ord=2
        )
        self.base_enemy_distance = self._enemy_distance
        self.base_distance = self._target_distance
        self.current_pos = self._agent_location
        self.prev_pos = None

        self.n_steps = 0

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        self.n_steps += 1
        if self.movement == "wasd":
            # interact with env when in human mode
            if self.render_mode == "human":
                events = pygame.event.get()
                keys = pygame.key.get_pressed()
                new_action = None
                if keys[pygame.K_LEFT]:
                    new_action = Actions.left.value
                if keys[pygame.K_RIGHT]:
                    new_action = Actions.right.value
                if keys[pygame.K_UP]:
                    new_action = Actions.down.value
                if keys[pygame.K_DOWN]:
                    new_action = Actions.up.value
                if new_action is None:
                    new_action = action
                direction = self._action_to_direction[new_action]
            else:
                direction = self._action_to_direction[action]
        elif self.movement == "stick":
            x, y = action
            magnitude = np.sqrt(x**2 + y**2)
            x = x / magnitude
            y = y / magnitude
            direction = np.array([x, y])
        else:
            raise TypeError

        self.prev_pos = self._agent_location
        self._agent_location = np.clip(
            self._agent_location + direction,
            0,
            self.window_size - 1,
        )

        self._target_distance = np.linalg.norm(
            self._agent_location - self._target_location, ord=2
        )
        self._enemy_distance = np.linalg.norm(
            self._agent_location - self._enemy_location, ord=2
        )

        prev_distance = np.linalg.norm(self.prev_pos - self._target_location, ord=2)
        prev_enemy_distance = np.linalg.norm(self.prev_pos - self._enemy_location, ord=2)

        self.reward = 0
        if self._target_distance < prev_distance:
            reward = 1
        elif self._target_distance == prev_distance:
            reward = 1
        else:
            reward = -1
        
        if self._enemy_distance < prev_enemy_distance:
            reward += 0.5
        elif self._enemy_distance == prev_enemy_distance:
            reward += 0
        else:
            reward -= 0.5
        
        reward -= np.log(self.n_steps)//5

        truncated = False
        if self._target_distance < self.window_size*0.04:
            reward += 10
            terminated = True
        elif self._enemy_distance < self.window_size*0.04:
            reward = -3
            terminated = True
            truncated = True
        else:
            terminated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = 1

        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self._target_location.astype(int)),  # * pix_square_size,
            pix_square_size * 10,
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location.astype(int)),  # * pix_square_size,
            pix_square_size * self.config.agent_radius,
        )

        # Finally, add some gridlines
        # for x in range(self.size + 1):
        #    pygame.draw.line(
        #        canvas,
        #        0,
        #        (0, pix_square_size * x),
        #        (self.window_size, pix_square_size * x),
        #        width=3,
        #    )
        #    pygame.draw.line(
        #        canvas,
        #        0,
        #        (pix_square_size * x, 0),
        #        (pix_square_size * x, self.window_size),
        #        width=3,
        #    )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
