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

        if not config:
            self.config = OmegaConf.load("configs/base_vampire.yaml")
        else:
            self.config = OmegaConf.load(config)

        self.base_reward = 0
        self.cum_reward = 0
        self.last_kills = deque()
        self.avg_kills = 0
        self.n_steps = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        if self.render_mode == "rgb_array":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.window_size, self.window_size, 3),
                dtype=np.uint8,
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent": spaces.Box(0, self.window_size, shape=(2,), dtype=float),
                    "agent_health": spaces.Box(
                        0, self.config.max_agent_health, shape=(1,), dtype=int
                    ),
                    "enemies": spaces.Box(
                        0, self.window_size, shape=(self.size, 2), dtype=float
                    ),
                    "enemies_sense": spaces.Box(0, self.size, shape=(8,), dtype=int),
                    "target": spaces.Box(0, self.window_size, shape=(2,), dtype=float),
                }
            )
            # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        if movement == "wasd":
            self.action_space = spaces.Discrete(4)
        elif movement == "stick":
            self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=float)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        else:
            return {
                "agent": self._agent_location,
                "agent_health": self._agent_health,
                "enemies": self._enemies_location,
                "enemies_sense": self._enemies_sense,
                "target": self._target_location,
            }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=2
            ),
            "enemies_direction": (self._enemies_location - self._agent_location)
            / np.linalg.norm(self._agent_location - self._enemies_location),
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(
            0, self.window_size, size=2, dtype=int
        ).astype(float)
        self._enemies_location = (
            self.np_random.integers(0, self.window_size, size=self.size * 2, dtype=int)
            .reshape(self.size, 2)
            .astype(float)
        )

        self._enemies_distances = np.linalg.norm(
            self._agent_location - self._enemies_location, ord=2, axis=1
        )
        self._enemies_sense = np.zeros((8,), dtype=int)
        self._enemies_health = (
            np.ones((self.size), dtype=int) * self.config.max_enemy_health
        )
        self._agent_health = np.array([self.config.max_agent_health], dtype=int)
        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while (
            np.linalg.norm((self._target_location - self._agent_location), ord=2) < 200
        ):
            self._target_location = self.np_random.integers(
                0, self.window_size, 2
            ).astype(float)

        self.base_distance = np.linalg.norm(
            (self._target_location - self._agent_location), ord=2
        )

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

        self._agent_location = np.clip(
            self._agent_location + direction * self.config.agent_speed,
            0,
            self.window_size - 1,
        )
        self._enemies_location += (
            -1.0 * self._get_info()["enemies_direction"] * self.config.enemy_speed
        )

        self._enemies_sense = []
        wdth = self.config.width
        half = wdth//2
        counts = []
        for ox,oy in product([-1,0,1], [-1,0,1]):
            #x+ox*wdth-step:x+ox*wdth+step, y+oy*wdth-step:y+oy*wdth+step) 
            if ox == 0 and oy == 0:
                continue
            h=np.linalg.norm(self._enemies_location-np.array([x+ox*wdth, y+oy*wdth]), ord=1)
            counts.append((h<half).sum())

        self._enemies_sense = np.array(counts)

        self._enemies_distances = np.linalg.norm(
            self._agent_location - self._enemies_location, ord=2, axis=1
        )

        attacks_from_enemies_mask = (
            self._enemies_distances < self.config.enemy_attack_range
        ).astype(int)
        self._agent_health -= attacks_from_enemies_mask.sum() * self.config.enemy_damage
        if (self._agent_health < 0).astype(bool)[0] == True:
            terminated = True
            reward = -100
        else:
            distance_to_target = np.linalg.norm(
                self._agent_location - self._target_location, ord=2
            )
            reward = 1 - distance_to_target / self.base_distance
            reward -= attacks_from_enemies_mask.sum() > 0
            if distance_to_target < 30:
                reward += 111
                terminated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

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

        # draw the agent's garlic radius
        if self.render_mode == "human":
            pygame.draw.circle(
                canvas,
                (120, 120, 120),
                (self._agent_location.astype(int)),  # * pix_square_size,
                pix_square_size * self.config.agent_attack_range,
            )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location.astype(int)),  # * pix_square_size,
            pix_square_size * self.config.agent_radius,
        )

        for x in range(self.size):
            pygame.draw.circle(
                canvas,
                (245, 245, 245),
                self._enemies_location[x].astype(int),
                pix_square_size * self.config.enemy_radius,
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
