from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from scipy import stats
from collections import deque


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class VampireWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, size=5, movement: str = "wasd"):
        self.size = size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window
        self.movement = movement
        self.attack = 10
        self.max_enemy_health = 50
        self.max_agent_health = 100
        self.enemy_speed = 20.0
        self.agent_attack_range = 100
        self.enemy_attack_range = 50
        self.base_reward = 0
        self.cum_reward = 0
        self.last_kills = deque()
        self.kills_window = 20
        self.avg_kills = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.window_size, shape=(2,), dtype=float),
                "enemies": spaces.Box(
                    0, self.window_size, shape=(self.size, 2), dtype=float
                ),
                "target": spaces.Box(0, self.window_size, shape=(2,), dtype=float),
                "agent_health": spaces.Box(
                    0, self.max_agent_health, shape=(1,), dtype=int
                ),
                "enemies_health": spaces.Box(
                    0, self.max_enemy_health, shape=(self.size,), dtype=int
                ),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        if movement == "wasd":
            self.action_space = spaces.MultiDiscrete(4)
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
        return {
            "agent": self._agent_location,
            "enemies": self._enemies_location,
            "target": self._target_location,
            "agent_health": self._agent_health,
            "enemies_health": self._enemies_health,
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "enemies_distances": np.linalg.norm(
                self._agent_location - self._enemies_location, ord=2, axis=1
            ),
            "enemies_direction": (self._enemies_location - self._agent_location)
            / np.linalg.norm(self._agent_location - self._enemies_location),
            "cum_reward": self.cum_reward,
            "avg_kills": self.avg_kills,
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

        self._enemies_health = np.ones((self.size), dtype=int) * self.max_enemy_health

        self._agent_health = np.array([self.max_agent_health], dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.window_size, 2
            ).astype(float)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        #
        #
        if self.movement == "wasd":
            direction = self._action_to_direction[action]
        elif self.movement == "stick":
            self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=float)

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.window_size - 1
        )

        self._enemies_location += (
            -1.0 * self._get_info()["enemies_direction"] * self.enemy_speed
        )

        # An episode is done iff the agent has reached the target
        #
        enemies_under_attack_mask = (
            self._get_info()["enemies_distances"] < self.agent_attack_range
        ).astype(int)

        assert enemies_under_attack_mask.shape == (self.size,)

        self._enemies_health = (
            self._enemies_health - self.attack * enemies_under_attack_mask
        )

        enemies_dead_mask = (self._enemies_health <= 0).astype(bool)

        assert len(enemies_dead_mask.shape) == 1
        assert len(enemies_dead_mask) == self.size

        num_dead_enemies = enemies_dead_mask.sum()

        self.last_kills.append(num_dead_enemies)
        if len(self.last_kills) > self.kills_window:
            self.last_kills.popleft()
        self.avg_kills = sum(self.last_kills) / self.kills_window

        new_locations = (
            self.np_random.integers(
                0, self.window_size, size=num_dead_enemies * 2, dtype=int
            )
            .reshape(num_dead_enemies, 2)
            .astype(float)
        )

        if num_dead_enemies > 0:
            self._enemies_location[enemies_dead_mask] = new_locations
            self._enemies_health[enemies_dead_mask] = self.max_enemy_health

        terminated = (self._agent_health < 0).astype(bool)[0]

        # TODO: add Kills per X last steps as reward
        self.cum_reward += self.avg_kills if not terminated else 0

        reward = self.cum_reward

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
        # First we draw the target
        pix_square_size = 1

        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self._target_location) * pix_square_size,
            pix_square_size * 10,
        )

        # draw the agent's garlic radius
        pygame.draw.circle(
            canvas,
            (120, 120, 120),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size * self.agent_attack_range,
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size * 30,
        )

        for x in range(self.size):
            pygame.draw.circle(
                canvas, (205, 245, 255), self._enemies_location[x], pix_square_size * 15
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
