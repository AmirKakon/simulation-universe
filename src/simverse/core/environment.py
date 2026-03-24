"""SimVerseEnv — base Gymnasium environment tying engine, robot, task, and scene together."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from simverse.core.engine import PhysicsEngine, SimulationState
from simverse.core.robot import Robot
from simverse.core.scene import Scene
from simverse.core.task import Task


class SimVerseEnv(gym.Env[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]):
    """Base Gymnasium environment for SimVerse.

    Composes a PhysicsEngine, Scene, Robot, and Task into a standard
    Gymnasium interface that any RL library can train on.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        engine: PhysicsEngine,
        scene: Scene,
        robot: Robot,
        task: Task,
        render_mode: str | None = None,
        render_width: int = 640,
        render_height: int = 480,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.scene = scene
        self.robot = robot
        self.task = task
        self.render_mode = render_mode
        self._render_width = render_width
        self._render_height = render_height

        self.action_space = robot.get_action_space()
        self.observation_space = self._build_observation_space()

        self._current_state: SimulationState | None = None

    def _build_observation_space(self) -> spaces.Space[Any]:
        """Merge robot observation space with task observation space additions."""
        robot_space = self.robot.get_observation_space()
        task_additions = self.task.get_observation_space_additions()

        if isinstance(robot_space, spaces.Dict) and task_additions:
            merged = dict(robot_space.spaces)
            merged.update(task_additions)
            return spaces.Dict(merged)

        return robot_space

    def _build_observation(self, state: SimulationState) -> dict[str, Any]:
        """Combine robot and task observations into a single dict."""
        obs = self.robot.get_observation(state.qpos, state.qvel)
        task_obs = self.task.get_task_observation(state)
        obs.update(task_obs)
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        state = self.engine.reset(seed=seed)
        self._current_state = state
        task_info = self.task.reset(state)

        obs = self._build_observation(state)
        info: dict[str, Any] = {"task_info": task_info}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        assert self._current_state is not None, "Must call reset() before step()"

        prev_state = self._current_state
        ctrl = self.robot.action_to_ctrl(action)
        state = self.engine.step(ctrl)
        self._current_state = state

        self.task.increment_step()
        result = self.task.compute_reward(prev_state, action, state)

        obs = self._build_observation(state)
        truncated = result.truncated or self.task.is_truncated()
        info: dict[str, Any] = {
            "success": result.success,
            "step_count": self.task.step_count(),
            **result.info,
        }

        if self.render_mode == "human":
            self.render()

        return obs, result.reward, result.terminated, truncated, info

    def render(self) -> NDArray[np.uint8] | None:
        frame = self.engine.render(
            width=self._render_width,
            height=self._render_height,
        )
        if self.render_mode == "rgb_array":
            return frame.rgb
        return None

    def close(self) -> None:
        self.engine.close()
