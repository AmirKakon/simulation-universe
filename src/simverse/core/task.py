"""Task protocol — defines objectives, rewards, and success criteria for environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from simverse.core.engine import SimulationState


@dataclass
class TaskConfig:
    """Declarative configuration for a task."""

    name: str
    description: str = ""
    max_episode_steps: int = 1000
    reward_scale: float = 1.0
    success_threshold: float = 0.95
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """The outcome of a single environment step from the task's perspective."""

    reward: float
    terminated: bool
    truncated: bool
    success: bool
    info: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """Abstract base for tasks that define what a robot should accomplish.

    A Task owns the reward function, success criteria, termination logic,
    and any task-specific observation augmentation (e.g. goal positions).
    """

    def __init__(self, config: TaskConfig) -> None:
        self.config = config
        self._step_count = 0

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def reset(self, state: SimulationState) -> dict[str, Any]:
        """Reset the task for a new episode. Return task-specific initial observations."""

    @abstractmethod
    def compute_reward(
        self,
        state: SimulationState,
        action: NDArray[np.floating[Any]],
        next_state: SimulationState,
    ) -> StepResult:
        """Evaluate one transition and return reward + termination signals."""

    @abstractmethod
    def get_task_observation(self, state: SimulationState) -> dict[str, Any]:
        """Return task-specific observations (e.g. goal position, target object pose)."""

    @abstractmethod
    def get_observation_space_additions(self) -> dict[str, Any]:
        """Return additional observation space entries this task adds."""

    def step_count(self) -> int:
        return self._step_count

    def increment_step(self) -> None:
        self._step_count += 1

    def is_truncated(self) -> bool:
        return self._step_count >= self.config.max_episode_steps
