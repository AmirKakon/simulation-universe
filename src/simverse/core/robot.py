"""Robot base class — defines the interface for all robots in SimVerse."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


@dataclass
class JointInfo:
    """Metadata for a single robot joint."""

    name: str
    type: str  # "revolute", "prismatic", "fixed"
    range: tuple[float, float]
    damping: float = 0.0
    stiffness: float = 0.0


@dataclass
class RobotConfig:
    """Declarative configuration for a robot model."""

    name: str
    model_path: Path
    description: str = ""
    manufacturer: str = ""
    dof: int = 0
    has_gripper: bool = False
    default_joint_positions: list[float] = field(default_factory=list)
    control_frequency: float = 20.0  # Hz


class Robot(ABC):
    """Base class for all robots in SimVerse.

    Subclasses define robot-specific kinematics, action/observation spaces,
    and how to map high-level actions to low-level actuator controls.
    """

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self._joint_info: list[JointInfo] = []

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model_path(self) -> Path:
        return self.config.model_path

    @property
    def dof(self) -> int:
        return self.config.dof

    @property
    def joints(self) -> list[JointInfo]:
        return list(self._joint_info)

    @abstractmethod
    def get_action_space(self) -> spaces.Space[Any]:
        """Return the Gymnasium action space for this robot."""

    @abstractmethod
    def get_observation_space(self) -> spaces.Space[Any]:
        """Return the Gymnasium observation space for robot-specific observations."""

    @abstractmethod
    def get_joint_positions(self, qpos: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Extract this robot's joint positions from the full simulation qpos vector."""

    @abstractmethod
    def get_joint_velocities(self, qvel: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Extract this robot's joint velocities from the full simulation qvel vector."""

    @abstractmethod
    def get_end_effector_position(
        self, qpos: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Return the 3D position of the end effector in world coordinates."""

    @abstractmethod
    def action_to_ctrl(self, action: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Convert a high-level action into low-level actuator controls."""

    @abstractmethod
    def get_observation(
        self,
        qpos: NDArray[np.floating[Any]],
        qvel: NDArray[np.floating[Any]],
    ) -> dict[str, NDArray[np.floating[Any]]]:
        """Build the robot-specific observation dictionary from simulation state."""
