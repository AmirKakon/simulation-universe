"""Scene composition — combines robots, objects, and environment geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from simverse.core.robot import Robot
from simverse.core.sensor import Sensor


@dataclass
class SceneObject:
    """A non-robot object in the scene (table, cup, box, etc.)."""

    name: str
    model_path: Path | None = None
    position: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    orientation: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # wxyz quaternion
    )
    scale: float = 1.0
    is_static: bool = False
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneConfig:
    """Declarative scene description that can be serialized to/from YAML."""

    name: str
    model_path: Path
    description: str = ""
    gravity: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=np.float64)
    )
    timestep: float = 0.002
    objects: list[SceneObject] = field(default_factory=list)


class Scene:
    """A composed simulation scene with robots, objects, and sensors.

    The Scene is the bridge between declarative configuration and the
    physics engine. It knows how to assemble a complete MJCF model
    from a base environment plus robot and object includes.
    """

    def __init__(self, config: SceneConfig) -> None:
        self.config = config
        self._robots: list[Robot] = []
        self._sensors: list[Sensor] = []
        self._objects: list[SceneObject] = list(config.objects)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model_path(self) -> Path:
        return self.config.model_path

    @property
    def robots(self) -> list[Robot]:
        return list(self._robots)

    @property
    def sensors(self) -> list[Sensor]:
        return list(self._sensors)

    @property
    def objects(self) -> list[SceneObject]:
        return list(self._objects)

    def add_robot(self, robot: Robot) -> None:
        self._robots.append(robot)

    def add_sensor(self, sensor: Sensor) -> None:
        self._sensors.append(sensor)

    def add_object(self, obj: SceneObject) -> None:
        self._objects.append(obj)

    def get_model_path(self) -> Path:
        """Return the path to the main MJCF model for this scene."""
        return self.config.model_path

    def get_robot(self, name: str) -> Robot:
        for robot in self._robots:
            if robot.name == name:
                return robot
        raise KeyError(f"Robot '{name}' not found in scene '{self.name}'")
