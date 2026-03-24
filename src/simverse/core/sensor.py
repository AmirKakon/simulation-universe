"""Virtual sensors — cameras, force/torque, touch, and joint sensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from simverse.core.engine import PhysicsEngine


class SensorType(Enum):
    CAMERA = "camera"
    FORCE_TORQUE = "force_torque"
    TOUCH = "touch"
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    IMU = "imu"


@dataclass
class SensorConfig:
    """Configuration for a virtual sensor."""

    name: str
    sensor_type: SensorType
    update_rate: float = 30.0  # Hz
    noise_stddev: float = 0.0
    params: dict[str, Any] | None = None


class Sensor(ABC):
    """Abstract base for virtual sensors attached to a simulation."""

    def __init__(self, config: SensorConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng()

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def sensor_type(self) -> SensorType:
        return self.config.sensor_type

    @abstractmethod
    def read(self, engine: PhysicsEngine) -> NDArray[np.floating[Any]]:
        """Take a reading from the sensor given the current engine state."""

    def _add_noise(self, data: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        if self.config.noise_stddev > 0:
            noise = self._rng.normal(0, self.config.noise_stddev, size=data.shape)
            return data + noise.astype(data.dtype)
        return data


class CameraSensor(Sensor):
    """RGB-D camera sensor that renders from a named camera in the scene."""

    def __init__(
        self,
        config: SensorConfig,
        width: int = 640,
        height: int = 480,
        camera_name: str | None = None,
    ) -> None:
        super().__init__(config)
        self.width = width
        self.height = height
        self.camera_name = camera_name

    def read(self, engine: PhysicsEngine) -> NDArray[np.floating[Any]]:
        frame = engine.render(
            width=self.width,
            height=self.height,
            camera_name=self.camera_name,
        )
        return frame.rgb.astype(np.float32) / 255.0


class ForceTorqueSensor(Sensor):
    """Force/torque sensor that reads from a MuJoCo sensor by name."""

    def __init__(self, config: SensorConfig, sensor_name: str) -> None:
        super().__init__(config)
        self.sensor_name = sensor_name

    def read(self, engine: PhysicsEngine) -> NDArray[np.floating[Any]]:
        state = engine.get_state()
        data = state.sensor_data.get(self.sensor_name, np.zeros(6, dtype=np.float64))
        return self._add_noise(data)
