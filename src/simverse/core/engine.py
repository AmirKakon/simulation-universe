"""Physics engine protocol — the abstraction boundary between SimVerse and simulation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SimulationState:
    """Snapshot of the full physics state at one timestep."""

    time: float
    qpos: NDArray[np.floating[Any]]
    qvel: NDArray[np.floating[Any]]
    ctrl: NDArray[np.floating[Any]]
    sensor_data: dict[str, NDArray[np.floating[Any]]] = field(default_factory=dict)


@dataclass
class RenderFrame:
    """A rendered image from the simulation."""

    rgb: NDArray[np.uint8]
    depth: NDArray[np.floating[Any]] | None = None
    width: int = 0
    height: int = 0


class PhysicsEngine(ABC):
    """Abstract interface for physics simulation backends.

    Implementations must handle model loading, stepping, state access, and rendering.
    The protocol is designed so that MuJoCo (CPU) and MJX (GPU) can both satisfy it.
    """

    @abstractmethod
    def load_model(self, model_path: Path, **kwargs: Any) -> None:
        """Load a scene/robot model from an MJCF or URDF file."""

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> SimulationState:
        """Reset the simulation to its initial state and return it."""

    @abstractmethod
    def step(self, ctrl: NDArray[np.floating[Any]]) -> SimulationState:
        """Advance the simulation by one timestep with the given control input."""

    @abstractmethod
    def get_state(self) -> SimulationState:
        """Return the current simulation state without advancing."""

    @abstractmethod
    def set_state(self, state: SimulationState) -> None:
        """Restore the simulation to a previously captured state."""

    @abstractmethod
    def render(
        self,
        width: int = 640,
        height: int = 480,
        camera_name: str | None = None,
    ) -> RenderFrame:
        """Render the current scene and return an image frame."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources held by the engine."""

    @property
    @abstractmethod
    def timestep(self) -> float:
        """The simulation timestep (seconds per step)."""

    @property
    @abstractmethod
    def n_actuators(self) -> int:
        """Number of actuators (control inputs) in the loaded model."""

    @property
    @abstractmethod
    def model_loaded(self) -> bool:
        """Whether a model has been successfully loaded."""
