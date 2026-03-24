"""MJX GPU-accelerated physics engine — stub for cloud training.

MJX (MuJoCo XLA) provides GPU-accelerated physics via JAX. This module
will be fully implemented when cloud GPU training infrastructure is set up.
For now it defines the interface and raises NotImplementedError with
clear instructions for enabling GPU support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from simverse.core.engine import PhysicsEngine, RenderFrame, SimulationState

logger = logging.getLogger(__name__)

_MJX_INSTALL_MSG = (
    "MJX (GPU-accelerated MuJoCo) requires JAX with CUDA support. "
    "Install with: pip install simverse[cloud]"
)


def _check_mjx_available() -> bool:
    try:
        import jax  # noqa: F401
        import mujoco.mjx  # noqa: F401

        return True
    except ImportError:
        return False


class MJXEngine(PhysicsEngine):
    """GPU-accelerated physics engine using MuJoCo XLA (MJX).

    Designed for large-scale parallel training on cloud GPU/TPU instances.
    Uses the same MJCF model files as the CPU MuJoCo engine.
    """

    def __init__(self, batch_size: int = 1024) -> None:
        if not _check_mjx_available():
            logger.warning("MJX not available. %s", _MJX_INSTALL_MSG)

        self.batch_size = batch_size
        self._model_path: Path | None = None
        self._loaded = False

    def load_model(self, model_path: Path, **kwargs: Any) -> None:
        raise NotImplementedError(
            f"MJX engine load_model not yet implemented. {_MJX_INSTALL_MSG}"
        )

    def reset(self, *, seed: int | None = None) -> SimulationState:
        raise NotImplementedError("MJX engine reset not yet implemented.")

    def step(self, ctrl: NDArray[np.floating[Any]]) -> SimulationState:
        raise NotImplementedError("MJX engine step not yet implemented.")

    def get_state(self) -> SimulationState:
        raise NotImplementedError("MJX engine get_state not yet implemented.")

    def set_state(self, state: SimulationState) -> None:
        raise NotImplementedError("MJX engine set_state not yet implemented.")

    def render(
        self,
        width: int = 640,
        height: int = 480,
        camera_name: str | None = None,
    ) -> RenderFrame:
        raise NotImplementedError("MJX engine render not yet implemented.")

    def close(self) -> None:
        self._loaded = False

    @property
    def timestep(self) -> float:
        raise NotImplementedError("MJX engine timestep not yet implemented.")

    @property
    def n_actuators(self) -> int:
        raise NotImplementedError("MJX engine n_actuators not yet implemented.")

    @property
    def model_loaded(self) -> bool:
        return self._loaded
