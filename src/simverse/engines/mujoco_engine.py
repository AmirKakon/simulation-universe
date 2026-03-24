"""MuJoCo CPU physics engine implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from simverse.core.engine import PhysicsEngine, RenderFrame, SimulationState

logger = logging.getLogger(__name__)


class MuJoCoEngine(PhysicsEngine):
    """Physics engine backed by MuJoCo running on CPU.

    This is the primary engine for local development and prototyping.
    It wraps the native MuJoCo Python bindings with the SimVerse engine protocol.
    """

    def __init__(self) -> None:
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._initial_qpos: NDArray[np.floating[Any]] | None = None
        self._initial_qvel: NDArray[np.floating[Any]] | None = None

    def load_model(self, model_path: Path, **kwargs: Any) -> None:
        path_str = str(model_path)
        if path_str.endswith(".xml") or path_str.endswith(".mjcf"):
            self._model = mujoco.MjModel.from_xml_path(path_str)
        elif path_str.endswith(".urdf"):
            self._model = mujoco.MjModel.from_xml_path(path_str)
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        self._initial_qpos = self._data.qpos.copy()
        self._initial_qvel = self._data.qvel.copy()
        self._renderer = None

        logger.info(
            "Loaded model: %s (nq=%d, nv=%d, nu=%d)",
            model_path.name,
            self._model.nq,
            self._model.nv,
            self._model.nu,
        )

    def _ensure_loaded(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        if self._model is None or self._data is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return self._model, self._data

    def reset(self, *, seed: int | None = None) -> SimulationState:
        model, data = self._ensure_loaded()
        assert self._initial_qpos is not None and self._initial_qvel is not None

        mujoco.mj_resetData(model, data)
        data.qpos[:] = self._initial_qpos
        data.qvel[:] = self._initial_qvel
        mujoco.mj_forward(model, data)

        return self._capture_state(model, data)

    def step(self, ctrl: NDArray[np.floating[Any]]) -> SimulationState:
        model, data = self._ensure_loaded()
        np.copyto(data.ctrl, ctrl[: model.nu])
        mujoco.mj_step(model, data)
        return self._capture_state(model, data)

    def get_state(self) -> SimulationState:
        model, data = self._ensure_loaded()
        return self._capture_state(model, data)

    def set_state(self, state: SimulationState) -> None:
        model, data = self._ensure_loaded()
        data.qpos[:] = state.qpos
        data.qvel[:] = state.qvel
        data.ctrl[:] = state.ctrl
        data.time = state.time
        mujoco.mj_forward(model, data)

    def render(
        self,
        width: int = 640,
        height: int = 480,
        camera_name: str | None = None,
    ) -> RenderFrame:
        model, data = self._ensure_loaded()

        needs_new = (
            self._renderer is None
            or self._renderer.width != width
            or self._renderer.height != height
        )
        if needs_new:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(model, width=width, height=height)

        if camera_name is not None:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id < 0:
                raise ValueError(f"Camera '{camera_name}' not found in model")
            self._renderer.update_scene(data, camera=cam_id)
        else:
            self._renderer.update_scene(data)

        rgb = self._renderer.render()

        self._renderer.enable_depth_rendering()
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()

        return RenderFrame(
            rgb=rgb.copy(),
            depth=depth.copy(),
            width=width,
            height=height,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._model = None
        self._data = None
        logger.info("MuJoCo engine closed")

    @property
    def timestep(self) -> float:
        model, _ = self._ensure_loaded()
        return float(model.opt.timestep)

    @property
    def n_actuators(self) -> int:
        model, _ = self._ensure_loaded()
        return int(model.nu)

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def mj_model(self) -> mujoco.MjModel:
        """Direct access to the underlying MjModel (for advanced use)."""
        model, _ = self._ensure_loaded()
        return model

    @property
    def mj_data(self) -> mujoco.MjData:
        """Direct access to the underlying MjData (for advanced use)."""
        _, data = self._ensure_loaded()
        return data

    def _capture_state(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> SimulationState:
        sensor_data: dict[str, NDArray[np.floating[Any]]] = {}
        for i in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                adr = model.sensor_adr[i]
                dim = model.sensor_dim[i]
                sensor_data[name] = data.sensordata[adr : adr + dim].copy()

        return SimulationState(
            time=float(data.time),
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            ctrl=data.ctrl.copy(),
            sensor_data=sensor_data,
        )
