"""Unit tests for the MuJoCo engine."""

from pathlib import Path

import numpy as np
import pytest

from simverse.core.engine import SimulationState
from simverse.engines.mujoco_engine import MuJoCoEngine

PANDA_MODEL = Path(__file__).resolve().parents[2] / "assets" / "robots" / "panda" / "panda.xml"
DESK_MODEL = Path(__file__).resolve().parents[2] / "assets" / "scenes" / "desk" / "desk_pickup.xml"


class TestMuJoCoEngine:
    def test_load_model(self) -> None:
        engine = MuJoCoEngine()
        engine.load_model(PANDA_MODEL)
        assert engine.model_loaded
        assert engine.n_actuators == 9  # 7 arm + 2 gripper
        assert engine.timestep > 0
        engine.close()

    def test_load_model_not_found(self) -> None:
        engine = MuJoCoEngine()
        with pytest.raises(Exception):
            engine.load_model(Path("/nonexistent/model.xml"))

    def test_reset(self) -> None:
        engine = MuJoCoEngine()
        engine.load_model(PANDA_MODEL)
        state = engine.reset()
        assert isinstance(state, SimulationState)
        assert state.time == 0.0 or state.time < 0.01
        assert len(state.qpos) > 0
        assert len(state.qvel) > 0
        engine.close()

    def test_step(self) -> None:
        engine = MuJoCoEngine()
        engine.load_model(PANDA_MODEL)
        engine.reset()
        ctrl = np.zeros(engine.n_actuators, dtype=np.float64)
        state = engine.step(ctrl)
        assert isinstance(state, SimulationState)
        assert state.time > 0
        engine.close()

    def test_get_set_state(self) -> None:
        engine = MuJoCoEngine()
        engine.load_model(PANDA_MODEL)
        engine.reset()

        ctrl = np.zeros(engine.n_actuators, dtype=np.float64)
        engine.step(ctrl)
        saved_state = engine.get_state()

        for _ in range(10):
            engine.step(ctrl)

        engine.set_state(saved_state)
        restored = engine.get_state()
        np.testing.assert_allclose(restored.qpos, saved_state.qpos, atol=1e-10)
        np.testing.assert_allclose(restored.qvel, saved_state.qvel, atol=1e-10)
        engine.close()

    def test_render(self) -> None:
        engine = MuJoCoEngine()
        engine.load_model(PANDA_MODEL)
        engine.reset()
        frame = engine.render(width=320, height=240)
        assert frame.rgb.shape == (240, 320, 3)
        assert frame.width == 320
        assert frame.height == 240
        engine.close()

    def test_sensor_data(self) -> None:
        engine = MuJoCoEngine()
        engine.load_model(PANDA_MODEL)
        state = engine.reset()
        assert len(state.sensor_data) > 0
        assert "joint1_pos" in state.sensor_data
        engine.close()

    def test_operations_before_load_raise(self) -> None:
        engine = MuJoCoEngine()
        assert not engine.model_loaded
        with pytest.raises(RuntimeError):
            engine.reset()
        with pytest.raises(RuntimeError):
            engine.step(np.zeros(1))
        with pytest.raises(RuntimeError):
            engine.get_state()

    def test_desk_scene_loads(self) -> None:
        engine = MuJoCoEngine()
        engine.load_model(DESK_MODEL)
        assert engine.model_loaded
        state = engine.reset()
        assert len(state.qpos) > 9  # arm + gripper + free objects
        engine.close()
