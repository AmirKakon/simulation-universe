"""Unit tests for the task system."""

import numpy as np

from simverse.core.engine import SimulationState
from simverse.core.task import TaskConfig
from simverse.envs.manipulation.desk_pickup import PickupTask


class TestPickupTask:
    def setup_method(self) -> None:
        self.task = PickupTask()

    def _make_state(self, obj_z: float = 0.8) -> SimulationState:
        # 7 arm joints + 2 gripper + 7 (freejoint: 3 pos + 4 quat) for red_cube
        qpos = np.zeros(9 + 7, dtype=np.float64)
        qpos[9] = 0.5  # obj x
        qpos[10] = 0.1  # obj y
        qpos[11] = obj_z  # obj z
        qpos[12] = 1.0  # qw
        return SimulationState(
            time=0.0,
            qpos=qpos,
            qvel=np.zeros(15, dtype=np.float64),
            ctrl=np.zeros(9, dtype=np.float64),
        )

    def test_reset(self) -> None:
        state = self._make_state()
        info = self.task.reset(state)
        assert "target_body" in info
        assert self.task.step_count() == 0

    def test_compute_reward_basic(self) -> None:
        state = self._make_state(0.8)
        next_state = self._make_state(0.8)
        action = np.zeros(8, dtype=np.float32)

        self.task.reset(state)
        result = self.task.compute_reward(state, action, next_state)

        assert isinstance(result.reward, float)
        assert not result.success
        assert not result.terminated

    def test_object_lifted_is_success(self) -> None:
        state = self._make_state(0.8)
        self.task.reset(state)

        lifted_state = self._make_state(1.0)
        # EE at same position as object (dist ~0)
        action = np.zeros(8, dtype=np.float32)
        result = self.task.compute_reward(state, action, lifted_state)

        assert result.info["is_lifted"]

    def test_object_fell_terminates(self) -> None:
        state = self._make_state(0.8)
        self.task.reset(state)

        fallen_state = self._make_state(0.3)
        action = np.zeros(8, dtype=np.float32)
        result = self.task.compute_reward(state, action, fallen_state)

        assert result.terminated

    def test_truncation(self) -> None:
        config = TaskConfig(name="test", max_episode_steps=5)
        task = PickupTask(config)
        state = self._make_state()
        task.reset(state)

        for _ in range(5):
            task.increment_step()

        assert task.is_truncated()

    def test_task_observation(self) -> None:
        state = self._make_state()
        self.task.reset(state)
        obs = self.task.get_task_observation(state)
        assert "target_position" in obs
        assert len(obs["target_position"]) == 3
