"""Unit tests for robot registry and Panda robot."""

import numpy as np
import pytest

from simverse.core.robot import RobotConfig
from simverse.robots.arms.panda import PandaRobot
from simverse.robots.registry import discover_robots, get_robot, list_robots


class TestRobotRegistry:
    def test_discover_and_list(self) -> None:
        discover_robots()
        robots = list_robots()
        assert len(robots) >= 1
        names = [r["name"] for r in robots]
        assert "panda" in names

    def test_get_robot(self) -> None:
        discover_robots()
        robot = get_robot("panda")
        assert isinstance(robot, PandaRobot)
        assert robot.name == "panda"
        assert robot.dof == 7

    def test_get_unknown_robot(self) -> None:
        with pytest.raises(KeyError):
            get_robot("nonexistent_robot")


class TestPandaRobot:
    def setup_method(self) -> None:
        self.robot = PandaRobot(PandaRobot.default_config())

    def test_default_config(self) -> None:
        config = PandaRobot.default_config()
        assert isinstance(config, RobotConfig)
        assert config.name == "panda"
        assert config.dof == 7
        assert config.has_gripper
        assert config.model_path.name == "panda.xml"

    def test_action_space(self) -> None:
        space = self.robot.get_action_space()
        assert space.shape == (8,)  # 7 arm + 1 gripper

    def test_observation_space(self) -> None:
        space = self.robot.get_observation_space()
        assert "joint_positions" in space.spaces
        assert "joint_velocities" in space.spaces
        assert "gripper_position" in space.spaces
        assert "end_effector_position" in space.spaces

    def test_action_to_ctrl(self) -> None:
        action = np.zeros(8, dtype=np.float32)
        action[7] = 1.0  # open gripper
        ctrl = self.robot.action_to_ctrl(action)
        assert len(ctrl) == 9  # 7 arm + 2 finger
        assert ctrl[7] == pytest.approx(0.04)  # max width
        assert ctrl[8] == pytest.approx(0.04)

    def test_get_joint_positions(self) -> None:
        qpos = np.ones(20, dtype=np.float64) * 0.5
        joint_pos = self.robot.get_joint_positions(qpos)
        assert len(joint_pos) == 7
        np.testing.assert_array_equal(joint_pos, qpos[:7])

    def test_get_observation(self) -> None:
        qpos = np.zeros(10, dtype=np.float64)
        qvel = np.zeros(10, dtype=np.float64)
        obs = self.robot.get_observation(qpos, qvel)
        assert "joint_positions" in obs
        assert "joint_velocities" in obs
        assert "gripper_position" in obs
        assert "end_effector_position" in obs
