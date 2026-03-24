"""Franka Emika Panda robot arm — 7-DOF with parallel-jaw gripper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from simverse.core.robot import JointInfo, Robot, RobotConfig
from simverse.robots.registry import register_robot

ASSETS_DIR = Path(__file__).resolve().parents[4] / "assets" / "robots" / "panda"

PANDA_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
]

PANDA_JOINT_RANGES = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
]

PANDA_DEFAULT_QPOS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


@register_robot("panda")
class PandaRobot(Robot):
    """Franka Emika Panda — 7-DOF robot arm with a parallel-jaw gripper.

    Standard research robot for manipulation tasks. Uses position control
    with a 7-dimensional continuous action space for joint positions
    plus 1 dimension for gripper open/close.
    """

    N_ARM_JOINTS: ClassVar[int] = 7
    N_GRIPPER_JOINTS: ClassVar[int] = 2  # two finger joints
    GRIPPER_MAX_WIDTH: ClassVar[float] = 0.04

    def __init__(self, config: RobotConfig) -> None:
        super().__init__(config)
        self._joint_info = [
            JointInfo(name=name, type="revolute", range=r)
            for name, r in zip(PANDA_JOINT_NAMES, PANDA_JOINT_RANGES)
        ]

    @classmethod
    def default_config(cls) -> RobotConfig:
        return RobotConfig(
            name="panda",
            model_path=ASSETS_DIR / "panda.xml",
            description="Franka Emika Panda 7-DOF robot arm with parallel-jaw gripper",
            manufacturer="Franka Emika",
            dof=7,
            has_gripper=True,
            default_joint_positions=PANDA_DEFAULT_QPOS,
            control_frequency=20.0,
        )

    def get_action_space(self) -> spaces.Box:
        # 7 arm joints + 1 gripper command (0=close, 1=open)
        low = np.array([r[0] for r in PANDA_JOINT_RANGES] + [0.0], dtype=np.float32)
        high = np.array([r[1] for r in PANDA_JOINT_RANGES] + [1.0], dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def get_observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "joint_positions": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.N_ARM_JOINTS,), dtype=np.float64
            ),
            "joint_velocities": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.N_ARM_JOINTS,), dtype=np.float64
            ),
            "gripper_position": spaces.Box(
                low=0.0, high=self.GRIPPER_MAX_WIDTH, shape=(1,), dtype=np.float64
            ),
            "end_effector_position": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
            ),
        })

    def get_joint_positions(self, qpos: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return qpos[: self.N_ARM_JOINTS].copy()

    def get_joint_velocities(self, qvel: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return qvel[: self.N_ARM_JOINTS].copy()

    def get_end_effector_position(
        self, qpos: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        # In a full implementation this would use forward kinematics via MuJoCo's
        # mj_body_sitepos. For now we return a placeholder based on joint config.
        # The actual EE position is computed in get_observation using engine data.
        return np.zeros(3, dtype=np.float64)

    def action_to_ctrl(self, action: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        arm_ctrl = action[:self.N_ARM_JOINTS].astype(np.float64)
        gripper_cmd = float(action[self.N_ARM_JOINTS]) if len(action) > self.N_ARM_JOINTS else 0.0
        gripper_pos = gripper_cmd * self.GRIPPER_MAX_WIDTH
        ctrl = np.concatenate([arm_ctrl, [gripper_pos, gripper_pos]])
        return ctrl

    def get_observation(
        self,
        qpos: NDArray[np.floating[Any]],
        qvel: NDArray[np.floating[Any]],
    ) -> dict[str, NDArray[np.floating[Any]]]:
        joint_pos = self.get_joint_positions(qpos)
        joint_vel = self.get_joint_velocities(qvel)
        gripper_pos = np.array(
            [qpos[self.N_ARM_JOINTS] + qpos[self.N_ARM_JOINTS + 1]]
            if len(qpos) > self.N_ARM_JOINTS + 1
            else [0.0],
            dtype=np.float64,
        )
        ee_pos = self.get_end_effector_position(qpos)
        return {
            "joint_positions": joint_pos,
            "joint_velocities": joint_vel,
            "gripper_position": gripper_pos,
            "end_effector_position": ee_pos,
        }
