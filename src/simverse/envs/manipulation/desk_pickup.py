"""DeskPickup environment — train a Panda arm to pick up objects from a desk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from simverse.core.engine import SimulationState
from simverse.core.environment import SimVerseEnv
from simverse.core.scene import Scene, SceneConfig
from simverse.core.task import StepResult, Task, TaskConfig
from simverse.engines.mujoco_engine import MuJoCoEngine
from simverse.robots.arms.panda import PandaRobot

SCENE_PATH = Path(__file__).resolve().parents[4] / "assets" / "scenes" / "desk" / "desk_pickup.xml"


class PickupTask(Task):
    """Task: move the end effector to the target object and lift it above the table."""

    TARGET_HEIGHT = 0.95  # target must be lifted to this z-coordinate
    REACH_THRESHOLD = 0.05  # distance to consider "reached"

    def __init__(self, config: TaskConfig | None = None) -> None:
        if config is None:
            config = TaskConfig(
                name="pickup",
                description="Pick up target object and lift it above the desk",
                max_episode_steps=500,
                reward_scale=1.0,
            )
        super().__init__(config)
        self._target_body_name = "red_cube"
        self._target_initial_pos = np.zeros(3, dtype=np.float64)
        self._ee_site_name = "end_effector"

    def reset(self, state: SimulationState) -> dict[str, Any]:
        self._step_count = 0
        self._target_initial_pos = self._get_target_pos(state)
        return {"target_body": self._target_body_name}

    def compute_reward(
        self,
        state: SimulationState,
        action: NDArray[np.floating[Any]],
        next_state: SimulationState,
    ) -> StepResult:
        target_pos = self._get_target_pos(next_state)
        ee_pos = self._get_ee_pos(next_state)

        dist_to_target = float(np.linalg.norm(ee_pos - target_pos))

        reaching_reward = -dist_to_target

        height_reward = 0.0
        lifted = target_pos[2] > self.TARGET_HEIGHT
        if lifted:
            height_reward = 10.0 * (target_pos[2] - self._target_initial_pos[2])

        action_penalty = -0.01 * float(np.sum(np.square(action)))

        reward = self.config.reward_scale * (reaching_reward + height_reward + action_penalty)

        success = lifted and dist_to_target < self.REACH_THRESHOLD
        fell_off = target_pos[2] < 0.5  # object fell off the table

        return StepResult(
            reward=reward,
            terminated=success or fell_off,
            truncated=False,
            success=success,
            info={
                "dist_to_target": dist_to_target,
                "target_height": float(target_pos[2]),
                "is_lifted": lifted,
            },
        )

    def get_task_observation(self, state: SimulationState) -> dict[str, Any]:
        target_pos = self._get_target_pos(state)
        return {
            "target_position": target_pos,
        }

    def get_observation_space_additions(self) -> dict[str, Any]:
        from gymnasium import spaces

        return {
            "target_position": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
            ),
        }

    def _get_target_pos(self, state: SimulationState) -> NDArray[np.floating[Any]]:
        # Target object (red_cube) is the first free body — its position
        # is stored in qpos as [x, y, z, qw, qx, qy, qz] starting at index 9
        # (7 arm joints + 2 finger joints = 9)
        obj_start = 9  # after 7 arm + 2 gripper joints
        return state.qpos[obj_start : obj_start + 3].copy()

    def _get_ee_pos(self, state: SimulationState) -> NDArray[np.floating[Any]]:
        # Approximate EE position from joint state (overridden in env with site data)
        return np.zeros(3, dtype=np.float64)


class DeskPickupEnv(SimVerseEnv):
    """Gymnasium environment: Panda arm picks up objects from a desk.

    Observation space:
        - joint_positions (7,): arm joint angles
        - joint_velocities (7,): arm joint velocities
        - gripper_position (1,): gripper opening width
        - end_effector_position (3,): EE position in world frame
        - target_position (3,): target object position

    Action space:
        - (8,): 7 joint position targets + 1 gripper command

    Reward:
        - Dense reaching reward (negative distance to target)
        - Bonus for lifting the object
        - Small action penalty for smooth control
    """

    def __init__(
        self,
        render_mode: str | None = None,
        target_object: str = "red_cube",
    ) -> None:
        engine = MuJoCoEngine()
        engine.load_model(SCENE_PATH)

        robot = PandaRobot(PandaRobot.default_config())
        task = PickupTask()
        task._target_body_name = target_object

        scene_config = SceneConfig(
            name="desk_pickup",
            model_path=SCENE_PATH,
            description="Desk with Panda arm and manipulable objects",
        )
        scene = Scene(scene_config)
        scene.add_robot(robot)

        super().__init__(
            engine=engine,
            scene=scene,
            robot=robot,
            task=task,
            render_mode=render_mode,
        )

        self._ee_site_id: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        import mujoco
        model = self.engine.mj_model  # type: ignore[attr-defined]
        self._ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )

        obs = self._update_ee_observation(obs)
        return obs, info

    def step(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self._update_ee_observation(obs)
        return obs, reward, terminated, truncated, info

    def _update_ee_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Replace the placeholder EE position with the actual site position."""
        if self._ee_site_id is not None and self._ee_site_id >= 0:
            data = self.engine.mj_data  # type: ignore[attr-defined]
            obs["end_effector_position"] = data.site_xpos[self._ee_site_id].copy()
        return obs
