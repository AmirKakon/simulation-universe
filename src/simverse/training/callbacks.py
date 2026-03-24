"""Training callbacks for logging, checkpointing, and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

logger = logging.getLogger(__name__)


class TrainingMetricsCallback(BaseCallback):
    """Logs custom training metrics (success rate, episode stats) to TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_successes: list[bool] = []
        self._episode_rewards: list[float] = []
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        if rewards is not None and len(rewards) > 0:
            self._current_episode_reward += float(rewards[0])

        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                success = info.get("success", False)

                self._episode_successes.append(success)
                self._episode_rewards.append(ep_reward)

                self.logger.record("rollout/ep_success", float(success))
                self.logger.record("rollout/ep_reward_custom", ep_reward)
                self.logger.record("rollout/ep_length_custom", ep_length)

                if len(self._episode_successes) >= 10:
                    recent = self._episode_successes[-100:]
                    self.logger.record("rollout/success_rate", sum(recent) / len(recent))

                self._current_episode_reward = 0.0

        return True


def create_callbacks(
    checkpoint_dir: Path,
    checkpoint_freq: int,
    eval_env: Any,
    eval_freq: int,
    eval_episodes: int,
    log_dir: Path,
) -> list[BaseCallback]:
    """Build the standard set of training callbacks."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="simverse_model",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=eval_episodes,
        eval_freq=eval_freq,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval_logs"),
        deterministic=True,
        verbose=1,
    )

    metrics_cb = TrainingMetricsCallback()

    return [checkpoint_cb, eval_cb, metrics_cb]
