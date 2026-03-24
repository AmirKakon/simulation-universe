"""Policy evaluation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Aggregated evaluation results."""

    n_episodes: int
    mean_reward: float
    std_reward: float
    mean_length: float
    success_rate: float
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_successes: list[bool] = field(default_factory=list)


def evaluate_policy(
    model: Any,
    env: gym.Env[Any, Any],
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
) -> EvalResult:
    """Run a trained policy for n_episodes and collect metrics."""
    rewards_list: list[float] = []
    lengths_list: list[int] = []
    successes_list: list[bool] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated

            if info.get("success", False):
                success = True

            if render:
                env.render()

        rewards_list.append(total_reward)
        lengths_list.append(steps)
        successes_list.append(success)

        logger.info(
            "Episode %d/%d: reward=%.2f, length=%d, success=%s",
            ep + 1, n_episodes, total_reward, steps, success,
        )

    return EvalResult(
        n_episodes=n_episodes,
        mean_reward=float(np.mean(rewards_list)),
        std_reward=float(np.std(rewards_list)),
        mean_length=float(np.mean(lengths_list)),
        success_rate=float(np.mean(successes_list)),
        episode_rewards=rewards_list,
        episode_lengths=lengths_list,
        episode_successes=successes_list,
    )


def load_and_evaluate(
    model_path: Path,
    env_id: str,
    n_episodes: int = 10,
    algorithm: str = "PPO",
) -> EvalResult:
    """Load a saved model and evaluate it."""
    from stable_baselines3 import PPO, SAC, TD3

    algo_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
    if algorithm not in algo_map:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(algo_map.keys())}")

    import simverse.envs  # noqa: F401 — triggers Gymnasium registration

    env = gym.make(env_id)
    model = algo_map[algorithm].load(str(model_path))
    result = evaluate_policy(model, env, n_episodes=n_episodes)
    env.close()
    return result
