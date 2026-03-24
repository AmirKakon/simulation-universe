"""Training orchestrator — sets up and runs RL training with Stable-Baselines3."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gymnasium as gym

from simverse.training.callbacks import create_callbacks
from simverse.training.config import Algorithm, TrainingConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates RL training runs.

    Handles environment creation, algorithm selection, callback wiring,
    and model saving. Designed to be driven by the CLI or the API server.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._model: Any = None
        self._env: gym.Env[Any, Any] | None = None
        self._eval_env: gym.Env[Any, Any] | None = None
        self._run_dir: Path | None = None

    def setup(self) -> None:
        """Create environments, algorithm, and callbacks."""
        import simverse.envs  # noqa: F401 — triggers Gymnasium registration

        logger.info(
            "Setting up training: env=%s, algo=%s",
            self.config.env_id, self.config.algorithm,
        )

        self._env = self._make_env()
        self._eval_env = self._make_env()
        self._model = self._create_algorithm()

        env_slug = self.config.env_id.replace("/", "_")
        self._run_dir = self.config.log_dir / f"{env_slug}_{self.config.algorithm.value}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Training setup complete. Run dir: %s", self._run_dir)

    def train(self) -> Path:
        """Run the full training loop and return path to the saved model."""
        if self._model is None or self._env is None or self._eval_env is None:
            raise RuntimeError("Call setup() before train()")
        assert self._run_dir is not None

        callbacks = create_callbacks(
            checkpoint_dir=self.config.checkpoint_dir,
            checkpoint_freq=self.config.checkpoint_freq,
            eval_env=self._eval_env,
            eval_freq=self.config.eval_freq,
            eval_episodes=self.config.eval_episodes,
            log_dir=self._run_dir,
        )

        logger.info("Starting training for %d timesteps...", self.config.total_timesteps)
        self._model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            log_interval=self.config.log_interval,
            progress_bar=True,
        )

        model_path = self._run_dir / "final_model"
        self._model.save(str(model_path))
        logger.info("Training complete. Model saved to: %s", model_path)

        self.config.to_yaml(self._run_dir / "training_config.yaml")

        return model_path

    def cleanup(self) -> None:
        """Close environments and free resources."""
        if self._env is not None:
            self._env.close()
        if self._eval_env is not None:
            self._eval_env.close()
        logger.info("Training resources cleaned up")

    def _make_env(self) -> gym.Env[Any, Any]:
        env_kwargs: dict[str, Any] = {}
        if self.config.render_mode:
            env_kwargs["render_mode"] = self.config.render_mode
        return gym.make(self.config.env_id, **env_kwargs)

    def _create_algorithm(self) -> Any:
        from stable_baselines3 import PPO, SAC, TD3

        common_kwargs: dict[str, Any] = {
            "env": self._env,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "gamma": self.config.gamma,
            "seed": self.config.seed,
            "verbose": 1,
            "tensorboard_log": str(self.config.log_dir / "tensorboard"),
            "device": self.config.device,
        }

        if self.config.algorithm == Algorithm.PPO:
            return PPO(
                "MultiInputPolicy",
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                n_epochs=self.config.n_epochs,
                n_steps=self.config.n_steps,
                **common_kwargs,
            )
        elif self.config.algorithm == Algorithm.SAC:
            return SAC(
                "MultiInputPolicy",
                buffer_size=self.config.buffer_size,
                tau=self.config.tau,
                ent_coef="auto",
                **common_kwargs,
            )
        elif self.config.algorithm == Algorithm.TD3:
            return TD3(
                "MultiInputPolicy",
                buffer_size=self.config.buffer_size,
                tau=self.config.tau,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
