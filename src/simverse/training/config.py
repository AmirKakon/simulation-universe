"""Training configuration — validated with Pydantic."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Algorithm(str, Enum):
    PPO = "PPO"
    SAC = "SAC"
    TD3 = "TD3"


class TrainingConfig(BaseModel):
    """Full configuration for a training run."""

    env_id: str = "SimVerse/DeskPickup-v0"
    algorithm: Algorithm = Algorithm.PPO
    total_timesteps: int = Field(default=100_000, ge=1)
    seed: int = 42
    n_envs: int = Field(default=1, ge=1)

    learning_rate: float = Field(default=3e-4, gt=0)
    batch_size: int = Field(default=64, ge=1)
    gamma: float = Field(default=0.99, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    clip_range: float = Field(default=0.2, ge=0, le=1)  # PPO only
    ent_coef: float = Field(default=0.0, ge=0)
    n_epochs: int = Field(default=10, ge=1)  # PPO only
    n_steps: int = Field(default=2048, ge=1)  # PPO only
    buffer_size: int = Field(default=1_000_000, ge=1)  # SAC/TD3 only
    tau: float = Field(default=0.005, ge=0, le=1)  # SAC/TD3 only

    log_dir: Path = Path("runs")
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_freq: int = Field(default=10_000, ge=1)
    eval_freq: int = Field(default=5_000, ge=1)
    eval_episodes: int = Field(default=10, ge=1)
    log_interval: int = Field(default=1, ge=1)

    render_mode: str | None = None
    device: str = "auto"

    model_config = {"extra": "allow"}

    @classmethod
    def from_yaml(cls, path: Path) -> TrainingConfig:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
