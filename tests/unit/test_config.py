"""Unit tests for training configuration."""

from pathlib import Path

import pytest

from simverse.training.config import Algorithm, TrainingConfig


class TestTrainingConfig:
    def test_defaults(self) -> None:
        config = TrainingConfig()
        assert config.env_id == "SimVerse/DeskPickup-v0"
        assert config.algorithm == Algorithm.PPO
        assert config.total_timesteps == 100_000
        assert config.learning_rate == 3e-4

    def test_validation(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(total_timesteps=-1)
        with pytest.raises(Exception):
            TrainingConfig(learning_rate=-0.001)

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
env_id: "SimVerse/DeskPickup-v0"
algorithm: "SAC"
total_timesteps: 50000
seed: 123
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = TrainingConfig.from_yaml(yaml_file)
        assert config.algorithm == Algorithm.SAC
        assert config.total_timesteps == 50_000
        assert config.seed == 123

    def test_to_yaml(self, tmp_path: Path) -> None:
        config = TrainingConfig(total_timesteps=5000)
        out_path = tmp_path / "output.yaml"
        config.to_yaml(out_path)
        assert out_path.exists()

        loaded = TrainingConfig.from_yaml(out_path)
        assert loaded.total_timesteps == 5000
