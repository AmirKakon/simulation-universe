"""CLI entry point for training: `simverse-train`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from simverse.training.config import Algorithm, TrainingConfig
from simverse.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="simverse-train",
        description="Train a robot policy in a SimVerse environment",
    )
    parser.add_argument("--env", default="SimVerse/DeskPickup-v0", help="Gymnasium environment ID")
    parser.add_argument("--algo", default="PPO", choices=["PPO", "SAC", "TD3"], help="RL algorithm")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--log-dir", type=str, default="runs", help="Logging directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.config:
        config = TrainingConfig.from_yaml(Path(args.config))
    else:
        config = TrainingConfig(
            env_id=args.env,
            algorithm=Algorithm(args.algo),
            total_timesteps=args.timesteps,
            seed=args.seed,
            learning_rate=args.lr,
            log_dir=Path(args.log_dir),
        )

    trainer = Trainer(config)
    try:
        trainer.setup()
        model_path = trainer.train()
        print(f"\nTraining complete! Model saved to: {model_path}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
