"""Watch a trained policy control the Panda arm in the interactive 3D viewer.

Usage:
    python scripts/watch_trained.py --model path/to/final_model.zip
    python scripts/watch_trained.py --model trained_models/final_model.zip --episodes 5
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO, SAC, TD3

import simverse.envs  # noqa: F401

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
SCENE_PATH = Path(__file__).resolve().parents[1] / "assets" / "scenes" / "desk" / "desk_pickup.xml"


def run(model_path: str, algorithm: str = "PPO", episodes: int = 10) -> None:
    algo_cls = ALGO_MAP.get(algorithm)
    if algo_cls is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"Loading trained model: {model_path}")
    trained_model = algo_cls.load(model_path)

    print("Creating environment...")
    env = gym.make("SimVerse/DeskPickup-v0")
    obs, info = env.reset(seed=42)

    mj_model = env.unwrapped.engine.mj_model
    mj_data = env.unwrapped.engine.mj_data

    print()
    print("Launching 3D viewer with trained policy...")
    print("  - Left-click + drag to rotate")
    print("  - Right-click + drag to pan")
    print("  - Scroll to zoom")
    print("  - Close the window to exit")
    print()

    episode = 0
    step = 0
    total_reward = 0.0
    successes = 0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        wall_start = time.time()

        while viewer.is_running() and episode < episodes:
            action, _ = trained_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            total_reward += reward

            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - wall_start
            sim_time = mj_data.time
            if sim_time > elapsed:
                time.sleep(sim_time - elapsed)

            if terminated or truncated:
                episode += 1
                success = info.get("success", False)
                if success:
                    successes += 1
                print(
                    f"  Episode {episode}/{episodes}: "
                    f"reward={total_reward:+.1f}, "
                    f"steps={step}, "
                    f"{'SUCCESS' if success else 'failed'}"
                )
                total_reward = 0.0
                step = 0
                if episode < episodes:
                    obs, info = env.reset()

    env.close()
    print(f"\nDone. Success rate: {successes}/{episodes} ({100*successes/max(1,episodes):.0f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained SimVerse policy")
    parser.add_argument("--model", required=True, help="Path to trained model (.zip)")
    parser.add_argument("--algo", default="PPO", choices=["PPO", "SAC", "TD3"])
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    args = parser.parse_args()

    run(args.model, args.algo, args.episodes)


if __name__ == "__main__":
    main()
