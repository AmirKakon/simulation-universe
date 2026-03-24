"""Integration test: full environment lifecycle (create, step, reset, close)."""

import gymnasium as gym

import simverse.envs  # noqa: F401 — triggers registration


class TestDeskPickupLifecycle:
    def test_make_and_reset(self) -> None:
        env = gym.make("SimVerse/DeskPickup-v0")
        obs, info = env.reset(seed=42)

        assert isinstance(obs, dict)
        assert "joint_positions" in obs
        assert "target_position" in obs
        assert len(obs["joint_positions"]) == 7
        assert len(obs["target_position"]) == 3

        env.close()

    def test_step_loop(self) -> None:
        env = gym.make("SimVerse/DeskPickup-v0")
        obs, info = env.reset(seed=42)

        total_reward = 0.0
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                obs, info = env.reset()

        assert isinstance(total_reward, float)
        env.close()

    def test_render_rgb_array(self) -> None:
        env = gym.make("SimVerse/DeskPickup-v0", render_mode="rgb_array")
        env.reset(seed=42)
        frame = env.render()
        assert frame is not None
        assert frame.shape[2] == 3  # RGB
        env.close()

    def test_multiple_resets(self) -> None:
        env = gym.make("SimVerse/DeskPickup-v0")
        for seed in [1, 2, 3]:
            obs, info = env.reset(seed=seed)
            assert "joint_positions" in obs
        env.close()

    def test_action_space_valid(self) -> None:
        env = gym.make("SimVerse/DeskPickup-v0")
        env.reset()
        assert env.action_space.shape == (8,)
        action = env.action_space.sample()
        assert len(action) == 8
        env.close()
