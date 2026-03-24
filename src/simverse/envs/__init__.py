"""SimVerse Gymnasium environments — auto-registered on import."""

from gymnasium.envs.registration import register

register(
    id="SimVerse/DeskPickup-v0",
    entry_point="simverse.envs.manipulation.desk_pickup:DeskPickupEnv",
    max_episode_steps=500,
)
