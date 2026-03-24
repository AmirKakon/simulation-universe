"""Robot registry — discover and instantiate robots by name."""

from __future__ import annotations

import logging
from typing import Any

from simverse.core.robot import Robot, RobotConfig

logger = logging.getLogger(__name__)

_ROBOT_REGISTRY: dict[str, type[Robot]] = {}


def register_robot(name: str) -> Any:
    """Decorator to register a Robot subclass under a given name."""

    def decorator(cls: type[Robot]) -> type[Robot]:
        if name in _ROBOT_REGISTRY:
            logger.warning("Overwriting robot registration: %s", name)
        _ROBOT_REGISTRY[name] = cls
        return cls

    return decorator


def get_robot(name: str, config: RobotConfig | None = None) -> Robot:
    """Instantiate a registered robot by name."""
    if name not in _ROBOT_REGISTRY:
        available = ", ".join(sorted(_ROBOT_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Robot '{name}' not registered. Available: {available}")

    robot_cls = _ROBOT_REGISTRY[name]
    if config is None:
        config = _get_default_config(name, robot_cls)
    return robot_cls(config)


def list_robots() -> list[dict[str, str]]:
    """Return metadata for all registered robots."""
    result = []
    for name, cls in sorted(_ROBOT_REGISTRY.items()):
        result.append({
            "name": name,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "description": cls.__doc__ or "",
        })
    return result


def _get_default_config(name: str, cls: type[Robot]) -> RobotConfig:
    """Try to get a default config from the robot class."""
    if hasattr(cls, "default_config"):
        return cls.default_config()  # type: ignore[attr-defined]
    raise ValueError(
        f"Robot '{name}' has no default_config() classmethod. "
        f"Provide a RobotConfig explicitly."
    )


def discover_robots() -> None:
    """Import all built-in robot modules to trigger registration."""
    import simverse.robots.arms  # noqa: F401
