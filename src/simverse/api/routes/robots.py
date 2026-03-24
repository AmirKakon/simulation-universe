"""Robot listing endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from simverse.robots import discover_robots, list_robots

router = APIRouter(prefix="/api/robots", tags=["robots"])


@router.get("")
async def get_robots() -> list[dict[str, str]]:
    """List all registered robot models with their metadata."""
    discover_robots()
    return list_robots()
