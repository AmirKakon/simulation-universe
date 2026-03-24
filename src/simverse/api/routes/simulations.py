"""Simulation management endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import gymnasium as gym
import numpy as np
from fastapi import APIRouter, HTTPException

from simverse.api.schemas.models import (
    CreateSimulationRequest,
    SimulationResponse,
    SimulationStatus,
    StepRequest,
    StepResponse,
)

router = APIRouter(prefix="/api/simulations", tags=["simulations"])

_active_simulations: dict[str, dict[str, Any]] = {}


@router.post("", response_model=SimulationResponse, status_code=201)
async def create_simulation(request: CreateSimulationRequest) -> SimulationResponse:
    """Create a new simulation instance."""
    import simverse.envs  # noqa: F401

    try:
        env = gym.make(request.env_id, render_mode=request.render_mode)
    except gym.error.NameNotFound:
        raise HTTPException(status_code=404, detail=f"Environment '{request.env_id}' not found")

    sim_id = str(uuid.uuid4())[:8]
    obs, info = env.reset()

    _active_simulations[sim_id] = {
        "env": env,
        "env_id": request.env_id,
        "status": SimulationStatus.CREATED,
        "step_count": 0,
        "created_at": datetime.now(timezone.utc),
        "last_obs": obs,
    }

    return SimulationResponse(
        id=sim_id,
        env_id=request.env_id,
        status=SimulationStatus.CREATED,
        step_count=0,
        created_at=_active_simulations[sim_id]["created_at"],
    )


@router.get("", response_model=list[SimulationResponse])
async def list_simulations() -> list[SimulationResponse]:
    """List all active simulations."""
    return [
        SimulationResponse(
            id=sim_id,
            env_id=data["env_id"],
            status=data["status"],
            step_count=data["step_count"],
            created_at=data["created_at"],
        )
        for sim_id, data in _active_simulations.items()
    ]


@router.get("/{sim_id}", response_model=SimulationResponse)
async def get_simulation(sim_id: str) -> SimulationResponse:
    """Get simulation status."""
    if sim_id not in _active_simulations:
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found")

    data = _active_simulations[sim_id]
    return SimulationResponse(
        id=sim_id,
        env_id=data["env_id"],
        status=data["status"],
        step_count=data["step_count"],
        created_at=data["created_at"],
    )


@router.post("/{sim_id}/step", response_model=StepResponse)
async def step_simulation(sim_id: str, request: StepRequest) -> StepResponse:
    """Advance the simulation by one step."""
    if sim_id not in _active_simulations:
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found")

    data = _active_simulations[sim_id]
    env = data["env"]

    action = np.array(request.action, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    data["step_count"] += 1
    data["last_obs"] = obs
    data["status"] = SimulationStatus.RUNNING

    if terminated or truncated:
        data["status"] = SimulationStatus.COMPLETED

    serialized_obs = {
        k: v.tolist() if hasattr(v, "tolist") else [float(v)]
        for k, v in obs.items()
    }
    serialized_info = {
        k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
        for k, v in info.items()
    }

    return StepResponse(
        observation=serialized_obs,
        reward=float(reward),
        terminated=terminated,
        truncated=truncated,
        info=serialized_info,
    )


@router.post("/{sim_id}/reset", response_model=SimulationResponse)
async def reset_simulation(sim_id: str) -> SimulationResponse:
    """Reset a simulation to initial state."""
    if sim_id not in _active_simulations:
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found")

    data = _active_simulations[sim_id]
    obs, info = data["env"].reset()
    data["step_count"] = 0
    data["status"] = SimulationStatus.CREATED
    data["last_obs"] = obs

    return SimulationResponse(
        id=sim_id,
        env_id=data["env_id"],
        status=data["status"],
        step_count=0,
        created_at=data["created_at"],
    )


@router.delete("/{sim_id}", status_code=204)
async def delete_simulation(sim_id: str) -> None:
    """Close and remove a simulation."""
    if sim_id not in _active_simulations:
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found")

    data = _active_simulations.pop(sim_id)
    data["env"].close()
