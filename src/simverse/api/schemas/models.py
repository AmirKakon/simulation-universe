"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SimulationStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CreateSimulationRequest(BaseModel):
    env_id: str = "SimVerse/DeskPickup-v0"
    render_mode: str | None = "rgb_array"


class StepRequest(BaseModel):
    action: list[float]


class StepResponse(BaseModel):
    observation: dict[str, list[float]]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class SimulationResponse(BaseModel):
    id: str
    env_id: str
    status: SimulationStatus
    step_count: int = 0
    created_at: datetime


class TrainingRequest(BaseModel):
    env_id: str = "SimVerse/DeskPickup-v0"
    algorithm: str = "PPO"
    total_timesteps: int = Field(default=100_000, ge=1)
    seed: int = 42
    learning_rate: float = Field(default=3e-4, gt=0)
    config_overrides: dict[str, object] | None = None


class TrainingResponse(BaseModel):
    id: str
    env_id: str
    algorithm: str
    status: TrainingStatus
    total_timesteps: int
    current_timestep: int = 0
    mean_reward: float | None = None
    success_rate: float | None = None
    model_path: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RobotInfo(BaseModel):
    name: str
    class_name: str = Field(alias="class")
    description: str


class TrainedModelInfo(BaseModel):
    id: str
    algorithm: str
    env_id: str
    path: str
    timesteps: int
    mean_reward: float | None = None
    created_at: datetime
