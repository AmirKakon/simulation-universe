"""Training management endpoints."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from simverse.api.schemas.models import TrainingRequest, TrainingResponse, TrainingStatus
from simverse.training.config import Algorithm, TrainingConfig
from simverse.training.trainer import Trainer

router = APIRouter(prefix="/api/training", tags=["training"])

_training_runs: dict[str, dict[str, Any]] = {}


def _run_training(run_id: str, config: TrainingConfig) -> None:
    """Execute training in a background thread."""
    run = _training_runs[run_id]
    run["status"] = TrainingStatus.RUNNING
    run["started_at"] = datetime.now(timezone.utc)

    trainer = Trainer(config)
    try:
        trainer.setup()
        model_path = trainer.train()
        run["status"] = TrainingStatus.COMPLETED
        run["model_path"] = str(model_path)
        run["current_timestep"] = config.total_timesteps
        run["completed_at"] = datetime.now(timezone.utc)
    except Exception as e:
        run["status"] = TrainingStatus.FAILED
        run["error"] = str(e)
        run["completed_at"] = datetime.now(timezone.utc)
    finally:
        trainer.cleanup()


@router.post("/start", response_model=TrainingResponse, status_code=201)
async def start_training(request: TrainingRequest) -> TrainingResponse:
    """Launch a training run in the background."""
    run_id = str(uuid.uuid4())[:8]

    overrides = dict(request.config_overrides) if request.config_overrides else {}
    config = TrainingConfig(
        env_id=request.env_id,
        algorithm=Algorithm(request.algorithm),
        total_timesteps=request.total_timesteps,
        seed=request.seed,
        learning_rate=request.learning_rate,
        **overrides,
    )

    _training_runs[run_id] = {
        "config": config,
        "env_id": request.env_id,
        "algorithm": request.algorithm,
        "status": TrainingStatus.QUEUED,
        "total_timesteps": request.total_timesteps,
        "current_timestep": 0,
        "model_path": None,
        "started_at": None,
        "completed_at": None,
    }

    thread = threading.Thread(target=_run_training, args=(run_id, config), daemon=True)
    thread.start()

    return TrainingResponse(
        id=run_id,
        env_id=request.env_id,
        algorithm=request.algorithm,
        status=TrainingStatus.QUEUED,
        total_timesteps=request.total_timesteps,
    )


@router.get("/{run_id}/status", response_model=TrainingResponse)
async def get_training_status(run_id: str) -> TrainingResponse:
    """Get the current status of a training run."""
    if run_id not in _training_runs:
        raise HTTPException(status_code=404, detail=f"Training run '{run_id}' not found")

    run = _training_runs[run_id]
    return TrainingResponse(
        id=run_id,
        env_id=run["env_id"],
        algorithm=run["algorithm"],
        status=run["status"],
        total_timesteps=run["total_timesteps"],
        current_timestep=run["current_timestep"],
        model_path=run.get("model_path"),
        started_at=run.get("started_at"),
        completed_at=run.get("completed_at"),
    )


@router.get("", response_model=list[TrainingResponse])
async def list_training_runs() -> list[TrainingResponse]:
    """List all training runs."""
    return [
        TrainingResponse(
            id=run_id,
            env_id=run["env_id"],
            algorithm=run["algorithm"],
            status=run["status"],
            total_timesteps=run["total_timesteps"],
            current_timestep=run["current_timestep"],
            model_path=run.get("model_path"),
            started_at=run.get("started_at"),
            completed_at=run.get("completed_at"),
        )
        for run_id, run in _training_runs.items()
    ]


@router.post("/{run_id}/cancel", response_model=TrainingResponse)
async def cancel_training(run_id: str) -> TrainingResponse:
    """Cancel a running training job."""
    if run_id not in _training_runs:
        raise HTTPException(status_code=404, detail=f"Training run '{run_id}' not found")

    run = _training_runs[run_id]
    if run["status"] in (TrainingStatus.COMPLETED, TrainingStatus.FAILED):
        raise HTTPException(status_code=400, detail="Training already finished")

    run["status"] = TrainingStatus.CANCELLED
    run["completed_at"] = datetime.now(timezone.utc)

    return TrainingResponse(
        id=run_id,
        env_id=run["env_id"],
        algorithm=run["algorithm"],
        status=run["status"],
        total_timesteps=run["total_timesteps"],
        current_timestep=run["current_timestep"],
        model_path=run.get("model_path"),
        started_at=run.get("started_at"),
        completed_at=run.get("completed_at"),
    )
