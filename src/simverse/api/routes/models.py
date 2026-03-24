"""Trained model management endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from simverse.api.schemas.models import TrainedModelInfo

router = APIRouter(prefix="/api/models", tags=["models"])

MODELS_DIR = Path("trained_models")


@router.get("", response_model=list[TrainedModelInfo])
async def list_models() -> list[TrainedModelInfo]:
    """List all saved trained models."""
    if not MODELS_DIR.exists():
        return []

    models: list[TrainedModelInfo] = []
    for model_dir in sorted(MODELS_DIR.iterdir()):
        if model_dir.is_dir():
            meta_path = model_dir / "metadata.yaml"
            if meta_path.exists():
                import yaml

                with open(meta_path) as f:
                    meta = yaml.safe_load(f)
                models.append(TrainedModelInfo(
                    id=model_dir.name,
                    algorithm=meta.get("algorithm", "unknown"),
                    env_id=meta.get("env_id", "unknown"),
                    path=str(model_dir),
                    timesteps=meta.get("timesteps", 0),
                    mean_reward=meta.get("mean_reward"),
                    created_at=meta.get("created_at", datetime.now(timezone.utc)),
                ))
    return models


@router.get("/{model_id}", response_model=TrainedModelInfo)
async def get_model(model_id: str) -> TrainedModelInfo:
    """Get details of a specific trained model."""
    model_dir = MODELS_DIR / model_id
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    meta_path = model_dir / "metadata.yaml"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Model metadata not found for '{model_id}'")

    import yaml

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    return TrainedModelInfo(
        id=model_id,
        algorithm=meta.get("algorithm", "unknown"),
        env_id=meta.get("env_id", "unknown"),
        path=str(model_dir),
        timesteps=meta.get("timesteps", 0),
        mean_reward=meta.get("mean_reward"),
        created_at=meta.get("created_at", datetime.now(timezone.utc)),
    )


@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: str) -> None:
    """Delete a trained model."""
    model_dir = MODELS_DIR / model_id
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    import shutil

    shutil.rmtree(model_dir)
