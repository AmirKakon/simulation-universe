"""FastAPI application — the main entry point for the SimVerse REST API."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from simverse.api.routes import models, robots, simulations, training

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SimVerse API",
    description=(
        "REST API for the SimVerse robotics simulation platform. "
        "Create simulations, train robot policies, and manage trained models."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulations.router)
app.include_router(training.router)
app.include_router(robots.router)
app.include_router(models.router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {
        "service": "SimVerse API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "healthy"}


def run_server() -> None:
    """Launch the API server via uvicorn."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("Starting SimVerse API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    run_server()
