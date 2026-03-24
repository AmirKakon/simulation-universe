# SimVerse

A robotics simulation platform for creating, testing, and training AI-powered robots. Built on MuJoCo and Gymnasium with a REST API for integration with AI agent ecosystems.

## Overview

SimVerse provides a modular framework for:

- **Simulating robots** in physics-accurate 3D environments (MuJoCo)
- **Training robot policies** using reinforcement learning (Stable-Baselines3)
- **Controlling simulations remotely** via a REST API (FastAPI)
- **Scaling training to cloud GPUs** via Docker and MJX

The platform is designed as the simulation backbone for an AI agent ecosystem — your Jarvis-style agent can create environments, train robots, and evaluate results through the API.

## Architecture

```
simverse/
├── core/          # Engine, Robot, Task, Scene, Sensor abstractions
├── engines/       # MuJoCo (CPU) and MJX (GPU) backends
├── robots/        # Robot definitions (Panda arm, extensible registry)
├── envs/          # Gymnasium environments (DeskPickup, more to come)
├── training/      # RL training pipeline (PPO, SAC, TD3)
├── api/           # FastAPI REST server
├── viewer/        # Local MuJoCo viewer
└── utils/         # Config loading, project helpers
```

### Design Principles

1. **Protocol-driven** — `PhysicsEngine`, `Robot`, and `Task` are abstract interfaces. Swap MuJoCo for Isaac Sim without changing environment code.
2. **Gymnasium-native** — Every environment is a standard `gymnasium.Env`. Use any RL library.
3. **API-first** — Everything is accessible through REST endpoints for agent orchestration.
4. **Sim-to-real ready** — MJCF/URDF robot models and modular control interfaces prepare for real hardware.

## Quick Start

### Installation

```bash
# Clone and install (CPU — works on any machine)
git clone <repo-url> simulation-universe
cd simulation-universe
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -e ".[all]"     # Everything: training, API, viewer, dev tools
```

Minimal install options:
```bash
pip install -e "."            # Core only (simulation + gymnasium)
pip install -e ".[training]"  # Add RL training (PyTorch, SB3)
pip install -e ".[api]"       # Add REST API server
pip install -e ".[cloud]"     # Add MJX GPU acceleration (requires CUDA)
```

### Run a Simulation (Python)

```python
import gymnasium as gym
import simverse.envs  # registers SimVerse environments

env = gym.make("SimVerse/DeskPickup-v0", render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Train a Policy (CLI)

```bash
# Train with PPO for 100K steps
simverse-train --env SimVerse/DeskPickup-v0 --algo PPO --timesteps 100000

# Train with a config file
simverse-train --config configs/training/default.yaml

# Train with SAC and custom learning rate
simverse-train --algo SAC --lr 1e-4 --timesteps 500000
```

### Train a Policy (Python)

```python
from simverse.training import Trainer, TrainingConfig, Algorithm

config = TrainingConfig(
    env_id="SimVerse/DeskPickup-v0",
    algorithm=Algorithm.PPO,
    total_timesteps=100_000,
    learning_rate=3e-4,
)

trainer = Trainer(config)
trainer.setup()
model_path = trainer.train()  # saves model + TensorBoard logs
trainer.cleanup()
```

### Start the API Server

```bash
# Direct
simverse-serve

# Or with uvicorn (auto-reload for development)
uvicorn simverse.api.app:app --reload --port 8000
```

Then visit `http://localhost:8000/docs` for interactive API documentation.

### API Usage Examples

```bash
# Create a simulation
curl -X POST http://localhost:8000/api/simulations \
  -H "Content-Type: application/json" \
  -d '{"env_id": "SimVerse/DeskPickup-v0"}'

# Step the simulation
curl -X POST http://localhost:8000/api/simulations/{id}/step \
  -H "Content-Type: application/json" \
  -d '{"action": [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.5]}'

# Start a training run
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"env_id": "SimVerse/DeskPickup-v0", "algorithm": "PPO", "total_timesteps": 100000}'

# Check training status
curl http://localhost:8000/api/training/{id}/status

# List available robots
curl http://localhost:8000/api/robots
```

### Docker

```bash
# Run the API server
docker compose up api

# Run a training job
docker compose --profile training run train --config configs/training/default.yaml

# Launch TensorBoard to monitor training
docker compose --profile monitoring up tensorboard
```

### Cloud GPU Training (Azure)

```bash
# Deploy a GPU VM and run training
python scripts/deploy_azure_gpu.py --resource-group simverse-rg --vm-name simverse-train
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/simulations` | Create a new simulation |
| `GET` | `/api/simulations` | List all simulations |
| `GET` | `/api/simulations/{id}` | Get simulation status |
| `POST` | `/api/simulations/{id}/step` | Step simulation forward |
| `POST` | `/api/simulations/{id}/reset` | Reset simulation |
| `DELETE` | `/api/simulations/{id}` | Delete simulation |
| `POST` | `/api/training/start` | Start a training run |
| `GET` | `/api/training/{id}/status` | Get training progress |
| `GET` | `/api/training` | List all training runs |
| `POST` | `/api/training/{id}/cancel` | Cancel training |
| `GET` | `/api/robots` | List available robots |
| `GET` | `/api/models` | List trained models |
| `GET` | `/api/models/{id}` | Get model details |
| `DELETE` | `/api/models/{id}` | Delete a model |

## Project Structure

```
simulation-universe/
├── src/simverse/              # Main Python package
│   ├── core/                  # Abstract interfaces
│   │   ├── engine.py          # PhysicsEngine protocol
│   │   ├── robot.py           # Robot base class
│   │   ├── task.py            # Task/reward protocol
│   │   ├── environment.py     # SimVerseEnv (Gymnasium base)
│   │   ├── scene.py           # Scene composition
│   │   └── sensor.py          # Virtual sensors
│   ├── engines/               # Physics backends
│   │   ├── mujoco_engine.py   # MuJoCo CPU
│   │   └── mjx_engine.py     # MJX GPU (stub)
│   ├── robots/                # Robot library
│   │   ├── registry.py        # Robot discovery + instantiation
│   │   └── arms/panda.py      # Franka Panda 7-DOF arm
│   ├── envs/                  # Gymnasium environments
│   │   └── manipulation/
│   │       └── desk_pickup.py # DeskPickup-v0
│   ├── training/              # RL training pipeline
│   │   ├── trainer.py         # Training orchestrator
│   │   ├── config.py          # Pydantic config
│   │   ├── callbacks.py       # SB3 callbacks
│   │   └── evaluation.py     # Policy evaluation
│   ├── api/                   # REST API
│   │   ├── app.py             # FastAPI app
│   │   ├── routes/            # Endpoint handlers
│   │   └── schemas/           # Request/response models
│   ├── viewer/                # Visualization
│   └── utils/                 # Helpers
├── assets/                    # Robot models + scene files (MJCF)
├── configs/                   # YAML configuration files
├── tests/                     # Unit + integration tests
├── scripts/                   # CLI tools + cloud deployment
├── Dockerfile                 # Multi-stage build
├── docker-compose.yml         # Local orchestration
└── pyproject.toml             # Project configuration
```

## Use Cases Roadmap

| Phase | Use Case | Status |
|-------|----------|--------|
| 1 | Platform foundation + DeskPickup environment | Done |
| 2 | Robot arm picks up objects from desk (trained policy) | Next |
| 3 | Robot arm stacks/sorts multiple objects | Planned |
| 4 | Dual robot arms — collaborative tasks | Planned |
| 5 | Dual arms fold laundry (cloth simulation) | Planned |
| 6 | Object scanning to 3D model (separate: `3d-forge`) | Planned |
| 7 | Text prompt to 3D printable model (separate: `3d-forge`) | Planned |
| 8 | Sim-to-real transfer to physical robots (separate: `robot-bridge`) | Planned |

## Related Projects (Future)

- **3d-forge** — Text-to-3D and scan-to-3D model generation (diffusion models, NeRF). Connects to SimVerse via API to import generated models as simulation objects.
- **robot-bridge** — Sim-to-real transfer pipeline. Exports trained policies from SimVerse and deploys them to physical robots via ROS2.

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Physics | MuJoCo (CPU) / MJX (GPU) | Accurate physics simulation |
| RL Interface | Gymnasium | Standard environment API |
| RL Training | Stable-Baselines3 | PPO, SAC, TD3 algorithms |
| API | FastAPI | REST API for agent integration |
| Config | Pydantic + YAML | Validated configuration |
| Metrics | TensorBoard | Training visualization |
| Containers | Docker | Reproducible environments |
| Cloud | Azure GPU VMs | Scalable training |

## License

MIT
