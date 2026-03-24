FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    libglew-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa

COPY pyproject.toml README.md ./
COPY src/ src/
COPY assets/ assets/
COPY configs/ configs/

# ---- Development/CPU stage ----
FROM base AS dev
RUN pip install -e ".[all]"
EXPOSE 8000
CMD ["python", "-m", "simverse.api.app"]

# ---- Training stage (CPU) ----
FROM base AS train-cpu
RUN pip install -e ".[training]"
ENTRYPOINT ["python", "-m", "simverse.training.cli"]

# ---- API server stage ----
FROM base AS api
RUN pip install -e ".[api]"
EXPOSE 8000
CMD ["uvicorn", "simverse.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- GPU training stage (requires nvidia base) ----
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS train-gpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libosmesa6-dev libglew-dev patchelf \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

COPY pyproject.toml README.md ./
COPY src/ src/
COPY assets/ assets/
COPY configs/ configs/

RUN pip install -e ".[training,cloud]"
ENTRYPOINT ["python", "-m", "simverse.training.cli"]
