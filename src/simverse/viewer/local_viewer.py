"""Local MuJoCo viewer wrapper for interactive visualization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer

logger = logging.getLogger(__name__)


def launch_viewer(model_path: Path, **kwargs: Any) -> None:
    """Launch the interactive MuJoCo viewer for a given model."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    logger.info("Launching MuJoCo viewer for: %s", model_path.name)
    mujoco.viewer.launch(model, data)


def launch_passive_viewer(
    model_path: Path,
) -> tuple[mujoco.MjModel, mujoco.MjData, Any]:
    """Launch a passive (non-blocking) MuJoCo viewer.

    Returns (model, data, viewer_handle) so the caller can step
    the simulation and the viewer updates automatically.
    """
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    handle = mujoco.viewer.launch_passive(model, data)
    return model, data, handle
