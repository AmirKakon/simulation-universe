"""Visual demo: watch the Panda arm interact with objects on a desk.

Run with:
    .venv\\Scripts\\python.exe scripts\\demo_visual.py
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE_PATH = Path(__file__).resolve().parents[1] / "assets" / "scenes" / "desk" / "desk_pickup.xml"

print("Loading desk pickup scene...")
model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print(f"  Joints: {model.nq}, Actuators: {model.nu}")
print(f"  Timestep: {model.opt.timestep}s")
print()
print("Launching interactive 3D viewer...")
print("  - Left-click + drag to rotate")
print("  - Right-click + drag to pan")
print("  - Scroll to zoom")
print("  - Double-click a body to track it")
print("  - Close the window to exit")
print()

JOINT_RANGES = [
    (-2.8973, 2.8973),   # joint1
    (-1.7628, 1.7628),   # joint2
    (-2.8973, 2.8973),   # joint3
    (-3.0718, -0.0698),  # joint4
    (-2.8973, 2.8973),   # joint5
    (-0.0175, 3.7525),   # joint6
    (-2.8973, 2.8973),   # joint7
    (0.0, 0.04),         # finger1
    (0.0, 0.04),         # finger2
]
ACTION_LOW = np.array([r[0] for r in JOINT_RANGES], dtype=np.float64)
ACTION_HIGH = np.array([r[1] for r in JOINT_RANGES], dtype=np.float64)

HOLD_STEPS = 200  # hold each action for 200 physics steps (0.4 seconds)

step = 0
action_step = 0
current_action = np.zeros(model.nu, dtype=np.float64)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running():
        # Pick a new random target every HOLD_STEPS physics steps
        if action_step % HOLD_STEPS == 0:
            current_action = np.random.uniform(
                low=ACTION_LOW, high=ACTION_HIGH,
            )
            gripper = np.random.choice([0.0, 0.04])  # open or closed
            current_action[7] = gripper
            current_action[8] = gripper

        np.copyto(data.ctrl, current_action)
        mujoco.mj_step(model, data)
        step += 1
        action_step += 1

        viewer.sync()

        # Real-time pacing
        elapsed = time.time() - start
        if data.time > elapsed:
            time.sleep(data.time - elapsed)

        if step % 1000 == 0:
            print(f"  Step {step:5d} | Sim time: {data.time:.2f}s")

print(f"\nDone. Ran {step} steps.")
