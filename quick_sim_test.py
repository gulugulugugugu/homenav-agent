import os
import glob
import numpy as np
import imageio.v2 as imageio

import habitat_sim
from habitat_sim.agent import AgentConfiguration
from habitat_sim.agent import ActionSpec, ActuationSpec


DATA_ROOT = os.path.expanduser("~/habitat_data")

scene_candidates = glob.glob(
    os.path.join(DATA_ROOT, "**", "*.glb"),
    recursive=True,
)

if not scene_candidates:
    raise FileNotFoundError(
        "No .glb scene found under ~/habitat_data. "
        "Please download habitat_test_scenes first."
    )

scene_path = scene_candidates[0]
print("Using scene:", scene_path)

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene_path
sim_cfg.enable_physics = False

width = 640
height = 480

rgb_sensor = habitat_sim.CameraSensorSpec()
rgb_sensor.uuid = "rgb"
rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
rgb_sensor.resolution = [height, width]
rgb_sensor.position = [0.0, 1.5, 0.0]

depth_sensor = habitat_sim.CameraSensorSpec()
depth_sensor.uuid = "depth"
depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
depth_sensor.resolution = [height, width]
depth_sensor.position = [0.0, 1.5, 0.0]

agent_cfg = AgentConfiguration()
agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]

agent_cfg.action_space = {
    "move_forward": ActionSpec(
        "move_forward",
        ActuationSpec(amount=0.25),
    ),
    "turn_left": ActionSpec(
        "turn_left",
        ActuationSpec(amount=15.0),
    ),
    "turn_right": ActionSpec(
        "turn_right",
        ActuationSpec(amount=15.0),
    ),
}

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

agent = sim.initialize_agent(0)

output_dir = "test_outputs"
os.makedirs(output_dir, exist_ok=True)

actions = [
    "turn_left",
    "turn_left",
    "move_forward",
    "move_forward",
    "turn_right",
    "move_forward",
]

for step_id, action in enumerate(actions):
    obs = sim.step(action)

    rgb = obs["rgb"][:, :, :3]
    depth = obs["depth"]

    depth_vis = np.clip(depth, 0, 10)
    depth_vis = (depth_vis / 10.0 * 255).astype(np.uint8)

    imageio.imwrite(
        os.path.join(output_dir, f"rgb_{step_id:03d}.png"),
        rgb,
    )
    imageio.imwrite(
        os.path.join(output_dir, f"depth_{step_id:03d}.png"),
        depth_vis,
    )

    print(f"step={step_id}, action={action}, saved images")

sim.close()
print("Done. Check test_outputs/")