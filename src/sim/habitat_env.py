import os
import glob
import numpy as np
import imageio.v2 as imageio

import habitat_sim
from habitat_sim.agent import AgentConfiguration
from habitat_sim.agent import ActionSpec, ActuationSpec


class HabitatEnvWrapper:
    def __init__(self, data_root=None, output_dir="demos/run_frames"):
        self.data_root = data_root or os.path.expanduser("~/habitat_data")
        self.output_dir = output_dir
        self.sim = None
        self.frame_id = 0
        self.last_obs = None

        os.makedirs(self.output_dir, exist_ok=True)

    def reset(self):
        scene_candidates = glob.glob(
            os.path.join(self.data_root, "**", "*.glb"),
            recursive=True,
        )

        if not scene_candidates:
            raise FileNotFoundError("No .glb scene found under ~/habitat_data")

        scene_path = scene_candidates[0]
        print(f"[Habitat] Using scene: {scene_path}")

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
            "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
            "turn_left": ActionSpec("turn_left", ActuationSpec(amount=15.0)),
            "turn_right": ActionSpec("turn_right", ActuationSpec(amount=15.0)),
        }

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        self.sim.initialize_agent(0)

        self.last_obs = self.sim.get_sensor_observations()
        self.save_frame()
        return self.last_obs

    def get_observation(self):
        return self.last_obs

    def step(self, action: str):
        self.last_obs = self.sim.step(action)
        self.save_frame()
        return self.last_obs

    def move_forward(self):
        return self.step("move_forward")

    def turn_left(self):
        return self.step("turn_left")

    def turn_right(self):
        return self.step("turn_right")

    def stop(self):
        return self.last_obs

    def save_frame(self):
        if self.last_obs is None:
            return

        rgb = self.last_obs["rgb"][:, :, :3]
        path = os.path.join(self.output_dir, f"frame_{self.frame_id:04d}.png")
        imageio.imwrite(path, rgb)
        self.frame_id += 1

    def close(self):
        if self.sim is not None:
            self.sim.close()
