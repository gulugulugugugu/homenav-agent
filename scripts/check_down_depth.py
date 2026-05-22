import os
import numpy as np
from PIL import Image

from sim.habitat_env import HabitatEnvWrapper


def depth_to_png(depth, path):
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.05) & (d < 10.0)

    out = np.zeros_like(d, dtype=np.uint8)
    if valid.any():
        lo = np.percentile(d[valid], 5)
        hi = np.percentile(d[valid], 95)
        norm = (d - lo) / max(1e-6, hi - lo)
        out = np.clip(norm * 255, 0, 255).astype(np.uint8)

    Image.fromarray(out).save(path)


def main():
    out = "demos/down_depth_check"
    os.makedirs(out, exist_ok=True)

    env = HabitatEnvWrapper(output_dir=os.path.join(out, "frames"))
    obs = env.reset()

    Image.fromarray(obs["rgb"][:, :, :3]).save(os.path.join(out, "front_rgb_000.png"))
    depth_to_png(obs["depth"], os.path.join(out, "front_depth_000.png"))
    depth_to_png(obs["down_depth"], os.path.join(out, "down_depth_000.png"))

    for i in range(5):
        env.move_forward()
        obs = env.get_observation()
        Image.fromarray(obs["rgb"][:, :, :3]).save(os.path.join(out, f"front_rgb_{i+1:03d}.png"))
        depth_to_png(obs["down_depth"], os.path.join(out, f"down_depth_{i+1:03d}.png"))

    env.close()
    print("Saved to demos/down_depth_check")


if __name__ == "__main__":
    main()
