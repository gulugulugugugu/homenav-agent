import argparse
import glob
import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=3)
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames, "*.png")))

    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {args.frames}")

    with open(args.log, "r", encoding="utf-8") as f:
        logs = json.load(f)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with imageio.get_writer(args.out, fps=args.fps) as writer:
        for i, frame_path in enumerate(frame_paths):
            img = Image.open(frame_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            log_idx = min(i, len(logs) - 1)
            item = logs[log_idx]

            msg = item.get("agent_message", "")
            det = item.get("detection")

            # Top black status bar
            draw.rectangle([0, 0, img.width, 64], fill=(0, 0, 0))
            draw.text(
                (10, 10),
                f"Step {item.get('step', i)} | {msg[:100]}",
                fill=(255, 255, 255),
            )

            if det:
                x1, y1, x2, y2 = det["bbox"]
                label = det["label"]
                score = det["score"]

                # Red bbox
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)

                # Label background
                label_y1 = max(0, y1 - 30)
                draw.rectangle([x1, label_y1, x2, y1], fill=(255, 0, 0))
                draw.text(
                    (x1 + 5, label_y1 + 6),
                    f"{label} {score:.2f}",
                    fill=(255, 255, 255),
                )

            writer.append_data(np.asarray(img))

    print(f"Saved annotated video to: {args.out}")


if __name__ == "__main__":
    main()
