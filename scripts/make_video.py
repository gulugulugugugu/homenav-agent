import argparse
import glob
import os

import imageio.v2 as imageio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames, "*.png")))

    if not frame_paths:
        raise FileNotFoundError(f"No png frames found in {args.frames}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with imageio.get_writer(args.out, fps=args.fps) as writer:
        for path in frame_paths:
            frame = imageio.imread(path)
            writer.append_data(frame)

    print(f"Saved video to: {args.out}")


if __name__ == "__main__":
    main()
