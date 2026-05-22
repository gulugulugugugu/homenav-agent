import argparse
import json
import os
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw

from sim.habitat_env import HabitatEnvWrapper
from perception.detector import DetrTargetDetector, DebugTargetDetector
from agent.embodied_agent import EmbodiedNavAgent


def safe_text(text: str) -> str:
    """
    PIL default font may fail on Chinese characters.
    Keep overlay text ASCII-safe.
    """
    return str(text).encode("ascii", errors="replace").decode("ascii")


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text))


def draw_detection_overlay(rgb, step, message, detection=None):
    rgb = np.asarray(rgb)

    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Top status bar
    draw.rectangle([0, 0, img.width, 72], fill=(0, 0, 0))

    status = f"step={step} | {safe_text(message)[:110]}"
    draw.text((10, 10), status, fill=(255, 255, 255))

    if detection is not None:
        x1, y1, x2, y2 = detection["bbox"]
        label = detection.get("label", "object")
        score = float(detection.get("score", 0.0))

        # Clamp bbox
        x1 = max(0, min(img.width - 1, int(x1)))
        x2 = max(0, min(img.width - 1, int(x2)))
        y1 = max(0, min(img.height - 1, int(y1)))
        y2 = max(0, min(img.height - 1, int(y2)))

        # Red bbox
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)

        label_text = f"{label} {score:.2f}"
        label_y0 = max(0, y1 - 30)
        draw.rectangle([x1, label_y0, max(x2, x1 + 120), y1], fill=(255, 0, 0))
        draw.text((x1 + 5, label_y0 + 7), label_text, fill=(255, 255, 255))

    return img


def save_decision_frame(out_dir, step, rgb, message, detection):
    path = os.path.join(out_dir, f"step_{step:04d}.png")
    img = draw_detection_overlay(rgb, step, message, detection)
    img.save(path)
    return path


def save_evidence_frame(out_dir, step, rgb, message, detection):
    if detection is None:
        return None

    label = safe_name(detection.get("label", "object"))
    score = float(detection.get("score", 0.0))
    path = os.path.join(out_dir, f"step_{step:04d}_{label}_{score:.2f}.png")

    img = draw_detection_overlay(rgb, step, message, detection)
    img.save(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default="请到沙发旁边")
    parser.add_argument("--out", type=str, default="demos/sofa_v2")
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument(
        "--detector",
        type=str,
        default="detr",
        choices=["detr", "debug"],
    )
    parser.add_argument("--score_threshold", type=float, default=0.60)
    args = parser.parse_args()

    frames_dir = os.path.join(args.out, "frames")
    decision_dir = os.path.join(args.out, "decision_frames")
    evidence_dir = os.path.join(args.out, "evidence")

    if os.path.exists(args.out):
        shutil.rmtree(args.out)

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(decision_dir, exist_ok=True)
    os.makedirs(evidence_dir, exist_ok=True)

    sim_env = HabitatEnvWrapper(output_dir=frames_dir)

    if args.detector == "detr":
        detector = DetrTargetDetector(score_threshold=args.score_threshold)
    else:
        detector = DebugTargetDetector()

    agent = EmbodiedNavAgent(sim_env, detector)

    logs = []

    sim_env.reset()

    first_msg = agent.receive_command(args.command)
    print("User>", args.command)
    print("Agent>", first_msg)

    # Save initial decision frame before any action.
    initial_obs = sim_env.get_observation()
    initial_decision_path = save_decision_frame(
        decision_dir,
        0,
        initial_obs["rgb"],
        first_msg,
        None,
    )

    logs.append({
        "step": 0,
        "user_command": args.command,
        "agent_message": first_msg,
        "state": agent.state.value,
        "target": agent.target,
        "detection": None,
        "decision_frame": initial_decision_path,
        "evidence_frame": None,
    })

    completed = False

    for step in range(1, args.max_steps + 1):
        # This is the exact observation the agent will use for decision making.
        # We save this image, then agent.step() runs detection on the same current observation.
        pre_obs = sim_env.get_observation()
        pre_rgb = np.array(pre_obs["rgb"], copy=True)

        msg = agent.step()
        print("Agent>", msg)

        detection = agent.last_detection

        decision_path = save_decision_frame(
            decision_dir,
            step,
            pre_rgb,
            msg,
            detection,
        )

        evidence_path = None
        if detection is not None:
            evidence_path = save_evidence_frame(
                evidence_dir,
                step,
                pre_rgb,
                msg,
                detection,
            )
            print("Evidence>", evidence_path)

        logs.append({
            "step": step,
            "agent_message": msg,
            "state": agent.state.value,
            "target": agent.target,
            "detection": detection,
            "decision_frame": decision_path,
            "evidence_frame": evidence_path,
        })

        if "还需要什么" in msg:
            completed = True
            break

    if not completed:
        fail_msg = "任务未在最大步数内完成。"
        print("Agent>", fail_msg)

        final_obs = sim_env.get_observation()
        final_decision_path = save_decision_frame(
            decision_dir,
            args.max_steps + 1,
            final_obs["rgb"],
            fail_msg,
            agent.last_detection,
        )

        final_evidence_path = None
        if agent.last_detection is not None:
            final_evidence_path = save_evidence_frame(
                evidence_dir,
                args.max_steps + 1,
                final_obs["rgb"],
                fail_msg,
                agent.last_detection,
            )

        logs.append({
            "step": args.max_steps + 1,
            "agent_message": fail_msg,
            "state": agent.state.value,
            "target": agent.target,
            "detection": agent.last_detection,
            "decision_frame": final_decision_path,
            "evidence_frame": final_evidence_path,
        })

    with open(os.path.join(args.out, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    sim_env.close()

    print(f"\nSaved action frames to: {frames_dir}")
    print(f"Saved decision frames to: {decision_dir}")
    print(f"Saved evidence frames to: {evidence_dir}")
    print(f"Saved log to: {os.path.join(args.out, 'run_log.json')}")


if __name__ == "__main__":
    main()
