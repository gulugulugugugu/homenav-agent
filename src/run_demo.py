import argparse
import json
import os
import shutil

from sim.habitat_env import HabitatEnvWrapper
from perception.detector import DebugTargetDetector
from agent.embodied_agent import EmbodiedNavAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default="请到沙发旁边")
    parser.add_argument("--out", type=str, default="demos/sofa_demo")
    parser.add_argument("--max_steps", type=int, default=40)
    args = parser.parse_args()

    frames_dir = os.path.join(args.out, "frames")

    if os.path.exists(args.out):
        shutil.rmtree(args.out)

    os.makedirs(frames_dir, exist_ok=True)

    sim_env = HabitatEnvWrapper(output_dir=frames_dir)
    detector = DebugTargetDetector()
    agent = EmbodiedNavAgent(sim_env, detector)

    logs = []

    sim_env.reset()

    first_msg = agent.receive_command(args.command)
    print("User>", args.command)
    print("Agent>", first_msg)

    logs.append({
        "step": 0,
        "user_command": args.command,
        "agent_message": first_msg,
        "state": agent.state.value,
        "target": agent.target,
    })

    for step in range(1, args.max_steps + 1):
        msg = agent.step()
        print("Agent>", msg)

        logs.append({
            "step": step,
            "agent_message": msg,
            "state": agent.state.value,
            "target": agent.target,
        })

        if "还需要什么" in msg:
            break

    with open(os.path.join(args.out, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    sim_env.close()

    print(f"\nSaved frames to: {frames_dir}")
    print(f"Saved log to: {os.path.join(args.out, 'run_log.json')}")


if __name__ == "__main__":
    main()
