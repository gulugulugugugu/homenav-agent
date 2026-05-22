import argparse
import json
from pathlib import Path

DEFAULT_RUNS = [
    "table_final",
    "sofa_final",
    "plant_strict_final",
    "tv_final",
]

TARGET_HINTS = {
    "table": "table",
    "sofa": "sofa",
    "plant": "potted plant",
    "tv": "tv",
}

def infer_target_from_episode(episode_id):
    name = episode_id.lower()
    for key, target in TARGET_HINTS.items():
        if key in name:
            return target
    return "unknown"

def infer_action_from_message(message):
    msg = message.lower()

    if "map-arrived" in msg or "任务完成" in msg:
        return "stop"
    if "moving forward" in msg or "move_forward" in msg:
        return "move_forward"
    if "turning left" in msg or "turn_left" in msg:
        return "turn_left"
    if "turning right" in msg or "turn_right" in msg:
        return "turn_right"
    if "rotating" in msg or "no-path" in msg:
        return "turn_left"

    return "unknown"

def infer_reward(message, success_now):
    msg = message.lower()

    if "map-arrived" in msg or "任务完成" in msg:
        return 10.0
    if "arrival-rejected" in msg:
        return -1.0
    if "path-blocked" in msg:
        return -0.5
    if "no-path" in msg:
        return -0.2
    if success_now:
        return 0.0

    return -0.01

def safe_detection_fields(detection):
    if not detection:
        return None

    return {
        "label": detection.get("label"),
        "score": detection.get("score"),
        "bbox": detection.get("bbox"),
    }

def build_record(episode_id, run_dir, item, episode_success):
    step = int(item.get("step", 0))
    message = item.get("agent_message", "")
    target = infer_target_from_episode(episode_id)
    planner_action = infer_action_from_message(message)

    decision_frame = run_dir / "decision_frames" / f"decision_{step:04d}.png"
    rgb_frame = run_dir / "frames" / f"frame_{step + 1:04d}.png"

    evidence_frame = item.get("evidence_frame")
    if evidence_frame:
        evidence_frame = str(Path(evidence_frame))

    done = bool(item.get("done", False))
    success_now = done or ("MAP-ARRIVED" in message) or ("任务完成" in message)

    return {
        "episode_id": episode_id,
        "step": step,
        "target": target,
        "observation": {
            "rgb_frame": str(rgb_frame),
            "decision_frame": str(decision_frame),
            "evidence_frame": evidence_frame,
        },
        "state": {
            "plan_type": item.get("plan"),
            "detection": safe_detection_fields(item.get("detection")),
            "debug": item.get("debug", {}),
        },
        "action": {
            "planner_action": planner_action,
            "executed_action": planner_action,
            "human_action": None,
            "intervened": False,
        },
        "learning_labels": {
            "success_now": success_now,
            "episode_success": episode_success,
            "reward": infer_reward(message, success_now),
            "failure_reason": None if episode_success else "not_successful_or_not_finished",
        },
        "raw_agent_message": message,
    }

def load_episode(run_dir):
    log_path = run_dir / "run_log.json"

    if not log_path.exists():
        return []

    with log_path.open("r", encoding="utf-8") as f:
        logs = json.load(f)

    if not logs:
        return []

    final_msg = logs[-1].get("agent_message", "")
    episode_success = (
        "MAP-ARRIVED" in final_msg
        or "任务完成" in final_msg
        or bool(logs[-1].get("done", False))
    )

    episode_id = run_dir.name
    return [build_record(episode_id, run_dir, item, episode_success) for item in logs]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos_dir", type=str, default="demos")
    parser.add_argument("--out_dir", type=str, default="learning_data")
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    args = parser.parse_args()

    demos_dir = Path(args.demos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    summary = {
        "description": "HomeNav Agent 持续学习数据接口导出结果。本数据不是 RL 训练结果，而是 intervention-ready episode 数据。",
        "num_episodes": 0,
        "num_steps": 0,
        "num_success": 0,
        "success_rate": 0.0,
        "episodes": [],
        "schema": {
            "planner_action": "导航 planner 原本建议的动作",
            "executed_action": "实际执行动作；当前自动运行版本中与 planner_action 相同",
            "human_action": "预留字段；未来人工干预时记录人类覆盖动作",
            "intervened": "当前为 false；未来人工干预时为 true",
        },
    }

    for run_name in args.runs:
        run_dir = demos_dir / run_name
        records = load_episode(run_dir)

        if not records:
            summary["episodes"].append({
                "episode_id": run_name,
                "status": "missing_or_empty",
                "num_steps": 0,
                "success": False,
            })
            continue

        success = bool(records[-1]["learning_labels"]["episode_success"])
        target = records[0]["target"]

        summary["episodes"].append({
            "episode_id": run_name,
            "target": target,
            "status": "exported",
            "num_steps": len(records),
            "success": success,
        })

        all_records.extend(records)

    exported = [e for e in summary["episodes"] if e["status"] == "exported"]
    summary["num_episodes"] = len(exported)
    summary["num_steps"] = len(all_records)
    summary["num_success"] = len([e for e in exported if e.get("success")])
    summary["success_rate"] = summary["num_success"] / summary["num_episodes"] if summary["num_episodes"] else 0.0

    jsonl_path = out_dir / "intervention_episodes.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(all_records)} records.")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {summary_path}")
    print(f"Success rate over exported episodes: {summary['success_rate']:.2f}")

if __name__ == "__main__":
    main()
