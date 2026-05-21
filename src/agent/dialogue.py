TARGET_ALIASES = {
    "sofa": ["sofa", "couch", "沙发"],
    "bed": ["bed", "床"],
    "table": ["table", "desk", "桌子", "桌"],
    "chair": ["chair", "椅子"],
}


def parse_target(command: str):
    command = command.lower()
    for target, aliases in TARGET_ALIASES.items():
        for alias in aliases:
            if alias.lower() in command:
                return target
    return None


def completion_message():
    return "任务完成。还需要什么？"
