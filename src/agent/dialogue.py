TARGET_ALIASES = {
    "sofa": ["sofa", "couch", "沙发"],
    "bed": ["bed", "床"],
    "table": ["table", "desk", "桌子", "桌"],
    "chair": ["chair", "椅子"],
}


def parse_target(command: str):
    """
    Parse natural-language object navigation commands into detector target names.

    Existing stable targets:
    - table
    - sofa

    Extra candidate targets:
    - tv
    - potted plant
    - chair

    Keep table/sofa canonical names unchanged so previous successful demos are not broken.
    """
    text = (command or "").lower().strip()

    aliases = [
        ("sofa", [
            "沙发", "sofa", "couch",
        ]),
        ("table", [
            "桌子", "桌", "饭桌", "餐桌", "table", "dining table",
        ]),
        ("tv", [
            "电视", "电视机", "电视旁边", "tv", "television", "monitor", "screen",
        ]),
        ("potted plant", [
            "植物", "盆栽", "花盆", "绿植", "plant", "potted plant",
        ]),
        ("chair", [
            "椅子", "座椅", "chair",
        ]),
    ]

    for target, keys in aliases:
        if any(k in text for k in keys):
            return target

    return None


def completion_message():
    return "任务完成。还需要什么？"
