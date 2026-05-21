from enum import Enum


class AgentState(Enum):
    IDLE = "idle"
    PARSE_COMMAND = "parse_command"
    SEARCH_TARGET = "search_target"
    APPROACH_TARGET = "approach_target"
    ARRIVED = "arrived"
    FAILED = "failed"
