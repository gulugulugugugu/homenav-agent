from agent.state_machine import AgentState
from agent.dialogue import parse_target, completion_message


class EmbodiedNavAgent:
    def __init__(self, sim_env, detector):
        self.sim_env = sim_env
        self.detector = detector
        self.state = AgentState.IDLE
        self.target = None
        self.approach_steps = 0

    def receive_command(self, command: str):
        self.state = AgentState.PARSE_COMMAND
        self.target = parse_target(command)

        if self.target is None:
            self.state = AgentState.FAILED
            return "我没识别出目标。你可以说：请到沙发旁边 / 请到床旁边 / 请到桌子旁边 / 请到椅子旁边。"

        self.state = AgentState.SEARCH_TARGET
        self.approach_steps = 0
        return f"收到，我会导航到 {self.target} 旁边。"

    def step(self):
        obs = self.sim_env.get_observation()

        if self.state == AgentState.SEARCH_TARGET:
            detection = self.detector.detect(obs["rgb"], self.target)

            if detection is None:
                self.sim_env.turn_left()
                return f"[SEARCH] Searching for {self.target}..."

            self.state = AgentState.APPROACH_TARGET
            return f"[DETECT] Detected {self.target}. Approaching..."

        if self.state == AgentState.APPROACH_TARGET:
            obs = self.sim_env.get_observation()
            detection = self.detector.detect(obs["rgb"], self.target)

            if detection is None:
                self.state = AgentState.SEARCH_TARGET
                return f"[LOST] Lost {self.target}. Searching again..."

            distance = self.detector.estimate_distance(obs["depth"], detection)
            center_x = detection["center_x"]

            self.approach_steps += 1

            if self.approach_steps >= 8 or distance < 1.0:
                self.sim_env.stop()
                self.state = AgentState.ARRIVED
                return completion_message()

            if center_x < 0.45:
                self.sim_env.turn_left()
                return f"[CONTROL] Target left. Turning left. distance={distance:.2f}"

            if center_x > 0.55:
                self.sim_env.turn_right()
                return f"[CONTROL] Target right. Turning right. distance={distance:.2f}"

            self.sim_env.move_forward()
            return f"[CONTROL] Moving forward to {self.target}. distance={distance:.2f}"

        if self.state == AgentState.ARRIVED:
            return completion_message()

        return "[IDLE]"
