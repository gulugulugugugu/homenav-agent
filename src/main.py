from sim.habitat_env import HabitatEnvWrapper
from perception.detector import DebugTargetDetector
from agent.embodied_agent import EmbodiedNavAgent


def main():
    sim_env = HabitatEnvWrapper()
    detector = DebugTargetDetector()
    agent = EmbodiedNavAgent(sim_env, detector)

    sim_env.reset()

    print("\nHomeNav-Agent is ready.")
    print("Try: 请到沙发旁边 / 请到床旁边 / go to the table")
    print("Type q to quit.\n")

    while True:
        command = input("User> ")

        if command.lower() in ["q", "quit", "exit"]:
            break

        print("Agent>", agent.receive_command(command))

        for _ in range(30):
            msg = agent.step()
            print("Agent>", msg)

            if "还需要什么" in msg:
                break

    sim_env.close()


if __name__ == "__main__":
    main()
