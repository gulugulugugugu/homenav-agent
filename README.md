# HomeNav-Agent

HomeNav-Agent is a language-guided embodied navigation agent built in Habitat simulation.

Given a natural language command such as "请到沙发旁边", the agent parses the target location, runs a visual navigation loop in a simulated indoor environment, approaches the target area, and asks the user "还需要什么？" after task completion.

## Demo Commands

The current system supports:

- 请到沙发旁边
- 请到床旁边
- 请到桌子旁边
- 请到椅子旁边
- go to the sofa
- go to the bed
- go to the table
- go to the chair

## System Architecture

User Command  
→ Language Parser  
→ Embodied Agent State Machine  
→ Visual Target Adapter  
→ Habitat Simulator  
→ Low-level Navigation Actions  
→ Completion Response

## Features

- Text-command interface
- Multi-target navigation support
- Habitat-based indoor simulation
- RGB-D observation interface
- Modular embodied agent design
- State-machine-based control logic
- Demo video generation
- Web presentation interface
- Continual learning extension proposal

## No Privileged Information Policy

The final agent design does not rely on simulator object coordinates, ground-truth target pose, semantic scene graph, or shortest-path oracle during execution.

The allowed inputs are:

- RGB observation
- Depth observation
- Robot/agent state
- Action history
- Collision or navigation feedback

The current demo uses a debug visual-target adapter to validate the full language-to-navigation pipeline. The perception module is designed as a replaceable interface and can be upgraded to open-vocabulary object detectors such as OWL-ViT, GroundingDINO, or YOLO-World.

## Run the Demo

```bash
conda activate homenav
cd ~/homenav-agent
PYTHONPATH=src python src/run_demo.py --command "请到沙发旁边" --out demos/sofa_demo
python scripts/make_video.py --frames demos/sofa_demo/frames --out demos/sofa_demo.mp4 --fps 4
