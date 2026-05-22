# 持续学习数据接口

本目录用于保存 HomeNav Agent 在导航过程中导出的持续学习数据接口。

本项目主线选择的是“导航方向”，并没有声称已经完成完整的 LeRobot 强化学习训练。本模块的作用是提供一个可扩展的数据层：把导航过程中的观测、动作、检测结果、规划状态和成功/失败信息整理成 episode 格式，方便未来接入人工干预数据、模仿学习或强化学习。

## 导出文件

运行命令：

    python scripts/export_intervention_dataset.py

会生成：

    learning_data/intervention_episodes.jsonl
    learning_data/summary.json

其中：

- intervention_episodes.jsonl：逐步记录每个导航 episode 中的状态、动作和标签。
- summary.json：统计每个 episode 的目标、步数和是否成功。

## 当前数据格式

每一步记录包含：

- 目标物体，例如 table、sofa、potted plant、tv
- 决策帧路径
- 检测证据帧路径
- planner 给出的动作
- 实际执行的动作
- 是否有人类干预
- 成功 / 失败标签
- reward 占位值
- 原始 agent message

当前自动运行的 demo 中没有人工干预，所以字段通常是：

    planner_action: turn_left
    executed_action: turn_left
    human_action: null
    intervened: false

如果未来加入人工干预界面，例如人在机器人走错时覆盖动作，则可以记录为：

    planner_action: turn_left
    executed_action: move_forward
    human_action: move_forward
    intervened: true

## 未来 LeRobot / 持续学习扩展

未来可以把每一次导航运行看成一个 episode：

- observation：RGB 图像、decision frame、检测证据、地图调试状态
- action：实际执行的底层导航动作
- intervention：人类覆盖动作
- reward：成功、失败、卡住、错误到达等信号

之后可以将这些数据转换为 LeRobot-style dataset，用于训练一个局部导航修正策略：

    当前观测 + planner 状态 -> 更好的底层动作

本模块目前只提供数据采集与导出接口，不声称已经完成强化学习训练或成功率提升。
