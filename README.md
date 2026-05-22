# HomeNav Agent：Habitat 中文具身导航系统

本项目实现了一个基于 Habitat 的中文具身导航 Agent。用户可以输入自然语言指令，例如：

- 请到桌子旁边
- 请到沙发旁边
- 请到植物旁边
- 请到电视旁边

Agent 会基于 RGB-D 视觉、机器人位姿、目标检测、地图构建和路径规划，导航到目标物体旁边。任务完成后，Agent 会回复：

    任务完成。还需要什么？

项目选择的是“导航方向”，并额外提供一个持续学习数据接口作为 bonus。

## 在线展示

项目网页：

https://gulugulugugugu.github.io/homenav-agent/

网页中包含 table、sofa、plant、tv 四个导航 demo 视频。

## GitHub 仓库

https://github.com/gulugulugugu/homenav-agent

## 功能特点

- Habitat 居家仿真环境
- 中文文字输入
- RGB-D 视觉感知
- COCO / Faster R-CNN 目标检测
- Depth-based top-down traversible map
- Object memory 与多帧确认
- 多实例目标选择
- Reachable standoff planning
- Path following
- 任务完成后询问用户“还需要什么”

## 系统流程

整体流程如下：

    中文指令
      -> 目标解析
      -> RGB 目标检测
      -> 多帧目标确认
      -> 多实例目标记忆
      -> Depth 建图
      -> 可通行区域分析
      -> 目标附近 reachable standoff cell 选择
      -> 路径规划与执行
      -> 完成任务并询问“还需要什么”

## 核心设计

### 1. 物体不是导航终点，而是语义锚点

系统不会直接追 bounding box 中心。检测到的物体只作为 semantic anchor。机器人真正要到达的是目标附近可通行、可到达的 free cell。

例如，“请到沙发旁边”不是撞向沙发中心，而是在沙发附近找到一个安全可达的停靠点。

### 2. Reachable Standoff Planning

系统会在当前地图的 reachable component 中寻找目标附近的 standoff cell。这样可以避免目标附近虽然看起来很近，但实际上被障碍物、墙或家具隔开的情况。

### 3. 多实例目标选择

如果环境中出现多个同类物体，例如多个植物，系统不会死守最早看到的目标，而是选择当前地图上更容易到达的 confirmed instance。这更接近人类的导航逻辑：如果旁边就有目标，就不会绕远路去另一个。

### 4. 严格到达验证

对于植物这类小物体，系统会使用更严格的到达距离验证，避免“地图上到达了某个 standoff cell，但实际上离目标还很远”的假到达问题。

## 运行方式

进入项目目录：

    cd ~/homenav-agent

设置环境变量：

    export KMP_DUPLICATE_LIB_OK=TRUE
    export OMP_NUM_THREADS=1
    export TORCH_HOME=~/torch_cache

运行桌子导航：

    PYTHONPATH=src python src/run_mapnav_demo.py \
      --command "请到桌子旁边" \
      --out demos/table_final \
      --max_steps 220 \
      --detector detr \
      --score_threshold 0.60

运行沙发导航：

    PYTHONPATH=src python src/run_mapnav_demo.py \
      --command "请到沙发旁边" \
      --out demos/sofa_final \
      --max_steps 220 \
      --detector detr \
      --score_threshold 0.60

运行植物导航：

    PYTHONPATH=src python src/run_mapnav_demo.py \
      --command "请到植物旁边" \
      --out demos/plant_strict_final \
      --max_steps 220 \
      --detector detr \
      --score_threshold 0.60

运行电视导航：

    PYTHONPATH=src python src/run_mapnav_demo.py \
      --command "请到电视旁边" \
      --out demos/tv_final \
      --max_steps 220 \
      --detector detr \
      --score_threshold 0.60

生成视频：

    python scripts/make_video.py \
      --frames demos/table_final/decision_frames \
      --out demos/table_final_decision.mp4 \
      --fps 3

## 持续学习数据接口 Bonus

本项目额外提供一个持续学习数据接口，但不声称已经完成完整 RL 训练。

运行：

    python scripts/export_intervention_dataset.py

会导出：

    learning_data/intervention_episodes.jsonl
    learning_data/summary.json

该接口会把导航过程中的 observation、planner action、executed action、检测结果、调试状态和成功 / 失败标签整理成 episode 数据。字段中预留了：

    human_action
    intervened

未来可以接入人工干预界面，将人类修正动作记录下来，并进一步转换为 LeRobot-style dataset，用于模仿学习或强化学习。

## 后续扩展

本项目主线聚焦于导航方向，当前版本已经完成 Habitat 场景中的中文目标导航、多目标支持、RGB-D 建图、可达停靠点规划和网页展示。

后续可以继续扩展：

- 接入更强的视觉检测或分割模型，提升小物体和遮挡场景下的稳定性。
- 将当前导出的 intervention-ready episodes 转换为 LeRobot-style dataset。
- 在人工干预数据基础上训练局部导航修正策略，进一步提升复杂场景下的成功率。

## 文件结构

    src/
      run_mapnav_demo.py          # map-based ObjectNav 主程序
      agent/                      # 指令解析与 agent 相关逻辑
      perception/                 # 目标检测
      sim/                        # Habitat 环境封装

    scripts/
      make_video.py               # demo 视频生成
      export_intervention_dataset.py

    docs/
      index.html                  # GitHub Pages 展示网页
      assets/                     # demo 视频资源

    learning_data/
      README.md
      intervention_episodes.jsonl
      summary.json

    demos/
      table_final/
      sofa_final/
      plant_strict_final/
      tv_final/

## 总结

HomeNav Agent 是一个模块化具身导航系统。它结合了 learned object detection、RGB-D 几何建图、目标记忆、多实例选择和经典路径规划，使机器人可以在 Habitat 居家环境中根据中文指令导航到多个目标物体旁边。
