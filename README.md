# AstraDrive CARLA Lab

`AstraDrive CARLA Lab` is a lightweight public snapshot of a CARLA-based autonomous driving simulation workspace.

This repository only keeps the core scripts and documentation needed to understand and run the project:

- Ego vehicle spawning and control
- NPC traffic generation
- Multimodal sensor data collection
- Logging and environment validation
- BeamFusion sandbox code for isolated algorithm experiments

Large local assets are intentionally excluded from the public repo:

- CARLA engine/source trees
- Generated sensor datasets
- Videos, PDFs, posters, and presentation files
- Temporary outputs, caches, and experiment artifacts

## Project Structure

```text
.
|-- README.md
|-- PROJECT_STRUCTURE.md
|-- autonomous_driving_test.py
|-- control_ego_vehicle.py
|-- spawn_npc_traffic.py
|-- sensor_io_handler.py
|-- run_with_logging.py
|-- 环境验证脚本.py
|-- quick_test.py
|-- test_*.py
|-- 启动CARLA服务器.bat
|-- 启动CARLA服务器-高性能.bat
|-- 运行自动驾驶测试.bat
|-- 运行NPC交通生成.bat
|-- 运行Ego Vehicle控制.bat
|-- 运行环境验证.bat
|-- CARLA安装说明.md
|-- CARLA完整使用指南.md
|-- 运行说明_性能_算法.md
`-- beamfusion_sandbox/
```

Detailed layout notes are in `PROJECT_STRUCTURE.md`.

## Core Scripts

- `autonomous_driving_test.py`: main integrated autonomous driving and multimodal collection script
- `spawn_npc_traffic.py`: spawn background traffic and walkers
- `control_ego_vehicle.py`: manual ego vehicle control entry point
- `sensor_io_handler.py`: sensor data persistence helpers
- `run_with_logging.py`: wrapper to run the main script with full logging
- `环境验证脚本.py`: environment and dependency checks

## BeamFusion Sandbox

`beamfusion_sandbox/` contains isolated experiments for BeamFusion-based CARLA control and evaluation. Public code is kept, but generated outputs are excluded.

## Quick Start

1. Start the CARLA server.
2. Verify the Python/CARLA environment.
3. Run the main script or one of the helper batch files.

Example:

```bash
python autonomous_driving_test.py --sync --num-vehicles 30 --num-walkers 20
```

## Local-Only Assets

If you are working from the original private workspace, these paths are expected to exist locally but are not published here:

- `carla-ue5-dev/`
- `carla_data/`
- `test_data/`
- `test_output/`
- `poster_assets/`
- `*.mp4`
- `*.pdf`
- `*.pptx`

## Notes

- This repository is a curated publishable snapshot, not a full backup of the local machine.
- The original local history is preserved on a backup branch before the public cleanup.
