# AstraDrive CARLA Lab Structure

## Included in Public Repo

### Root scripts

- `autonomous_driving_test.py`
- `control_ego_vehicle.py`
- `spawn_npc_traffic.py`
- `sensor_io_handler.py`
- `run_with_logging.py`
- `环境验证脚本.py`
- `quick_test.py`
- `test_carla_connection.py`
- `test_connection.py`
- `test_data_collection.py`
- `test_fixed_features.py`
- `test_fixed_script.py`
- `test_npc_movement.py`
- `test_npc_traffic.py`
- `test_quick.py`

### Launch helpers

- `启动CARLA服务器.bat`
- `启动CARLA服务器-高性能.bat`
- `运行自动驾驶测试.bat`
- `运行NPC交通生成.bat`
- `运行Ego Vehicle控制.bat`
- `运行环境验证.bat`
- `运行交通生成示例.bat`
- `运行手动控制示例.bat`

### Documentation

- `README.md`
- `PROJECT_STRUCTURE.md`
- `CARLA安装说明.md`
- `CARLA完整使用指南.md`
- `修复说明.md`
- `测试报告.md`
- `运行说明_性能_算法.md`

### BeamFusion code

- `beamfusion_sandbox/README.md`
- `beamfusion_sandbox/CARLA_baseline_review_2026-02-26.md`
- `beamfusion_sandbox/scripts/*.py`
- `beamfusion_sandbox/closed_loop_control_lab/*.py`
- `beamfusion_sandbox/closed_loop_control_lab/*.md`
- `beamfusion_sandbox/beamfusion_shadow_lab/*.py`
- `beamfusion_sandbox/beamfusion_shadow_lab/*.md`
- `beamfusion_sandbox/isolated_loop/*.py`
- `beamfusion_sandbox/isolated_loop/*.md`

## Excluded from Public Repo

- `carla-ue5-dev/`
- `carla_data/`
- `test_data/`
- `test_output/`
- `poster_assets/`
- `__pycache__/`
- `.cursor/`
- all generated logs and output folders
- videos, PDFs, and presentation files
- CE301 grading/poster/abstract working files

## Publishing Principle

The public repository is intentionally limited to:

- runnable core code
- concise setup and structure docs
- small helper scripts
- source code for isolated algorithm experiments

This keeps the public snapshot understandable and reasonably small while preserving the full local workspace privately.
