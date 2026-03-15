# CARLA BeamFusion Sandbox

This folder is an isolated sandbox for integrating the 6G BeamFusion model into CARLA demos.

Safety rule:
- Do not modify original files under `C:\CARLA` root scripts.
- Only add/modify files inside this folder.

Current purpose:
- Inspect existing CARLA run quality.
- Provide a clean bridge script for BeamFusion inference from `E:\6G\Code`.

Key scripts:
- `scripts/inspect_existing_runs.py`: summarize latest `carla_data/run_*` logs.
- `scripts/beamfusion_carla_stub.py`: CARLA connection + model bridge stub.

