# Isolated Loop Runner (No Original Code Changes)

This folder is a strict isolation layer for CARLA loop testing before any BeamFusion-into-driving integration.

Rules:
- Do not edit `C:\CARLA\autonomous_driving_test.py`.
- Do not edit original baseline driving scripts in `C:\CARLA`.
- Only run/iterate inside this folder.

## What this runner does
- Calls original `autonomous_driving_test.py` as a subprocess.
- Tries multiple controller profiles automatically:
  - `user_baseline_adv`
  - `adv_no_global_route`
  - `adv_global_route`
  - `simple_loop_fallback`
- Auto-parses latest `C:\CARLA\carla_data\run_*` logs.
- Marks success when run has enough frames and average speed indicates the ego is moving on-loop.
- Adds collision guard via `--max-collisions` (default `0`).

## Usage

1) If CARLA server is already running:

```bash
python C:\CARLA\beamfusion_sandbox\isolated_loop\run_isolated_loop.py --map-name Town05 --duration 180 --tag town05_iso_v1
```

2) Auto-start CARLA server from executable:

```bash
python C:\CARLA\beamfusion_sandbox\isolated_loop\run_isolated_loop.py --auto-start-server --server-exe D:\carla\CARLA_0.9.16\CarlaUE4.exe --server-launch-mode plain --map-name Town05 --duration 180 --tag town05_iso_v1
```

3) Reproduce your known baseline command inside isolation runner:

```bash
python C:\CARLA\beamfusion_sandbox\isolated_loop\run_isolated_loop.py --auto-start-server --server-launch-mode plain --map-name Town05 --duration 180 --num-vehicles 6 --num-walkers 4 --target-speed 55 --tag user_baseline_iso
```

4) Strict no-crash mode:

```bash
python C:\CARLA\beamfusion_sandbox\isolated_loop\run_isolated_loop.py --auto-start-server --server-launch-mode plain --map-name Town05 --duration 120 --target-speed 55 --max-collisions 0 --tag town05_no_crash_check
```

## Output
- JSON report:
  - `C:\CARLA\beamfusion_sandbox\isolated_loop\outputs\<tag>\report.json`
- Markdown summary:
  - `C:\CARLA\beamfusion_sandbox\isolated_loop\outputs\<tag>\report.md`
- Per-attempt raw logs:
  - `C:\CARLA\beamfusion_sandbox\isolated_loop\outputs\<tag>\*.log`
