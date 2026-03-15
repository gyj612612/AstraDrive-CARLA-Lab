# BeamFusion Closed-Loop Lab (Isolated)

This folder is a strict isolation area for **closed-loop** validation.

Rules:
- Do not edit original files in `C:\CARLA`.
- Closed-loop experiments run only from this folder.

## What this lab contains

- `autonomous_driving_beamfusion_closed_loop.py`
  - Copied from original baseline and modified in isolation.
  - BeamFusion output now directly affects control policy:
    - speed cap (risk-aware),
    - lane-change permission gate.
- `run_closed_loop_beamfusion.py`
  - Auto-iterative runner with profile fallback and report generation.
  - Collision-threshold acceptance via `--max-collisions`.
- `run_closed_loop_autotune.py`
  - Automatic parameter search for turn/overtake robustness.
  - Scores candidates by displacement/speed/stuck ratio/collision.

## Quick start

```bash
python C:\CARLA\beamfusion_sandbox\closed_loop_control_lab\run_closed_loop_beamfusion.py --auto-start-server --server-exe D:\carla\CARLA_0.9.16\CarlaUE4.exe --server-launch-mode plain --map-name Town05 --duration 120 --num-vehicles 6 --num-walkers 4 --target-speed 55 --max-collisions 0 --tag bf_closed_loop_town05
```

Recommended turn-robust setting (no-route junction optimization):

```bash
python C:\CARLA\beamfusion_sandbox\closed_loop_control_lab\run_closed_loop_beamfusion.py --map-name Town03 --duration 60 --num-vehicles 6 --num-walkers 4 --target-speed 50 --bf-dominant-level strong --junction-turn-deg 14 --junction-speed-cap 16 --junction-lookahead 6.5 --no-show --max-collisions 0 --tag bf_turnfix_v2_runner60
```

## Output

- `C:\CARLA\beamfusion_sandbox\closed_loop_control_lab\outputs\<tag>\report.json`
- `C:\CARLA\beamfusion_sandbox\closed_loop_control_lab\outputs\<tag>\report.md`
- New run folder in `C:\CARLA\carla_data\run_*` containing:
  - `performance.log` (with BeamFusion columns),
  - `beamfusion_closed_loop_log.jsonl` (policy inference log).

## Autotune

```bash
python C:\CARLA\beamfusion_sandbox\closed_loop_control_lab\run_closed_loop_autotune.py --auto-start-server --server-exe D:\carla\CARLA_0.9.16\CarlaUE4.exe --server-launch-mode plain --map-name Town05 --duration 60 --target-speed 50 --max-collisions 0 --tag bf_autotune_quick
```
