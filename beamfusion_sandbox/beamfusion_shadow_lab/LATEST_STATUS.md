# Latest Status (2026-03-02)

## Isolation Guarantee
- Original file untouched: `C:\CARLA\autonomous_driving_test.py`
- All new work is under:
  - `C:\CARLA\beamfusion_sandbox\isolated_loop\`
  - `C:\CARLA\beamfusion_sandbox\beamfusion_shadow_lab\`

## What is running now
- `run_isolated_loop.py`: auto-iterative loop-stability checker.
- `run_beamfusion_shadow_pipeline.py`: loop-driving + BeamFusion shadow inference (no control takeover).
- Both scripts now enforce collision-threshold acceptance via `--max-collisions` (default `0`).

## Verified Runs

1. Isolated loop checker (collision guard check):
- Report: `C:\CARLA\beamfusion_sandbox\isolated_loop\outputs\iso_collision_guard_check\report.json`
- Result: success (`user_baseline_adv`), `final_collision_counter=0`.

2. BeamFusion shadow pipeline (60s quick check):
- Report: `C:\CARLA\beamfusion_sandbox\beamfusion_shadow_lab\outputs\shadow_collision_guard_check\report.json`
- Result: success, `pred_count=121`, `final_collision_counter=0`, no shadow inference errors.

3. BeamFusion shadow pipeline (120s, speed 55):
- Report: `C:\CARLA\beamfusion_sandbox\beamfusion_shadow_lab\outputs\shadow_collision_guard_120_t55\report.json`
- Result: success, selected profile `user_baseline_adv`, `pred_count=240`, `final_collision_counter=0`.

## Recommended Command (current best)

```bash
python C:\CARLA\beamfusion_sandbox\beamfusion_shadow_lab\run_beamfusion_shadow_pipeline.py --auto-start-server --server-exe D:\carla\CARLA_0.9.16\CarlaUE4.exe --server-launch-mode plain --map-name Town05 --duration 120 --num-vehicles 6 --num-walkers 4 --target-speed 55 --max-collisions 0 --tag shadow_town05_no_crash
```
