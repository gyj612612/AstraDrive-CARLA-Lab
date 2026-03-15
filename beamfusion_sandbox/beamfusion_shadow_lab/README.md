# BeamFusion Shadow Lab (Fully Isolated)

Purpose:
- Integrate BeamFusion into CARLA in **shadow mode** first.
- Keep original baseline code untouched (`C:\CARLA\autonomous_driving_test.py` is not edited).
- Ensure ego can run loop while BeamFusion inference runs in parallel for logging/debug.

Pipeline:
1. Run isolated selector to find a stable loop-driving profile.
2. Launch original autonomous script with that profile.
3. In parallel, attach an extra RGB sensor to `role_name=hero` and run BeamFusion prediction.
4. Save `shadow_*.jsonl` + per-attempt logs + final report.

Safety gate:
- `--max-collisions` controls acceptance threshold for both selector and shadow attempt summaries.
- Default is `0` (strict no-crash acceptance).

## Run

```bash
python C:\CARLA\beamfusion_sandbox\beamfusion_shadow_lab\run_beamfusion_shadow_pipeline.py --auto-start-server --server-exe D:\carla\CARLA_0.9.16\CarlaUE4.exe --server-launch-mode plain --map-name Town05 --duration 120 --num-vehicles 6 --num-walkers 4 --target-speed 55 --tag shadow_town05_v1
```

Recommended no-crash run:

```bash
python C:\CARLA\beamfusion_sandbox\beamfusion_shadow_lab\run_beamfusion_shadow_pipeline.py --auto-start-server --server-exe D:\carla\CARLA_0.9.16\CarlaUE4.exe --server-launch-mode plain --map-name Town05 --duration 120 --num-vehicles 6 --num-walkers 4 --target-speed 55 --max-collisions 0 --tag shadow_town05_no_crash
```

## Output

- `C:\CARLA\beamfusion_sandbox\beamfusion_shadow_lab\outputs\<tag>\report.json`
- `C:\CARLA\beamfusion_sandbox\beamfusion_shadow_lab\outputs\<tag>\report.md`
- `C:\CARLA\beamfusion_sandbox\beamfusion_shadow_lab\outputs\<tag>\shadow_*.jsonl`
