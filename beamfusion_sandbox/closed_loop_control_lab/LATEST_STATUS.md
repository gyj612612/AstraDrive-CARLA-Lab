# Latest Status (2026-03-02)

## Isolation
- Original baseline untouched: `C:\CARLA\autonomous_driving_test.py`
- Closed-loop integration only in:
  - `C:\CARLA\beamfusion_sandbox\closed_loop_control_lab\autonomous_driving_beamfusion_closed_loop.py`
  - `C:\CARLA\beamfusion_sandbox\closed_loop_control_lab\run_closed_loop_beamfusion.py`

## Closed-Loop Definition (Current)
- Not shadow-only anymore.
- BeamFusion inference now feeds control policy every tick window:
  - speed cap gate,
  - lane-change permission gate.
- Steering and low-level path tracking remain on stable baseline controller.

## Latest Verification
1. Run tag: `bf_closed_loop_curbguard_120s`
- Report: `C:\CARLA\beamfusion_sandbox\closed_loop_control_lab\outputs\bf_closed_loop_curbguard_120s\report.json`
- Result: success, profile `bf_adv_no_lc`
- Key metrics:
  - `final_collision_counter = 0`
  - `avg_speed_kmh = 14.10`
  - `displacement_m = 167.17`

2. Run tag: `bf_closed_loop_junctionfix_120s`
- Report: `C:\CARLA\beamfusion_sandbox\closed_loop_control_lab\outputs\bf_closed_loop_junctionfix_120s\report.json`
- Result: success, profile `bf_adv_lc`
- Key metrics:
  - `final_collision_counter = 0`
  - `avg_speed_kmh = 10.48`
  - `displacement_m = 144.66`

3. Data logs exist in run dirs:
- `C:\CARLA\carla_data\run_20260302_050141\performance.log`
- `C:\CARLA\carla_data\run_20260302_050141\beamfusion_closed_loop_log.jsonl`

4. Run tag: `bf_turnfix_v2_runner60` (Town03, right-turn stress)
- Report: `C:\CARLA\beamfusion_sandbox\closed_loop_control_lab\outputs\bf_turnfix_v2_runner60\report.json`
- Result: success, profile `bf_adv_lc` (no fallback)
- Key metrics:
  - `final_collision_counter = 0`
  - `avg_speed_kmh = 17.91`
  - `displacement_m = 77.67`
  - `rows = 16` (passes runner acceptance)

5. Direct turn scenario regression:
- Run dir: `E:\6G\carla_data\run_20260302_054328`
- Key metrics from `performance.log` + `route.csv`:
  - `rows = 15`
  - `avg_speed_kmh = 14.97`
  - `displacement_m = 48.58`
  - `collision_count = 0`

## Notes
- NaN handling fixed at source:
  - sanitize `gps_mean/std` and `power_mean/std`,
  - safe normalization in online feature builder.
- Added concentrated lane-keeping safety:
  - lane-edge risk counter,
  - hard curb guard (forbid steering toward curb when edge risk high),
  - intersection deadlock release,
  - stronger low-speed unstick recovery.
- Default `bf_lane_change_risk_th` was raised to `0.82` to avoid over-restricting lane-change under valid beam confidence.
- 2026-03-02 turn/stall fixes added:
  - fixed `ego_wp_now` use-before-assign in junction mode trigger,
  - propagated dynamic `track_lookahead` into all local lane-follow calls,
  - added signed-turn preference in no-route junctions,
  - strengthened no-route turn steering authority (bounded),
  - enabled low-speed throttle floor under curb-guard/recenter conditions to avoid deadlock at right turns,
  - runner now passes junction args and enforces `--no-view-windows` for better test FPS stability.
