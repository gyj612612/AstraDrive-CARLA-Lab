# CARLA Existing Baseline Review (2026-02-26)

Scope:
- Reviewed root scripts under `C:\CARLA`.
- Reviewed latest run logs under `C:\CARLA\carla_data\run_*`.
- Did not modify original baseline files.

## Baseline code status

Primary script:
- `autonomous_driving_test.py` (2961 lines, 46 functions)

Earlier variants:
- `autonomous_driving_test_v4.py` (2016 lines)
- `autonomous_driving_test_v3.py` (2000 lines)

Main algorithm traits:
- Multi-sensor collection: RGB, depth, semantic, LiDAR, radar.
- Rule-based obstacle handling + lane-follow control.
- Lane-change safety gating (front/back clearance, hold, timeout, progress checks).
- Collision callback and detailed runtime logging.

Compared with v4/v3, current main adds safety and lane stability helpers (selected):
- `_select_stable_lane_wp`, `_lane_change_guidance_wp`, `_lane_center_offset_*`
- `compute_follow_speed_kmh`, `get_front_overlap_risk`, `collision_callback`

## Existing run quality (from logs)

Data source:
- `C:\CARLA\carla_data\run_*\performance.log`
- `C:\CARLA\carla_data\run_*\collision.csv`

Aggregated statistics:
- Runs with performance logs: `110`
- Zero-collision runs: `106`
- Nonzero-collision runs: `4`
- Worst run: `run_20260209_063745` with collision counter `2172`

Recent trend (latest 15 runs):
- Zero-collision runs: `13/15`
- Median approx FPS: `22.64`
- Median avg speed: `1.89 km/h`

Interpretation:
- Safety improved in most recent runs (few/no collisions).
- But recent controller behavior is conservative/sticky, with very low average speed.
- This is suitable for safety demo, but not yet strong for efficiency/progress metrics.

## Sandbox created for BeamFusion integration

New isolated folder:
- `C:\CARLA\beamfusion_sandbox`

Files:
- `README.md`
- `scripts\inspect_existing_runs.py`
- `scripts\beamfusion_carla_stub.py`

This sandbox is intended for future integration of:
- Query-conditioned beam prediction (DETR query tokens)
- Dual consistency (fusion consistency + simulator consistency)
- Missing-modality robustness

