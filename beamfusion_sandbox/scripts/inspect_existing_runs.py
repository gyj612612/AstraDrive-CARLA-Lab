from __future__ import annotations

import csv
from pathlib import Path
import statistics

ROOT = Path(r"C:\CARLA\carla_data")


def to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def fmt(x):
    return "NA" if x is None else f"{x:.2f}"


def summarize(run_dir: Path):
    perf = run_dir / "performance.log"
    coll = run_dir / "collision.csv"

    rows = []
    if perf.exists() and perf.stat().st_size > 0:
        with perf.open("r", encoding="utf-8", errors="ignore") as f:
            rows = list(csv.DictReader(f))

    deltas = [to_float(r.get("delta_ms")) for r in rows]
    deltas = [x for x in deltas if x is not None and x > 0]
    speeds = [to_float(r.get("speed_kmh")) for r in rows]
    speeds = [x for x in speeds if x is not None]
    lane_change = [to_float(r.get("lane_changing")) for r in rows]
    lane_change = [x for x in lane_change if x is not None]
    coll_count = [to_float(r.get("collision_count")) for r in rows]
    coll_count = [x for x in coll_count if x is not None]

    avg_delta = statistics.mean(deltas) if deltas else None
    fps = (1000.0 / avg_delta) if avg_delta else None
    avg_speed = statistics.mean(speeds) if speeds else None
    lc_ratio = (sum(1 for x in lane_change if x > 0.5) / len(lane_change)) if lane_change else None
    final_coll = int(coll_count[-1]) if coll_count else 0

    coll_rows = 0
    if coll.exists():
        with coll.open("r", encoding="utf-8", errors="ignore") as f:
            coll_rows = max(0, sum(1 for _ in f) - 1)

    return {
        "run": run_dir.name,
        "rows": len(rows),
        "fps": fmt(fps),
        "avg_speed": fmt(avg_speed),
        "lane_change_ratio": fmt(lc_ratio),
        "final_collision_counter": final_coll,
        "collision_csv_rows": coll_rows,
    }


def main():
    runs = sorted([p for p in ROOT.glob("run_*") if p.is_dir()])
    print(f"found_runs={len(runs)}")
    for run in runs[-15:]:
        s = summarize(run)
        print(
            f"{s['run']}: rows={s['rows']}, fps={s['fps']}, speed={s['avg_speed']}, "
            f"lanechg={s['lane_change_ratio']}, coll_counter={s['final_collision_counter']}, "
            f"collision_rows={s['collision_csv_rows']}"
        )


if __name__ == "__main__":
    main()
