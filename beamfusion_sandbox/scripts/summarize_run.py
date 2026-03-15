from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def fnum(v):
    try:
        return float(v)
    except Exception:
        return None


def summarize(run_dir: Path):
    perf = run_dir / 'performance.log'
    rows = list(csv.DictReader(perf.open('r', encoding='utf-8', errors='ignore')))

    deltas = [fnum(r.get('delta_ms')) for r in rows]
    deltas = [x for x in deltas if x is not None and x > 0]
    speeds = [fnum(r.get('speed_kmh')) for r in rows]
    speeds = [x for x in speeds if x is not None]
    collisions = [fnum(r.get('collision_count')) for r in rows]
    collisions = [x for x in collisions if x is not None]
    lane_changing = [fnum(r.get('lane_changing')) for r in rows]
    lane_changing = [x for x in lane_changing if x is not None]

    avg_delta = statistics.mean(deltas) if deltas else None
    fps = 1000.0 / avg_delta if avg_delta else None

    return {
        'run': run_dir.name,
        'rows': len(rows),
        'avg_delta_ms': avg_delta,
        'approx_fps': fps,
        'avg_speed_kmh': statistics.mean(speeds) if speeds else None,
        'max_speed_kmh': max(speeds) if speeds else None,
        'final_collision_counter': int(collisions[-1]) if collisions else 0,
        'lane_change_ratio': (sum(1 for x in lane_changing if x > 0.5) / len(lane_changing)) if lane_changing else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--out-json', default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    s = summarize(run_dir)
    print(json.dumps(s, indent=2, ensure_ascii=False))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
