from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_closed_loop_beamfusion.py"
OUT_ROOT = ROOT / "outputs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class EvalMetrics:
    success: bool
    collisions: int
    displacement_m: float
    avg_speed_kmh: float
    stuck_ratio: float
    lane_change_ratio: float
    sharp_turn_ratio: float
    sharp_turn_stuck_ratio: float
    score: float


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def compute_metrics(report: Dict[str, Any], turn_threshold: float) -> EvalMetrics:
    if not report.get("success"):
        return EvalMetrics(
            success=False,
            collisions=999,
            displacement_m=0.0,
            avg_speed_kmh=0.0,
            stuck_ratio=1.0,
            lane_change_ratio=0.0,
            sharp_turn_ratio=0.0,
            sharp_turn_stuck_ratio=1.0,
            score=-1e9,
        )

    run_dir = Path(str(report.get("success_run_dir", "")))
    rs = None
    for a in report.get("attempts", []):
        if str(a.get("run_detected")) == str(run_dir):
            rs = a.get("run_summary") or {}
            break
    if rs is None:
        rs = (report.get("attempts", [{}])[-1].get("run_summary") if report.get("attempts") else {}) or {}

    displacement = _safe_float(rs.get("displacement_m"), 0.0)
    avg_speed = _safe_float(rs.get("avg_speed_kmh"), 0.0)
    collisions = int(_safe_float(rs.get("final_collision_counter"), 0.0))

    perf = run_dir / "performance.log"
    rows = []
    if perf.exists():
        rows = list(csv.DictReader(perf.open("r", encoding="utf-8", errors="ignore")))

    if not rows:
        return EvalMetrics(
            success=True,
            collisions=collisions,
            displacement_m=displacement,
            avg_speed_kmh=avg_speed,
            stuck_ratio=1.0,
            lane_change_ratio=0.0,
            sharp_turn_ratio=0.0,
            sharp_turn_stuck_ratio=1.0,
            score=-5e6,
        )

    speeds = [_safe_float(r.get("speed_kmh"), 0.0) for r in rows]
    lane_changing = [_safe_float(r.get("lane_changing"), 0.0) for r in rows]
    turn_deg = [_safe_float(r.get("bf_turn_deg"), 0.0) for r in rows]

    total = max(1, len(rows))
    stuck_ratio = sum(1 for s in speeds if s < 2.0) / total
    lane_change_ratio = sum(1 for x in lane_changing if x > 0.5) / total
    sharp_idx = [i for i, d in enumerate(turn_deg) if d >= float(turn_threshold)]
    sharp_turn_ratio = len(sharp_idx) / total
    if sharp_idx:
        sharp_turn_stuck_ratio = sum(1 for i in sharp_idx if speeds[i] < 2.0) / len(sharp_idx)
    else:
        sharp_turn_stuck_ratio = 0.0

    score = (
        1.0 * displacement
        + 2.2 * avg_speed
        + 30.0 * lane_change_ratio
        - 130.0 * stuck_ratio
        - 110.0 * sharp_turn_stuck_ratio
        - 150.0 * collisions
    )
    if collisions > 0:
        score -= 5000.0

    return EvalMetrics(
        success=True,
        collisions=collisions,
        displacement_m=displacement,
        avg_speed_kmh=avg_speed,
        stuck_ratio=stuck_ratio,
        lane_change_ratio=lane_change_ratio,
        sharp_turn_ratio=sharp_turn_ratio,
        sharp_turn_stuck_ratio=sharp_turn_stuck_ratio,
        score=score,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-tune closed-loop BeamFusion control params.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map-name", default="Town05")
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--delta-time", type=float, default=0.02)
    parser.add_argument("--num-vehicles", type=int, default=6)
    parser.add_argument("--num-walkers", type=int, default=4)
    parser.add_argument("--target-speed", type=float, default=50.0)
    parser.add_argument("--max-collisions", type=int, default=0)
    parser.add_argument("--auto-start-server", action="store_true")
    parser.add_argument("--server-exe", default=r"D:\carla\CARLA_0.9.16\CarlaUE4.exe")
    parser.add_argument("--server-launch-mode", choices=["plain", "custom"], default="plain")
    parser.add_argument("--server-wait-sec", type=float, default=40.0)
    parser.add_argument("--post-start-wait-sec", type=float, default=25.0)
    parser.add_argument("--tag", default="bf_autotune")
    args = parser.parse_args()

    if not RUNNER.exists():
        raise FileNotFoundError(f"Runner not found: {RUNNER}")

    tag_base = f"{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = OUT_ROOT / tag_base
    ensure_dir(out_dir)

    candidates: List[Dict[str, Any]] = [
        {
            "name": "assist_balanced",
            "overrides": {
                "--bf-dominant-level": "assist",
                "--max-steer": 0.48,
                "--steer-rate": 0.10,
                "--lane-change-front-clear": 14.0,
                "--lane-change-back-clear": 7.5,
                "--bf-force-overtake-wait": 2.3,
            },
        },
        {
            "name": "turn_priority",
            "overrides": {
                "--bf-dominant-level": "strong",
                "--max-steer": 0.56,
                "--steer-rate": 0.13,
                "--bf-turn-sharp-deg": 8.5,
                "--bf-turn-speed-cap": 20.0,
                "--bf-turn-steer-boost": 0.12,
                "--bf-turn-rate-boost": 0.06,
            },
        },
        {
            "name": "overtake_priority",
            "overrides": {
                "--bf-dominant-level": "strong",
                "--max-steer": 0.52,
                "--steer-rate": 0.12,
                "--lane-change-front-clear": 12.0,
                "--lane-change-back-clear": 6.5,
                "--lane-change-speed": 22.0,
                "--bf-force-overtake-wait": 1.8,
                "--bf-overtake-front-relax": 0.62,
                "--bf-overtake-back-relax": 0.60,
            },
        },
        {
            "name": "aggressive_mix",
            "overrides": {
                "--bf-dominant-level": "strong",
                "--max-steer": 0.58,
                "--steer-rate": 0.14,
                "--lane-change-front-clear": 11.5,
                "--lane-change-back-clear": 6.0,
                "--lane-change-speed": 24.0,
                "--bf-turn-sharp-deg": 8.0,
                "--bf-turn-speed-cap": 19.0,
                "--bf-force-overtake-wait": 1.6,
            },
        },
        {
            "name": "safe_conservative",
            "overrides": {
                "--bf-dominant-level": "assist",
                "--max-steer": 0.46,
                "--steer-rate": 0.09,
                "--lane-change-front-clear": 15.0,
                "--lane-change-back-clear": 8.0,
                "--bf-turn-sharp-deg": 10.5,
                "--bf-turn-speed-cap": 21.0,
                "--bf-force-overtake-wait": 2.8,
            },
        },
    ]

    records: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for idx, cand in enumerate(candidates, start=1):
        run_tag = f"{tag_base}_{idx:02d}_{cand['name']}"
        cmd = [
            sys.executable,
            str(RUNNER),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--map-name",
            args.map_name,
            "--duration",
            str(args.duration),
            "--delta-time",
            str(args.delta_time),
            "--num-vehicles",
            str(args.num_vehicles),
            "--num-walkers",
            str(args.num_walkers),
            "--target-speed",
            str(args.target_speed),
            "--max-collisions",
            str(args.max_collisions),
            "--server-exe",
            str(args.server_exe),
            "--server-launch-mode",
            str(args.server_launch_mode),
            "--server-wait-sec",
            str(args.server_wait_sec),
            "--post-start-wait-sec",
            str(args.post_start_wait_sec),
            "--tag",
            run_tag,
        ]
        if args.auto_start_server:
            cmd.append("--auto-start-server")
        for k, v in cand["overrides"].items():
            cmd.extend([str(k), str(v)])

        cp = subprocess.run(cmd, capture_output=True, text=True)
        run_report_path = OUT_ROOT / run_tag / "report.json"
        if run_report_path.exists():
            run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
        else:
            run_report = {"success": False, "error": "report_missing"}
        metrics = compute_metrics(
            run_report,
            turn_threshold=float(cand["overrides"].get("--bf-turn-sharp-deg", 10.0)),
        )

        rec = {
            "candidate": cand["name"],
            "tag": run_tag,
            "command": cmd,
            "return_code": cp.returncode,
            "metrics": metrics.__dict__,
            "overrides": cand["overrides"],
            "report_path": str(run_report_path),
        }
        records.append(rec)

        if (best is None) or (metrics.score > best["metrics"]["score"]):
            best = rec

    summary = {
        "time_utc": utc_now(),
        "tag_base": tag_base,
        "best": best,
        "candidates": records,
    }
    write_json(out_dir / "autotune_report.json", summary)

    md = [
        f"# Closed-Loop Autotune Report ({tag_base})",
        "",
        f"- best: `{best['candidate'] if best else None}`",
        f"- best_score: `{best['metrics']['score'] if best else None}`",
        "",
        "| candidate | success | collisions | avg_speed | displacement | stuck_ratio | lane_change_ratio | sharp_turn_stuck | score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        m = r["metrics"]
        md.append(
            f"| {r['candidate']} | {m['success']} | {m['collisions']} | {m['avg_speed_kmh']:.2f} | {m['displacement_m']:.2f} | "
            f"{m['stuck_ratio']:.3f} | {m['lane_change_ratio']:.3f} | {m['sharp_turn_stuck_ratio']:.3f} | {m['score']:.2f} |"
        )
    (out_dir / "autotune_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[Done] {out_dir / 'autotune_report.json'}")


if __name__ == "__main__":
    main()

