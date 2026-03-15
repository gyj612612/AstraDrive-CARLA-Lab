from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


CARLA_ROOT = Path(r"C:\CARLA")
AUTON_SCRIPT = CARLA_ROOT / "autonomous_driving_test.py"
DATA_ROOT = CARLA_ROOT / "carla_data"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEFAULT_EXE = Path(r"D:\carla\CARLA_0.9.16\CarlaUE4.exe")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_run_dirs() -> List[Path]:
    if not DATA_ROOT.exists():
        return []
    runs = [p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.stat().st_mtime)
    return runs


def latest_new_run(before: List[Path]) -> Optional[Path]:
    before_set = {p.resolve() for p in before}
    after = list_run_dirs()
    new_runs = [p for p in after if p.resolve() not in before_set]
    if not new_runs:
        return None
    new_runs.sort(key=lambda p: p.stat().st_mtime)
    return new_runs[-1]


def check_server_ready(host: str, port: int, timeout_sec: float = 4.0) -> bool:
    try:
        import carla  # type: ignore

        client = carla.Client(host, port)
        client.set_timeout(timeout_sec)
        _ = client.get_world()
        return True
    except Exception:
        return False


def wait_server(host: str, port: int, wait_sec: float) -> bool:
    t0 = time.time()
    while time.time() - t0 <= wait_sec:
        if check_server_ready(host, port, timeout_sec=2.0):
            return True
        time.sleep(2.0)
    return False


def start_server_process(
    exe_path: Path,
    port: int,
    launch_mode: str,
    quality: str,
    res_x: int,
    res_y: int,
    use_vulkan: bool,
    no_vsync: bool,
) -> subprocess.Popen[str]:
    if not exe_path.exists():
        raise FileNotFoundError(f"CARLA executable not found: {exe_path}")

    cmd = [str(exe_path), f"-carla-rpc-port={port}"]
    if launch_mode == "custom":
        cmd.extend(
            [
                f"-quality-level={quality}",
                "-Windowed",
                f"-ResX={res_x}",
                f"-ResY={res_y}",
            ]
        )
        if use_vulkan:
            cmd.append("-vulkan")
        if no_vsync:
            cmd.append("-NoVSync")

    return subprocess.Popen(
        cmd,
        cwd=str(exe_path.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def stop_server_process(proc: Optional[subprocess.Popen[str]]) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=8.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def ensure_server(
    host: str,
    port: int,
    auto_start: bool,
    exe_path: Path,
    launch_mode: str,
    quality: str,
    res_x: int,
    res_y: int,
    use_vulkan: bool,
    no_vsync: bool,
    wait_sec: float,
    post_start_wait_sec: float,
    current_proc: Optional[subprocess.Popen[str]],
) -> tuple[bool, Optional[subprocess.Popen[str]]]:
    if check_server_ready(host, port, timeout_sec=2.0):
        return True, current_proc

    if not auto_start:
        return False, current_proc

    # Restart managed server process when simulator is not reachable.
    stop_server_process(current_proc)
    proc = start_server_process(
        exe_path=exe_path,
        port=port,
        launch_mode=launch_mode,
        quality=quality,
        res_x=res_x,
        res_y=res_y,
        use_vulkan=use_vulkan,
        no_vsync=no_vsync,
    )
    ok = wait_server(host, port, wait_sec=wait_sec)
    if ok and post_start_wait_sec > 0:
        time.sleep(float(post_start_wait_sec))
    return ok, proc


def summarize_run(run_dir: Path, max_collisions: int = 0) -> Dict[str, Any]:
    perf = run_dir / "performance.log"
    route = run_dir / "route.csv"
    if not perf.exists():
        return {
            "run_dir": str(run_dir),
            "ok": False,
            "reason": "performance.log missing",
        }

    rows = list(csv.DictReader(perf.open("r", encoding="utf-8", errors="ignore")))
    if not rows:
        return {
            "run_dir": str(run_dir),
            "ok": False,
            "reason": "performance.log has no rows",
        }

    def fnum(v: Any) -> Optional[float]:
        try:
            return float(v)
        except Exception:
            return None

    deltas = [fnum(r.get("delta_ms")) for r in rows]
    deltas = [x for x in deltas if x is not None and x > 0]
    speeds = [fnum(r.get("speed_kmh")) for r in rows]
    speeds = [x for x in speeds if x is not None]
    collisions = [fnum(r.get("collision_count")) for r in rows]
    collisions = [x for x in collisions if x is not None]

    avg_delta = sum(deltas) / max(1, len(deltas))
    avg_speed = sum(speeds) / max(1, len(speeds))
    final_collision = int(collisions[-1]) if collisions else 0

    route_rows = []
    displacement_m = 0.0
    if route.exists():
        route_rows = list(csv.DictReader(route.open("r", encoding="utf-8", errors="ignore")))
        if route_rows:
            try:
                x0 = float(route_rows[0]["x"])
                y0 = float(route_rows[0]["y"])
                x1 = float(route_rows[-1]["x"])
                y1 = float(route_rows[-1]["y"])
                displacement_m = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            except Exception:
                displacement_m = 0.0

    # 环线可开动判定：采样记录足够 + 平均速度可用 + 轨迹位移明显
    ok = (
        len(rows) >= 15
        and avg_speed >= 5.0
        and displacement_m >= 40.0
        and final_collision <= int(max_collisions)
    )
    return {
        "run_dir": str(run_dir),
        "ok": bool(ok),
        "rows": len(rows),
        "avg_delta_ms": avg_delta,
        "approx_fps": (1000.0 / avg_delta) if avg_delta > 0 else None,
        "avg_speed_kmh": avg_speed,
        "route_rows": len(route_rows),
        "displacement_m": displacement_m,
        "final_collision_counter": final_collision,
        "max_collisions_allowed": int(max_collisions),
    }


def build_profiles(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return [
        {
            # User-provided known baseline command (plus isolated wrappers).
            "name": "user_baseline_adv",
            "extra": [
                "--ego-adv",
                "--target-speed",
                str(args.target_speed),
                "--npc-lane-change",
                "--lane-change-prefer",
                args.lane_change_prefer,
            ],
        },
        {
            "name": "adv_no_global_route",
            "extra": [
                "--ego-adv",
                "--target-speed",
                str(max(35.0, args.target_speed - 8.0)),
                "--npc-lane-change",
                "--npc-off-route",
                "--lane-change-prefer",
                args.lane_change_prefer,
            ],
        },
        {
            "name": "adv_global_route",
            "extra": [
                "--ego-adv",
                "--global-route",
                "--target-speed",
                str(args.target_speed),
                "--npc-lane-change",
                "--npc-off-route",
                "--lane-change-prefer",
                args.lane_change_prefer,
            ],
        },
        {
            "name": "simple_loop_fallback",
            "extra": [
                "--ego-loop",
                "--target-speed",
                str(max(25.0, args.target_speed - 15.0)),
            ],
        },
    ]


def run_one_attempt(
    profile: Dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
) -> Dict[str, Any]:
    before_runs = list_run_dirs()
    cmd = [sys.executable, str(AUTON_SCRIPT)] + profile["common"] + profile["extra"]
    t0 = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(CARLA_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    stdout_lower = (completed.stdout or "").lower()
    conn_failed = ("connection failed" in stdout_lower) or ("time-out of 10000ms" in stdout_lower)

    attempt_log = run_dir / f"{profile['name']}.log"
    attempt_log.write_text(
        f"[time_utc] {utc_now()}\n"
        f"[cmd] {' '.join(cmd)}\n"
        f"[return_code] {completed.returncode}\n\n"
        f"[stdout]\n{completed.stdout}\n\n"
        f"[stderr]\n{completed.stderr}\n",
        encoding="utf-8",
        errors="ignore",
    )

    new_run = latest_new_run(before_runs)
    summary = None
    if new_run is not None:
        summary = summarize_run(new_run, max_collisions=args.max_collisions)

    return {
        "profile": profile["name"],
        "cmd": cmd,
        "return_code": completed.returncode,
        "connection_failed": conn_failed,
        "elapsed_sec": elapsed,
        "run_detected": str(new_run) if new_run else None,
        "run_summary": summary,
        "attempt_log": str(attempt_log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated CARLA loop runner (no modification to original scripts).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map-name", default="Town05")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--delta-time", type=float, default=0.02)
    parser.add_argument("--num-vehicles", type=int, default=6)
    parser.add_argument("--num-walkers", type=int, default=4)
    parser.add_argument("--target-speed", type=float, default=50.0)
    parser.add_argument("--vis-stride", type=int, default=6)
    parser.add_argument("--lane-change-prefer", choices=["right", "left"], default="right")
    parser.add_argument("--max-collisions", type=int, default=0)

    parser.add_argument("--no-save", action="store_true", default=True)
    parser.add_argument("--no-show", action="store_true", default=False)
    parser.add_argument("--hud", action="store_true", default=True)

    parser.add_argument("--auto-start-server", action="store_true")
    parser.add_argument("--server-exe", default=str(DEFAULT_EXE))
    parser.add_argument("--server-wait-sec", type=float, default=35.0)
    parser.add_argument("--post-start-wait-sec", type=float, default=30.0)
    parser.add_argument("--server-launch-mode", choices=["plain", "custom"], default="plain")
    parser.add_argument("--server-quality", default="Low")
    parser.add_argument("--server-res-x", type=int, default=1280)
    parser.add_argument("--server-res-y", type=int, default=720)
    parser.add_argument("--server-vulkan", action="store_true")
    parser.add_argument("--server-no-vsync", action="store_true")

    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if not AUTON_SCRIPT.exists():
        raise FileNotFoundError(f"Original script missing: {AUTON_SCRIPT}")

    tag = args.tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / tag
    ensure_dir(run_dir)

    server_proc: Optional[subprocess.Popen[str]] = None
    server_was_running = check_server_ready(args.host, args.port, timeout_sec=2.0)
    ok_server, server_proc = ensure_server(
        host=args.host,
        port=args.port,
        auto_start=bool(args.auto_start_server),
        exe_path=Path(args.server_exe),
        launch_mode=args.server_launch_mode,
        quality=args.server_quality,
        res_x=args.server_res_x,
        res_y=args.server_res_y,
        use_vulkan=bool(args.server_vulkan),
        no_vsync=bool(args.server_no_vsync),
        wait_sec=float(args.server_wait_sec),
        post_start_wait_sec=float(args.post_start_wait_sec),
        current_proc=server_proc,
    )
    if not ok_server:
        raise RuntimeError(
            "CARLA server not reachable. Start server first or use --auto-start-server."
        )

    profiles = build_profiles(args)
    for p in profiles:
        common_args = [
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--map",
            args.map_name,
            "--sync",
            "--duration",
            str(args.duration),
            "--delta-time",
            str(args.delta_time),
            "--vis-stride",
            str(args.vis_stride),
            "--num-vehicles",
            str(args.num_vehicles),
            "--num-walkers",
            str(args.num_walkers),
        ]
        if args.no_save:
            common_args.append("--no-save")
        if args.no_show:
            common_args.append("--no-show")
        if args.hud:
            common_args.append("--hud")
        p["common"] = common_args

    attempts: List[Dict[str, Any]] = []
    success_attempt: Optional[Dict[str, Any]] = None
    for profile in profiles:
        ok_server, server_proc = ensure_server(
            host=args.host,
            port=args.port,
            auto_start=bool(args.auto_start_server),
            exe_path=Path(args.server_exe),
            launch_mode=args.server_launch_mode,
            quality=args.server_quality,
            res_x=args.server_res_x,
            res_y=args.server_res_y,
            use_vulkan=bool(args.server_vulkan),
            no_vsync=bool(args.server_no_vsync),
            wait_sec=float(args.server_wait_sec),
            post_start_wait_sec=float(args.post_start_wait_sec),
            current_proc=server_proc,
        )
        if not ok_server:
            attempts.append(
                {
                    "profile": profile["name"],
                    "cmd": None,
                    "return_code": -999,
                    "connection_failed": True,
                    "elapsed_sec": 0.0,
                    "run_detected": None,
                    "run_summary": None,
                    "attempt_log": None,
                    "error": "server unavailable before attempt",
                }
            )
            continue

        result = run_one_attempt(profile, args=args, run_dir=run_dir)
        attempts.append(result)

        rs = result.get("run_summary") or {}
        if result["return_code"] == 0 and (not result.get("connection_failed", False)) and rs.get("ok", False):
            success_attempt = result
            break

    report = {
        "time_utc": utc_now(),
        "carla_root": str(CARLA_ROOT),
        "original_script": str(AUTON_SCRIPT),
        "isolation_output_dir": str(run_dir),
        "server_was_running": bool(server_was_running),
        "auto_started_server": bool(server_proc is not None),
        "attempt_count": len(attempts),
        "success": success_attempt is not None,
        "success_profile": success_attempt["profile"] if success_attempt else None,
        "success_run_dir": (success_attempt.get("run_detected") if success_attempt else None),
        "attempts": attempts,
    }
    write_json(run_dir / "report.json", report)

    md_lines = [
        f"# Isolated Loop Report ({tag})",
        "",
        f"- success: `{report['success']}`",
        f"- success_profile: `{report['success_profile']}`",
        f"- success_run_dir: `{report['success_run_dir']}`",
        f"- attempt_count: `{report['attempt_count']}`",
        "",
        "| profile | return_code | run_detected | ok | rows | avg_speed_kmh | collisions |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for a in attempts:
        s = a.get("run_summary") or {}
        md_lines.append(
            f"| {a['profile']} | {a['return_code']} | `{a.get('run_detected')}` | "
            f"{s.get('ok')} | {s.get('rows')} | {s.get('avg_speed_kmh')} | {s.get('final_collision_counter')} |"
        )
    (run_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[Done] report: {run_dir / 'report.json'}")

    # Only terminate server if this script started it.
    stop_server_process(server_proc)


if __name__ == "__main__":
    main()
