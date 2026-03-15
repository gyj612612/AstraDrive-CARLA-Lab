#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrapper to run autonomous_driving_test.py with full stdout/stderr logging.
Usage examples:
  python run_with_logging.py --duration 60 --num-vehicles 20 --num-walkers 15 --sync
  python run_with_logging.py --npc-autopilot --walker-autopilot
Logs are written under: C:\\CARLA\\logs\\run_YYYYmmdd_HHMMSS.log
"""

import argparse
import datetime
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run autonomous_driving_test.py with logging")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="2000")
    parser.add_argument("--num-vehicles", default="20")
    parser.add_argument("--num-walkers", default="15")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--autopilot", action="store_true")
    parser.add_argument("--npc-autopilot", action="store_true")
    parser.add_argument("--walker-autopilot", action="store_true")
    parser.add_argument("--map", default=None)
    parser.add_argument("--duration", default="60")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Any extra args passed through")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.abspath(os.path.join("logs", f"run_{stamp}.log"))

    cmd = [
        sys.executable,
        "autonomous_driving_test.py",
        "--host", args.host,
        "--port", args.port,
        "--num-vehicles", args.num_vehicles,
        "--num-walkers", args.num_walkers,
        "--duration", args.duration,
    ]
    if args.sync:
        cmd.append("--sync")
    if args.autopilot:
        cmd.append("--autopilot")
    if args.npc_autopilot:
        cmd.append("--npc-autopilot")
    if args.walker_autopilot:
        cmd.append("--walker-autopilot")
    if args.map:
        cmd.extend(["--map", args.map])
    if args.extra:
        cmd.extend(args.extra)

    print(f"[LOGGER] Writing log to: {log_path}")
    print(f"[LOGGER] Command: {' '.join(cmd)}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        process.wait()
        if process.returncode != 0:
            print(f"[LOGGER] Process exited with code {process.returncode}")

    print(f"[LOGGER] Done. Log saved at {log_path}")


if __name__ == "__main__":
    main()
