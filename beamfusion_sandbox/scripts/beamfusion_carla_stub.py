from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="CARLA -> BeamFusion integration stub")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--beamfusion-src", default=r"E:\6G\Code\src")
    args = parser.parse_args()

    # 1) Check CARLA Python API
    import carla  # noqa: F401
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()
    print(f"[OK] CARLA connected. map={world.get_map().name}")

    # 2) Check BeamFusion import path
    beamfusion_src = Path(args.beamfusion_src)
    if not beamfusion_src.exists():
        raise FileNotFoundError(f"BeamFusion src not found: {beamfusion_src}")
    sys.path.insert(0, str(beamfusion_src))

    from beamfusion import CarlaBeamAdapter  # type: ignore

    print(f"[OK] BeamFusion import works. adapter={CarlaBeamAdapter.__name__}")

    # 3) Placeholder input shape check (replace by real sensor stream later)
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    gps = np.zeros((12,), dtype=np.float32)
    power = np.zeros((256,), dtype=np.float32)
    print(f"[OK] placeholder sample built: rgb={rgb.shape}, gps={gps.shape}, power={power.shape}")

    print("[NEXT] Load trained checkpoint and wire real sensor callbacks in this sandbox.")


if __name__ == "__main__":
    main()
