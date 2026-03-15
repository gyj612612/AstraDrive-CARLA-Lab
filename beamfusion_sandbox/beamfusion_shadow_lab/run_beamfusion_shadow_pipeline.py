from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


CARLA_ROOT = Path(r"C:\CARLA")
AUTON_SCRIPT = CARLA_ROOT / "autonomous_driving_test.py"
DATA_ROOT = CARLA_ROOT / "carla_data"
ISOLATED_LOOP_SCRIPT = Path(r"C:\CARLA\beamfusion_sandbox\isolated_loop\run_isolated_loop.py")
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEFAULT_CARLA_EXE = Path(r"D:\carla\CARLA_0.9.16\CarlaUE4.exe")
DEFAULT_BEAM_CONFIG = Path(
    r"E:\6G\Code\outputs\ablation_a6_a7\ccfa_auto_v1_a6\03_a6_score_weighted_mean_s2028\config.json"
)
DEFAULT_BEAM_CKPT = Path(
    r"E:\6G\Code\outputs\ablation_a6_a7\ccfa_auto_v1_a6\03_a6_score_weighted_mean_s2028\checkpoints\best.pt"
)
DEFAULT_BEAM_SRC = Path(r"E:\6G\Code\src")


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


def summarize_run(run_dir: Path, max_collisions: int = 0) -> Dict[str, Any]:
    perf = run_dir / "performance.log"
    route = run_dir / "route.csv"
    if not perf.exists():
        return {"run_dir": str(run_dir), "ok": False, "reason": "performance.log missing"}

    rows = list(csv.DictReader(perf.open("r", encoding="utf-8", errors="ignore")))
    if not rows:
        return {"run_dir": str(run_dir), "ok": False, "reason": "performance.log has no rows"}

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


def check_server_ready(host: str, port: int, timeout_sec: float = 3.0) -> bool:
    try:
        import carla  # type: ignore

        client = carla.Client(host, port)
        client.set_timeout(timeout_sec)
        _ = client.get_world()
        return True
    except Exception:
        return False


def build_profile_extras(profile: str, target_speed: float, lane_change_prefer: str) -> List[str]:
    if profile == "user_baseline_adv":
        return [
            "--ego-adv",
            "--target-speed",
            str(target_speed),
            "--npc-lane-change",
            "--lane-change-prefer",
            lane_change_prefer,
        ]
    if profile == "adv_no_global_route":
        return [
            "--ego-adv",
            "--target-speed",
            str(max(35.0, target_speed - 8.0)),
            "--npc-lane-change",
            "--npc-off-route",
            "--lane-change-prefer",
            lane_change_prefer,
        ]
    if profile == "adv_global_route":
        return [
            "--ego-adv",
            "--global-route",
            "--target-speed",
            str(target_speed),
            "--npc-lane-change",
            "--npc-off-route",
            "--lane-change-prefer",
            lane_change_prefer,
        ]
    # simple_loop_fallback
    return ["--ego-loop", "--target-speed", str(max(25.0, target_speed - 15.0))]


def select_stable_profile(args: argparse.Namespace, out_dir: Path) -> str:
    tag = f"{args.tag}_selector"
    cmd = [
        sys.executable,
        str(ISOLATED_LOOP_SCRIPT),
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
        "--lane-change-prefer",
        args.lane_change_prefer,
        "--server-launch-mode",
        args.server_launch_mode,
        "--server-exe",
        str(args.server_exe),
        "--server-wait-sec",
        str(args.server_wait_sec),
        "--post-start-wait-sec",
        str(args.post_start_wait_sec),
        "--tag",
        tag,
    ]
    if args.no_save:
        cmd.append("--no-save")
    if args.no_show:
        cmd.append("--no-show")
    if args.hud:
        cmd.append("--hud")
    if args.auto_start_server:
        cmd.append("--auto-start-server")

    selector_log = out_dir / "selector.log"
    completed = subprocess.run(cmd, capture_output=True, text=True)
    selector_log.write_text(
        f"[time_utc] {utc_now()}\n[cmd] {' '.join(cmd)}\n[return_code] {completed.returncode}\n\n"
        f"[stdout]\n{completed.stdout}\n\n[stderr]\n{completed.stderr}\n",
        encoding="utf-8",
        errors="ignore",
    )
    report_path = Path(r"C:\CARLA\beamfusion_sandbox\isolated_loop\outputs") / tag / "report.json"
    if not report_path.exists():
        return "simple_loop_fallback"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("success") and report.get("success_profile"):
        return str(report["success_profile"])
    return "simple_loop_fallback"


@dataclass
class FeatureStats:
    gps_mean: np.ndarray
    gps_std: np.ndarray
    power_mean: np.ndarray
    power_std: np.ndarray
    power_dim: int


class FeatureBuilder:
    def __init__(self, stats: FeatureStats) -> None:
        self.stats = stats

    def _normalize(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        d = min(len(x), len(mean), len(std))
        out = np.zeros_like(mean, dtype=np.float32)
        out[:d] = (x[:d] - mean[:d]) / std[:d]
        return out

    def build_gps(self, hero, actors) -> np.ndarray:
        hloc = hero.get_location()
        nearest = None
        best_d = 1e9
        for a in actors:
            if a.id == hero.id:
                continue
            try:
                loc = a.get_location()
            except Exception:
                continue
            d = math.hypot(loc.x - hloc.x, loc.y - hloc.y)
            if d < best_d:
                best_d = d
                nearest = loc
        if nearest is None:
            nearest = hloc
        raw = np.array(
            [hloc.x, hloc.y, hloc.z, 1.0, 1.0, 1.0, nearest.x, nearest.y, nearest.z, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        return self._normalize(raw, self.stats.gps_mean, self.stats.gps_std)

    def _base_power_256(self, hero, actors) -> np.ndarray:
        htf = hero.get_transform()
        hloc = htf.location
        hyaw = math.radians(htf.rotation.yaw)
        linear = np.full((4, 64), 1e-3, dtype=np.float32)
        for a in actors:
            if a.id == hero.id:
                continue
            try:
                loc = a.get_location()
            except Exception:
                continue
            dx = loc.x - hloc.x
            dy = loc.y - hloc.y
            d = math.hypot(dx, dy)
            if d < 0.8 or d > 120.0:
                continue
            ang = math.atan2(dy, dx) - hyaw
            while ang > math.pi:
                ang -= 2 * math.pi
            while ang < -math.pi:
                ang += 2 * math.pi
            b = int(((ang + math.pi) / (2 * math.pi)) * 64) % 64
            strength = 1.0 / (d * d + 1.0)
            for c in range(4):
                idx = (b + c * 11) % 64
                if strength > linear[c, idx]:
                    linear[c, idx] = strength
        return linear.reshape(-1)

    def build_power(self, hero, actors) -> np.ndarray:
        base = self._base_power_256(hero, actors)
        if self.stats.power_dim == 256:
            raw = base
        elif self.stats.power_dim == 512:
            raw = np.concatenate([base, 10.0 * np.log10(np.clip(base, 1e-6, None))], axis=0)
        else:
            reps = int(math.ceil(self.stats.power_dim / float(base.shape[0])))
            raw = np.tile(base, reps)[: self.stats.power_dim]
        return self._normalize(raw.astype(np.float32), self.stats.power_mean, self.stats.power_std)


def load_beam_adapter(args: argparse.Namespace):
    import torch

    if not args.beamfusion_src.exists():
        raise FileNotFoundError(f"BeamFusion src not found: {args.beamfusion_src}")
    sys.path.insert(0, str(args.beamfusion_src))

    from beamfusion.carla_adapter import CarlaAdapterConfig, CarlaBeamAdapter
    from beamfusion.config import TrainConfig
    from beamfusion.data import prepare_scenario36
    from beamfusion.models import DetrIemfBeamModel

    cfg_path = args.beam_config
    if not cfg_path.exists():
        raise FileNotFoundError(f"Beam config not found: {cfg_path}")
    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = TrainConfig(**raw_cfg)

    prepared = prepare_scenario36(
        scenario_root=cfg.scenario_root,
        seed=cfg.seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        image_key_a=cfg.image_key_a,
        image_key_b=cfg.image_key_b,
        power_use_log=cfg.power_use_log,
        power_log_clip_min=cfg.power_log_clip_min,
        max_samples=None,
    )
    num_classes = int(prepared.labels.max() + 1)
    model = DetrIemfBeamModel(
        gps_dim=prepared.gps.shape[1],
        power_dim=prepared.power.shape[1],
        num_classes=num_classes,
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
        detr_repo=cfg.detr_repo,
        detr_variant=cfg.detr_variant,
        detr_pretrained=cfg.detr_pretrained,
        detr_checkpoint_path=cfg.detr_checkpoint_path,
        detr_checkpoint_strict=cfg.detr_checkpoint_strict,
        topk_queries=cfg.topk_queries,
        freeze_detr=cfg.freeze_detr,
        use_dual_view=cfg.use_dual_view,
        query_pool_mode=cfg.query_pool_mode,
        query_pool_heads=cfg.query_pool_heads,
        modality_dropout_p=0.0,
        ae_enabled=cfg.ae_enabled,
        ae_use_vae=cfg.ae_use_vae,
        ae_latent_dim=cfg.ae_latent_dim,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    ckpt = args.beam_checkpoint
    if not ckpt.exists():
        raise FileNotFoundError(f"Beam checkpoint not found: {ckpt}")
    loaded = torch.load(str(ckpt), map_location=device, weights_only=False)
    state = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
    model.load_state_dict(state, strict=False)
    model.eval()

    adapter = CarlaBeamAdapter(
        model=model,
        device=str(device),
        cfg=CarlaAdapterConfig(image_size=cfg.image_size, topk=args.topk),
    )
    stats = FeatureStats(
        gps_mean=prepared.gps_mean.astype(np.float32),
        gps_std=prepared.gps_std.astype(np.float32),
        power_mean=prepared.power_mean.astype(np.float32),
        power_std=prepared.power_std.astype(np.float32),
        power_dim=int(prepared.power.shape[1]),
    )
    return adapter, stats


class ShadowMonitor:
    def __init__(
        self,
        host: str,
        port: int,
        adapter,
        feature_builder: FeatureBuilder,
        out_jsonl: Path,
        infer_hz: float = 2.0,
        image_w: int = 960,
        image_h: int = 540,
    ) -> None:
        self.host = host
        self.port = port
        self.adapter = adapter
        self.feature_builder = feature_builder
        self.out_jsonl = out_jsonl
        self.infer_hz = max(0.2, infer_hz)
        self.image_w = int(image_w)
        self.image_h = int(image_h)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_frame: Optional[int] = None
        self._latest_ts: float = 0.0
        self._sensor = None
        self._hero = None
        self._errors: List[str] = []
        self.pred_count = 0

    def _on_image(self, image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        with self._lock:
            self._latest_rgb = arr
            self._latest_frame = int(image.frame)
            self._latest_ts = time.time()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _run(self) -> None:
        import carla  # type: ignore

        ensure_dir(self.out_jsonl.parent)
        logf = self.out_jsonl.open("w", encoding="utf-8")
        last_inf = 0.0
        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(6.0)
            world = client.get_world()

            # Wait hero from original script spawn.
            hero = None
            t0 = time.time()
            while (time.time() - t0) < 80.0 and not self._stop.is_set():
                vehicles = world.get_actors().filter("vehicle.*")
                for v in vehicles:
                    try:
                        role = v.attributes.get("role_name", "")
                    except Exception:
                        role = ""
                    if role == "hero":
                        hero = v
                        break
                if hero is not None:
                    break
                time.sleep(0.5)
            if hero is None:
                self._errors.append("hero_not_found")
                return
            self._hero = hero

            bp = world.get_blueprint_library().find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(self.image_w))
            bp.set_attribute("image_size_y", str(self.image_h))
            bp.set_attribute("fov", "100")
            tf = carla.Transform(carla.Location(x=1.5, z=2.2))
            self._sensor = world.spawn_actor(bp, tf, attach_to=hero)
            self._sensor.listen(self._on_image)

            while not self._stop.is_set():
                now = time.time()
                if (now - last_inf) < (1.0 / self.infer_hz):
                    time.sleep(0.01)
                    continue

                with self._lock:
                    rgb = None if self._latest_rgb is None else self._latest_rgb.copy()
                    fid = self._latest_frame
                if rgb is None:
                    time.sleep(0.01)
                    continue

                try:
                    hero_tf = hero.get_transform()
                    actors = list(world.get_actors().filter("vehicle.*"))
                    gps = self.feature_builder.build_gps(hero, actors)
                    power = self.feature_builder.build_power(hero, actors)
                    pred = self.adapter.predict(rgb=rgb, gps=gps, power=power)
                    row = {
                        "time_utc": utc_now(),
                        "frame": int(fid) if fid is not None else None,
                        "ego_x": float(hero_tf.location.x),
                        "ego_y": float(hero_tf.location.y),
                        "ego_yaw": float(hero_tf.rotation.yaw),
                        "topk_beams": pred["topk_beams"].tolist(),
                        "topk_probs": pred["topk_probs"].tolist(),
                        "gate_weights": pred["gate_weights"].tolist(),
                    }
                    logf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    logf.flush()
                    self.pred_count += 1
                except Exception as e:
                    self._errors.append(f"infer_error:{e}")

                last_inf = now

        except Exception as e:
            self._errors.append(f"monitor_error:{e}")
        finally:
            try:
                if self._sensor is not None:
                    self._sensor.stop()
                    self._sensor.destroy()
            except Exception:
                pass
            try:
                logf.close()
            except Exception:
                pass

    def summary(self) -> Dict[str, Any]:
        return {"pred_count": int(self.pred_count), "errors": self._errors}


def run_one_shadow_attempt(
    args: argparse.Namespace,
    profile: str,
    adapter,
    feature_builder: FeatureBuilder,
    out_dir: Path,
) -> Dict[str, Any]:
    before_runs = list_run_dirs()
    cmd = [
        sys.executable,
        str(AUTON_SCRIPT),
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
        cmd.append("--no-save")
    if args.no_show:
        cmd.append("--no-show")
    if args.hud:
        cmd.append("--hud")
    cmd.extend(build_profile_extras(profile, args.target_speed, args.lane_change_prefer))

    mon = ShadowMonitor(
        host=args.host,
        port=args.port,
        adapter=adapter,
        feature_builder=feature_builder,
        out_jsonl=out_dir / f"shadow_{profile}.jsonl",
        infer_hz=args.infer_hz,
        image_w=args.shadow_image_w,
        image_h=args.shadow_image_h,
    )
    mon.start()
    t0 = time.time()
    cp = subprocess.run(cmd, cwd=str(CARLA_ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0
    time.sleep(2.0)
    mon.stop()

    attempt_log = out_dir / f"attempt_{profile}.log"
    attempt_log.write_text(
        f"[time_utc] {utc_now()}\n"
        f"[cmd] {' '.join(cmd)}\n"
        f"[return_code] {cp.returncode}\n\n[stdout]\n{cp.stdout}\n\n[stderr]\n{cp.stderr}\n",
        encoding="utf-8",
        errors="ignore",
    )

    run_dir = latest_new_run(before_runs)
    run_summary = summarize_run(run_dir, max_collisions=args.max_collisions) if run_dir is not None else None
    shadow_summary = mon.summary()
    ok = bool(
        cp.returncode == 0
        and run_summary is not None
        and run_summary.get("ok", False)
        and shadow_summary.get("pred_count", 0) >= 5
    )
    return {
        "profile": profile,
        "cmd": cmd,
        "return_code": cp.returncode,
        "elapsed_sec": elapsed,
        "run_detected": str(run_dir) if run_dir else None,
        "run_summary": run_summary,
        "shadow_summary": shadow_summary,
        "attempt_log": str(attempt_log),
        "ok": ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BeamFusion shadow integration pipeline in isolated CARLA sandbox."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map-name", default="Town05")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--delta-time", type=float, default=0.02)
    parser.add_argument("--num-vehicles", type=int, default=6)
    parser.add_argument("--num-walkers", type=int, default=4)
    parser.add_argument("--target-speed", type=float, default=55.0)
    parser.add_argument("--vis-stride", type=int, default=6)
    parser.add_argument("--lane-change-prefer", choices=["right", "left"], default="right")
    parser.add_argument("--max-collisions", type=int, default=0)
    parser.add_argument("--no-save", action="store_true", default=True)
    parser.add_argument("--no-show", action="store_true", default=False)
    parser.add_argument("--hud", action="store_true", default=True)

    parser.add_argument("--auto-start-server", action="store_true")
    parser.add_argument("--server-exe", type=Path, default=DEFAULT_CARLA_EXE)
    parser.add_argument("--server-launch-mode", choices=["plain", "custom"], default="plain")
    parser.add_argument("--server-wait-sec", type=float, default=35.0)
    parser.add_argument("--post-start-wait-sec", type=float, default=30.0)

    parser.add_argument("--beamfusion-src", type=Path, default=DEFAULT_BEAM_SRC)
    parser.add_argument("--beam-config", type=Path, default=DEFAULT_BEAM_CONFIG)
    parser.add_argument("--beam-checkpoint", type=Path, default=DEFAULT_BEAM_CKPT)
    parser.add_argument("--device", default="")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--infer-hz", type=float, default=2.0)
    parser.add_argument("--shadow-image-w", type=int, default=960)
    parser.add_argument("--shadow-image-h", type=int, default=540)

    parser.add_argument("--skip-selector", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if not AUTON_SCRIPT.exists():
        raise FileNotFoundError(f"Original script missing: {AUTON_SCRIPT}")
    if not ISOLATED_LOOP_SCRIPT.exists():
        raise FileNotFoundError(f"Selector script missing: {ISOLATED_LOOP_SCRIPT}")

    if args.auto_start_server and args.server_exe.exists():
        # Use selector/runner side to start server with same strategy.
        pass
    if not check_server_ready(args.host, args.port, timeout_sec=2.0) and not args.auto_start_server:
        raise RuntimeError("CARLA server not reachable. Start it first or add --auto-start-server.")

    tag = args.tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / tag
    ensure_dir(out_dir)

    selected_profile = "simple_loop_fallback"
    if not args.skip_selector:
        selected_profile = select_stable_profile(args, out_dir=out_dir)

    adapter, stats = load_beam_adapter(args)
    feature_builder = FeatureBuilder(stats)

    profiles = [selected_profile]
    for p in ["user_baseline_adv", "adv_no_global_route", "adv_global_route", "simple_loop_fallback"]:
        if p not in profiles:
            profiles.append(p)

    attempts = []
    success = None
    for p in profiles:
        result = run_one_shadow_attempt(args, profile=p, adapter=adapter, feature_builder=feature_builder, out_dir=out_dir)
        attempts.append(result)
        if result["ok"]:
            success = result
            break

    report = {
        "time_utc": utc_now(),
        "tag": tag,
        "carla_root": str(CARLA_ROOT),
        "autonomous_script": str(AUTON_SCRIPT),
        "max_collisions_allowed": int(args.max_collisions),
        "selected_profile": selected_profile,
        "success": success is not None,
        "success_profile": (success["profile"] if success else None),
        "success_run_dir": (success["run_detected"] if success else None),
        "attempts": attempts,
        "beam_model": {
            "config": str(args.beam_config),
            "checkpoint": str(args.beam_checkpoint),
            "src": str(args.beamfusion_src),
            "device": args.device or "auto",
        },
    }
    write_json(out_dir / "report.json", report)

    md = [
        f"# BeamFusion Shadow Report ({tag})",
        "",
        f"- selected_profile: `{selected_profile}`",
        f"- max_collisions_allowed: `{int(args.max_collisions)}`",
        f"- success: `{report['success']}`",
        f"- success_profile: `{report['success_profile']}`",
        f"- success_run_dir: `{report['success_run_dir']}`",
        "",
        "| profile | ok | return_code | pred_count | avg_speed_kmh | displacement_m | collisions | run_dir |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for a in attempts:
        rs = a.get("run_summary") or {}
        ss = a.get("shadow_summary") or {}
        md.append(
            f"| {a['profile']} | {a.get('ok')} | {a['return_code']} | {ss.get('pred_count')} | "
            f"{rs.get('avg_speed_kmh')} | {rs.get('displacement_m')} | {rs.get('final_collision_counter')} | `{a.get('run_detected')}` |"
        )
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[Done] report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
