#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA autonomous driving and multimodal data collection test.

Features:
1. Spawn NPC vehicles and walkers (optional autopilot).
2. Spawn an ego vehicle we control.
3. Attach multimodal sensors to the ego vehicle.
4. Collect and save multimodal data.
5. Support manual or autopilot control of the ego vehicle.
"""

import carla
try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except Exception:
    GlobalRoutePlanner = None
import random
import time
import os
import sys
import json
import math
import numpy as np
from queue import Queue
import argparse
import pygame
from datetime import datetime
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from pygame._sdl2 import Window, Renderer, Texture
    SDL2_AVAILABLE = True
except Exception:
    SDL2_AVAILABLE = False


sensor_queue = Queue()


DATA_DIR = "carla_data"
RGB_DIR = os.path.join(DATA_DIR, "rgb")
DEPTH_DIR = os.path.join(DATA_DIR, "depth")
SEMANTIC_DIR = os.path.join(DATA_DIR, "semantic")
LIDAR_DIR = os.path.join(DATA_DIR, "lidar")
RADAR_DIR = os.path.join(DATA_DIR, "radar")


for dir_path in [RGB_DIR, DEPTH_DIR, SEMANTIC_DIR, LIDAR_DIR, RADAR_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def log_status(message: str):
    """Lightweight startup log to file. 写启动日志到文件。"""
    try:
        with open(os.path.join(DATA_DIR, "startup.log"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} {message}\n")
    except Exception:
        pass

sensor_stats = {}
latest_frames = {}
SHOW_VISUALS = False
RUN_NAME = None
RUN_DIR = None
ADV_ROUTE = []
ADV_ROUTE_INDEX = 0
NO_SAVE = False
SAVE_STRIDE = 3
route_log = None
actor_log = None
collision_log = None
latest_obstacle_dist = None
last_obstacle_time = 0.0
latest_lidar_dist = None
latest_radar_dist = None
collision_count = 0
last_collision_frame = -1
last_collision_desc = ""
STOP_ON_COLLISION = False
obstacle_confirm_count = 0
lane_change_until = 0.0
lane_change_target = None
lane_change_target_lane = None
lane_change_started_at = 0.0
lane_change_origin_lane = None
lane_change_hold_until = 0.0
lane_lock_lane = None
lane_lock_until = 0.0
lane_keep_lane_key = None
lane_recover_lane = None
lane_recover_until = 0.0
lane_change_retry_until = 0.0
lane_change_fail_streak = 0
lane_change_start_offset = 0.0
lane_change_best_offset = 0.0
lane_change_last_progress_time = 0.0
last_steer = 0.0
ROUTE_WPS = []
ROUTE_INDEX = 0
ROUTE_TARGET = None
CROSSING_SPAWN_INFO = []
CROSSING_PEDS = []
last_ped_spawn_time = 0.0
dynamic_ped_count = 0
stuck_counter = 0
block_npc_start_time = 0.0
prev_lane_edge_offset = 0.0
lane_edge_risk_count = 0
beam_latest_rgb = None
beam_latest_frame = -1
lane_front_rgb = None
lane_front_frame = -1


@dataclass
class BeamFeatureStats:
    gps_mean: np.ndarray
    gps_std: np.ndarray
    power_mean: np.ndarray
    power_std: np.ndarray
    power_dim: int


class BeamFeatureBuilder:
    def __init__(self, stats: BeamFeatureStats) -> None:
        self.stats = stats

    def _normalize(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        d = min(len(x), len(mean), len(std))
        out = np.zeros_like(mean, dtype=np.float32)
        x_safe = np.nan_to_num(x[:d], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mean_safe = np.nan_to_num(mean[:d], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        std_safe = np.nan_to_num(std[:d], nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
        std_safe = np.clip(std_safe, 1e-6, None)
        out[:d] = (x_safe - mean_safe) / std_safe
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
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


class BeamFusionClosedLoopPolicy:
    def __init__(self, args) -> None:
        self.args = args
        self.adapter = None
        self.feature_builder: Optional[BeamFeatureBuilder] = None
        self.enabled = False
        self.last_infer_ts = 0.0
        self.state: Dict[str, Any] = {
            "available": False,
            "top1_beam": -1,
            "top1_prob": 0.0,
            "entropy": 1.0,
            "risk": 1.0,
            "speed_cap_kmh": float(args.target_speed),
            "allow_lane_change": False,
            "gate_weights": [],
        }
        self.log_fp = None

    def _norm_entropy(self, probs: np.ndarray) -> float:
        if probs.size <= 1:
            return 0.0
        p = np.clip(probs.astype(np.float64), 1e-8, 1.0)
        h = -float(np.sum(p * np.log(p)))
        return float(h / np.log(float(p.size)))

    def _safe_float(self, v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    def open_log(self, run_dir: Optional[str]) -> None:
        if not self.enabled or run_dir is None:
            return
        try:
            p = os.path.join(run_dir, "beamfusion_closed_loop_log.jsonl")
            self.log_fp = open(p, "w", encoding="utf-8")
        except Exception:
            self.log_fp = None

    def close_log(self) -> None:
        try:
            if self.log_fp is not None:
                self.log_fp.close()
        except Exception:
            pass
        self.log_fp = None

    def _load_adapter(self):
        import torch

        src = Path(self.args.bf_src)
        cfg_path = Path(self.args.bf_config)
        ckpt_path = Path(self.args.bf_checkpoint)
        if (not src.exists()) or (not cfg_path.exists()) or (not ckpt_path.exists()):
            missing = []
            if not src.exists():
                missing.append(f"src={src}")
            if not cfg_path.exists():
                missing.append(f"config={cfg_path}")
            if not ckpt_path.exists():
                missing.append(f"ckpt={ckpt_path}")
            raise FileNotFoundError("BeamFusion assets missing: " + ", ".join(missing))

        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from beamfusion.carla_adapter import CarlaAdapterConfig, CarlaBeamAdapter
        from beamfusion.config import TrainConfig
        from beamfusion.data import prepare_scenario36
        from beamfusion.models import DetrIemfBeamModel

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
        device = torch.device(self.args.bf_device if self.args.bf_device else ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
        loaded = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
        model.load_state_dict(state, strict=False)
        model.eval()
        adapter = CarlaBeamAdapter(
            model=model,
            device=str(device),
            cfg=CarlaAdapterConfig(image_size=cfg.image_size, topk=max(1, int(self.args.bf_topk))),
        )
        stats = BeamFeatureStats(
            gps_mean=np.nan_to_num(prepared.gps_mean.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            gps_std=np.clip(
                np.nan_to_num(prepared.gps_std.astype(np.float32), nan=1.0, posinf=1.0, neginf=1.0),
                1e-6,
                None,
            ),
            power_mean=np.nan_to_num(prepared.power_mean.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            power_std=np.clip(
                np.nan_to_num(prepared.power_std.astype(np.float32), nan=1.0, posinf=1.0, neginf=1.0),
                1e-6,
                None,
            ),
            power_dim=int(prepared.power.shape[1]),
        )
        return adapter, BeamFeatureBuilder(stats)

    def initialize(self) -> bool:
        if not self.args.bf_enable:
            self.enabled = False
            return False
        try:
            self.adapter, self.feature_builder = self._load_adapter()
            self.enabled = True
            print("[BeamFusion] closed-loop policy enabled")
            return True
        except Exception as e:
            self.enabled = False
            print(f"[BeamFusion][WARN] init failed, fallback to baseline control: {e}")
            return False

    def infer(self, ego_vehicle, all_vehicles, rgb_frame, now_time: float) -> Dict[str, Any]:
        if (not self.enabled) or (self.adapter is None) or (self.feature_builder is None):
            return self.state
        if rgb_frame is None:
            return self.state
        min_dt = 1.0 / max(0.2, float(self.args.bf_infer_hz))
        if (now_time - self.last_infer_ts) < min_dt:
            return self.state
        try:
            gps = self.feature_builder.build_gps(ego_vehicle, all_vehicles)
            power = self.feature_builder.build_power(ego_vehicle, all_vehicles)
            pred = self.adapter.predict(rgb=rgb_frame, gps=gps, power=power)
            topk_beams = np.asarray(pred["topk_beams"])
            topk_beams = np.nan_to_num(topk_beams, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64)
            topk_probs = np.asarray(pred["topk_probs"], dtype=np.float32)
            topk_probs = np.nan_to_num(topk_probs, nan=0.0, posinf=0.0, neginf=0.0)
            topk_probs = np.clip(topk_probs, 0.0, 1.0)
            p_sum = float(np.sum(topk_probs))
            if p_sum > 1e-6:
                topk_probs = topk_probs / p_sum
            elif topk_probs.size > 0:
                topk_probs = np.zeros_like(topk_probs)
                topk_probs[0] = 1.0
            gate_weights = np.asarray(pred["gate_weights"], dtype=np.float32)
            gate_weights = np.nan_to_num(gate_weights, nan=0.0, posinf=0.0, neginf=0.0)
            gate_weights = np.clip(gate_weights, 0.0, 1.0)
            g_sum = float(np.sum(gate_weights))
            if g_sum > 1e-6:
                gate_weights = gate_weights / g_sum
            elif gate_weights.size > 0:
                gate_weights = np.full_like(gate_weights, 1.0 / float(gate_weights.size))

            conf = float(topk_probs[0]) if topk_probs.size > 0 else 0.0
            entropy = self._norm_entropy(topk_probs)
            gate_uncert = 1.0 - float(np.max(gate_weights)) if gate_weights.size > 0 else 0.5
            risk = (1.0 - conf) + float(self.args.bf_entropy_weight) * entropy + float(self.args.bf_gate_weight) * gate_uncert
            risk = max(0.0, min(1.5, risk))
            speed_scale = float(self.args.bf_speed_max_scale) - float(self.args.bf_risk_speed_gain) * risk
            speed_scale = max(float(self.args.bf_speed_min_scale), min(float(self.args.bf_speed_max_scale), speed_scale))
            speed_cap = float(self.args.target_speed) * speed_scale
            allow_lane_change = bool(
                conf >= float(self.args.bf_lane_change_min_conf) and risk <= float(self.args.bf_lane_change_risk_th)
            )

            self.state = {
                "available": True,
                "top1_beam": int(topk_beams[0]) if topk_beams.size > 0 else -1,
                "top1_prob": conf,
                "entropy": entropy,
                "risk": risk,
                "speed_cap_kmh": speed_cap,
                "allow_lane_change": allow_lane_change,
                "gate_weights": gate_weights.tolist(),
            }
            self.last_infer_ts = now_time
            if self.log_fp is not None:
                row = {
                    "time_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "top1_beam": self.state["top1_beam"],
                    "top1_prob": conf,
                    "entropy": entropy,
                    "risk": risk,
                    "speed_cap_kmh": speed_cap,
                    "allow_lane_change": allow_lane_change,
                    "gate_weights": self.state["gate_weights"],
                }
                self.log_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.log_fp.flush()
        except Exception as e:
            if self.log_fp is not None:
                self.log_fp.write(json.dumps({"time_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z", "error": str(e)}, ensure_ascii=False) + "\n")
                self.log_fp.flush()
        return self.state


class LaneVisionAssist:
    """Lightweight lane-mark based offset estimator from front RGB camera."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled and (cv2 is not None))
        self.prev_offset = 0.0
        self.sign_score = 0.0
        self.sign = 1.0

    def available(self) -> bool:
        return self.enabled

    def _fit_line_x_at_y(self, segments, y_ref: float) -> Optional[float]:
        if not segments:
            return None
        m_sum = 0.0
        b_sum = 0.0
        w_sum = 0.0
        for (x1, y1, x2, y2, w) in segments:
            dx = (x2 - x1)
            if abs(dx) < 1e-5:
                continue
            m = (y2 - y1) / dx
            b = y1 - m * x1
            m_sum += m * w
            b_sum += b * w
            w_sum += w
        if w_sum <= 1e-6:
            return None
        m = m_sum / w_sum
        b = b_sum / w_sum
        if abs(m) < 1e-5:
            return None
        return float((y_ref - b) / m)

    def estimate(self, rgb: np.ndarray, map_offset_norm: Optional[float] = None) -> Dict[str, float]:
        if (not self.enabled) or rgb is None or cv2 is None:
            return {"available": 0.0, "offset_norm": 0.0, "confidence": 0.0}
        try:
            img = rgb
            h, w = img.shape[:2]
            if h < 64 or w < 64:
                return {"available": 0.0, "offset_norm": 0.0, "confidence": 0.0}

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 35, 120)
            hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            # White/yellow lane marking mask.
            white = cv2.inRange(hls, np.array([0, 165, 0], dtype=np.uint8), np.array([255, 255, 110], dtype=np.uint8))
            yellow = cv2.inRange(hls, np.array([12, 45, 70], dtype=np.uint8), np.array([45, 255, 255], dtype=np.uint8))
            color_mask = cv2.bitwise_or(white, yellow)

            roi_mask = np.zeros_like(edges)
            roi = np.array(
                [[
                    (int(0.03 * w), h - 1),
                    (int(0.36 * w), int(0.57 * h)),
                    (int(0.64 * w), int(0.57 * h)),
                    (int(0.97 * w), h - 1),
                ]],
                dtype=np.int32,
            )
            cv2.fillPoly(roi_mask, roi, 255)
            edges = cv2.bitwise_and(edges, roi_mask)
            color_mask = cv2.bitwise_and(color_mask, roi_mask)
            lane_bin = cv2.bitwise_or(edges, color_mask)

            lines = cv2.HoughLinesP(
                lane_bin,
                rho=1,
                theta=np.pi / 180.0,
                threshold=16,
                minLineLength=max(12, int(0.04 * h)),
                maxLineGap=max(16, int(0.05 * h)),
            )

            left, right = [], []
            len_left = 0.0
            len_right = 0.0
            if lines is not None:
                for seg in lines:
                    x1, y1, x2, y2 = [int(v) for v in seg[0]]
                    dx = float(x2 - x1)
                    dy = float(y2 - y1)
                    if abs(dx) < 1.0:
                        continue
                    slope = dy / dx
                    if abs(slope) < 0.18 or abs(slope) > 5.0:
                        continue
                    length = float((dx * dx + dy * dy) ** 0.5)
                    x_mid = 0.5 * (x1 + x2)
                    if slope < 0 and x_mid < 0.55 * w:
                        left.append((x1, y1, x2, y2, length))
                        len_left += length
                    elif slope > 0 and x_mid > 0.45 * w:
                        right.append((x1, y1, x2, y2, length))
                        len_right += length

            y_ref = float(int(0.84 * h))
            xl = self._fit_line_x_at_y(left, y_ref)
            xr = self._fit_line_x_at_y(right, y_ref)

            # Fallback: histogram peaks on color mask when line fitting is weak.
            if xl is None or xr is None:
                hist = np.sum(color_mask[int(0.62 * h) :, :], axis=0).astype(np.float32)
                if hist.size == w:
                    l_peak = int(np.argmax(hist[: w // 2])) if np.max(hist[: w // 2]) > 20.0 else -1
                    r_peak_rel = int(np.argmax(hist[w // 2 :])) if np.max(hist[w // 2 :]) > 20.0 else -1
                    r_peak = (w // 2 + r_peak_rel) if r_peak_rel >= 0 else -1
                    if xl is None and l_peak >= 0:
                        xl = float(l_peak)
                    if xr is None and r_peak >= 0:
                        xr = float(r_peak)

            if xl is not None and xr is not None:
                lane_center = 0.5 * (xl + xr)
                conf = min(1.0, 0.30 + 0.0020 * (len_left + len_right))
            elif xl is not None:
                lane_center = xl + 0.48 * w
                conf = min(0.68, 0.18 + 0.0028 * len_left)
            elif xr is not None:
                lane_center = xr - 0.48 * w
                conf = min(0.68, 0.18 + 0.0028 * len_right)
            else:
                return {"available": 0.0, "offset_norm": self.prev_offset, "confidence": 0.0}

            offset_norm = float((lane_center - 0.5 * w) / (0.5 * w))
            offset_norm = max(-1.5, min(1.5, offset_norm))

            if map_offset_norm is not None and abs(float(map_offset_norm)) < 1.2 and conf > 0.25:
                self.sign_score += float(map_offset_norm) * float(offset_norm)
                self.sign_score = max(-30.0, min(30.0, self.sign_score))
                if self.sign_score < -0.8:
                    self.sign = -1.0
                elif self.sign_score > 0.8:
                    self.sign = 1.0

            offset_norm *= self.sign
            offset_norm = 0.72 * self.prev_offset + 0.28 * offset_norm
            self.prev_offset = float(offset_norm)
            return {"available": 1.0, "offset_norm": float(offset_norm), "confidence": float(conf)}
        except Exception:
            return {"available": 0.0, "offset_norm": self.prev_offset, "confidence": 0.0}


def set_run_directories(run_name: str):
    """Set per-run directories for data output. 为每次运行创建独立数据目录。"""
    global RUN_NAME, RUN_DIR, RGB_DIR, DEPTH_DIR, SEMANTIC_DIR, LIDAR_DIR, RADAR_DIR, route_log, actor_log, collision_log
    RUN_NAME = run_name
    RUN_DIR = os.path.join(DATA_DIR, run_name)
    RGB_DIR = os.path.join(RUN_DIR, "rgb")
    DEPTH_DIR = os.path.join(RUN_DIR, "depth")
    SEMANTIC_DIR = os.path.join(RUN_DIR, "semantic")
    LIDAR_DIR = os.path.join(RUN_DIR, "lidar")
    RADAR_DIR = os.path.join(RUN_DIR, "radar")
    for dir_path in [RGB_DIR, DEPTH_DIR, SEMANTIC_DIR, LIDAR_DIR, RADAR_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    route_log = open(os.path.join(RUN_DIR, "route.csv"), "w", newline="", encoding="utf-8")
    writer = csv.writer(route_log)
    writer.writerow(["frame", "x", "y", "z", "speed_kmh"])
    actor_log = open(os.path.join(RUN_DIR, "actors.csv"), "w", newline="", encoding="utf-8")
    aw = csv.writer(actor_log)
    aw.writerow(["frame", "type", "id", "x", "y", "z", "note"])
    collision_log = open(os.path.join(RUN_DIR, "collision.csv"), "w", newline="", encoding="utf-8")
    cw = csv.writer(collision_log)
    cw.writerow(["frame", "timestamp", "other_id", "other_type", "intensity", "x", "y", "z"])


def image_to_array(image, converter=None):
    """Return an RGB numpy array from a CARLA image."""
    if converter is not None:
        image.convert(converter)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]  # BGR -> RGB
    return array


def update_obstacle_from_lidar(lidar_data, lane_half_width=1.4):
    """Update obstacle distance using LiDAR points in front corridor. 用激光点更新前方障碍距离。"""
    global latest_obstacle_dist, last_obstacle_time, latest_lidar_dist
    pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    pts = pts.reshape((-1, 4))
    # Forward corridor filter (ignore near field to avoid ground/self noise)
    # Reject near-field self returns and curb-side clutter
    forward = pts[:, 0] > 5.0
    lateral = np.abs(pts[:, 1]) < lane_half_width
    height = (pts[:, 2] > 0.35) & (pts[:, 2] < 1.8)
    mask = forward & lateral & height
    if not np.any(mask):
        return
    d = np.sqrt(pts[mask, 0] ** 2 + pts[mask, 1] ** 2)
    if d.size == 0:
        return
    min_d = float(np.min(d))
    if min_d < 5.0:
        return
    latest_lidar_dist = min_d
    latest_obstacle_dist = min_d
    last_obstacle_time = time.time()


def update_obstacle_from_radar(radar_data, max_azimuth=0.20):
    """Update obstacle distance using radar detections. 用雷达更新前方障碍距离。"""
    global latest_obstacle_dist, last_obstacle_time, latest_radar_dist
    min_d = None
    for det in radar_data:
        if abs(det.azimuth) > max_azimuth:
            continue
        if min_d is None or det.depth < min_d:
            min_d = det.depth
    if min_d is not None and min_d >= 5.0:
        latest_radar_dist = min_d
        latest_obstacle_dist = min_d
        last_obstacle_time = time.time()


def collision_callback(event):
    """Collision callback with run-level logging. 碰撞事件回调并写入日志。"""
    global collision_count, last_collision_frame, last_collision_desc, collision_log
    collision_count += 1
    last_collision_frame = int(event.frame)
    other = event.other_actor
    other_id = int(other.id) if other is not None else -1
    other_type = other.type_id if other is not None else "unknown"
    impulse = event.normal_impulse
    intensity = float((impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2) ** 0.5)
    last_collision_desc = f"{other_type}:{other_id}:{intensity:.2f}"
    try:
        loc = event.transform.location
        if collision_log is not None:
            collision_log.write(
                f"{event.frame},{event.timestamp:.4f},{other_id},{other_type},{intensity:.4f},{loc.x:.3f},{loc.y:.3f},{loc.z:.3f}\n"
            )
            collision_log.flush()
    except Exception:
        pass
    print(f"[COLLISION] frame={event.frame} other={other_type}({other_id}) intensity={intensity:.2f}")

def sensor_callback(sensor_data, sensor_name, data_dir):
    """Sensor data callback with disk IO. 传感器回调，处理保存/可视化。"""
    global beam_latest_rgb, beam_latest_frame, lane_front_rgb, lane_front_frame
    frame_id = sensor_data.frame
    

    if sensor_name not in sensor_stats:
        sensor_stats[sensor_name] = {'count': 0, 'last_frame': 0}
    sensor_stats[sensor_name]['count'] += 1
    sensor_stats[sensor_name]['last_frame'] = frame_id
    
    try:
        name = sensor_name.lower()
        if "beam_rgb" in name:
            beam_latest_rgb = image_to_array(sensor_data)
            beam_latest_frame = int(frame_id)
            sensor_queue.put((frame_id, sensor_name))
            return
        if "rgb_front" in name:
            lane_front_rgb = image_to_array(sensor_data)
            lane_front_frame = int(frame_id)
        if 'view_fp' in name or 'view_tp' in name:
            latest_frames[name] = image_to_array(sensor_data)
            sensor_queue.put((frame_id, sensor_name))
            return
        save_this = (frame_id % SAVE_STRIDE == 0)
        if NO_SAVE:
            # Only update latest frame for visualization
            if 'rgb' in name and SHOW_VISUALS:
                latest_frames['rgb_front'] = image_to_array(sensor_data)
            elif 'depth' in name and SHOW_VISUALS:
                latest_frames['depth_front'] = image_to_array(sensor_data, carla.ColorConverter.LogarithmicDepth)
            elif 'semantic' in name and SHOW_VISUALS:
                latest_frames['semantic_front'] = image_to_array(sensor_data, carla.ColorConverter.CityScapesPalette)
            elif 'lidar' in name:
                update_obstacle_from_lidar(sensor_data)
            elif 'radar' in name:
                update_obstacle_from_radar(sensor_data)
            sensor_queue.put((frame_id, sensor_name))
            return

        if 'rgb' in name and save_this:
            file_path = os.path.join(RGB_DIR, f"{sensor_name}_{frame_id:06d}.png")
            sensor_data.save_to_disk(file_path)
            if SHOW_VISUALS:
                latest_frames['rgb_front'] = image_to_array(sensor_data)
        elif 'depth' in name and save_this:
            file_path = os.path.join(DEPTH_DIR, f"{sensor_name}_{frame_id:06d}.png")
            sensor_data.save_to_disk(file_path, carla.ColorConverter.LogarithmicDepth)
            if SHOW_VISUALS:
                latest_frames['depth_front'] = image_to_array(sensor_data, carla.ColorConverter.LogarithmicDepth)
        elif 'semantic' in name and save_this:
            file_path = os.path.join(SEMANTIC_DIR, f"{sensor_name}_{frame_id:06d}.png")
            sensor_data.save_to_disk(file_path, carla.ColorConverter.CityScapesPalette)
            if SHOW_VISUALS:
                latest_frames['semantic_front'] = image_to_array(sensor_data, carla.ColorConverter.CityScapesPalette)
        elif 'lidar' in name:
            update_obstacle_from_lidar(sensor_data)
            if save_this:
                file_path = os.path.join(LIDAR_DIR, f"{sensor_name}_{frame_id:06d}.ply")
                sensor_data.save_to_disk(file_path)
        elif 'radar' in name:
            update_obstacle_from_radar(sensor_data)
            if save_this:
                file_path = os.path.join(RADAR_DIR, f"{sensor_name}_{frame_id:06d}.txt")
                with open(file_path, 'w') as f:
                    for detection in sensor_data:
                        f.write(f"{detection.velocity} {detection.azimuth} {detection.altitude} {detection.depth}\n")
        

        sensor_queue.put((frame_id, sensor_name))
        

        if frame_id % 100 == 0:
            print(f"[IO] Saved {sensor_name} data, frame: {frame_id}")
    except Exception as e:
        print(f"[ERROR] Failed to save {sensor_name} data at frame {frame_id}: {e}")

def setup_sensors(
    world,
    ego_vehicle,
    blueprint_library,
    view_width=1280,
    view_height=720,
    view_windows=True,
    beam_enabled=False,
    beam_width=960,
    beam_height=540,
):
    """Attach multimodal sensors to the ego vehicle. 在 Ego 车上挂载多模态传感器。"""
    sensors = []
    
    print("\nInstalling multimodal sensors...")
    

    try:
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', '1920')
        rgb_bp.set_attribute('image_size_y', '1080')
        rgb_bp.set_attribute('fov', '110')
        rgb_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(pitch=0))
        rgb_camera = world.spawn_actor(rgb_bp, rgb_transform, attach_to=ego_vehicle)
        rgb_camera.listen(lambda data: sensor_callback(data, 'rgb_front', RGB_DIR))
        sensors.append(rgb_camera)
        print("  [OK] RGB camera installed")
    except Exception as e:
        print(f"  [ERROR] Failed to install RGB camera: {e}")

    if beam_enabled:
        try:
            beam_bp = blueprint_library.find('sensor.camera.rgb')
            beam_bp.set_attribute('image_size_x', str(int(beam_width)))
            beam_bp.set_attribute('image_size_y', str(int(beam_height)))
            beam_bp.set_attribute('fov', '100')
            beam_transform = carla.Transform(carla.Location(x=1.5, z=2.2), carla.Rotation(pitch=0))
            beam_camera = world.spawn_actor(beam_bp, beam_transform, attach_to=ego_vehicle)
            beam_camera.listen(lambda data: sensor_callback(data, 'beam_rgb', RGB_DIR))
            sensors.append(beam_camera)
            print("  [OK] BeamFusion RGB camera installed")
        except Exception as e:
            print(f"  [WARN] Failed to install BeamFusion RGB camera: {e}")

    if view_windows:
        try:
            fp_bp = blueprint_library.find('sensor.camera.rgb')
            fp_bp.set_attribute('image_size_x', str(view_width))
            fp_bp.set_attribute('image_size_y', str(view_height))
            fp_bp.set_attribute('fov', '90')
            fp_transform = carla.Transform(carla.Location(x=1.6, z=1.3), carla.Rotation(pitch=0))
            fp_camera = world.spawn_actor(fp_bp, fp_transform, attach_to=ego_vehicle)
            fp_camera.listen(lambda data: sensor_callback(data, 'view_fp', RGB_DIR))
            sensors.append(fp_camera)
            print("  [OK] First-person view camera installed")
        except Exception as e:
            print(f"  [ERROR] Failed to install first-person camera: {e}")

        try:
            tp_bp = blueprint_library.find('sensor.camera.rgb')
            tp_bp.set_attribute('image_size_x', str(view_width))
            tp_bp.set_attribute('image_size_y', str(view_height))
            tp_bp.set_attribute('fov', '90')
            tp_transform = carla.Transform(carla.Location(x=-6.0, z=2.5), carla.Rotation(pitch=-10))
            tp_camera = world.spawn_actor(tp_bp, tp_transform, attach_to=ego_vehicle)
            tp_camera.listen(lambda data: sensor_callback(data, 'view_tp', RGB_DIR))
            sensors.append(tp_camera)
            print("  [OK] Third-person view camera installed")
        except Exception as e:
            print(f"  [ERROR] Failed to install third-person camera: {e}")


    try:
        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '1920')
        depth_bp.set_attribute('image_size_y', '1080')
        depth_bp.set_attribute('fov', '110')
        depth_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(pitch=0))
        depth_camera = world.spawn_actor(depth_bp, depth_transform, attach_to=ego_vehicle)
        depth_camera.listen(lambda data: sensor_callback(data, 'depth_front', DEPTH_DIR))
        sensors.append(depth_camera)
        print("  [OK] Depth camera installed")
    except Exception as e:
        print(f"  [ERROR] Failed to install depth camera: {e}")
    

    try:
        semantic_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        semantic_bp.set_attribute('image_size_x', '1920')
        semantic_bp.set_attribute('image_size_y', '1080')
        semantic_bp.set_attribute('fov', '110')
        semantic_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(pitch=0))
        semantic_camera = world.spawn_actor(semantic_bp, semantic_transform, attach_to=ego_vehicle)
        semantic_camera.listen(lambda data: sensor_callback(data, 'semantic_front', SEMANTIC_DIR))
        sensors.append(semantic_camera)
        print("  [OK] Semantic segmentation camera installed")
    except Exception as e:
        print(f"  [ERROR] Failed to install semantic segmentation camera: {e}")
    

    try:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '1000000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('range', '100')
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
        lidar.listen(lambda data: sensor_callback(data, 'lidar_top', LIDAR_DIR))
        sensors.append(lidar)
        print("  [OK] LiDAR installed")
    except Exception as e:
        print(f"  [ERROR] Failed to install LiDAR: {e}")
    

    try:
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '30')
        radar_bp.set_attribute('points_per_second', '1500')
        radar_transform = carla.Transform(carla.Location(x=2.5, z=0.5))
        radar = world.spawn_actor(radar_bp, radar_transform, attach_to=ego_vehicle)
        radar.listen(lambda data: sensor_callback(data, 'radar_front', RADAR_DIR))
        sensors.append(radar)
        print("  [OK] Radar installed")
    except Exception as e:
        print(f"  [ERROR] Failed to install radar: {e}")

    try:
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)
        collision_sensor.listen(collision_callback)
        sensors.append(collision_sensor)
        print("  [OK] Collision sensor installed")
    except Exception as e:
        print(f"  [ERROR] Failed to install collision sensor: {e}")
    
    print(f"\nInstalled {len(sensors)} sensors")
    return sensors

def spawn_npc_vehicles(world, blueprint_library, num_vehicles=30, min_distance=0.0):
    """Spawn NPC vehicles (kept static on-lane for controlled testing). 生成静态 NPC 车辆。"""
    spawn_points = world.get_map().get_spawn_points()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    

    vehicle_blueprints = [v for v in vehicle_blueprints if int(v.get_attribute('number_of_wheels')) == 4]
    
    spawned_vehicles = []
    num_spawned = 0
    used_points = []

    # Shuffle spawn points once to spread vehicles across the whole map
    random.shuffle(spawn_points)
    for spawn_point in spawn_points:
        if num_spawned >= num_vehicles:
            break
        if min_distance > 0.0:
            ok = True
            for p in used_points:
                if p.location.distance(spawn_point.location) < min_distance:
                    ok = False
                    break
            if not ok:
                continue
        vehicle_bp = random.choice(vehicle_blueprints)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            continue
        # Keep vehicles in-lane and static so they serve as obstacles for testing
        vehicle.set_autopilot(False)
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        spawned_vehicles.append(vehicle)
        num_spawned += 1
        used_points.append(spawn_point)
    
    print(f"[OK] Spawned {num_spawned} NPC vehicles")
    return spawned_vehicles


def move_block_npcs_in_front(world, ego_vehicle, npc_vehicles, start_distance=12.0, spacing=8.0):
    """Move existing NPCs to positions ahead of ego in same lane. 将已有NPC移到Ego前方同车道。"""
    if not npc_vehicles:
        return 0
    map_obj = world.get_map()
    ego_tf = ego_vehicle.get_transform()
    fwd = ego_tf.get_forward_vector()
    moved = 0
    for i, npc in enumerate(npc_vehicles):
        dist = start_distance + i * spacing
        center = ego_tf.location + fwd * dist
        wp = map_obj.get_waypoint(center, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        tf = wp.transform
        tf.location.z += 0.3
        try:
            npc.set_transform(tf)
            npc.set_autopilot(False)
            npc.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            moved += 1
        except Exception:
            continue
    if moved > 0:
        print(f"[OK] Moved {moved} NPCs ahead of ego")
    return moved


def clear_npcs_near_ego_corridor(ego_vehicle, npc_vehicles, forward_clear=85.0, back_clear=20.0, lateral_clear=8.0):
    """Remove NPCs near ego start corridor to keep lane-change demo space clear. 清理Ego起步走廊附近NPC。"""
    if not npc_vehicles:
        return npc_vehicles, 0
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    kept = []
    removed = []
    for v in npc_vehicles:
        try:
            loc = v.get_location()
        except Exception:
            kept.append(v)
            continue
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        dot = forward.x * dx + forward.y * dy
        lat = abs(-forward.y * dx + forward.x * dy)
        if (-back_clear <= dot <= forward_clear) and (lat <= lateral_clear):
            removed.append(v)
        else:
            kept.append(v)
    for v in removed:
        try:
            v.destroy()
        except Exception:
            pass
    if removed:
        print(f"[OK] Cleared {len(removed)} nearby NPCs around ego corridor")
    return kept, len(removed)


def spawn_blocking_npcs_ahead(world, blueprint_library, ego_vehicle, count=3, start_distance=15.0, spacing=12.0):
    """Spawn multiple static NPCs ahead in the ego lane. 在Ego车道前方生成多辆静态车。"""
    map_obj = world.get_map()
    ego_wp = map_obj.get_waypoint(ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        print("[WARN] No ego waypoint for blocking NPCs")
        return []
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    vehicle_blueprints = [v for v in vehicle_blueprints if int(v.get_attribute('number_of_wheels')) == 4]
    spawned = []
    for i in range(count):
        base_dist = start_distance + i * spacing
        placed = False
        # Try multiple distances if the spawn point is occupied
        for step in range(8):
            dist = base_dist + step * 2.0
            next_wps = ego_wp.next(dist)
            if not next_wps:
                continue
            # Keep blocker on ego lane id whenever possible.
            wp = None
            for cand in next_wps:
                if cand.road_id == ego_wp.road_id and cand.lane_id == ego_wp.lane_id:
                    wp = cand
                    break
            if wp is None:
                wp = next_wps[0]
            spawn_tf = wp.transform
            spawn_tf.location.z += 0.3
            bp = random.choice(vehicle_blueprints)
            npc = world.try_spawn_actor(bp, spawn_tf)
            if npc is None:
                continue
            try:
                npc.set_autopilot(False)
                npc.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            except Exception:
                pass
            spawned.append(npc)
            placed = True
            break
        if not placed:
            print(f"[WARN] Failed to place blocking NPC at ~{base_dist:.1f}m")
    if spawned:
        print(f"[OK] Spawned {len(spawned)} blocking NPCs ahead")
    return spawned


def spawn_npc_vehicles_off_route(world, blueprint_library, num_vehicles, route, min_distance=0.0):
    """Spawn NPC vehicles away from ego route lanes. 在非Ego路线车道生成NPC。"""
    if not route:
        return spawn_npc_vehicles(world, blueprint_library, num_vehicles, min_distance=min_distance)
    route_lanes = {(wp.road_id, wp.lane_id) for wp in route if wp is not None}
    spawn_points = world.get_map().get_spawn_points()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    vehicle_blueprints = [v for v in vehicle_blueprints if int(v.get_attribute('number_of_wheels')) == 4]
    safe_points = []
    map_obj = world.get_map()
    for sp in spawn_points:
        wp = map_obj.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        if (wp.road_id, wp.lane_id) in route_lanes:
            continue
        safe_points.append(sp)
    random.shuffle(safe_points)
    spawned_vehicles = []
    num_spawned = 0
    used_points = []
    for sp in safe_points:
        if num_spawned >= num_vehicles:
            break
        if min_distance > 0.0:
            ok = True
            for p in used_points:
                if p.location.distance(sp.location) < min_distance:
                    ok = False
                    break
            if not ok:
                continue
        vehicle_bp = random.choice(vehicle_blueprints)
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle is not None:
            vehicle.set_autopilot(False)
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            spawned_vehicles.append(vehicle)
            num_spawned += 1
            used_points.append(sp)
    print(f"[OK] Spawned {num_spawned} NPC vehicles off route")
    return spawned_vehicles


def spawn_blocking_npc(world, blueprint_library, ego_vehicle, distance=18.0):
    """Spawn a blocking NPC in ego lane ahead. 在Ego车道前方生成阻挡NPC。"""
    map_obj = world.get_map()
    ego_wp = map_obj.get_waypoint(ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        print("[WARN] No ego waypoint for blocking NPC")
        return None
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    vehicle_blueprints = [v for v in vehicle_blueprints if int(v.get_attribute('number_of_wheels')) == 4]
    for delta in [0.0, 5.0, 10.0]:
        next_wps = ego_wp.next(distance + delta)
        if not next_wps:
            continue
        wp = next_wps[0]
        vehicle_bp = random.choice(vehicle_blueprints)
        vehicle = world.try_spawn_actor(vehicle_bp, wp.transform)
        if vehicle is not None:
            vehicle.set_autopilot(False)
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            print(f"[OK] Blocking NPC spawned at {distance + delta:.1f}m ahead")
            return vehicle
    print("[WARN] Failed to spawn blocking NPC")
    return None


def remove_front_npc(world, ego_vehicle, npc_vehicles, max_distance=40.0, lane_half_width=2.0):
    """Remove closest NPC directly in front of ego. 删除Ego前方最近NPC。"""
    if not npc_vehicles:
        return None
    ego_tf = ego_vehicle.get_transform()
    forward = ego_tf.get_forward_vector()
    ego_loc = ego_tf.location
    best = None
    best_d = None
    for v in list(npc_vehicles):
        try:
            loc = v.get_location()
        except Exception:
            continue
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        dot = forward.x * dx + forward.y * dy
        if dot <= 0:
            continue
        lateral = abs(-forward.y * dx + forward.x * dy)
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_distance or lateral > lane_half_width:
            continue
        if best_d is None or dist < best_d:
            best_d = dist
            best = v
    if best is not None:
        try:
            best.destroy()
        except Exception:
            pass
        try:
            npc_vehicles.remove(best)
        except ValueError:
            pass
        print(f"[OK] Removed front NPC at {best_d:.1f}m")
        return best_d
    return None


def move_npcs_off_ego_lane(world, ego_vehicle, npc_vehicles):
    """Relocate NPCs away from ego lane to right/opposite lanes. 将NPC移出Ego车道。"""
    if not npc_vehicles:
        return 0
    moved = 0
    map_obj = world.get_map()
    ego_wp = map_obj.get_waypoint(ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return 0
    ego_lane_key = (ego_wp.road_id, ego_wp.lane_id)
    for v in list(npc_vehicles):
        try:
            v_wp = map_obj.get_waypoint(v.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception:
            continue
        if v_wp is None:
            continue
        if (v_wp.road_id, v_wp.lane_id) != ego_lane_key:
            continue
        target_wp = None
        right = v_wp.get_right_lane()
        left = v_wp.get_left_lane()
        if right and right.lane_type == carla.LaneType.Driving:
            target_wp = right
        elif left and left.lane_type == carla.LaneType.Driving:
            target_wp = left
        if target_wp is None:
            continue
        try:
            v.set_transform(target_wp.transform)
            v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            moved += 1
        except Exception:
            continue
    if moved:
        print(f"[OK] Moved {moved} NPC vehicles off ego lane")
    return moved


def move_npcs_off_route(world, route, npc_vehicles):
    """Relocate NPCs away from ego route lanes. 将NPC移出Ego路线车道。"""
    if not npc_vehicles or not route:
        return 0
    map_obj = world.get_map()
    route_lanes = {(wp.road_id, wp.lane_id) for wp in route if wp is not None}
    moved = 0
    for v in list(npc_vehicles):
        try:
            v_wp = map_obj.get_waypoint(v.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception:
            continue
        if v_wp is None:
            continue
        if (v_wp.road_id, v_wp.lane_id) not in route_lanes:
            continue
        target_wp = None
        right = v_wp.get_right_lane()
        left = v_wp.get_left_lane()
        if right and right.lane_type == carla.LaneType.Driving:
            target_wp = right
        elif left and left.lane_type == carla.LaneType.Driving:
            target_wp = left
        if target_wp is None:
            continue
        try:
            v.set_transform(target_wp.transform)
            v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            moved += 1
        except Exception:
            continue
    if moved:
        print(f"[OK] Moved {moved} NPC vehicles off ego route")
    return moved

def spawn_npc_walkers(world, blueprint_library, num_walkers=20):
    """Spawn NPC walkers. 生成 NPC 行人。"""
    walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
    spawn_points = []
    

    for i in range(num_walkers * 2):
        spawn_point = carla.Transform()
        spawn_point.location = world.get_random_location_from_navigation()
        if spawn_point.location is not None:
            spawn_points.append(spawn_point)
    
    spawned_walkers = []
    walker_controllers = []
    
    for i in range(min(num_walkers, len(spawn_points))):
        walker_bp = random.choice(walker_blueprints)
        spawn_point = spawn_points[i]
        
        walker = world.try_spawn_actor(walker_bp, spawn_point)
        if walker is not None:
            spawned_walkers.append(walker)
            

            controller_bp = blueprint_library.find('controller.ai.walker')
            controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
            if controller is not None:
                walker_controllers.append(controller)
    
    print(f"[OK] Spawned {len(spawned_walkers)} NPC walkers")
    return spawned_walkers, walker_controllers


def _find_nav_location_near(world, target, max_tries=20, max_dist=10.0):
    """Find a navigation location near target. 在目标点附近寻找可导航位置。"""
    best = None
    best_d = None
    for _ in range(max_tries):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        d = loc.distance(target)
        if d <= max_dist and (best_d is None or d < best_d):
            best = loc
            best_d = d
    return best


def _crossing_endpoints_from_wp(wp, lateral_ratio=0.6):
    """Compute crossing start/end across all driving lanes. 基于整条道路宽度生成横穿端点。"""
    lane_width = max(float(getattr(wp, "lane_width", 3.5)), 2.5)
    fwd = wp.transform.get_forward_vector()
    right = carla.Vector3D(x=fwd.y, y=-fwd.x, z=0.0)

    # Walk to rightmost/leftmost driving lanes
    rightmost = wp
    leftmost = wp
    cur = wp
    while True:
        nxt = cur.get_right_lane()
        if nxt is None or nxt.lane_type != carla.LaneType.Driving:
            break
        rightmost = nxt
        cur = nxt
    cur = wp
    while True:
        nxt = cur.get_left_lane()
        if nxt is None or nxt.lane_type != carla.LaneType.Driving:
            break
        leftmost = nxt
        cur = nxt

    # Extend to road edge from lane centers
    start_loc = rightmost.transform.location + right * (lane_width * lateral_ratio)
    end_loc = leftmost.transform.location - right * (lane_width * lateral_ratio)
    start_loc.z += 0.5
    end_loc.z += 0.5
    return start_loc, end_loc


def spawn_crossing_pedestrian(world, blueprint_library, ego_vehicle, distance=12.0, lateral=4.0, speed=1.4):
    """Spawn a pedestrian crossing in front of ego. 在Ego前方生成横穿行人。"""
    map_obj = world.get_map()
    ego_tf = ego_vehicle.get_transform()
    fwd = ego_tf.get_forward_vector()
    right = carla.Vector3D(x=fwd.y, y=-fwd.x, z=0.0)
    start_loc = None
    end_loc = None
    for d in [distance, max(4.0, distance - 2.0), max(4.0, distance - 4.0)]:
        center = ego_tf.location + fwd * d
        for lat in [abs(lateral), max(1.5, abs(lateral) * 0.7), max(1.5, abs(lateral) * 0.4)]:
            s = center + right * lat
            wp = map_obj.get_waypoint(s, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is not None:
                start_loc, end_loc = _crossing_endpoints_from_wp(wp, lateral_ratio=0.6)
                break
        if start_loc is not None and end_loc is not None:
            break
    if start_loc is None or end_loc is None:
        print("[WARN] No navigation location for crossing pedestrian")
        return None, None

    walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
    walker = world.try_spawn_actor(walker_bp, carla.Transform(start_loc))
    if walker is None:
        print("[WARN] Failed to spawn crossing pedestrian")
        return None, None

    try:
        direction = end_loc - start_loc
        length = max((direction.x ** 2 + direction.y ** 2 + direction.z ** 2) ** 0.5, 1e-3)
        direction = carla.Vector3D(direction.x / length, direction.y / length, 0.0)
        walker.apply_control(carla.WalkerControl(direction=direction, speed=speed))
        print(f"[OK] Crossing pedestrian spawned at ({start_loc.x:.1f},{start_loc.y:.1f}) -> ({end_loc.x:.1f},{end_loc.y:.1f})")
        CROSSING_SPAWN_INFO.append(("ped", walker.id, start_loc, "cross_start"))
        CROSSING_SPAWN_INFO.append(("ped", walker.id, end_loc, "cross_end"))
    except Exception as e:
        print(f"[WARN] Crossing pedestrian control failed: {e}")
    return walker, None


def spawn_crossing_pedestrians(world, blueprint_library, ego_vehicle, count=4, spacing=20.0, start_distance=6.0, lateral=3.0, speed=1.4):
    """Spawn multiple crossing pedestrians ahead along route. 在路线前方生成多名横穿行人。"""
    map_obj = world.get_map()
    ego_wp = map_obj.get_waypoint(ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        print("[WARN] No ego waypoint for crossing pedestrians")
        return []
    spawned = []
    for i in range(count):
        dist = start_distance + i * spacing
        next_wps = ego_wp.next(dist)
        if not next_wps:
            continue
        wp = next_wps[0]
        start_loc, end_loc = _crossing_endpoints_from_wp(wp, lateral_ratio=0.6)
        walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
        walker = world.try_spawn_actor(walker_bp, carla.Transform(start_loc))
        if walker is None:
            continue
        try:
            direction = end_loc - start_loc
            length = max((direction.x ** 2 + direction.y ** 2 + direction.z ** 2) ** 0.5, 1e-3)
            direction = carla.Vector3D(direction.x / length, direction.y / length, 0.0)
            walker.apply_control(carla.WalkerControl(direction=direction, speed=speed))
        except Exception:
            pass
        spawned.append((walker, start_loc, end_loc, speed))
        CROSSING_SPAWN_INFO.append(("ped", walker.id, start_loc, "cross_start"))
        CROSSING_SPAWN_INFO.append(("ped", walker.id, end_loc, "cross_end"))
    if spawned:
        print(f"[OK] Spawned {len(spawned)} crossing pedestrians")
    return spawned


def spawn_scattered_crossers(world, blueprint_library, count=10, min_dist=25.0, speed=1.4):
    """Spawn crossing pedestrians scattered across the map. 在地图各处生成横穿行人。"""
    spawned = []
    used = []
    map_obj = world.get_map()
    spawn_points = map_obj.get_spawn_points()
    random.shuffle(spawn_points)
    for sp in spawn_points:
        if len(spawned) >= count:
            break
        wp = map_obj.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        center = wp.transform.location
        if min_dist > 0:
            too_close = False
            for p in used:
                if p.distance(center) < min_dist:
                    too_close = True
                    break
            if too_close:
                continue
        start_loc, end_loc = _crossing_endpoints_from_wp(wp, lateral_ratio=0.6)
        walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
        walker = world.try_spawn_actor(walker_bp, carla.Transform(start_loc))
        if walker is None:
            continue
        try:
            direction = end_loc - start_loc
            length = max((direction.x ** 2 + direction.y ** 2 + direction.z ** 2) ** 0.5, 1e-3)
            direction = carla.Vector3D(direction.x / length, direction.y / length, 0.0)
            walker.apply_control(carla.WalkerControl(direction=direction, speed=speed))
        except Exception:
            pass
        spawned.append((walker, start_loc, end_loc, speed))
        used.append(center)
        CROSSING_SPAWN_INFO.append(("ped", walker.id, start_loc, "cross_start"))
        CROSSING_SPAWN_INFO.append(("ped", walker.id, end_loc, "cross_end"))
    if spawned:
        print(f"[OK] Spawned {len(spawned)} scattered crossers")
    return spawned


def get_front_ped_distance(ego_vehicle, walkers, max_distance=20.0, lane_half_width=2.5):
    """Return nearest pedestrian distance in front corridor. 获取前方行人最近距离。"""
    if not walkers:
        return None
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    best = None
    for w in walkers:
        try:
            loc = w.get_location()
        except Exception:
            continue
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        dot = forward.x * dx + forward.y * dy
        if dot <= 0:
            continue
        lateral = abs(-forward.y * dx + forward.x * dy)
        if lateral > lane_half_width:
            continue
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_distance:
            continue
        if best is None or dist < best:
            best = dist
    return best


def get_front_vehicle_distance(ego_vehicle, vehicles, max_distance=30.0, lane_half_width=2.0):
    """Return nearest vehicle distance in front corridor. 获取前方车辆最近距离。"""
    if not vehicles:
        return None
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    best = None
    for v in vehicles:
        try:
            loc = v.get_location()
        except Exception:
            continue
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        dot = forward.x * dx + forward.y * dy
        if dot <= 0:
            continue
        lateral = abs(-forward.y * dx + forward.x * dy)
        if lateral > lane_half_width:
            continue
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_distance:
            continue
        if best is None or dist < best:
            best = dist
    return best


def get_front_vehicle_info(ego_vehicle, vehicles, max_distance=40.0, lane_half_width=2.2):
    """Return nearest front vehicle info (distance/relative speed). 获取前车距离与相对速度。"""
    if not vehicles:
        return None
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    ego_vel = ego_vehicle.get_velocity()
    ego_vf = ego_vel.x * forward.x + ego_vel.y * forward.y
    best = None
    for v in vehicles:
        try:
            loc = v.get_location()
            vv = v.get_velocity()
        except Exception:
            continue
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        dot = forward.x * dx + forward.y * dy
        if dot <= 0:
            continue
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_distance:
            continue
        lateral = abs(-forward.y * dx + forward.x * dy)
        if lateral > lane_half_width:
            # Near-field fallback: do not lose a lead vehicle when ego is between lanes.
            near_dist = max(8.0, lane_half_width * 3.0)
            near_lat = max(lane_half_width + 0.8, lane_half_width * 1.45)
            if not (dist <= near_dist and lateral <= near_lat):
                continue
        lead_vf = vv.x * forward.x + vv.y * forward.y
        rel_speed = max(0.0, ego_vf - lead_vf)  # closing speed (m/s)
        cand = {
            "distance": dist,
            "lead_speed_kmh": max(0.0, lead_vf * 3.6),
            "rel_speed_kmh": rel_speed * 3.6,
            "vehicle": v,
        }
        if best is None or dist < best["distance"]:
            best = cand
    return best


def get_front_overlap_risk(ego_vehicle, vehicles, max_distance=14.0, extra_margin=0.6):
    """Return nearest front vehicle distance that can geometrically overlap in lane-change.
    返回并线阶段可能发生几何重叠的前车最近距离。
    """
    if not vehicles:
        return None
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    try:
        ego_half_w = float(getattr(ego_vehicle.bounding_box.extent, "y", 1.0))
    except Exception:
        ego_half_w = 1.0
    best = None
    for v in vehicles:
        try:
            loc = v.get_location()
            vh_half_w = float(getattr(v.bounding_box.extent, "y", 1.0))
        except Exception:
            continue
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        longitudinal = forward.x * dx + forward.y * dy
        if longitudinal <= 0.0:
            continue
        lateral = abs(-forward.y * dx + forward.x * dy)
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_distance:
            continue
        # If lateral gap is not larger than combined half-width + margin,
        # corner contact is still possible during unfinished lane change.
        lat_gate = ego_half_w + vh_half_w + extra_margin
        if lateral <= lat_gate:
            if best is None or dist < best:
                best = dist
    return best


def compute_follow_speed_kmh(base_speed_kmh, cur_speed_kmh, lead_info, min_gap=5.0, time_gap=1.2, slow_dist=28.0, min_speed_kmh=8.0):
    """Adaptive car-following speed target. 自适应跟驰速度目标。"""
    if lead_info is None:
        return base_speed_kmh
    dist = lead_info["distance"]
    lead_speed = lead_info["lead_speed_kmh"]
    closing = lead_info["rel_speed_kmh"]
    desired_gap = max(min_gap, min_gap + time_gap * (cur_speed_kmh / 3.6) + 0.2 * (closing / 3.6))
    if dist <= min_gap:
        return 0.0
    if dist <= desired_gap:
        return max(min_speed_kmh, min(base_speed_kmh, lead_speed))
    if dist < slow_dist:
        ratio = (dist - desired_gap) / max(slow_dist - desired_gap, 1e-3)
        ratio = max(0.0, min(1.0, ratio))
        follow_speed = max(min_speed_kmh, min(base_speed_kmh, lead_speed + 5.0))
        return follow_speed + ratio * (base_speed_kmh - follow_speed)
    return base_speed_kmh


def _lane_change_target(world, ego_vehicle, npc_vehicles, prefer="right", front_clear=15.0, back_clear=8.0, allow_outward=True):
    """Find adjacent lane target waypoint if clear. 查找相邻安全车道。"""
    map_obj = world.get_map()
    ego_wp = map_obj.get_waypoint(ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return None
    ego_yaw = ego_vehicle.get_transform().rotation.yaw
    candidates = []
    right = ego_wp.get_right_lane()
    left = ego_wp.get_left_lane()
    if prefer == "left":
        candidates = [left, right]
    else:
        candidates = [right, left]
    # Stability-first: prefer lane change toward road center to avoid curb-side merges.
    inward = []
    outward = []
    for c in candidates:
        if c is None:
            continue
        if ego_wp.lane_id != 0 and c.lane_id != 0 and (ego_wp.lane_id * c.lane_id) > 0:
            if abs(c.lane_id) < abs(ego_wp.lane_id):
                inward.append(c)
            elif abs(c.lane_id) > abs(ego_wp.lane_id):
                outward.append(c)
            else:
                inward.append(c)
        else:
            outward.append(c)
    if inward:
        candidates = inward + ([] if not allow_outward else outward)
    elif not allow_outward:
        candidates = []
    for target_wp in candidates:
        if target_wp is None or target_wp.lane_type != carla.LaneType.Driving:
            continue
        if not allow_outward and abs(target_wp.lane_id) > abs(ego_wp.lane_id):
            continue
        if target_wp.is_junction:
            continue
        if float(getattr(target_wp, "lane_width", 3.5)) < 2.8:
            continue
        # Must be same road and same direction (lane_id sign) to avoid U-turns.
        if target_wp.road_id != ego_wp.road_id:
            continue
        if ego_wp.lane_id == 0 or target_wp.lane_id == 0:
            continue
        if ego_wp.lane_id * target_wp.lane_id < 0:
            continue
        # Yaw alignment check to ensure same direction.
        yaw_diff = (target_wp.transform.rotation.yaw - ego_yaw + 180.0) % 360.0 - 180.0
        if abs(yaw_diff) > 45.0:
            continue
        # Continuity check: avoid side lanes that end immediately.
        probe = target_wp
        continuity_ok = True
        for _ in range(8):
            nxt = probe.next(8.0)
            if not nxt:
                continuity_ok = False
                break
            probe = min(
                nxt,
                key=lambda w: abs((w.transform.rotation.yaw - ego_yaw + 180.0) % 360.0 - 180.0),
            )
            if probe.lane_type != carla.LaneType.Driving:
                continuity_ok = False
                break
            if probe.is_junction:
                continuity_ok = False
                break
            if probe.lane_id == 0 or target_wp.lane_id == 0 or probe.lane_id * target_wp.lane_id < 0:
                continuity_ok = False
                break
            if float(getattr(probe, "lane_width", 3.5)) < 2.6:
                continuity_ok = False
                break
        if not continuity_ok:
            continue
        # Aim a bit ahead in the target lane so we don't stall at lateral-only offset.
        ahead_wps = target_wp.next(8.0)
        if ahead_wps:
            target_wp = ahead_wps[0]
        target_lane = (target_wp.road_id, target_wp.lane_id)
        clear = True
        for v in npc_vehicles:
            try:
                v_wp = map_obj.get_waypoint(v.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            except Exception:
                continue
            if v_wp is None:
                continue
            if (v_wp.road_id, v_wp.lane_id) != target_lane:
                continue
            # relative position to ego
            ego_tf = ego_vehicle.get_transform()
            loc = v.get_location()
            dx = loc.x - ego_tf.location.x
            dy = loc.y - ego_tf.location.y
            forward = ego_tf.get_forward_vector()
            dot = forward.x * dx + forward.y * dy
            if -back_clear <= dot <= front_clear:
                clear = False
                break
        if clear:
            return target_wp
    return None


def _find_target_lane_wp(ego_wp, target_lane_key):
    """Find target lane waypoint near ego by scanning adjacent lanes. 在Ego附近查找目标车道。"""
    if ego_wp is None or target_lane_key is None:
        return None
    if (ego_wp.road_id, ego_wp.lane_id) == target_lane_key:
        return ego_wp
    candidates = [ego_wp]
    cur = ego_wp
    for _ in range(4):
        nxt = cur.get_left_lane()
        if nxt is None or nxt.lane_type != carla.LaneType.Driving or nxt.road_id != ego_wp.road_id:
            break
        candidates.append(nxt)
        cur = nxt
    cur = ego_wp
    for _ in range(4):
        nxt = cur.get_right_lane()
        if nxt is None or nxt.lane_type != carla.LaneType.Driving or nxt.road_id != ego_wp.road_id:
            break
        candidates.append(nxt)
        cur = nxt
    for wp in candidates:
        if (wp.road_id, wp.lane_id) == target_lane_key:
            return wp
    return None


def _select_stable_lane_wp(map_obj, vehicle_loc, lane_key=None, stick_offset=2.15):
    """Select a stable lane waypoint to avoid lane-flip oscillation. 选择稳定参考车道避免左右跳车道。"""
    nearest_wp = map_obj.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if nearest_wp is None:
        return None, lane_key
    if lane_key is None:
        return nearest_wp, (nearest_wp.road_id, nearest_wp.lane_id)
    pref_wp = _find_target_lane_wp(nearest_wp, lane_key)
    if pref_wp is not None:
        pref_off = _lane_center_offset_abs(vehicle_loc, pref_wp)
        nearest_off = _lane_center_offset_abs(vehicle_loc, nearest_wp)
        lane_width = max(2.6, float(getattr(pref_wp, "lane_width", 3.5)))
        # Keep lane key to avoid oscillation, but do not keep stale lane too long.
        rebind_threshold = max(float(stick_offset), lane_width * 0.95)
        if pref_off <= rebind_threshold:
            # If nearest lane is clearly better aligned, rebind immediately.
            if nearest_off + 0.85 < pref_off:
                return nearest_wp, (nearest_wp.road_id, nearest_wp.lane_id)
            return pref_wp, lane_key
    # Rebind only when previous lane center is too far.
    return nearest_wp, (nearest_wp.road_id, nearest_wp.lane_id)


def _lane_change_guidance_wp(world, ego_vehicle, target_lane_key, lookahead=10.0):
    """Build a moving lane-change guidance waypoint on target lane. 构建动态变道引导点。"""
    map_obj = world.get_map()
    ego_wp = map_obj.get_waypoint(ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    lane_wp = _find_target_lane_wp(ego_wp, target_lane_key)
    if lane_wp is None:
        return None
    nxt = lane_wp.next(max(6.0, float(lookahead)))
    return nxt[0] if nxt else lane_wp


def _lane_center_offset_abs(vehicle_loc, lane_wp):
    """Absolute lateral offset from lane center (m). 与车道中心横向偏差绝对值。"""
    if lane_wp is None:
        return 999.0
    dx = vehicle_loc.x - lane_wp.transform.location.x
    dy = vehicle_loc.y - lane_wp.transform.location.y
    yaw = np.deg2rad(lane_wp.transform.rotation.yaw)
    lat = -dx * np.sin(yaw) + dy * np.cos(yaw)
    return abs(float(lat))


def _lane_center_offset_signed(vehicle_loc, lane_wp):
    """Signed lateral offset from lane center (m). 与车道中心横向偏差（有符号）。"""
    if lane_wp is None:
        return 0.0
    dx = vehicle_loc.x - lane_wp.transform.location.x
    dy = vehicle_loc.y - lane_wp.transform.location.y
    yaw = np.deg2rad(lane_wp.transform.rotation.yaw)
    lat = -dx * np.sin(yaw) + dy * np.cos(yaw)
    return float(lat)


def _yaw_error_abs_deg(yaw_vehicle, yaw_lane):
    """Absolute yaw error in degrees. 车辆与车道方向夹角绝对值。"""
    err = (yaw_lane - yaw_vehicle + 180.0) % 360.0 - 180.0
    return abs(float(err))


def _control_to_waypoint(vehicle, target_wp, target_speed_kmh, max_steer=0.35):
    """Steer toward a target waypoint (lane-change). 朝目标路点转向。"""
    tf = vehicle.get_transform()
    v = vehicle.get_velocity()
    speed = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5 * 3.6
    dx = target_wp.transform.location.x - tf.location.x
    dy = target_wp.transform.location.y - tf.location.y
    yaw = np.deg2rad(tf.rotation.yaw)
    x_local = dx * np.cos(-yaw) - dy * np.sin(-yaw)
    y_local = dx * np.sin(-yaw) + dy * np.cos(-yaw)
    # If target is behind, recover with forward crawl (no brake), otherwise car can stall.
    if x_local < 1.0:
        if abs(y_local) < 0.25:
            recover_steer = 0.0
        else:
            recover_steer = max_steer * (0.75 if y_local >= 0.0 else -0.75)
        return carla.VehicleControl(throttle=0.35, steer=float(recover_steer), brake=0.0)
    ld2 = max(x_local**2 + y_local**2, 1e-6)
    steer = np.arctan2(2.0 * y_local, ld2**0.5)
    steer = max(-max_steer, min(max_steer, steer))
    if speed < target_speed_kmh - 5:
        throttle = 0.45
        brake = 0.0
    elif speed > target_speed_kmh + 5:
        throttle = 0.0
        brake = 0.35
    else:
        throttle = 0.25
        brake = 0.0
    # If steering is large, reduce throttle to avoid sliding into poles
    if abs(steer) > 0.25:
        throttle = min(throttle, 0.25)
        if speed > 4.0:
            brake = max(brake, 0.05)
    if speed < 2.0 and target_speed_kmh > 0.1:
        throttle = max(throttle, 0.35)
        brake = 0.0
    return carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))


def simple_lane_follow_control(world, vehicle, target_speed_kmh=20.0):
    """A lightweight lane-follow controller for ego to drive a loop. 轻量级车道跟随。"""
    map_obj = world.get_map()
    transform = vehicle.get_transform()
    waypoint = map_obj.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if waypoint is None:
        return carla.VehicleControl(throttle=0.0, brake=1.0)

    next_wp = waypoint.next(4.0)[0] if waypoint.next(4.0) else waypoint
    v = vehicle.get_velocity()
    speed = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5 * 3.6  # km/h

    # Direction steering
    forward = transform.get_forward_vector()
    to_target = next_wp.transform.location - transform.location
    dot = forward.x * to_target.x + forward.y * to_target.y
    cross = forward.x * to_target.y - forward.y * to_target.x
    steer = max(-1.0, min(1.0, 0.05 * cross))

    # Speed control
    throttle = 0.5 if speed < target_speed_kmh else 0.1
    brake = 0.0
    if speed > target_speed_kmh + 5:
        throttle = 0.0
        brake = 0.4

    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


def build_route(world, vehicle, look_ahead=400, step=4.0):
    """Build a closed loop route starting from current vehicle location. 构建前向路线。"""
    route = []
    map_obj = world.get_map()
    wp = map_obj.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return route
    route.append(wp)
    for _ in range(look_ahead):
        nxt = route[-1].next(step)
        if not nxt:
            break
        wp_next = nxt[0]
        route.append(wp_next)
    return route


def _next_same_lane(wp, step=10.0):
    """Advance waypoint while preferring same lane id. 前进时优先保持同一车道。"""
    nxt = wp.next(step)
    if not nxt:
        return None
    for cand in nxt:
        if cand.road_id == wp.road_id and cand.lane_id == wp.lane_id:
            return cand
    return nxt[0]


def _adjacent_ok(wp, prefer="right"):
    """Check whether adjacent lane exists and has same driving direction. 检查相邻同向车道。"""
    if wp is None:
        return False
    sides = ["right", "left"] if prefer == "right" else ["left", "right"]
    for side in sides:
        adj = wp.get_right_lane() if side == "right" else wp.get_left_lane()
        if adj is None or adj.lane_type != carla.LaneType.Driving:
            continue
        if adj.road_id != wp.road_id:
            continue
        if wp.lane_id == 0 or adj.lane_id == 0:
            continue
        if wp.lane_id * adj.lane_id < 0:
            continue
        return True
    return False


def get_multilane_spawn_points(world, prefer="right", min_forward=35.0, stable_forward=70.0, step=10.0):
    """Return spawn points with a stable adjacent lane for future overtaking. 返回前方连续可变道出生点。"""
    map_obj = world.get_map()
    points = []
    all_spawns = map_obj.get_spawn_points()
    for sp in all_spawns:
        wp = map_obj.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        if wp.is_junction:
            continue
        # Need enough forward road so blocker and lane change are both testable.
        if not wp.next(min_forward):
            continue
        if not _adjacent_ok(wp, prefer=prefer):
            continue
        cur = wp
        ok = True
        covered = 0.0
        while covered < stable_forward:
            if cur is None or cur.is_junction:
                ok = False
                break
            if not _adjacent_ok(cur, prefer=prefer):
                ok = False
                break
            cur = _next_same_lane(cur, step=step)
            covered += step
        if not ok:
            continue
        points.append(sp)
    return points


def _pick_far_spawn_point(map_obj, origin_loc, min_dist=150.0):
    """Pick a spawn point far from origin. 选择远处出生点作为目标。"""
    spawn_points = map_obj.get_spawn_points()
    if not spawn_points:
        return None
    best = None
    best_d = -1.0
    candidates = []
    for sp in spawn_points:
        d = sp.location.distance(origin_loc)
        if d > min_dist:
            candidates.append(sp)
        if d > best_d:
            best = sp
            best_d = d
    if candidates:
        return random.choice(candidates)
    return best


def build_random_route(world, start_loc, look_ahead=1200, step=4.0):
    """Build a long route by randomly choosing branches. 随机分支生成长路线。"""
    route = []
    map_obj = world.get_map()
    wp = map_obj.get_waypoint(start_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return route
    route.append(wp)
    for _ in range(look_ahead):
        nxt = route[-1].next(step)
        if not nxt:
            break
        # Choose a random branch to avoid small loops
        wp_next = random.choice(nxt)
        route.append(wp_next)
    return route


def build_global_route(world, start_loc, min_dist=150.0, resolution=2.0):
    """Build a long global route using GlobalRoutePlanner. 使用全局规划生成长路线。"""
    if GlobalRoutePlanner is None:
        print("[WARN] GlobalRoutePlanner not available, fallback to random route")
        return build_random_route(world, start_loc, look_ahead=1200, step=4.0)
    map_obj = world.get_map()
    target_sp = _pick_far_spawn_point(map_obj, start_loc, min_dist=min_dist)
    if target_sp is None:
        return []
    try:
        grp = GlobalRoutePlanner(map_obj, resolution)
        grp.setup()
        route = grp.trace_route(start_loc, target_sp.location)
        wps = [wp for (wp, _opt) in route if wp is not None]
        return wps
    except Exception as e:
        print(f"[WARN] Global route build failed: {e}, fallback to random route")
        return build_random_route(world, start_loc, look_ahead=1200, step=4.0)


def _signed_yaw_delta_deg(yaw_from, yaw_to):
    """Signed yaw delta in degrees, normalized to [-180, 180]."""
    return float((float(yaw_to) - float(yaw_from) + 180.0) % 360.0 - 180.0)


def advanced_lane_follow_control(
    map_obj,
    vehicle,
    target_speed_kmh=50.0,
    lookahead=8.0,
    lane_wp_hint=None,
    junction_mode=False,
    turn_pref=0.0,
):
    """Stanley-like lane keeping with curvature-aware speed. 斯坦利式车道保持+弯道降速。"""
    wp = lane_wp_hint if lane_wp_hint is not None else map_obj.get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return carla.VehicleControl(throttle=0.0, brake=1.0)

    tf = vehicle.get_transform()
    vel = vehicle.get_velocity()
    speed_kmh = (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5 * 3.6
    speed_ms = max(speed_kmh / 3.6, 0.1)

    dyn_lookahead = max(6.0, min(22.0, lookahead + speed_kmh * 0.10))
    if junction_mode or wp.is_junction:
        dyn_lookahead = min(dyn_lookahead, max(6.0, lookahead))
    target_wps = wp.next(dyn_lookahead)
    target_wp = wp
    if target_wps:
        yaw_v = tf.rotation.yaw
        best_wp = target_wps[0]
        best_score = 1e9
        for cand in target_wps:
            heading_signed = _signed_yaw_delta_deg(yaw_v, cand.transform.rotation.yaw)
            heading_err = abs(heading_signed)
            curv_pen = 0.0
            nxt = cand.next(6.0)
            if nxt:
                curv_pen = min(abs(_signed_yaw_delta_deg(cand.transform.rotation.yaw, n.transform.rotation.yaw)) for n in nxt)
            score = heading_err + 0.35 * curv_pen
            if cand.is_junction:
                score += (-1.0 if junction_mode else 2.5)
            if abs(float(turn_pref)) > 0.2:
                if heading_signed * float(turn_pref) < -2.0:
                    score += min(10.0, 0.55 * (abs(heading_signed) + 2.0))
                elif heading_signed * float(turn_pref) > 6.0:
                    score -= min(2.5, 0.10 * abs(heading_signed))
                if junction_mode and abs(heading_signed) < 4.0:
                    score += 3.5
            if score < best_score:
                best_score = score
                best_wp = cand
        target_wp = best_wp

    # Heading error: blend near tangent and lookahead tangent.
    yaw_v = tf.rotation.yaw
    heading_near = _signed_yaw_delta_deg(yaw_v, wp.transform.rotation.yaw)
    heading_far = _signed_yaw_delta_deg(yaw_v, target_wp.transform.rotation.yaw)
    far_weight = 0.65 if junction_mode else 0.35
    heading_err = np.deg2rad((1.0 - far_weight) * heading_near + far_weight * heading_far)

    # Cross-track error in lane frame (+left, -right)
    dx_lane = tf.location.x - wp.transform.location.x
    dy_lane = tf.location.y - wp.transform.location.y
    yaw_lane = np.deg2rad(wp.transform.rotation.yaw)
    cte = -dx_lane * np.sin(yaw_lane) + dy_lane * np.cos(yaw_lane)

    # Stanley steering
    k_cte = 0.9
    k_heading = 1.0
    steer = k_heading * heading_err + np.arctan2(k_cte * cte, speed_ms + 1.0)

    # Small lookahead pull for smoother turns
    yaw = np.deg2rad(tf.rotation.yaw)
    dx = target_wp.transform.location.x - tf.location.x
    dy = target_wp.transform.location.y - tf.location.y
    x_local = dx * np.cos(-yaw) - dy * np.sin(-yaw)
    y_local = dx * np.sin(-yaw) + dy * np.cos(-yaw)
    ld2 = max(x_local**2 + y_local**2, 1e-6)
    lookahead_gain = 0.34 if junction_mode else 0.25
    steer += lookahead_gain * np.arctan2(2.0 * y_local, ld2**0.5)
    steer = max(-0.6, min(0.6, steer))

    # Curvature-aware speed target
    curve_speed = target_speed_kmh
    try:
        w1 = wp.next(8.0)[0] if wp.next(8.0) else wp
        w2 = wp.next(18.0)[0] if wp.next(18.0) else w1
        yaw1 = w1.transform.rotation.yaw
        yaw2 = w2.transform.rotation.yaw
        dyaw = abs((yaw2 - yaw1 + 180.0) % 360.0 - 180.0)
        curvature = dyaw / 10.0  # deg per ~10m
        if curvature > 8.0:
            curve_speed = min(curve_speed, 20.0)
        elif curvature > 4.0:
            curve_speed = min(curve_speed, 30.0)
        elif curvature > 2.0:
            curve_speed = min(curve_speed, 40.0)
    except Exception:
        pass
    if wp.is_junction or junction_mode:
        curve_speed = min(curve_speed, 20.0 if junction_mode else 22.0)

    # Longitudinal control
    target_speed_kmh = min(target_speed_kmh, curve_speed)
    if speed_kmh < target_speed_kmh - 8:
        throttle = 0.75
        brake = 0.0
    elif speed_kmh < target_speed_kmh - 2:
        throttle = 0.55
        brake = 0.0
    elif speed_kmh > target_speed_kmh + 6:
        throttle = 0.0
        brake = 0.30
    else:
        throttle = 0.30
        brake = 0.0

    if speed_kmh < 2.0:
        throttle = max(throttle, 0.45)
        brake = 0.0

    if abs(steer) > 0.35:
        throttle = min(throttle, 0.25)
        brake = max(brake, 0.06)

    return carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))


def estimate_upcoming_turn_signed_deg(map_obj, vehicle, d1=8.0, d2=20.0):
    """Estimate signed near-future turn angle using waypoint yaw delta. 估计前方弯道方向与强度（有符号）。"""
    try:
        wp0 = map_obj.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp0 is None:
            return 0.0
        w1s = wp0.next(max(4.0, float(d1)))
        w2s = wp0.next(max(8.0, float(d2)))
        if (not w1s) or (not w2s):
            return 0.0
        yaw1 = float(w1s[0].transform.rotation.yaw)
        yaw2 = float(w2s[0].transform.rotation.yaw)
        return _signed_yaw_delta_deg(yaw1, yaw2)
    except Exception:
        return 0.0


def estimate_upcoming_turn_deg(map_obj, vehicle, d1=8.0, d2=20.0):
    """Estimate near-future turn intensity using waypoint yaw delta. 估计前方弯道强度。"""
    return abs(estimate_upcoming_turn_signed_deg(map_obj, vehicle, d1=d1, d2=d2))

def main():
    parser = argparse.ArgumentParser(description="CARLA autonomous driving and multimodal data collection test. 自动驾驶与多模态采集。")
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--num-vehicles', type=int, default=30, help='Number of NPC vehicles')
    parser.add_argument('--num-walkers', type=int, default=20, help='Number of NPC walkers')
    parser.add_argument('--sync', action='store_true', help='Enable synchronous mode')
    parser.add_argument('--autopilot', action='store_true', help='Enable autopilot for ego vehicle')
    parser.add_argument('--npc-autopilot', action='store_true', help='Enable autopilot for NPC vehicles')
    parser.add_argument('--walker-autopilot', action='store_true', help='Enable autopilot for NPC walkers')
    parser.add_argument('--map', default=None, help='Map name, e.g. Town01, Town10')
    parser.add_argument('--duration', type=int, default=None, help='Run duration in seconds (optional)')
    parser.add_argument('--delta-time', type=float, default=0.02, help='Fixed delta seconds in sync mode')
    parser.add_argument('--ego-spawn-index', type=int, default=-1, help='Fixed ego spawn point index (>=0 to use)')
    parser.add_argument('--list-spawn-points', action='store_true', help='List spawn point indices and coordinates, then exit')
    parser.add_argument('--no-show', action='store_true', help='Disable live sensor visuals (default: enabled)')
    parser.add_argument('--no-save', action='store_true', help='Disable disk saving to test performance')
    parser.add_argument('--ego-loop', action='store_true', help='Enable simple lane-follow loop for ego vehicle')
    parser.add_argument('--ego-adv', action='store_true', help='Enable advanced Stanley/PID lane-follow controller')
    parser.add_argument('--target-speed', type=float, default=70.0, help='Target speed for ego (km/h)')
    parser.add_argument('--save-stride', type=int, default=3, help='Save every Nth frame to reduce IO stutter (每隔N帧写盘)')
    parser.add_argument('--vis-stride', type=int, default=3, help='Update visualization every N frames to reduce load (每隔N帧刷新显示)')
    parser.add_argument('--hud', action='store_true', help='Show overlay HUD (FPS/speed/frame) on display')
    parser.add_argument('--hazard-dist', type=float, default=6.0, help='Obstacle distance threshold for slowing (m)')
    parser.add_argument('--hazard-speed', type=float, default=30.0, help='Target speed when obstacle detected (km/h)')
    parser.add_argument('--remove-front-npc', action='store_true', help='Remove nearest NPC in front of ego')
    parser.add_argument('--no-view-windows', action='store_true', help='Disable separate FP/TP view windows')
    parser.add_argument('--view-width', type=int, default=1280, help='FP/TP window width')
    parser.add_argument('--view-height', type=int, default=720, help='FP/TP window height')
    parser.add_argument('--view-fullscreen', action='store_true', help='Fullscreen FP/TP windows')
    parser.add_argument('--hud-pos-x', type=int, default=20, help='HUD window position X')
    parser.add_argument('--hud-pos-y', type=int, default=20, help='HUD window position Y')
    parser.add_argument('--fp-pos-x', type=int, default=980, help='First-person window position X')
    parser.add_argument('--fp-pos-y', type=int, default=20, help='First-person window position Y')
    parser.add_argument('--tp-pos-x', type=int, default=980, help='Third-person window position X')
    parser.add_argument('--tp-pos-y', type=int, default=520, help='Third-person window position Y')
    parser.add_argument('--npc-off-route', action='store_true', help='Move NPCs off ego lane')
    parser.add_argument('--block-npc', action='store_true', help='Spawn blocking NPCs ahead in ego lane')
    parser.add_argument('--block-npc-distance', type=float, default=18.0, help='Blocking NPC start distance ahead (m)')
    parser.add_argument('--block-npc-count', type=int, default=3, help='Number of blocking NPCs ahead')
    parser.add_argument('--block-npc-spacing', type=float, default=12.0, help='Spacing between blocking NPCs (m)')
    parser.add_argument('--block-npc-relocate', action='store_true', help='Relocate existing NPCs ahead if block spawn fails')
    parser.add_argument('--crossing-ped', action='store_true', help='Spawn a crossing pedestrian ahead')
    parser.add_argument('--ped-distance', type=float, default=8.0, help='Crossing pedestrian distance ahead (m)')
    parser.add_argument('--ped-lateral', type=float, default=3.0, help='Crossing pedestrian lateral offset (m)')
    parser.add_argument('--ped-speed', type=float, default=1.4, help='Crossing pedestrian speed (m/s)')
    parser.add_argument('--ped-detect-dist', type=float, default=18.0, help='Pedestrian detect distance (m)')
    parser.add_argument('--ped-stop-dist', type=float, default=8.0, help='Pedestrian stop distance (m)')
    parser.add_argument('--ped-lane-half-width', type=float, default=4.0, help='Pedestrian detection corridor half width (m)')
    parser.add_argument('--crossing-ped-count', type=int, default=1, help='Number of crossing pedestrians')
    parser.add_argument('--crossing-ped-spacing', type=float, default=20.0, help='Spacing between crossing pedestrians (m)')
    parser.add_argument('--crossing-ped-loop', action='store_true', help='Loop crossing pedestrians back and forth')
    parser.add_argument('--ped-spawn-interval', type=float, default=0.0, help='Spawn a crossing pedestrian every N seconds (0=off)')
    parser.add_argument('--ped-max-active', type=int, default=8, help='Max active crossing pedestrians')
    parser.add_argument('--crossing-scatter-count', type=int, default=0, help='Scattered crossing pedestrians across map')
    parser.add_argument('--crossing-scatter-min-dist', type=float, default=25.0, help='Min distance between scattered crossers (m)')
    parser.add_argument('--npc-min-distance', type=float, default=0.0, help='Min distance between NPC vehicles (m)')
    parser.add_argument('--ped-lane-change', action='store_true', help='Change lane to avoid pedestrian if possible')
    parser.add_argument('--npc-lane-change', action='store_true', help='Change lane to avoid blocking NPC')
    parser.add_argument('--npc-change-dist', type=float, default=16.0, help='NPC lane-change trigger distance (m)')
    parser.add_argument('--npc-stop-dist', type=float, default=8.0, help='NPC stop distance if no lane change (m)')
    parser.add_argument('--npc-slow-dist', type=float, default=28.0, help='NPC slow-down distance (m)')
    parser.add_argument('--npc-min-speed', type=float, default=8.0, help='Min crawl speed near lead vehicle (km/h)')
    parser.add_argument('--follow-time-gap', type=float, default=1.2, help='Time headway for following (s)')
    parser.add_argument('--follow-min-gap', type=float, default=5.0, help='Minimum standstill gap to lead vehicle (m)')
    parser.add_argument('--lane-change-closing', type=float, default=4.0, help='Min closing speed to trigger NPC lane change (km/h)')
    parser.add_argument('--lane-change-min-speed', type=float, default=8.0, help='Min speed (km/h) to allow lane change')
    parser.add_argument('--lane-change-duration', type=float, default=2.5, help='Lane change duration (s)')
    parser.add_argument('--lane-change-max-duration', type=float, default=5.0, help='Lane change max hold time (s)')
    parser.add_argument('--lane-change-speed', type=float, default=18.0, help='Target speed during lane change (km/h)')
    parser.add_argument('--lane-change-steer', type=float, default=0.45, help='Max steering clamp during lane change')
    parser.add_argument('--lane-change-lookahead', type=float, default=10.0, help='Guidance waypoint lookahead during lane change (m)')
    parser.add_argument('--lane-change-hold', type=float, default=4.0, help='Hold time before allowing another lane change (s)')
    parser.add_argument('--lane-change-center-tol', type=float, default=0.40, help='Lane-change completion lateral tolerance (m)')
    parser.add_argument('--lane-change-yaw-tol', type=float, default=12.0, help='Lane-change completion yaw tolerance (deg)')
    parser.add_argument('--lane-change-fail-cooldown', type=float, default=2.6, help='Base cooldown after failed lane change (s)')
    parser.add_argument('--lane-change-progress-timeout', type=float, default=2.8, help='Abort lane change if no lateral progress after this time (s)')
    parser.add_argument('--lane-change-min-progress', type=float, default=0.15, help='Required lateral progress to keep lane change active (m)')
    parser.add_argument('--lane-change-prefer', choices=['right', 'left'], default='right', help='Preferred lane change direction')
    parser.add_argument('--lane-change-front-clear', type=float, default=15.0, help='Min front clear distance for lane change (m)')
    parser.add_argument('--lane-change-back-clear', type=float, default=8.0, help='Min back clear distance for lane change (m)')
    parser.add_argument('--lane-stick-offset', type=float, default=2.15, help='Lane-stick max center offset before lane rebind (m)')
    parser.add_argument('--global-route', action='store_true', help='Use long global route planning (disable to avoid sharp turns)')
    parser.add_argument('--max-steer', type=float, default=0.45, help='Max steering magnitude clamp')
    parser.add_argument('--steer-rate', type=float, default=0.08, help='Max steering change per tick')
    parser.add_argument('--fixed-scenario', action='store_true', help='Fixed demo: block NPC + crossing pedestrian near ego')
    parser.add_argument('--stop-on-collision', action='store_true', help='Stop run immediately when ego collides')
    parser.add_argument('--bf-enable', action='store_true', help='Enable BeamFusion closed-loop policy')
    parser.add_argument('--bf-src', default=r'E:\6G\Code\src', help='BeamFusion source root')
    parser.add_argument('--bf-config', default=r'E:\6G\Code\outputs\ablation_a6_a7\ccfa_auto_v1_a6\03_a6_score_weighted_mean_s2028\config.json', help='BeamFusion config json')
    parser.add_argument('--bf-checkpoint', default=r'E:\6G\Code\outputs\ablation_a6_a7\ccfa_auto_v1_a6\03_a6_score_weighted_mean_s2028\checkpoints\best.pt', help='BeamFusion checkpoint path')
    parser.add_argument('--bf-device', default='', help='BeamFusion device override, e.g. cuda or cpu')
    parser.add_argument('--bf-topk', type=int, default=5, help='BeamFusion top-k outputs')
    parser.add_argument('--bf-infer-hz', type=float, default=2.0, help='BeamFusion inference frequency')
    parser.add_argument('--bf-image-w', type=int, default=960, help='BeamFusion camera width')
    parser.add_argument('--bf-image-h', type=int, default=540, help='BeamFusion camera height')
    parser.add_argument('--bf-speed-min-scale', type=float, default=0.55, help='Minimum speed scale under high risk')
    parser.add_argument('--bf-speed-max-scale', type=float, default=1.0, help='Maximum speed scale under low risk')
    parser.add_argument('--bf-risk-speed-gain', type=float, default=0.45, help='Risk-to-speed attenuation gain')
    parser.add_argument('--bf-entropy-weight', type=float, default=0.35, help='Entropy contribution to risk')
    parser.add_argument('--bf-gate-weight', type=float, default=0.30, help='Gate uncertainty contribution to risk')
    parser.add_argument('--bf-lane-change-min-conf', type=float, default=0.45, help='Min confidence to allow lane change')
    parser.add_argument('--bf-lane-change-risk-th', type=float, default=0.82, help='Max risk to allow lane change')
    parser.add_argument('--bf-dominant-level', choices=['off', 'assist', 'strong'], default='assist', help='How strongly BeamFusion modulates control policy')
    parser.add_argument('--bf-steer-boost-max', type=float, default=0.10, help='Max extra steer authority when beam risk is low')
    parser.add_argument('--bf-steer-rate-boost', type=float, default=0.05, help='Max extra steer-rate authority when beam risk is low')
    parser.add_argument('--bf-turn-sharp-deg', type=float, default=10.0, help='Sharp-turn threshold in degrees')
    parser.add_argument('--bf-turn-speed-cap', type=float, default=22.0, help='Speed cap for sharp turns')
    parser.add_argument('--bf-turn-steer-boost', type=float, default=0.08, help='Additional steer cap boost on sharp turns')
    parser.add_argument('--bf-turn-rate-boost', type=float, default=0.04, help='Additional steer-rate boost on sharp turns')
    parser.add_argument('--bf-force-overtake-wait', type=float, default=2.5, help='Seconds blocked by front NPC before forcing lane-change attempt')
    parser.add_argument('--bf-overtake-front-relax', type=float, default=0.70, help='Front-clear ratio when forced overtake is active')
    parser.add_argument('--bf-overtake-back-relax', type=float, default=0.70, help='Back-clear ratio when forced overtake is active')
    parser.add_argument('--lv-disable', action='store_true', help='Disable visual lane assist')
    parser.add_argument('--lv-conf-min', type=float, default=0.22, help='Minimum lane-vision confidence to affect control')
    parser.add_argument('--lv-weight-max', type=float, default=0.45, help='Maximum fusion weight for lane-vision offset')
    parser.add_argument('--lv-edge-ratio-th', type=float, default=0.88, help='Edge ratio threshold to activate hard curb guard')
    parser.add_argument('--lv-curb-steer-k', type=float, default=0.26, help='Hard curb guard steering correction gain')
    parser.add_argument('--junction-turn-disable', action='store_true', help='Disable dedicated no-route junction turn optimization')
    parser.add_argument('--junction-turn-deg', type=float, default=18.0, help='Turn degree threshold to enable no-route turn mode')
    parser.add_argument('--junction-speed-cap', type=float, default=18.0, help='Speed cap in no-route junction turn mode')
    parser.add_argument('--junction-lookahead', type=float, default=7.0, help='Lookahead distance in no-route junction turn mode')
    args = parser.parse_args()

    # Default show visuals unless explicitly disabled
    args.show = not args.no_show
    global SAVE_STRIDE, NO_SAVE, STOP_ON_COLLISION, obstacle_confirm_count, lane_change_until, lane_change_target, lane_change_target_lane, lane_change_started_at, lane_change_origin_lane, lane_change_hold_until, lane_lock_lane, lane_lock_until, lane_keep_lane_key, lane_recover_lane, lane_recover_until, lane_change_retry_until, lane_change_fail_streak, lane_change_start_offset, lane_change_best_offset, lane_change_last_progress_time, last_ped_spawn_time, dynamic_ped_count, ROUTE_WPS, ROUTE_INDEX, ROUTE_TARGET, stuck_counter, collision_count, last_collision_frame, last_collision_desc, last_steer, beam_latest_rgb, beam_latest_frame, block_npc_start_time, prev_lane_edge_offset, lane_edge_risk_count, lane_front_rgb, lane_front_frame
    SAVE_STRIDE = max(1, args.save_stride)
    NO_SAVE = args.no_save
    STOP_ON_COLLISION = bool(args.stop_on_collision)
    args.vis_stride = max(1, args.vis_stride)
    obstacle_confirm_count = 0
    lane_change_until = 0.0
    lane_change_target = None
    lane_change_target_lane = None
    lane_change_started_at = 0.0
    lane_change_origin_lane = None
    lane_change_hold_until = 0.0
    lane_lock_lane = None
    lane_lock_until = 0.0
    lane_keep_lane_key = None
    lane_recover_lane = None
    lane_recover_until = 0.0
    lane_change_retry_until = 0.0
    lane_change_fail_streak = 0
    lane_change_start_offset = 0.0
    lane_change_best_offset = 0.0
    lane_change_last_progress_time = 0.0
    last_steer = 0.0
    ROUTE_WPS = []
    ROUTE_INDEX = 0
    ROUTE_TARGET = None
    stuck_counter = 0
    collision_count = 0
    last_collision_frame = -1
    last_collision_desc = ""
    beam_latest_rgb = None
    beam_latest_frame = -1
    block_npc_start_time = 0.0
    prev_lane_edge_offset = 0.0
    lane_edge_risk_count = 0
    lane_front_rgb = None
    lane_front_frame = -1
    last_ped_spawn_time = 0.0
    dynamic_ped_count = 0

    log_status("main start")

    if args.fixed_scenario:
        args.crossing_ped = False
        args.block_npc = True
        args.ped_distance = 5.0
        args.ped_lateral = 1.8
        args.npc_lane_change = True
        args.num_walkers = 0
        args.crossing_ped_count = 0
        args.crossing_ped_spacing = 20.0
        args.crossing_ped_loop = True
        args.ped_spawn_interval = 0.0
        args.ped_max_active = 0
        args.crossing_scatter_count = 25
        args.crossing_scatter_min_dist = 25.0
        args.npc_min_distance = 30.0
    
    print("=" * 60)
    print("CARLA autonomous driving and multimodal data collection test")
    print("=" * 60)
    

    print("\n[1/6] Connecting to CARLA server...")
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"[OK] Connected to CARLA server at {args.host}:{args.port}")
        log_status("connected")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print("Please make sure the CARLA server is running.")
        log_status(f"connect failed: {e}")
        return
    

    if args.map:
        print(f"\n[2/6] Loading map: {args.map}...")
        try:
            world = client.load_world(args.map)
            print(f"[OK] Map {args.map} loaded")
            log_status("map loaded")
        except Exception as e:
            print(f"[ERROR] Failed to load map: {e}")
            print("Using default map.")
            log_status(f"map load failed: {e}")
    
    if args.list_spawn_points:
        spawn_points = world.get_map().get_spawn_points()
        print("\nSpawn points (index: x, y, z, yaw):")
        for i, sp in enumerate(spawn_points):
            loc = sp.location
            rot = sp.rotation
            print(f"  {i}: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f}), yaw={rot.yaw:.1f}")
        return


    blueprint_library = world.get_blueprint_library()
    

    if args.sync:
        print("\n[3/6] Enabling synchronous mode...")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.delta_time
        world.apply_settings(settings)
        

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        print("[OK] Synchronous mode enabled")
        log_status("sync enabled")
    

    print("\n[4/6] Spawning NPC vehicles and walkers...")
    npc_vehicles = []
    npc_walkers, walker_controllers = spawn_npc_walkers(world, blueprint_library, args.num_walkers)
    log_status(f"walkers spawned: {len(npc_walkers)}")

    
    if not args.npc_off_route:
        npc_vehicles = spawn_npc_vehicles(world, blueprint_library, args.num_vehicles, min_distance=args.npc_min_distance)
    log_status(f"npc vehicles spawned: {len(npc_vehicles)}")

    print("\nConfiguring Traffic Manager...")
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_random_device_seed(0)
    

    tm_port = traffic_manager.get_port()
    print(f"Traffic Manager port: {tm_port}")
    
    # NPC vehicle autopilot (optional)
    if args.npc_autopilot:
        autopilot_count = 0
        for vehicle in npc_vehicles:
            try:
                vehicle.set_autopilot(True, tm_port)
                autopilot_count += 1
            except Exception as e:
                print(f"Warning: Failed to set autopilot for vehicle {vehicle.id}: {e}")

        print(f"[OK] {autopilot_count}/{len(npc_vehicles)} NPC vehicles set to autopilot")

        # Wait briefly so vehicles begin moving
        if args.sync:
            for _ in range(10):
                world.tick()
        else:
            time.sleep(1.0)
    else:
        print("[OK] NPC vehicles remain static (autopilot disabled)")

    # NPC walker autopilot (optional)
    if args.walker_autopilot:
        print("\nEnabling NPC walkers autopilot...")
        walker_count = 0
        for controller in walker_controllers:
            try:
                controller.start()
                controller.go_to_location(world.get_random_location_from_navigation())
                controller.set_max_speed(1.5 + random.random())
                walker_count += 1
            except Exception as e:
                print(f"Warning: Failed to start walker controller: {e}")

        print(f"[OK] {walker_count}/{len(walker_controllers)} NPC walkers set to autopilot")
    else:
        print("\nNPC walkers remain static (autopilot disabled)")

    print("\n[5/6] Spawning Ego vehicle...")
    spawn_points = world.get_map().get_spawn_points()
    ego_spawn_pool = list(spawn_points)
    if args.ego_spawn_index is None or args.ego_spawn_index < 0:
        if args.block_npc or args.npc_lane_change or args.ped_lane_change:
            try:
                multilane = get_multilane_spawn_points(world, prefer=args.lane_change_prefer, min_forward=max(30.0, args.block_npc_distance + 12.0))
                if multilane:
                    ego_spawn_pool = multilane
                    print(f"[OK] Using multi-lane spawn pool: {len(ego_spawn_pool)} points")
            except Exception as e:
                print(f"[WARN] Failed to build multi-lane spawn pool: {e}")
    ego_spawn_point = random.choice(spawn_points)
    

    ego_bp = blueprint_library.filter('vehicle.*')[0]
    ego_bp.set_attribute('role_name', 'hero')
    
    ego_vehicle = None
    spawn_candidates = []
    if args.ego_spawn_index is not None and args.ego_spawn_index >= 0 and args.ego_spawn_index < len(spawn_points):
        spawn_candidates = [spawn_points[args.ego_spawn_index]]
    else:
        spawn_candidates = list(ego_spawn_pool if ego_spawn_pool else spawn_points)
        random.shuffle(spawn_candidates)
    for ego_spawn_point in spawn_candidates:
        try:
            ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
            if ego_vehicle:
                break
        except Exception:
            continue
    if ego_vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle after retries")
    print(f"[OK] Ego vehicle created: {ego_vehicle.type_id}")
    log_status("ego spawned")
    

    if args.autopilot:
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())
        print("[OK] Ego vehicle set to autopilot")
    else:
        print("[OK] Ego vehicle ready for manual control")
    if args.ego_loop:
        ego_vehicle.set_autopilot(False)
        print("[OK] Ego vehicle will follow simple lane loop (controller enabled)")
    if args.ego_adv:
        ego_vehicle.set_autopilot(False)
        print("[OK] Ego vehicle will use advanced Stanley/PID lane follow")

    # Let the simulator settle so transforms are valid before placing blockers
    if args.sync:
        try:
            for _ in range(3):
                world.tick()
        except Exception:
            pass

    # Build a long global route to avoid small local loops
    if args.ego_adv and args.global_route:
        try:
            ROUTE_WPS = build_global_route(world, ego_vehicle.get_location(), min_dist=150.0, resolution=2.0)
            ROUTE_INDEX = 0
            if ROUTE_WPS:
                ROUTE_TARGET = ROUTE_WPS[-1]
                print(f"[OK] Global route built: {len(ROUTE_WPS)} waypoints")
            else:
                print("[WARN] Global route unavailable, fallback to local lane follow")
        except Exception as e:
            print(f"[WARN] Global route init failed: {e}")

    if args.npc_off_route and not args.block_npc:
        try:
            route = build_route(world, ego_vehicle, look_ahead=300, step=4.0)
            npc_vehicles = spawn_npc_vehicles_off_route(world, blueprint_library, args.num_vehicles, route, min_distance=args.npc_min_distance)
        except Exception as e:
            print(f"[WARN] npc_off_route failed: {e}")
    # In block scenario, keep ordinary NPCs away from ego route so lane-change test is clean.
    if args.block_npc and (not args.npc_off_route) and args.num_vehicles > 0:
        try:
            for v in npc_vehicles:
                v.destroy()
            npc_vehicles = []
            route = build_route(world, ego_vehicle, look_ahead=500, step=4.0)
            off_min_dist = max(args.npc_min_distance, 25.0)
            npc_vehicles = spawn_npc_vehicles_off_route(
                world,
                blueprint_library,
                args.num_vehicles,
                route,
                min_distance=off_min_dist,
            )
            npc_vehicles, removed_local = clear_npcs_near_ego_corridor(
                ego_vehicle,
                npc_vehicles,
                forward_clear=max(70.0, args.block_npc_distance + 45.0),
                back_clear=20.0,
                lateral_clear=8.0,
            )
            if removed_local > 0:
                print(f"[OK] Block scenario: removed {removed_local} local NPCs near ego")
            print(f"[OK] Block scenario: spawned {len(npc_vehicles)} NPCs off ego route")
        except Exception as e:
            print(f"[WARN] block scenario off-route NPC failed: {e}")
    if (not args.npc_off_route) and (not args.block_npc):
        npc_vehicles = spawn_npc_vehicles(world, blueprint_library, args.num_vehicles, min_distance=args.npc_min_distance)
    if args.remove_front_npc:
        try:
            remove_front_npc(world, ego_vehicle, npc_vehicles)
        except Exception as e:
            print(f"[WARN] remove_front_npc failed: {e}")
    if args.block_npc:
        try:
            blocks = spawn_blocking_npcs_ahead(
                world,
                blueprint_library,
                ego_vehicle,
                count=max(1, args.block_npc_count),
                start_distance=args.block_npc_distance,
                spacing=args.block_npc_spacing,
            )
            for block in blocks:
                npc_vehicles.append(block)
                CROSSING_SPAWN_INFO.append(("npc", block.id, block.get_location(), "block_npc"))
            if args.block_npc_relocate:
                # Reposition existing NPCs ahead to guarantee visible blocking vehicles
                moved = move_block_npcs_in_front(
                    world,
                    ego_vehicle,
                    npc_vehicles,
                    start_distance=args.block_npc_distance,
                    spacing=args.block_npc_spacing,
                )
                if moved > 0:
                    for v in npc_vehicles[:moved]:
                        CROSSING_SPAWN_INFO.append(("npc", v.id, v.get_location(), "block_npc_relocate"))
        except Exception as e:
            print(f"[WARN] block_npc failed: {e}")
    log_status(f"after block_npc total npc: {len(npc_vehicles)}")
    if args.crossing_ped:
        try:
            if args.crossing_ped_count > 1:
                spawned = spawn_crossing_pedestrians(
                    world,
                    blueprint_library,
                    ego_vehicle,
                    count=args.crossing_ped_count,
                    spacing=args.crossing_ped_spacing,
                    start_distance=args.ped_distance,
                    lateral=args.ped_lateral,
                    speed=args.ped_speed,
                )
                for w, s_loc, e_loc, spd in spawned:
                    npc_walkers.append(w)
                    CROSSING_PEDS.append({"walker": w, "start": s_loc, "end": e_loc, "speed": spd, "to_end": True, "last": 0.0})
            else:
                walker, controller = spawn_crossing_pedestrian(
                    world,
                    blueprint_library,
                    ego_vehicle,
                    distance=args.ped_distance,
                    lateral=args.ped_lateral,
                    speed=args.ped_speed,
                )
                if walker is not None:
                    npc_walkers.append(walker)
                    CROSSING_PEDS.append({"walker": walker, "start": None, "end": None, "speed": args.ped_speed, "to_end": True, "last": 0.0})
                if controller is not None:
                    walker_controllers.append(controller)
        except Exception as e:
            print(f"[WARN] crossing_ped failed: {e}")
    if args.crossing_scatter_count > 0:
        try:
            spawned = spawn_scattered_crossers(
                world,
                blueprint_library,
                count=args.crossing_scatter_count,
                min_dist=args.crossing_scatter_min_dist,
                speed=args.ped_speed,
            )
            for w, s_loc, e_loc, spd in spawned:
                npc_walkers.append(w)
                CROSSING_PEDS.append({"walker": w, "start": s_loc, "end": e_loc, "speed": spd, "to_end": True, "last": 0.0})
        except Exception as e:
            print(f"[WARN] crossing_scatter failed: {e}")
    log_status(f"crossing walkers total: {len(npc_walkers)}")
    

    # Prepare per-run directories
    print("[DEBUG] Preparing run directories...")
    run_stamp = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        set_run_directories(run_stamp)
        print(f"\n[Run] Output directory: {os.path.abspath(RUN_DIR)}")
        log_status(f"run dir: {RUN_DIR}")
    except Exception as e:
        print(f"[ERROR] Failed to set run directories: {e}")
        log_status(f"run dir failed: {e}")
        raise

    beam_policy = BeamFusionClosedLoopPolicy(args)
    beam_enabled = beam_policy.initialize()
    beam_policy.open_log(RUN_DIR)
    if beam_enabled:
        print("[BeamFusion] closed-loop control policy active (speed/lane-change gating)")
    else:
        print("[BeamFusion] policy inactive, baseline controller only")
    lane_vision = LaneVisionAssist(enabled=not args.lv_disable)
    if lane_vision.available():
        print("[LaneVision] visual lane assist active")
    else:
        print("[LaneVision] visual lane assist inactive (disabled or cv2 unavailable)")

    print("\n[6/6] Attaching sensors to Ego vehicle...")
    view_windows = (not args.no_view_windows) and args.show
    if view_windows and not SDL2_AVAILABLE:
        print("[WARN] pygame._sdl2 not available, view windows disabled")
        view_windows = False
    sensors = setup_sensors(
        world,
        ego_vehicle,
        blueprint_library,
        view_width=args.view_width,
        view_height=args.view_height,
        view_windows=view_windows,
        beam_enabled=beam_enabled,
        beam_width=args.bf_image_w,
        beam_height=args.bf_image_h,
    )
    print(f"\nInstalled {len(sensors)} sensors")
    log_status(f"sensors attached: {len(sensors)}")


    spectator = world.get_spectator()
    transform = ego_vehicle.get_transform()
    chase_location = transform.location + carla.Location(x=-8, z=3)
    chase_rotation = carla.Rotation(pitch=-10, yaw=transform.rotation.yaw)
    spectator.set_transform(carla.Transform(chase_location, chase_rotation))
    
    print("\n" + "=" * 60)
    print("Environment setup complete")
    print("=" * 60)
    print(f"\nData output dir: {os.path.abspath(RUN_DIR)}")
    print("\nSensors:")
    print("  - RGB camera (front)")
    print("  - Depth camera")
    print("  - Semantic segmentation camera")
    print("  - LiDAR")
    print("  - Radar")
    print(f"\nNPC vehicles: {len(npc_vehicles)}")
    print(f"NPC walkers: {len(npc_walkers)}")
    print("\nPress Ctrl+C to stop data collection...")
    print("=" * 60)

    if actor_log:
        try:
            ego_loc = ego_vehicle.get_location()
            actor_log.write(f"0,ego,{ego_vehicle.id},{ego_loc.x:.3f},{ego_loc.y:.3f},{ego_loc.z:.3f},spawn\n")
            for v in npc_vehicles:
                loc = v.get_location()
                actor_log.write(f"0,npc,{v.id},{loc.x:.3f},{loc.y:.3f},{loc.z:.3f},spawn\n")
            for w in npc_walkers:
                loc = w.get_location()
                actor_log.write(f"0,ped,{w.id},{loc.x:.3f},{loc.y:.3f},{loc.z:.3f},spawn\n")
            for t, aid, loc, note in CROSSING_SPAWN_INFO:
                actor_log.write(f"0,{t},{aid},{loc.x:.3f},{loc.y:.3f},{loc.z:.3f},{note}\n")
            actor_log.flush()
        except Exception:
            pass
    
    # Performance log
    perf_log = open(os.path.join(RUN_DIR, "performance.log"), "w", encoding="utf-8")
    perf_log.write("frame,delta_ms,queue_size,speed_kmh,obstacle_m,steer,lane_offset,lane_keep_road,lane_keep_lane,lane_keep_offset,npc_dist,ped_dist,closing_kmh,lane_changing,lc_road,lc_lane,lc_hold_s,collision_count,last_collision_frame,bf_enabled,bf_top1_beam,bf_top1_prob,bf_entropy,bf_risk,bf_speed_cap_kmh,bf_allow_lane_change,bf_turn_deg,bf_blocked_wait_s,lv_available,lv_conf,lv_offset_norm,lv_weight,lv_fused_norm\n")
    last_time = time.time()

    try:

        frame_count = 0
        start_time = time.time()
        if args.show:
            global SHOW_VISUALS
            SHOW_VISUALS = True
            os.environ["SDL_VIDEO_WINDOW_POS"] = f"{args.hud_pos_x},{args.hud_pos_y}"
            pygame.init()
            window_size = (960, 540)
            screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption("CARLA Multimodal Live View")
            font = pygame.font.SysFont(None, 20)
            log_status("pygame windows init")
            fp_window = None
            tp_window = None
            fp_renderer = None
            tp_renderer = None
            if view_windows:
                fp_window = Window("First Person", size=(args.view_width, args.view_height))
                tp_window = Window("Third Person", size=(args.view_width, args.view_height))
                if args.view_fullscreen:
                    fp_window.set_fullscreen(True)
                    tp_window.set_fullscreen(True)
                else:
                    fp_window.position = (args.fp_pos_x, args.fp_pos_y)
                    tp_window.position = (args.tp_pos_x, args.tp_pos_y)
                fp_renderer = Renderer(fp_window)
                tp_renderer = Renderer(tp_window)

        while True:
            now_tick = time.time()
            delta_ms = (now_tick - last_time) * 1000.0
            last_time = now_tick

            if args.sync:
                world.tick()
            else:
                time.sleep(0.02)
            
            frame_count += 1
            npc_dist = None
            ped_dist = None
            closing_kmh = 0.0
            lane_changing = False
            bf_state = {
                "available": False,
                "top1_beam": -1,
                "top1_prob": 0.0,
                "entropy": 1.0,
                "risk": 1.0,
                "speed_cap_kmh": float(args.target_speed),
                "allow_lane_change": True,
            }
            turn_deg = 0.0
            blocked_wait = 0.0
            lv_available = 0.0
            lv_conf = 0.0
            lv_offset_norm = 0.0
            lv_weight = 0.0
            lv_fused_norm = 0.0

            if STOP_ON_COLLISION and collision_count > 0:
                print(f"\n[STOP] Collision detected at frame {last_collision_frame}: {last_collision_desc}")
                break

            # Spawn new crossing pedestrians periodically near current ego
            if args.crossing_ped and args.ped_spawn_interval > 0.0:
                if (now_tick - start_time) > 2.0 and (now_tick - last_ped_spawn_time) >= args.ped_spawn_interval:
                    if dynamic_ped_count < args.ped_max_active:
                        walker, _ = spawn_crossing_pedestrian(
                            world,
                            blueprint_library,
                            ego_vehicle,
                            distance=args.ped_distance,
                            lateral=args.ped_lateral,
                            speed=args.ped_speed,
                        )
                        if walker is not None:
                            npc_walkers.append(walker)
                            CROSSING_PEDS.append({"walker": walker, "start": None, "end": None, "speed": args.ped_speed, "to_end": True, "last": 0.0})
                            dynamic_ped_count += 1
                            if actor_log:
                                try:
                                    loc = walker.get_location()
                                    actor_log.write(f"{frame_count},ped,{walker.id},{loc.x:.3f},{loc.y:.3f},{loc.z:.3f},spawn\n")
                                    actor_log.flush()
                                except Exception:
                                    pass
                        last_ped_spawn_time = now_tick

            # Update looping crossing pedestrians
            if args.crossing_ped_loop and CROSSING_PEDS:
                for ped in CROSSING_PEDS:
                    w = ped.get("walker")
                    if w is None:
                        continue
                    try:
                        loc = w.get_location()
                    except Exception:
                        continue
                    start = ped.get("start")
                    end = ped.get("end")
                    if start is None or end is None:
                        continue
                    target = end if ped.get("to_end", True) else start
                    if loc.distance(target) < 1.0:
                        ped["to_end"] = not ped.get("to_end", True)
                        target = end if ped["to_end"] else start
                    if now_tick - ped.get("last", 0.0) > 0.5:
                        direction = target - loc
                        length = max((direction.x ** 2 + direction.y ** 2 + direction.z ** 2) ** 0.5, 1e-3)
                        direction = carla.Vector3D(direction.x / length, direction.y / length, 0.0)
                        try:
                            w.apply_control(carla.WalkerControl(direction=direction, speed=ped.get("speed", 1.4)))
                            ped["last"] = now_tick
                        except Exception:
                            pass

            if args.duration is not None and (time.time() - start_time) >= args.duration:
                print("\n[OK] Duration reached, stopping data collection...")
                break
            
            if args.show:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n[OK] Quit event received, stopping data collection...")
                        raise KeyboardInterrupt

                if frame_count % args.vis_stride == 0:
                    screen.fill((20, 20, 20))

                    def make_surface(name, label, pos, size=(320, 180)):
                        if name not in latest_frames:
                            text = font.render(f"{label}: waiting...", True, (220, 220, 220))
                            screen.blit(text, pos)
                            return
                        arr = latest_frames[name]
                        surf = pygame.surfarray.make_surface(np.swapaxes(arr, 0, 1))
                        surf = pygame.transform.smoothscale(surf, size)
                        screen.blit(surf, pos)
                        count = sensor_stats.get(name, {}).get('count', 0)
                        text = font.render(f"{label} (frames: {count})", True, (255, 255, 0))
                        screen.blit(text, (pos[0], pos[1] + size[1] + 5))

                    make_surface('rgb_front', "RGB", (0, 0))
                    make_surface('depth_front', "Depth", (320, 0))
                    make_surface('semantic_front', "Semantic", (0, 180))

                    radar_count = sensor_stats.get('radar_front', {}).get('count', 0)
                    lidar_count = sensor_stats.get('lidar_top', {}).get('count', 0)
                    text = font.render(f"Radar frames: {radar_count} | LiDAR frames: {lidar_count}", True, (0, 200, 255))
                    screen.blit(text, (320, 180 + 10))

                    if args.hud:
                        velocity = ego_vehicle.get_velocity()
                        speed_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                        ltxt = f"{latest_lidar_dist:.1f}m" if latest_lidar_dist is not None else "n/a"
                        rtxt = f"{latest_radar_dist:.1f}m" if latest_radar_dist is not None else "n/a"
                        hud_text = font.render(
                            f"Frame: {frame_count}  Speed: {speed_kmh:.1f} km/h  L:{ltxt} R:{rtxt} C:{obstacle_confirm_count}",
                            True,
                            (255, 255, 255),
                        )
                        screen.blit(hud_text, (10, window_size[1] - 25))

                    pygame.display.flip()

            if args.ego_loop:
                control = simple_lane_follow_control(world, ego_vehicle, target_speed_kmh=20.0)
                v_loop = ego_vehicle.get_velocity()
                cur_speed_loop = 3.6 * (v_loop.x**2 + v_loop.y**2 + v_loop.z**2)**0.5
                # Clamp and smooth steering to reduce curb hits
                control.steer = max(-args.max_steer, min(args.max_steer, control.steer))
                delta_steer = control.steer - last_steer
                if delta_steer > args.steer_rate:
                    control.steer = last_steer + args.steer_rate
                elif delta_steer < -args.steer_rate:
                    control.steer = last_steer - args.steer_rate
                # Slow down when steering sharply
                if abs(control.steer) > 0.25:
                    control.throttle = min(control.throttle, 0.25)
                    if cur_speed_loop > 4.0:
                        control.brake = max(control.brake, 0.05)
                last_steer = control.steer
                ego_vehicle.apply_control(control)
            if args.ego_adv:
                target_speed = args.target_speed
                beam_allow_lane_change = True
                # Ignore obstacle logic during initial warmup
                ignore_obstacles = (time.time() - start_time) < 3.0
                # Drop stale obstacle updates
                if time.time() - last_obstacle_time > 0.5:
                    latest = None
                else:
                    latest = latest_obstacle_dist

                # Multimodal confirm: require both lidar & radar near before slowing
                if not ignore_obstacles:
                    if latest_lidar_dist is not None and latest_radar_dist is not None:
                        if latest_lidar_dist < args.hazard_dist and latest_radar_dist < args.hazard_dist:
                            obstacle_confirm_count += 1
                        else:
                            obstacle_confirm_count = max(0, obstacle_confirm_count - 1)
                    else:
                        obstacle_confirm_count = max(0, obstacle_confirm_count - 1)
                front_obstacle_confirmed = (latest is not None and latest < args.hazard_dist and obstacle_confirm_count >= 3)
                front_obstacle_brake = (latest is not None and latest < args.hazard_dist and obstacle_confirm_count >= 5)

                ped_dist = get_front_ped_distance(
                    ego_vehicle,
                    npc_walkers,
                    max_distance=args.ped_detect_dist,
                    lane_half_width=args.ped_lane_half_width,
                )
                lead_info = get_front_vehicle_info(
                    ego_vehicle,
                    npc_vehicles,
                    max_distance=max(args.npc_slow_dist, args.npc_change_dist) + 10.0,
                )
                overlap_risk_dist = get_front_overlap_risk(
                    ego_vehicle,
                    npc_vehicles,
                    max_distance=max(10.0, args.npc_stop_dist + 6.0),
                    extra_margin=0.65,
                )
                npc_dist = lead_info["distance"] if lead_info is not None else None
                closing_kmh = lead_info["rel_speed_kmh"] if lead_info is not None else 0.0
                now_time = time.time()
                velocity = ego_vehicle.get_velocity()
                cur_speed_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                dynamic_max_steer = float(args.max_steer)
                dynamic_steer_rate = float(args.steer_rate)
                dynamic_lc_lookahead = float(args.lane_change_lookahead)
                dynamic_lc_speed = float(args.lane_change_speed)
                if beam_enabled:
                    bf_state = beam_policy.infer(
                        ego_vehicle=ego_vehicle,
                        all_vehicles=[ego_vehicle] + list(npc_vehicles),
                        rgb_frame=beam_latest_rgb,
                        now_time=now_time,
                    )
                    try:
                        target_speed = min(float(target_speed), float(bf_state.get("speed_cap_kmh", target_speed)))
                    except Exception:
                        pass
                    beam_allow_lane_change = bool(bf_state.get("allow_lane_change", True))
                    if args.bf_dominant_level != "off":
                        dom_mult = 1.0 if args.bf_dominant_level == "assist" else 1.6
                        risk = max(0.0, min(1.0, float(bf_state.get("risk", 1.0))))
                        confidence = max(0.0, min(1.0, float(bf_state.get("top1_prob", 0.0))))
                        control_gain = dom_mult * (1.0 - 0.55 * risk) * (0.45 + 0.55 * confidence)
                        dynamic_max_steer = min(
                            0.70,
                            max(0.25, dynamic_max_steer + float(args.bf_steer_boost_max) * control_gain),
                        )
                        dynamic_steer_rate = min(
                            0.25,
                            max(0.06, dynamic_steer_rate + float(args.bf_steer_rate_boost) * control_gain),
                        )
                if lane_recover_lane is not None and now_time >= lane_recover_until:
                    lane_recover_lane = None
                    lane_recover_until = 0.0
                map_obj = world.get_map()
                try:
                    ego_wp_now = map_obj.get_waypoint(ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
                except Exception:
                    ego_wp_now = None
                turn_signed = estimate_upcoming_turn_signed_deg(map_obj, ego_vehicle, d1=8.0, d2=22.0)
                turn_deg = abs(turn_signed)
                junction_turn_mode = (
                    (not args.junction_turn_disable)
                    and (not args.global_route)
                    and (
                        (ego_wp_now is not None and ego_wp_now.is_junction)
                        or (turn_deg >= float(args.junction_turn_deg))
                    )
                )
                junction_turn_pref = 0.0
                if junction_turn_mode and abs(turn_signed) >= max(6.0, float(args.junction_turn_deg) * 0.45):
                    junction_turn_pref = 1.0 if turn_signed > 0.0 else -1.0
                elif junction_turn_mode:
                    junction_turn_pref = -0.35 if args.lane_change_prefer == "right" else 0.35
                track_lookahead = 10.0
                if junction_turn_mode:
                    target_speed = min(float(target_speed), float(args.junction_speed_cap))
                    track_lookahead = float(args.junction_lookahead)
                    dynamic_max_steer = min(0.68, max(0.30, dynamic_max_steer + 0.10))
                    dynamic_steer_rate = min(0.24, max(0.08, dynamic_steer_rate + 0.05))
                if beam_enabled and args.bf_dominant_level != "off" and turn_deg >= float(args.bf_turn_sharp_deg):
                    target_speed = min(float(target_speed), float(args.bf_turn_speed_cap))
                    dynamic_max_steer = min(0.72, dynamic_max_steer + float(args.bf_turn_steer_boost))
                    dynamic_steer_rate = min(0.28, dynamic_steer_rate + float(args.bf_turn_rate_boost))
                    dynamic_lc_lookahead = max(7.0, dynamic_lc_lookahead * 0.85)
                    dynamic_lc_speed = min(dynamic_lc_speed, max(14.0, float(args.bf_turn_speed_cap)))
                if lane_lock_lane is not None and now_time >= lane_lock_until:
                    lane_lock_lane = None
                if lane_change_target_lane is not None:
                    lane_change_done = False
                    center_ok = False
                    yaw_ok = False
                    target_lane_wp_now = None
                    if ego_wp_now is not None:
                        target_lane_wp_now = _find_target_lane_wp(ego_wp_now, lane_change_target_lane)
                        lane_change_done = (ego_wp_now.road_id == lane_change_target_lane[0] and ego_wp_now.lane_id == lane_change_target_lane[1])
                        if lane_change_done and target_lane_wp_now is None:
                            target_lane_wp_now = ego_wp_now
                        if target_lane_wp_now is not None:
                            cur_lc_offset = _lane_center_offset_abs(ego_vehicle.get_location(), target_lane_wp_now)
                            if (lane_change_best_offset <= 0.0) or (cur_lc_offset + 0.03 < lane_change_best_offset):
                                lane_change_best_offset = cur_lc_offset
                                lane_change_last_progress_time = now_time
                        if lane_change_done:
                            lane_ref_wp = target_lane_wp_now if target_lane_wp_now is not None else ego_wp_now
                            center_err = _lane_center_offset_abs(ego_vehicle.get_location(), lane_ref_wp)
                            yaw_err = _yaw_error_abs_deg(ego_vehicle.get_transform().rotation.yaw, lane_ref_wp.transform.rotation.yaw)
                            # Complete slightly earlier to avoid overshoot after merge on curves.
                            center_ok = center_err <= max(args.lane_change_center_tol, 0.55)
                            yaw_ok = yaw_err <= max(args.lane_change_yaw_tol, 16.0)
                    if lane_change_done and center_ok and yaw_ok and (now_time - lane_change_started_at) > 0.7:
                        finished_lane = lane_change_target_lane
                        lane_change_target = None
                        lane_change_target_lane = None
                        lane_change_origin_lane = None
                        lane_change_until = 0.0
                        lane_change_start_offset = 0.0
                        lane_change_best_offset = 0.0
                        lane_change_last_progress_time = 0.0
                        lane_change_fail_streak = 0
                        lane_change_retry_until = now_time + 0.3
                        lane_change_hold_until = now_time + args.lane_change_hold
                        lane_lock_lane = finished_lane
                        lane_lock_until = lane_change_hold_until
                        lane_keep_lane_key = finished_lane
                        lane_recover_lane = finished_lane
                        lane_recover_until = now_time + 2.2
                        if args.global_route:
                            try:
                                new_route = build_global_route(world, ego_vehicle.get_location(), min_dist=150.0, resolution=2.0)
                                if new_route:
                                    ROUTE_WPS = new_route
                                    ROUTE_INDEX = 0
                            except Exception:
                                pass
                    else:
                        lc_elapsed = now_time - lane_change_started_at
                        lc_progress = max(0.0, lane_change_start_offset - lane_change_best_offset)
                        progress_stalled = (
                            lc_elapsed > max(1.0, args.lane_change_progress_timeout)
                            and (now_time - lane_change_last_progress_time) > 0.7
                            and lc_progress < args.lane_change_min_progress
                            and cur_speed_kmh < max(6.0, args.npc_min_speed)
                        )
                        timed_out = lc_elapsed > args.lane_change_max_duration
                        if progress_stalled or timed_out:
                            lane_change_target = None
                            lane_change_target_lane = None
                            lane_change_origin_lane = None
                            lane_change_until = 0.0
                            lane_change_start_offset = 0.0
                            lane_change_best_offset = 0.0
                            lane_change_last_progress_time = 0.0
                            lane_change_fail_streak = min(8, lane_change_fail_streak + 1)
                            fail_cooldown = max(1.0, args.lane_change_fail_cooldown) * (1.0 + 0.45 * min(4, lane_change_fail_streak))
                            lane_change_retry_until = now_time + min(12.0, fail_cooldown)
                            lane_change_hold_until = max(lane_change_hold_until, lane_change_retry_until)
                            if ego_wp_now is not None:
                                lane_keep_lane_key = (ego_wp_now.road_id, ego_wp_now.lane_id)
                                lane_recover_lane = lane_keep_lane_key
                                lane_recover_until = now_time + 1.2
                            if progress_stalled:
                                print("[WARN] Lane change aborted: insufficient lateral progress")
                        else:
                            lane_changing = True

                if npc_dist is not None and npc_dist < max(args.npc_change_dist, args.npc_stop_dist + 1.0):
                    if block_npc_start_time <= 0.0:
                        block_npc_start_time = now_time
                else:
                    block_npc_start_time = 0.0
                blocked_wait = (now_time - block_npc_start_time) if block_npc_start_time > 0.0 else 0.0
                forced_lane_change = (npc_dist is not None and npc_dist < (args.npc_stop_dist + 1.5))
                if (
                    beam_enabled
                    and args.bf_dominant_level != "off"
                    and beam_allow_lane_change
                    and blocked_wait >= float(args.bf_force_overtake_wait)
                    and npc_dist is not None
                ):
                    forced_lane_change = True
                can_try_lane_change = (
                    (now_time >= lane_change_hold_until)
                    and (now_time >= lane_change_retry_until)
                    and (lane_lock_lane is None)
                    and (cur_speed_kmh >= args.lane_change_min_speed or forced_lane_change)
                )
                lane_change_failed_this_tick = False
                if args.ped_lane_change and beam_allow_lane_change and can_try_lane_change and ped_dist is not None and ped_dist < args.ped_detect_dist:
                    if lane_change_target is None:
                        lc_front_clear = args.lane_change_front_clear
                        lc_back_clear = args.lane_change_back_clear
                        if forced_lane_change:
                            lc_front_clear = max(6.0, lc_front_clear * 0.45)
                            lc_back_clear = max(3.0, lc_back_clear * 0.5)
                        if forced_lane_change and beam_enabled and args.bf_dominant_level != "off":
                            lc_front_clear = max(5.5, lc_front_clear * float(args.bf_overtake_front_relax))
                            lc_back_clear = max(2.8, lc_back_clear * float(args.bf_overtake_back_relax))
                        target_wp = _lane_change_target(
                            world,
                            ego_vehicle,
                            npc_vehicles,
                            prefer=args.lane_change_prefer,
                            front_clear=lc_front_clear,
                            back_clear=lc_back_clear,
                            allow_outward=True,
                        )
                        if target_wp is not None:
                            lane_change_target = target_wp
                            lane_change_target_lane = (target_wp.road_id, target_wp.lane_id)
                            if ego_wp_now is not None:
                                lane_change_origin_lane = (ego_wp_now.road_id, ego_wp_now.lane_id)
                            lane_change_started_at = now_time
                            lane_change_until = now_time + args.lane_change_duration
                            start_wp = _find_target_lane_wp(ego_wp_now, lane_change_target_lane) if ego_wp_now is not None else None
                            if start_wp is None:
                                start_wp = target_wp
                            lane_change_start_offset = _lane_center_offset_abs(ego_vehicle.get_location(), start_wp)
                            lane_change_best_offset = lane_change_start_offset
                            lane_change_last_progress_time = now_time
                            print("[INFO] Lane change triggered to avoid pedestrian")
                        elif forced_lane_change and not lane_change_failed_this_tick:
                            lane_change_fail_streak = min(8, lane_change_fail_streak + 1)
                            fail_cooldown = max(1.0, args.lane_change_fail_cooldown) * (1.0 + 0.30 * min(4, lane_change_fail_streak))
                            lane_change_retry_until = now_time + min(10.0, fail_cooldown)
                            lane_change_hold_until = max(lane_change_hold_until, lane_change_retry_until)
                            lane_change_failed_this_tick = True
                lane_change_trigger_dist = max(
                    args.npc_stop_dist + 3.0,
                    min(args.npc_change_dist, args.npc_slow_dist * 0.65),
                )
                if args.npc_lane_change and beam_allow_lane_change and can_try_lane_change and npc_dist is not None and npc_dist < lane_change_trigger_dist and (closing_kmh > args.lane_change_closing or npc_dist < args.npc_stop_dist + 1.0):
                    if lane_change_target is None:
                        lc_front_clear = args.lane_change_front_clear
                        lc_back_clear = args.lane_change_back_clear
                        if forced_lane_change:
                            lc_front_clear = max(6.0, lc_front_clear * 0.45)
                            lc_back_clear = max(3.0, lc_back_clear * 0.5)
                        if forced_lane_change and beam_enabled and args.bf_dominant_level != "off":
                            lc_front_clear = max(5.5, lc_front_clear * float(args.bf_overtake_front_relax))
                            lc_back_clear = max(2.8, lc_back_clear * float(args.bf_overtake_back_relax))
                        target_wp = _lane_change_target(
                            world,
                            ego_vehicle,
                            npc_vehicles,
                            prefer=args.lane_change_prefer,
                            front_clear=lc_front_clear,
                            back_clear=lc_back_clear,
                            allow_outward=True,
                        )
                        if target_wp is not None:
                            lane_change_target = target_wp
                            lane_change_target_lane = (target_wp.road_id, target_wp.lane_id)
                            if ego_wp_now is not None:
                                lane_change_origin_lane = (ego_wp_now.road_id, ego_wp_now.lane_id)
                            lane_change_started_at = now_time
                            lane_change_until = now_time + args.lane_change_duration
                            start_wp = _find_target_lane_wp(ego_wp_now, lane_change_target_lane) if ego_wp_now is not None else None
                            if start_wp is None:
                                start_wp = target_wp
                            lane_change_start_offset = _lane_center_offset_abs(ego_vehicle.get_location(), start_wp)
                            lane_change_best_offset = lane_change_start_offset
                            lane_change_last_progress_time = now_time
                            print("[INFO] Lane change triggered to avoid NPC")
                        elif forced_lane_change and not lane_change_failed_this_tick:
                            lane_change_fail_streak = min(8, lane_change_fail_streak + 1)
                            fail_cooldown = max(1.0, args.lane_change_fail_cooldown) * (1.0 + 0.30 * min(4, lane_change_fail_streak))
                            lane_change_retry_until = now_time + min(10.0, fail_cooldown)
                            lane_change_hold_until = max(lane_change_hold_until, lane_change_retry_until)
                            lane_change_failed_this_tick = True
                lane_changing = lane_changing or (lane_change_target is not None)
                # Stable lane reference: avoid nearest-waypoint lane flip on wide curves.
                if lane_change_target_lane is not None:
                    lane_keep_lane_key = lane_change_target_lane
                elif lane_recover_lane is not None and now_time < lane_recover_until:
                    lane_keep_lane_key = lane_recover_lane
                elif lane_lock_lane is not None and now_time < lane_lock_until:
                    lane_keep_lane_key = lane_lock_lane
                if lane_change_target_lane is not None and ego_wp_now is not None:
                    # While changing lane, keep target lane as reference to avoid
                    # snapping lane key back to current lane and stalling longitudinally.
                    stable_lane_wp = _find_target_lane_wp(ego_wp_now, lane_change_target_lane)
                    if stable_lane_wp is not None:
                        lane_keep_lane_key = lane_change_target_lane
                    else:
                        stable_lane_wp, lane_keep_lane_key = _select_stable_lane_wp(
                            map_obj,
                            ego_vehicle.get_location(),
                            lane_keep_lane_key,
                            stick_offset=args.lane_stick_offset,
                        )
                else:
                    stable_lane_wp, lane_keep_lane_key = _select_stable_lane_wp(
                        map_obj,
                        ego_vehicle.get_location(),
                        lane_keep_lane_key,
                        stick_offset=args.lane_stick_offset,
                    )
                lane_edge_offset = 0.0
                lane_edge_ratio = 0.0
                lane_edge_wp = ego_wp_now
                half_width = 1.75
                if lane_edge_wp is not None:
                    lane_edge_offset = _lane_center_offset_signed(ego_vehicle.get_location(), lane_edge_wp)
                    lane_width = max(2.6, float(getattr(lane_edge_wp, "lane_width", 3.5)))
                    half_width = max(1.3, lane_width * 0.5)
                    map_norm = lane_edge_offset / half_width
                    lane_edge_ratio = abs(map_norm)
                    lv_fused_norm = float(map_norm)

                    if lane_vision.available() and lane_front_rgb is not None:
                        lv_state = lane_vision.estimate(lane_front_rgb, map_offset_norm=float(map_norm))
                        lv_available = float(lv_state.get("available", 0.0))
                        lv_conf = float(lv_state.get("confidence", 0.0))
                        lv_offset_norm = float(lv_state.get("offset_norm", 0.0))
                        if lv_available > 0.5 and lv_conf >= float(args.lv_conf_min):
                            lv_weight = min(float(args.lv_weight_max), float(args.lv_weight_max) * lv_conf)
                            lv_fused_norm = (1.0 - lv_weight) * float(map_norm) + lv_weight * lv_offset_norm
                            lane_edge_offset = float(lv_fused_norm) * half_width
                            lane_edge_ratio = abs(float(lv_fused_norm))
                        else:
                            lv_weight = 0.0
                            lv_fused_norm = float(map_norm)

                    edge_growth = abs(lane_edge_offset) - abs(prev_lane_edge_offset)
                    if lane_edge_ratio > 0.84 and edge_growth > 0.008:
                        lane_edge_risk_count = min(20, lane_edge_risk_count + 1)
                    else:
                        lane_edge_risk_count = max(0, lane_edge_risk_count - 1)
                    if lane_edge_ratio > 0.98:
                        lane_keep_lane_key = (lane_edge_wp.road_id, lane_edge_wp.lane_id)
                # If lead vehicle is on a different lane while we are still crossing into lane_keep,
                # do not let old-lane lead lock longitudinal speed.
                if lead_info is not None and lane_keep_lane_key is not None and stable_lane_wp is not None:
                    try:
                        lead_wp = map_obj.get_waypoint(
                            lead_info["vehicle"].get_location(),
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving,
                        )
                    except Exception:
                        lead_wp = None
                    if lead_wp is not None:
                        lead_key = (lead_wp.road_id, lead_wp.lane_id)
                        keep_off_abs = _lane_center_offset_abs(ego_vehicle.get_location(), stable_lane_wp)
                        if lead_key != lane_keep_lane_key and keep_off_abs > 0.75 and (npc_dist is None or npc_dist > 3.0):
                            lead_info = None
                            npc_dist = None
                            closing_kmh = 0.0
                obstacle_brake_ok = (npc_dist is None) or (npc_dist < args.npc_slow_dist)
                if ped_dist is not None and ped_dist < args.ped_stop_dist:
                    target_speed = 0.0
                elif ped_dist is not None and ped_dist < args.ped_detect_dist:
                    target_speed = min(target_speed, 10.0)
                # Adaptive lead-vehicle following
                target_speed = compute_follow_speed_kmh(
                    target_speed,
                    cur_speed_kmh,
                    lead_info,
                    min_gap=args.follow_min_gap,
                    time_gap=args.follow_time_gap,
                    slow_dist=args.npc_slow_dist,
                    min_speed_kmh=args.npc_min_speed,
                )
                if lane_changing and overlap_risk_dist is not None:
                    if overlap_risk_dist < max(4.8, args.follow_min_gap):
                        target_speed = 0.0
                    elif overlap_risk_dist < max(8.0, args.follow_min_gap + 2.0):
                        target_speed = min(target_speed, max(6.0, args.npc_min_speed * 0.8))
                if (not lane_changing) and lane_edge_ratio > 0.82:
                    # Lane-edge guard: slow down before clipping curb/guardrail.
                    target_speed = min(target_speed, 24.0 if lane_edge_ratio > 0.92 else 30.0)
                if npc_dist is not None and npc_dist < args.npc_stop_dist:
                    if not lane_changing:
                        if args.npc_lane_change and beam_allow_lane_change and can_try_lane_change:
                            target_speed = max(args.npc_min_speed, min(target_speed, args.hazard_speed))
                        else:
                            target_speed = 0.0
                    else:
                        if npc_dist < max(args.follow_min_gap, args.npc_stop_dist * 0.7):
                            target_speed = 0.0
                        else:
                            target_speed = max(args.npc_min_speed, min(target_speed, args.hazard_speed))

                if (not ignore_obstacles) and front_obstacle_confirmed and obstacle_brake_ok:
                    target_speed = min(target_speed, args.hazard_speed)

                if lane_changing:
                    # Use dynamic target on target lane so change can finish without snapping back.
                    dynamic_lc_wp = None
                    try:
                        dynamic_lc_wp = _lane_change_guidance_wp(world, ego_vehicle, lane_change_target_lane, lookahead=dynamic_lc_lookahead)
                    except Exception:
                        dynamic_lc_wp = None
                    if dynamic_lc_wp is not None:
                        lane_change_target = dynamic_lc_wp
                    if lane_change_target is not None:
                        lc_speed = min(target_speed, args.hazard_speed, dynamic_lc_speed)
                        lc_max_steer = min(dynamic_max_steer, args.lane_change_steer)
                        # When already close to target lane center, reduce steering aggression
                        # to avoid oversteer into curb/guardrail.
                        if ego_wp_now is not None and lane_change_target_lane is not None:
                            target_lane_wp = _find_target_lane_wp(ego_wp_now, lane_change_target_lane)
                            if target_lane_wp is not None:
                                target_lane_offset = _lane_center_offset_abs(ego_vehicle.get_location(), target_lane_wp)
                                if target_lane_offset < 0.95:
                                    lc_max_steer = min(lc_max_steer, 0.30)
                                    lc_speed = min(lc_speed, 22.0)
                                if target_lane_offset < 0.55:
                                    lc_max_steer = min(lc_max_steer, 0.22)
                                    lc_speed = min(lc_speed, 18.0)
                        control = _control_to_waypoint(ego_vehicle, lane_change_target, lc_speed, max_steer=lc_max_steer)
                    else:
                        lane_changing = False
                        control = advanced_lane_follow_control(
                            map_obj,
                            ego_vehicle,
                            target_speed_kmh=target_speed,
                            lookahead=track_lookahead,
                            lane_wp_hint=stable_lane_wp,
                            junction_mode=junction_turn_mode,
                            turn_pref=junction_turn_pref,
                        )
                else:
                    recovering = lane_recover_lane is not None and now_time < lane_recover_until
                    lane_locked = lane_lock_lane is not None and now_time < lane_lock_until
                    if recovering:
                        rec_wp = _lane_change_guidance_wp(
                            world,
                            ego_vehicle,
                            lane_recover_lane,
                            lookahead=max(8.0, dynamic_lc_lookahead * 0.8),
                        )
                        if rec_wp is not None:
                            rec_speed = min(target_speed, max(14.0, min(dynamic_lc_speed, 24.0)))
                            control = _control_to_waypoint(
                                ego_vehicle,
                                rec_wp,
                                rec_speed,
                                max_steer=min(dynamic_max_steer, 0.24),
                            )
                            rec_center = _lane_center_offset_abs(ego_vehicle.get_location(), rec_wp)
                            rec_yaw = _yaw_error_abs_deg(
                                ego_vehicle.get_transform().rotation.yaw,
                                rec_wp.transform.rotation.yaw,
                            )
                            if rec_center < 0.32 and rec_yaw < 8.0:
                                lane_recover_lane = None
                                lane_recover_until = 0.0
                        else:
                            lane_recover_lane = None
                            lane_recover_until = 0.0
                            lane_locked = lane_lock_lane is not None and now_time < lane_lock_until
                    if (not recovering) and lane_locked:
                        lock_wp = _lane_change_guidance_wp(world, ego_vehicle, lane_lock_lane, lookahead=max(10.0, dynamic_lc_lookahead))
                        if lock_wp is not None:
                            control = _control_to_waypoint(
                                ego_vehicle,
                                lock_wp,
                                target_speed,
                                max_steer=min(dynamic_max_steer, 0.35),
                            )
                        else:
                            lane_lock_lane = None
                            lane_lock_until = 0.0
                            control = advanced_lane_follow_control(
                                map_obj,
                                ego_vehicle,
                                target_speed_kmh=target_speed,
                                lookahead=track_lookahead,
                                lane_wp_hint=stable_lane_wp,
                                junction_mode=junction_turn_mode,
                                turn_pref=junction_turn_pref,
                            )
                    elif not recovering:
                        lane_lock_lane = None
                        lane_lock_until = 0.0
                    # Follow global route if enabled, otherwise fallback to local lane follow.
                    if (not recovering) and (not lane_locked):
                        use_route = args.global_route and ROUTE_WPS is not None and len(ROUTE_WPS) > 0
                        if use_route:
                            # Advance route index when close to current target
                            try:
                                cur_wp = ROUTE_WPS[ROUTE_INDEX]
                                dist = ego_vehicle.get_location().distance(cur_wp.transform.location)
                                if dist < 6.0 and ROUTE_INDEX < len(ROUTE_WPS) - 1:
                                    ROUTE_INDEX += 1
                            except Exception:
                                pass
                            # If route finished, rebuild a new long route
                            if ROUTE_INDEX >= len(ROUTE_WPS) - 1:
                                new_route = build_global_route(world, ego_vehicle.get_location(), min_dist=150.0, resolution=2.0)
                                if new_route:
                                    ROUTE_WPS = new_route
                                    ROUTE_INDEX = 0
                            try:
                                target_wp = ROUTE_WPS[min(ROUTE_INDEX, len(ROUTE_WPS) - 1)]
                                control = _control_to_waypoint(ego_vehicle, target_wp, target_speed)
                            except Exception:
                                control = advanced_lane_follow_control(
                                    map_obj,
                                    ego_vehicle,
                                    target_speed_kmh=target_speed,
                                    lookahead=track_lookahead,
                                    lane_wp_hint=stable_lane_wp,
                                    junction_mode=junction_turn_mode,
                                    turn_pref=junction_turn_pref,
                                )
                        else:
                            control = advanced_lane_follow_control(
                                map_obj,
                                ego_vehicle,
                                target_speed_kmh=target_speed,
                                lookahead=track_lookahead,
                                lane_wp_hint=stable_lane_wp,
                                junction_mode=junction_turn_mode,
                                turn_pref=junction_turn_pref,
                            )

                if (not ignore_obstacles) and front_obstacle_brake and obstacle_brake_ok and not lane_changing and (npc_dist is None or npc_dist >= args.npc_change_dist):
                    control.brake = max(control.brake, 0.4)
                    control.throttle = 0.0
                if ped_dist is not None and ped_dist < args.ped_stop_dist:
                    control.brake = 1.0
                    control.throttle = 0.0
                if npc_dist is not None and npc_dist < args.npc_stop_dist and not lane_changing and not (args.npc_lane_change and can_try_lane_change):
                    control.brake = max(control.brake, 0.5)
                    control.throttle = 0.0
                if npc_dist is not None and npc_dist < max(2.8, args.follow_min_gap * 0.6) and not lane_changing:
                    # Emergency rear-end prevention
                    control.brake = 1.0
                    control.throttle = 0.0
                if lane_changing and npc_dist is not None and npc_dist < max(4.0, args.follow_min_gap):
                    # During lane change, prioritize not hitting lead vehicle.
                    control.brake = max(control.brake, 0.55)
                    control.throttle = min(control.throttle, 0.10)
                if lane_changing and overlap_risk_dist is not None and overlap_risk_dist < max(5.2, args.follow_min_gap):
                    # Guard against clipping front vehicle rear corner during unfinished merge.
                    control.brake = max(control.brake, 0.72)
                    control.throttle = 0.0
                if lane_changing and npc_dist is not None and npc_dist < 1.8:
                    control.brake = 1.0
                    control.throttle = 0.0
                if lane_changing and lane_edge_ratio > 0.94:
                    # During lane changes, push away from road edge to prevent wall clipping.
                    edge_level = min(1.0, (lane_edge_ratio - 0.94) / 0.10)
                    edge_correction = -np.sign(lane_edge_offset) * (0.06 + 0.10 * edge_level)
                    control.steer += edge_correction
                    control.throttle = min(control.throttle, 0.16)
                    if cur_speed_kmh > 16.0:
                        control.brake = max(control.brake, 0.30)

                if (not lane_changing) and lane_edge_ratio > max(0.80, float(args.lv_edge_ratio_th) - 0.06):
                    # Active recentering when close to lane edge.
                    edge_level = min(1.0, (lane_edge_ratio - 0.86) / 0.18)
                    edge_correction = -np.sign(lane_edge_offset) * (0.10 + 0.16 * edge_level)
                    control.steer += edge_correction
                    edge_throttle_cap = 0.22 if lane_edge_ratio < 0.95 else 0.10
                    if cur_speed_kmh > 3.0:
                        control.throttle = min(control.throttle, edge_throttle_cap)
                    else:
                        # Keep crawling authority at very low speed, otherwise the car can stall at curve edges.
                        control.brake = 0.0
                        control.throttle = max(control.throttle, 0.24)
                    if lane_edge_ratio > 0.95 and cur_speed_kmh > 18.0:
                        control.brake = max(control.brake, 0.35)

                if (not lane_changing) and lane_edge_ratio > float(args.lv_edge_ratio_th):
                    # Hard curb guard: never keep steering toward curb side.
                    edge_sign = np.sign(lane_edge_offset)
                    if edge_sign != 0.0 and (control.steer * edge_sign) > 0.0:
                        control.steer = 0.0
                    edge_level = min(1.0, (lane_edge_ratio - float(args.lv_edge_ratio_th)) / 0.10)
                    control.steer += -edge_sign * (float(args.lv_curb_steer_k) + 0.10 * edge_level)
                    curb_throttle_cap = 0.14 if lane_edge_ratio < 0.96 else 0.08
                    if cur_speed_kmh > 3.0:
                        control.throttle = min(control.throttle, curb_throttle_cap)
                    else:
                        control.brake = 0.0
                        control.throttle = max(control.throttle, 0.26)
                    if cur_speed_kmh > 8.0:
                        control.brake = max(control.brake, 0.22)
                    if cur_speed_kmh > 14.0:
                        control.brake = max(control.brake, 0.36)
                    if lane_edge_risk_count >= 4:
                        control.brake = max(control.brake, 0.28)

                # Junction crawl assist: avoid prolonged near-zero deadlock during turn commitment.
                if (
                    junction_turn_mode
                    and (not lane_changing)
                    and npc_far_for_recovery
                    and (ped_dist is None or ped_dist > args.ped_detect_dist)
                    and cur_speed_kmh < 1.0
                ):
                    control.brake = 0.0
                    control.throttle = max(control.throttle, 0.30)

                # Anti-stuck: sharp turns and curb-edge contacts can lock ego at very low speed.
                npc_far_for_recovery = (npc_dist is None) or (npc_dist > (args.npc_stop_dist + 1.6))
                if (
                    (not lane_changing)
                    and npc_far_for_recovery
                    and (ped_dist is None or ped_dist > args.ped_detect_dist)
                    and cur_speed_kmh < 1.2
                    and obstacle_confirm_count < 3
                ):
                    # Deadlock release at intersections / split waypoints: force crawl forward.
                    if ego_wp_now is not None and ego_wp_now.is_junction:
                        try:
                            jn = ego_wp_now.next(max(6.0, float(track_lookahead)))
                            if jn:
                                j_target = jn[0]
                                if len(jn) > 1:
                                    yaw_v = ego_vehicle.get_transform().rotation.yaw
                                    pref = float(junction_turn_pref)
                                    def _jn_score(cand_wp):
                                        heading_signed = _signed_yaw_delta_deg(yaw_v, cand_wp.transform.rotation.yaw)
                                        score = abs(heading_signed)
                                        if abs(pref) > 0.2:
                                            if heading_signed * pref < -2.0:
                                                score += 18.0
                                            elif heading_signed * pref > 6.0:
                                                score -= 3.0
                                            if abs(heading_signed) < 4.0:
                                                score += 2.5
                                        return score
                                    j_target = min(jn, key=_jn_score)
                                j_ctrl = _control_to_waypoint(
                                    ego_vehicle,
                                    j_target,
                                    target_speed_kmh=max(12.0, min(args.target_speed, 18.0)),
                                    max_steer=min(dynamic_max_steer, 0.34),
                                )
                                j_ctrl.brake = 0.0
                                j_ctrl.throttle = max(j_ctrl.throttle, 0.38)
                                control = j_ctrl
                        except Exception:
                            pass
                    else:
                        control.brake = 0.0
                        control.throttle = max(control.throttle, 0.34)

                if (not lane_changing) and cur_speed_kmh < 4.0 and abs(control.steer) > 0.26 and npc_far_for_recovery:
                    control.steer = max(-0.24, min(0.24, control.steer))
                    control.brake = 0.0
                    control.throttle = max(control.throttle, 0.32)

                if (not lane_changing) and npc_far_for_recovery:
                    if cur_speed_kmh < 2.2:
                        stuck_counter += 1
                        if lane_edge_ratio > 0.88:
                            stuck_counter += 1
                        if turn_deg >= max(8.0, args.bf_turn_sharp_deg - 1.5) and cur_speed_kmh < 3.0:
                            stuck_counter += 1
                    else:
                        stuck_counter = max(0, stuck_counter - 2)
                else:
                    stuck_counter = 0

                # Curb-edge escape: actively steer away from road edge with throttle.
                if (not lane_changing) and npc_far_for_recovery and lane_edge_ratio > float(args.lv_edge_ratio_th) and stuck_counter > 24:
                    edge_sign = np.sign(lane_edge_offset) if abs(lane_edge_offset) > 1e-3 else 0.0
                    if edge_sign != 0.0:
                        control.steer += -edge_sign * 0.28
                    control.brake = 0.0
                    control.throttle = max(control.throttle, 0.38)
                    if stuck_counter > 70 and cur_speed_kmh < 1.0:
                        control.reverse = True
                        control.throttle = max(control.throttle, 0.24)
                        if edge_sign != 0.0:
                            control.steer = -edge_sign * 0.30

                if stuck_counter > 45:
                    try:
                        ego_wp_rec = world.get_map().get_waypoint(
                            ego_vehicle.get_location(),
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving,
                        )
                        if ego_wp_rec is not None:
                            rec_wps = ego_wp_rec.next(10.0)
                            rec_wp = rec_wps[0] if rec_wps else ego_wp_rec
                            rec_ctrl = _control_to_waypoint(
                                ego_vehicle,
                                rec_wp,
                                target_speed_kmh=max(args.npc_min_speed, 14.0),
                                max_steer=min(dynamic_max_steer, 0.30),
                            )
                            rec_ctrl.brake = 0.0
                            rec_ctrl.throttle = max(rec_ctrl.throttle, 0.40)
                            control = rec_ctrl
                    except Exception:
                        pass

                # Close-range lane-change recovery: short reverse to create lateral room.
                # Use a wider trigger than bumper-touch to avoid getting pinned behind a stopped lead.
                if lane_changing and npc_dist is not None and npc_dist < max(3.5, args.follow_min_gap + 0.5) and cur_speed_kmh < 1.0:
                    control.reverse = True
                    control.brake = 0.0
                    control.throttle = max(control.throttle, 0.25)
                    control.steer = max(-0.30, min(0.30, control.steer))

                # Apply steering constraints for ego_adv as well
                control.steer = max(-dynamic_max_steer, min(dynamic_max_steer, control.steer))
                delta_steer = control.steer - last_steer
                steer_rate_limit = dynamic_steer_rate
                if lane_changing:
                    # Allow faster but still bounded steering response while merging.
                    steer_rate_limit = max(steer_rate_limit, 0.14 if args.bf_dominant_level == "off" else 0.17)
                if delta_steer > steer_rate_limit:
                    control.steer = last_steer + steer_rate_limit
                elif delta_steer < -steer_rate_limit:
                    control.steer = last_steer - steer_rate_limit
                if abs(control.steer) > 0.25:
                    control.throttle = min(control.throttle, 0.25)
                    if cur_speed_kmh > 4.0:
                        control.brake = max(control.brake, 0.05)
                last_steer = control.steer
                prev_lane_edge_offset = lane_edge_offset
                ego_vehicle.apply_control(control)

            if args.show and view_windows:
                def render_view(renderer, frame, size):
                    if frame is None:
                        return
                    surf = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
                    if surf.get_width() != size[0] or surf.get_height() != size[1]:
                        surf = pygame.transform.smoothscale(surf, size)
                    tex = Texture.from_surface(renderer, surf)
                    renderer.clear()
                    renderer.blit(tex)
                    renderer.present()

                render_view(fp_renderer, latest_frames.get('view_fp'), (args.view_width, args.view_height))
                render_view(tp_renderer, latest_frames.get('view_tp'), (args.view_width, args.view_height))

            transform = ego_vehicle.get_transform()
            rel = carla.Location(x=-6.0, y=0.0, z=2.5)
            world_rel = carla.Transform(transform.location, transform.rotation).transform(rel)
            chase_rotation = carla.Rotation(pitch=-8.0, yaw=transform.rotation.yaw)
            spectator.set_transform(carla.Transform(world_rel, chase_rotation))
            

            if args.sync:
                while not sensor_queue.empty():
                    try:
                        frame_id, sensor_name = sensor_queue.get_nowait()
                    except:
                        break
            

            if frame_count % 50 == 0:
                location = ego_vehicle.get_location()
                velocity = ego_vehicle.get_velocity()
                speed_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                qsize = sensor_queue.qsize()
                obs_m = f"{latest_obstacle_dist:.2f}" if latest_obstacle_dist is not None else ""
                # Lane offset for debugging curb hits
                lane_offset = ""
                lane_keep_road = ""
                lane_keep_lane = ""
                lane_keep_offset = ""
                try:
                    wp = world.get_map().get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if wp is not None:
                        lane_tf = wp.transform
                        dx_lane = location.x - lane_tf.location.x
                        dy_lane = location.y - lane_tf.location.y
                        yaw_lane = np.deg2rad(lane_tf.rotation.yaw)
                        lat = -dx_lane * np.sin(yaw_lane) + dy_lane * np.cos(yaw_lane)
                        lane_offset = f"{lat:.2f}"
                        if lane_keep_lane_key is not None:
                            keep_wp = _find_target_lane_wp(wp, lane_keep_lane_key)
                            if keep_wp is not None:
                                lane_keep_road = str(keep_wp.road_id)
                                lane_keep_lane = str(keep_wp.lane_id)
                                k_tf = keep_wp.transform
                                k_dx = location.x - k_tf.location.x
                                k_dy = location.y - k_tf.location.y
                                k_yaw = np.deg2rad(k_tf.rotation.yaw)
                                k_lat = -k_dx * np.sin(k_yaw) + k_dy * np.cos(k_yaw)
                                lane_keep_offset = f"{k_lat:.2f}"
                except Exception:
                    lane_offset = ""
                    lane_keep_road = ""
                    lane_keep_lane = ""
                    lane_keep_offset = ""
                npc_m = f"{npc_dist:.2f}" if npc_dist is not None else ""
                ped_m = f"{ped_dist:.2f}" if ped_dist is not None else ""
                lc_road = ""
                lc_lane = ""
                if lane_change_target_lane is not None:
                    lc_road = str(lane_change_target_lane[0])
                    lc_lane = str(lane_change_target_lane[1])
                lc_hold = max(0.0, lane_change_hold_until - time.time())
                perf_log.write(
                    f"{frame_count},{delta_ms:.2f},{qsize},{speed_kmh:.2f},{obs_m},{last_steer:.3f},{lane_offset},{lane_keep_road},{lane_keep_lane},{lane_keep_offset},{npc_m},{ped_m},{closing_kmh:.2f},{1 if lane_changing else 0},{lc_road},{lc_lane},{lc_hold:.2f},{collision_count},{last_collision_frame},"
                    f"{1 if beam_enabled else 0},{int(bf_state.get('top1_beam', -1))},{float(bf_state.get('top1_prob', 0.0)):.4f},{float(bf_state.get('entropy', 1.0)):.4f},{float(bf_state.get('risk', 1.0)):.4f},{float(bf_state.get('speed_cap_kmh', args.target_speed)):.2f},{1 if bool(bf_state.get('allow_lane_change', True)) else 0},{float(turn_deg):.2f},{float(blocked_wait):.2f},"
                    f"{float(lv_available):.0f},{float(lv_conf):.3f},{float(lv_offset_norm):.3f},{float(lv_weight):.3f},{float(lv_fused_norm):.3f}\n"
                )
                perf_log.flush()
                if route_log:
                    route_log.write(f"{frame_count},{location.x:.3f},{location.y:.3f},{location.z:.3f},{speed_kmh:.2f}\n")
                if actor_log:
                    try:
                        ego_loc = ego_vehicle.get_location()
                        actor_log.write(f"{frame_count},ego,{ego_vehicle.id},{ego_loc.x:.3f},{ego_loc.y:.3f},{ego_loc.z:.3f},track\n")
                        for v in npc_vehicles:
                            loc = v.get_location()
                            actor_log.write(f"{frame_count},npc,{v.id},{loc.x:.3f},{loc.y:.3f},{loc.z:.3f},track\n")
                        for w in npc_walkers:
                            loc = w.get_location()
                            actor_log.write(f"{frame_count},ped,{w.id},{loc.x:.3f},{loc.y:.3f},{loc.z:.3f},track\n")
                        actor_log.flush()
                    except Exception:
                        pass

            if frame_count % 100 == 0:
                location = ego_vehicle.get_location()
                velocity = ego_vehicle.get_velocity()
                speed_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5

                sensor_info = ", ".join([f"{name}: {stats['count']}" for name, stats in sensor_stats.items()])
                npc_moving = sum(1 for v in npc_vehicles if v.get_velocity().length() > 0.1)

                print(f"[Status] Frame: {frame_count}, Location: ({location.x:.1f}, {location.y:.1f}), Speed: {speed_kmh:.1f} km/h, NPC moving: {npc_moving}/{len(npc_vehicles)}, Collisions: {collision_count}")
                if beam_enabled:
                    print(
                        f"         BeamFusion: beam={bf_state.get('top1_beam', -1)} conf={float(bf_state.get('top1_prob', 0.0)):.3f} risk={float(bf_state.get('risk', 1.0)):.3f} "
                        f"speed_cap={float(bf_state.get('speed_cap_kmh', args.target_speed)):.1f} lc={bool(bf_state.get('allow_lane_change', True))} "
                        f"turn_deg={float(turn_deg):.1f} blocked={float(blocked_wait):.1f}s "
                        f"lv_conf={float(lv_conf):.2f} lv_off={float(lv_offset_norm):.2f} lv_w={float(lv_weight):.2f}"
                    )
                print(f"         Sensors: {sensor_info}")
    
    except KeyboardInterrupt:
        print("\n\nCleaning up...")
    
    finally:

        print("Destroying sensors...")
        for sensor in sensors:
            sensor.destroy()
        
        print("Destroying Ego vehicle...")
        ego_vehicle.destroy()
        
        print("Destroying NPC vehicles...")
        for vehicle in npc_vehicles:
            vehicle.destroy()
        
        print("Destroying NPC walkers...")
        for walker in npc_walkers:
            walker.destroy()
        for controller in walker_controllers:
            controller.stop()
            controller.destroy()

        if args.show:
            try:
                if view_windows:
                    try:
                        if fp_window:
                            fp_window.destroy()
                        if tp_window:
                            tp_window.destroy()
                    except Exception:
                        pass
                pygame.quit()
            except:
                pass
        try:
            perf_log.close()
        except:
            pass
        try:
            if route_log:
                route_log.close()
        except:
            pass
        try:
            if actor_log:
                actor_log.close()
        except:
            pass
        try:
            if collision_log:
                collision_log.close()
        except:
            pass
        try:
            beam_policy.close_log()
        except Exception:
            pass
        

        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(False)
        

        print("\n" + "=" * 60)
        print("Data collection summary")
        print("=" * 60)
        for sensor_name, stats in sensor_stats.items():
            print(f"  {sensor_name}: {stats['count']} frames collected")
        
        print(f"\nData saved to: {os.path.abspath(RUN_DIR)}")
        print("Cleanup completed")

if __name__ == '__main__':
    main()
