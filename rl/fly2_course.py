"""Main-branch fly2 course controller for make fly and make auto (VQ2 EKF pose)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

from rl import spec
from simulator.transforms import quat_to_yaw
from simulator.vq2_pose import VQ2PoseEstimator, spawn_position_ned

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GATE_MAP_PATH = os.path.join(_DATA_DIR, "gate_map.json")

HOVER_T = 0.27
KP_Z, KD_Z = 0.025, 0.030
K_ATT = 0.6
K_YAW = 0.4
SIGN_ROLL = -1.0
SIGN_PITCH = +1.0
SIGN_YAW = -1.0
RATE_CLIP = 0.30
YAW_CLIP = 0.5


def rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
    yaw = quat_to_yaw(w, x, y, z)
    return roll, pitch, yaw


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def detect_climb_course(gate_map: list) -> bool:
    """True when gate 1 is notably higher than gate 0 (climb course)."""
    if len(gate_map) < 2:
        return False
    z0 = float(gate_map[0]["pos"][2])
    z1 = float(gate_map[1]["pos"][2])
    return z1 > z0 + 2.0


def track_gates_to_gate_map(track_gates: list) -> list:
    out = []
    for i, g in enumerate(track_gates):
        if hasattr(g, "pos_ned"):
            pos = g.pos_ned
            orient = g.orient_quat
            width = g.width_m
            height = g.height_m
            gate_id = getattr(g, "gate_id", i)
        else:
            pos = g.get("position_ned")
            orient = g.get("orientation_ned")
            width = g.get("width", spec.GATE_SIZE_M)
            height = g.get("height", spec.GATE_SIZE_M)
            gate_id = g.get("gate_id", i)
        if not pos or not orient:
            continue
        if abs(pos[0]) + abs(pos[1]) + abs(pos[2]) < 0.01:
            continue
        out.append(
            {
                "id": gate_id,
                "pos": list(pos[:3]),
                "quat": list(orient[:4]),
                "w": width,
                "h": height,
            }
        )
    return out


def resolve_gate_map(data: dict) -> list:
    """Live track burst when poses valid; else rl/data/gate_map.json for VQ2."""
    if data.get("track_positions_valid") is not False:
        track = data.get("track_gates") or data.get("gates") or []
        gm = track_gates_to_gate_map(track)
        if gm:
            return gm
    if not os.path.isfile(GATE_MAP_PATH):
        return []
    try:
        with open(GATE_MAP_PATH, encoding="utf-8") as f:
            return json.load(f)["gates"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return []


@dataclass
class Fly2Config:
    speed: float = 2.8
    lean: float = 0.12
    klat: float = 0.04
    zoff: float = -1.0
    flipz: bool = False


def compute_course_rates(
    pos_ned,
    vel_ned,
    quat,
    active: int,
    gate_map: list,
    hold_z: float,
    cfg: Fly2Config,
):
    """One course control step. Returns (roll_rate, pitch_rate, yaw_rate, thrust)."""
    p = np.asarray(pos_ned, float)
    v = np.asarray(vel_ned, float)
    roll, pitch, yaw = rpy(quat)
    z, vz = p[2], v[2]

    n = len(gate_map)
    if active >= n:
        return 0.0, 0.0, 0.0, HOVER_T

    g = np.asarray(gate_map[active]["pos"], float)
    dx, dy = g[0] - p[0], g[1] - p[1]
    bearing = math.atan2(dy, dx)
    yaw_err = wrap(bearing - yaw)
    speed = float(np.linalg.norm(v[:2]))
    e_cross = dx * math.sin(yaw) - dy * math.cos(yaw)
    align = max(0.0, 1.0 - abs(yaw_err) / 0.4)
    dist = math.hypot(dx, dy)
    v_des = cfg.speed * align * min(1.0, 0.3 + dist / 10.0)
    lean = float(np.clip(0.05 * (v_des - speed), -0.05, cfg.lean))
    tgt_pitch = -lean
    tgt_roll = float(np.clip(cfg.klat * e_cross, -0.12, 0.12))
    tgt_z = (-g[2] if cfg.flipz else g[2]) + cfg.zoff

    roll_cmd = float(
        np.clip(SIGN_ROLL * K_ATT * (tgt_roll - roll), -RATE_CLIP, RATE_CLIP)
    )
    pitch_cmd = float(
        np.clip(SIGN_PITCH * K_ATT * (tgt_pitch - pitch), -RATE_CLIP, RATE_CLIP)
    )
    yaw_cmd = float(np.clip(SIGN_YAW * K_YAW * yaw_err, -YAW_CLIP, YAW_CLIP))
    thrust = float(np.clip(HOVER_T + KP_Z * (z - tgt_z) + KD_Z * vz, 0.18, 0.5))
    return roll_cmd, pitch_cmd, yaw_cmd, thrust


class Fly2CoursePilot:
    """Drop-in pilot: main fly2 course law + VQ2 EKF pose when odom absent."""

    def __init__(self, controller, data, config: Fly2Config | None = None):
        self.controller = controller
        self.data = data
        self.config = config or Fly2Config()
        self.gate_map: list = []
        self.hold_z = 0.0
        self._last_active = -1
        self._unsafe_ticks = 0
        self._pose = VQ2PoseEstimator()
        controller.set_control_mode("attitude")
        controller.set_attitude_rates(0, 0, 0, HOVER_T)
        print("[fly2] main course pilot ready (VQ2 EKF + gate_map)", flush=True)

    @property
    def gates_passed(self) -> int:
        return 0

    def on_attempt_start(self) -> None:
        self.gate_map = resolve_gate_map(self.data)
        self.config.flipz = detect_climb_course(self.gate_map)
        if self.config.flipz:
            print("[fly2] climb course detected — flipz=True", flush=True)
        odo = self.data.get("odometry")
        if odo is not None:
            self.hold_z = odo.get("z", 0.0)
        else:
            self.hold_z = float(
                spawn_position_ned(self.gate_map, flipz=self.config.flipz)[2]
            )
        self._pose.reset(self.gate_map, hold_z=self.hold_z, flipz=self.config.flipz)
        self._last_active = -1
        self._unsafe_ticks = 0
        src = (
            "track burst"
            if self.data.get("track_positions_valid") is not False and self.gate_map
            else GATE_MAP_PATH
        )
        print(
            f"[fly2] loaded {len(self.gate_map)} gates from {src} hold_z={self.hold_z:.1f}",
            flush=True,
        )
        if not self.gate_map:
            print(
                "ERROR: no gate map — run `make capture-gates` in VQ2 TRAINING first",
                flush=True,
            )

    def reset_for_attempt(self) -> None:
        self.gate_map = []
        self._last_active = -1
        self._unsafe_ticks = 0
        self._pose = VQ2PoseEstimator()
        self.data.pop("gate_target", None)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, 0, 0, HOVER_T)

    def _current_odometry(self) -> dict | None:
        odo = self.data.get("odometry")
        if odo is not None:
            return odo
        if not self.gate_map:
            return None
        return self._pose.tick(self.data, self.gate_map)

    def tick(self) -> None:
        odo = self._current_odometry()
        if not odo or not self.gate_map:
            self.controller.set_attitude_rates(0, 0, 0, HOVER_T)
            return

        quat = (odo["qw"], odo["qx"], odo["qy"], odo["qz"])
        pos = (odo["x"], odo["y"], odo["z"])
        vel = (odo.get("vx", 0), odo.get("vy", 0), odo.get("vz", 0))
        active = int(self.data.get("active_gate_index", 0) or 0)

        if active != self._last_active:
            print(f"[fly2] ACTIVE GATE -> {active}", flush=True)
            self._last_active = active

        roll_cmd, pitch_cmd, yaw_cmd, thrust = compute_course_rates(
            pos,
            vel,
            quat,
            active,
            self.gate_map,
            self.hold_z,
            self.config,
        )

        z = pos[2]
        gb_z = (spec.quat_to_R(np.asarray(quat)).T @ np.array([0.0, 0, 1.0]))[2]
        unsafe = gb_z < 0.0 or z < self.hold_z - 30 or z > self.hold_z + 30
        if unsafe:
            self._unsafe_ticks += 1
            if self._unsafe_ticks >= 5:
                self.controller.set_attitude_rates(0, 0, 0, HOVER_T)
            else:
                self.controller.set_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)
        else:
            self._unsafe_ticks = 0
            self.controller.set_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)
