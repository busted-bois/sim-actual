"""Main-branch fly2 course controller for make fly and make auto (VQ2 EKF pose)."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from rl import spec
from rl.calibration import load_calibration
from simulator.transforms import quat_to_yaw
from simulator.vq2_pose import VQ2PoseEstimator, spawn_position_ned

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GATE_MAP_PATH = os.path.join(_DATA_DIR, "gate_map.json")

HOVER_T = 0.27
KP_Z, KD_Z = 0.025, 0.030
K_ATT = 0.6
K_YAW = 0.4
# Body-rate damping (PD form, mirrors flightlab.controllers.PDController).
# 0.0 = pure-P behavior. Overridden by measured k_d from
# flightlab/calibration.json once the harness passes B2/B3 live
# (measured > guessed) — see _default_k_d().
K_D = 0.0
# Measured command-sign conventions vs ODOMETRY attitude (pitch normal;
# roll + yaw inverted). Valid when the attitude fed to the rate law comes
# from sim odometry (Training mode).
SIGN_ROLL = -1.0
SIGN_PITCH = +1.0
SIGN_YAW = -1.0
# HISTORICAL / FALSIFIED: the vq2 branch measured "inverted on all axes vs
# gyro-integrated attitude" (2026-07-01) and derived these signs -- but
# applying them to the EKF attitude flew the drone UPSIDE DOWN live
# (2026-07-02). Estimated attitude (accel-anchored) is truth-convention:
# use the default odometry signs. Kept only for reference/experiments.
EST_SIGNS = (-1.0, -1.0, -1.0)
RATE_CLIP = 0.30
YAW_CLIP = 0.5

# Signs measured live by the flightlab B0 harness override the constants above
# (measured > guessed). Written by `make attitude-harness`; absent until run.
_FLIGHTLAB_SIGNS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "flightlab", "signs.json"
)
_measured_signs_cache: tuple[float, float, float] | None | str = "unset"


def measured_signs() -> tuple[float, float, float] | None:
    """(roll, pitch, yaw) signs from flightlab/signs.json, or None if absent."""
    try:
        with open(_FLIGHTLAB_SIGNS_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return float(d["roll"]), float(d["pitch"]), float(d["yaw"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _default_signs() -> tuple[float, float, float]:
    global _measured_signs_cache
    if _measured_signs_cache == "unset":
        m = measured_signs()
        if m is not None:
            print(
                f"[fly2] using measured signs (flightlab/signs.json): {m}", flush=True
            )
        _measured_signs_cache = m
    return _measured_signs_cache or (SIGN_ROLL, SIGN_PITCH, SIGN_YAW)


_measured_k_d_cache: float | str = "unset"


def _default_k_d() -> float:
    """k_d from flightlab/calibration.json when measured, else module K_D."""
    global _measured_k_d_cache
    if _measured_k_d_cache == "unset":
        kd = load_calibration().get("k_d")
        if kd is not None:
            print(
                f"[fly2] using measured k_d (flightlab/calibration.json): {kd}",
                flush=True,
            )
        _measured_k_d_cache = float(kd) if kd is not None else K_D
    return _measured_k_d_cache


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


def rates_from_attitude_targets(
    roll,
    pitch,
    z,
    vz,
    tgt_roll,
    tgt_pitch,
    yaw_err,
    tgt_z,
    signs=None,
    k_d=None,
    gyro=(0.0, 0.0, 0.0),
):
    """Attitude-angle targets -> rate commands with measured sign conventions.

    `signs` selects the convention for the attitude SOURCE: default = the
    flightlab-measured signs when signs.json exists, else the odometry
    constants; pass EST_SIGNS when roll/pitch/yaw come from a gyro-integrated
    estimate (VQ2 regime). `k_d`/`gyro` add PD rate damping; k_d=None uses
    module K_D (0.0 = unchanged pure-P behavior).

    The damping term sits INSIDE the sign multiply: the plant responds
    rate = sign*cmd, so opposing a measured rate needs cmd = -sign*kd*rate.
    Outside the sign (the old form) it anti-damps on sign=-1 axes."""
    s_roll, s_pitch, s_yaw = signs if signs is not None else _default_signs()
    kd = _default_k_d() if k_d is None else k_d
    roll_cmd = float(
        np.clip(
            s_roll * (K_ATT * (tgt_roll - roll) - kd * gyro[0]),
            -RATE_CLIP,
            RATE_CLIP,
        )
    )
    pitch_cmd = float(
        np.clip(
            s_pitch * (K_ATT * (tgt_pitch - pitch) - kd * gyro[1]),
            -RATE_CLIP,
            RATE_CLIP,
        )
    )
    yaw_cmd = float(
        np.clip(s_yaw * (K_YAW * yaw_err - kd * gyro[2]), -YAW_CLIP, YAW_CLIP)
    )
    thrust = float(np.clip(HOVER_T + KP_Z * (z - tgt_z) + KD_Z * vz, 0.18, 0.5))
    return roll_cmd, pitch_cmd, yaw_cmd, thrust


@dataclass
class Fly2Config:
    speed: float = 2.8
    lean: float = 0.12
    klat: float = 0.04
    # Altitude trim around the gate OPENING CENTRE (NED: negative = higher).
    # Was -1.0 back when tgt_z aimed at the raw gate-map z; that z is the gate
    # BASE, so the old aim sat ~0.4 m below the centre (h/2 = 1.36 > 1.0).
    zoff: float = 0.0
    flipz: bool = False


def gate_target_z(gate: dict, cfg: Fly2Config) -> float:
    """NED z of the gate OPENING CENTRE (+ zoff trim).

    Gate-map/track-burst z is the gate BASE — rl/data/gate_map.json gate 0
    sits at z=-0.03 (ground level), impossible for a 2.72 m opening's centre.
    The centre is h/2 above the base (NED up = negative)."""
    z = float(gate["pos"][2])
    base_z = -z if cfg.flipz else z
    h = float(gate.get("h") or spec.GATE_SIZE_M)
    return base_z - h / 2.0 + cfg.zoff


class RateCmdShaper:
    """Low-pass + slew-limit rate cmds, slew-limit thrust, just before send.

    Same shaping math as flightlab.controllers.ShapedPDController, self-clocked
    so any caller loop rate works. Removes the step discontinuities that the
    90 Hz P-loop and gate retargets produce; tau=0.05 adds ~50 ms lag, harmless
    at the K_ATT=0.6 loop bandwidth. Thrust gets slew only (no lag): 0.6/s
    spreads the ~0.125 gate-advance thrust step over ~0.2 s.
    """

    def __init__(
        self, tau_s: float = 0.05, slew: float = 2.0, thrust_slew: float = 0.6
    ):
        self.tau = tau_s
        self.slew = slew
        self.thrust_slew = thrust_slew
        self.reset()

    def reset(self) -> None:
        self._filt = [0.0, 0.0, 0.0]
        self._last = [0.0, 0.0, 0.0]
        self._last_thrust: float | None = None
        self._t_prev: float | None = None

    def _dt(self) -> float:
        now = time.monotonic()
        dt = 1.0 / 90.0 if self._t_prev is None else now - self._t_prev
        self._t_prev = now
        return min(max(dt, 1.0 / 180.0), 1.0 / 30.0)

    def apply(
        self, roll_rate: float, pitch_rate: float, yaw_rate: float, thrust: float
    ) -> tuple[float, float, float, float]:
        dt = self._dt()
        alpha = dt / (self.tau + dt)
        out = []
        for i, v in enumerate((roll_rate, pitch_rate, yaw_rate)):
            self._filt[i] += alpha * (v - self._filt[i])
            step = self._filt[i] - self._last[i]
            lim = self.slew * dt
            self._last[i] += min(max(step, -lim), lim)
            out.append(self._last[i])
        if self._last_thrust is None:
            # First command passes through — slewing up from 0 would delay
            # takeoff thrust by ~0.5 s.
            self._last_thrust = float(thrust)
        else:
            lim = self.thrust_slew * dt
            self._last_thrust += min(max(thrust - self._last_thrust, -lim), lim)
        return out[0], out[1], out[2], self._last_thrust


class GateCarrot:
    """Slew the aim point between gates so retargets don't step the setpoints.

    When active_gate_index advances, bearing / cross-track / tgt_z all jump
    (~5 m of tgt_z on the climb course = an instant 0.125 thrust step). The
    carrot moves the aim from where it was toward the new gate's (xy,
    opening-centre z) at bounded speed instead. v_xy must exceed
    Fly2Config.speed (2.8) or the drone overruns its own aim.
    """

    def __init__(self, v_xy: float = 6.0, v_z: float = 2.0):
        self.v_xy = v_xy
        self.v_z = v_z
        self.reset()

    def reset(self) -> None:
        self._aim: list[float] | None = None
        self._t_prev: float | None = None

    def update(
        self, active: int, gate_map: list, cfg: Fly2Config
    ) -> tuple[float, float, float]:
        """Aim (x, y, tgt_z) for the current tick."""
        g = gate_map[active]
        tx, ty = float(g["pos"][0]), float(g["pos"][1])
        tz = gate_target_z(g, cfg)
        now = time.monotonic()
        dt = (
            0.0
            if self._t_prev is None
            else min(max(now - self._t_prev, 0.0), 1.0 / 30.0)
        )
        self._t_prev = now
        if self._aim is None:
            # First target (takeoff -> gate 0): aim straight at it.
            self._aim = [tx, ty, tz]
            return tx, ty, tz
        ax, ay, az = self._aim
        dx, dy = tx - ax, ty - ay
        d_xy = math.hypot(dx, dy)
        step_xy = self.v_xy * dt
        if d_xy <= step_xy or d_xy == 0.0:
            ax, ay = tx, ty
        else:
            ax += dx / d_xy * step_xy
            ay += dy / d_xy * step_xy
        dz = tz - az
        step_z = self.v_z * dt
        az = tz if abs(dz) <= step_z else az + math.copysign(step_z, dz)
        self._aim = [ax, ay, az]
        return ax, ay, az


def compute_course_rates(
    pos_ned,
    vel_ned,
    quat,
    active: int,
    gate_map: list,
    hold_z: float,
    cfg: Fly2Config,
    gyro=(0.0, 0.0, 0.0),
    carrot: GateCarrot | None = None,
    k_d=None,
):
    """One course control step. Returns (roll_rate, pitch_rate, yaw_rate, thrust).

    `carrot=None` aims straight at gate_map[active] (legacy behavior);
    passing a GateCarrot slews the aim across gate transitions instead."""
    p = np.asarray(pos_ned, float)
    v = np.asarray(vel_ned, float)
    roll, pitch, yaw = rpy(quat)
    z, vz = p[2], v[2]

    n = len(gate_map)
    if active >= n:
        return 0.0, 0.0, 0.0, HOVER_T

    if carrot is not None:
        gx, gy, tgt_z = carrot.update(active, gate_map, cfg)
    else:
        g = gate_map[active]["pos"]
        gx, gy = float(g[0]), float(g[1])
        tgt_z = gate_target_z(gate_map[active], cfg)
    dx, dy = gx - p[0], gy - p[1]
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

    return rates_from_attitude_targets(
        roll, pitch, z, vz, tgt_roll, tgt_pitch, yaw_err, tgt_z, k_d=k_d, gyro=gyro
    )


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
        self._carrot = GateCarrot()
        self._shaper = RateCmdShaper()
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
        self._carrot.reset()
        self._shaper.reset()
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
        self._carrot.reset()
        self._shaper.reset()
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

        gyro = (
            odo.get("roll_speed", 0.0),
            odo.get("pitch_speed", 0.0),
            odo.get("yaw_speed", 0.0),
        )
        roll_cmd, pitch_cmd, yaw_cmd, thrust = self._shaper.apply(
            *compute_course_rates(
                pos,
                vel,
                quat,
                active,
                self.gate_map,
                self.hold_z,
                self.config,
                gyro=gyro,
                carrot=self._carrot,
            )
        )

        z = pos[2]
        gb_z = (spec.quat_to_R(np.asarray(quat)).T @ np.array([0.0, 0, 1.0]))[2]
        unsafe = gb_z < 0.0 or z < self.hold_z - 30 or z > self.hold_z + 30
        if unsafe:
            self._unsafe_ticks += 1
            if self._unsafe_ticks >= 5:
                # Emergency level-hover stays unshaped: react immediately.
                self.controller.set_attitude_rates(0, 0, 0, HOVER_T)
            else:
                self.controller.set_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)
        else:
            self._unsafe_ticks = 0
            self.controller.set_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)
