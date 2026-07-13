"""VQ2 reactive vision pilot — main-branch camera-centering, fly2 inner loop.

Steers from gate_target (nx, ny, r_frac) without world pose or gate map.
Attitude from IMU complementary filter; altitude from pressure_alt.
Gate progression follows sim active_gate_index (authoritative in VQ2).
"""

from __future__ import annotations

import math
import time as _time

# Measured VQ2 dynamics (rl/fly2.py / rl.dynamics_id)
HOVER_T = 0.27
CRUISE_T = 0.30
KP_Z, KD_Z = 0.025, 0.030
K_ATT = 0.6
SIGN_ROLL = -1.0
SIGN_PITCH = +1.0
SIGN_YAW = -1.0
RATE_CLIP = 0.30
YAW_CLIP = 0.5

# Forward motion — main pilot pitch rates (VQ2 integrates rates, no auto-level)
CRUISE_PITCH_RATE = -0.2
SEARCH_FORWARD_PITCH = -0.04

GRAVITY = 9.81
CF_ALPHA = 0.02
DEFAULT_Z_HOLD = -5.0

# Vision guidance (main pilot technique)
VISION_YAW_GAIN = math.radians(40)
VISION_CENTER_DEADBAND = 0.30
VISION_MAX_AGE_S = 0.5
VISION_VY_GAIN = 6.0
VISION_MAX_ALT_ADJUST = 2.0
VISION_PROXIMITY_R_FRAC = 0.10
STABILIZE_HOLD_S = 0.1
POST_GATE_HOVER_S = 2.5

# Search when no gate in view
SEARCH_SWEEP_YAW_RATE = 0.8
SEARCH_SWEEP_PERIOD_S = 2.0
SEARCH_WARMUP_S = 1.5

OBSTACLE_CLEAR_ZONE = 0.25
DEBUG_LOG_INTERVAL_S = 2.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class _AttitudeEstimator:
    """Complementary filter: gyro integrate + accel gravity tilt."""

    def __init__(self) -> None:
        self.roll = 0.0
        self.pitch = 0.0
        self._last_imu_us: int | None = None

    def reset(self) -> None:
        self.roll = 0.0
        self.pitch = 0.0
        self._last_imu_us = None

    def update(self, imu: dict) -> tuple[float, float]:
        t_us = imu.get("time_us")
        if t_us is None:
            return self.roll, self.pitch

        dt = 0.0
        if self._last_imu_us is not None:
            dt = (int(t_us) - int(self._last_imu_us)) * 1e-6
        self._last_imu_us = int(t_us)

        gx = float(imu.get("gx", 0.0))
        gy = float(imu.get("gy", 0.0))
        ax = float(imu.get("ax", 0.0))
        ay = float(imu.get("ay", 0.0))
        az = float(imu.get("az", 0.0))

        if 0.0 < dt < 0.5:
            self.roll += gx * dt
            self.pitch += gy * dt

        norm = math.hypot(ax, ay, az)
        if norm > 0.5 * GRAVITY:
            ax_n, ay_n, az_n = ax / norm, ay / norm, az / norm
            roll_meas = math.atan2(ay_n, az_n)
            pitch_meas = math.atan2(-ax_n, math.hypot(ay_n, az_n))
            self.roll = CF_ALPHA * self.roll + (1.0 - CF_ALPHA) * roll_meas
            self.pitch = CF_ALPHA * self.pitch + (1.0 - CF_ALPHA) * pitch_meas

        return self.roll, self.pitch


class _AltitudeEstimator:
    """pressure_alt relative to latch at attempt start."""

    def __init__(self) -> None:
        self.z = DEFAULT_Z_HOLD
        self.vz = 0.0
        self._pressure_ref: float | None = None
        self._hold_z = DEFAULT_Z_HOLD
        self._last_z: float | None = None
        self._last_t: float | None = None

    def reset(self, hold_z: float = DEFAULT_Z_HOLD) -> None:
        self.z = hold_z
        self.vz = 0.0
        self._pressure_ref = None
        self._hold_z = hold_z
        self._last_z = None
        self._last_t = None

    def update(self, imu: dict | None) -> tuple[float, float]:
        if imu is None:
            return self.z, self.vz

        pa = imu.get("pressure_alt")
        if pa is not None:
            if self._pressure_ref is None:
                self._pressure_ref = float(pa)
            self.z = -(float(pa) - self._pressure_ref) + self._hold_z

        now = _time.monotonic()
        if self._last_z is not None and self._last_t is not None:
            dt = now - self._last_t
            if dt > 1e-4:
                vz_raw = (self.z - self._last_z) / dt
                self.vz = 0.85 * self.vz + 0.15 * vz_raw
        self._last_z = self.z
        self._last_t = now
        return self.z, self.vz

    def thrust(self, z_target: float, cruise: bool = False) -> float:
        error = self.z - z_target
        base = CRUISE_T if cruise else HOVER_T
        return float(_clamp(base + KP_Z * error + KD_Z * self.vz, 0.18, 0.5))


class VQ2VisionPilot:
    """make auto pilot: vision-centering + measured VQ2 rate loop."""

    def __init__(self, controller, data):
        self.controller = controller
        self.data = data
        self._att = _AttitudeEstimator()
        self._alt = _AltitudeEstimator()
        self._hold_z = DEFAULT_Z_HOLD
        self._advancing = False
        self._stabilize_start: float | None = None
        self._post_gate_time: float | None = None
        self._last_active_gate = 0
        self._searching = False
        self._search_yaw_dir = 1.0
        self._search_start_time: float | None = None
        self._last_debug_log = 0.0
        self._mode_str = "init"
        controller.set_control_mode("attitude")
        controller.set_attitude_rates(0, 0, 0, HOVER_T)
        print(
            "[vq2] reactive vision pilot ready (camera centering + IMU)",
            flush=True,
        )

    @property
    def gates_passed(self) -> int:
        return int(self.data.get("active_gate_index", 0) or 0)

    def on_attempt_start(self) -> None:
        imu = self.data.get("imu")
        if imu is not None and imu.get("pressure_alt") is not None:
            self._pressure_ref_at_start(float(imu["pressure_alt"]))
        else:
            self._alt.reset(self._hold_z)
        self._att.reset()
        self._advancing = False
        self._stabilize_start = None
        self._post_gate_time = None
        self._last_active_gate = int(self.data.get("active_gate_index", 0) or 0)
        self._searching = False
        self._search_yaw_dir = 1.0
        self._search_start_time = None
        self._last_debug_log = 0.0
        print(f"[vq2] attempt start hold_z={self._hold_z:.1f}", flush=True)

    def _pressure_ref_at_start(self, pa: float) -> None:
        self._alt.reset(self._hold_z)
        self._alt._pressure_ref = pa

    def reset_for_attempt(self) -> None:
        self._att.reset()
        self._alt.reset(self._hold_z)
        self._advancing = False
        self._stabilize_start = None
        self._post_gate_time = None
        self._last_active_gate = 0
        self._searching = False
        self._search_yaw_dir = 1.0
        self._search_start_time = None
        self._last_debug_log = 0.0
        self._mode_str = "reset"
        self.data.pop("gate_target", None)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, 0, 0, HOVER_T)

    def _level_roll_rate(self, roll: float) -> float:
        return float(_clamp(SIGN_ROLL * K_ATT * (0.0 - roll), -RATE_CLIP, RATE_CLIP))

    def _level_pitch_rate(self, pitch: float) -> float:
        return float(_clamp(SIGN_PITCH * K_ATT * (0.0 - pitch), -RATE_CLIP, RATE_CLIP))

    def _check_sim_gate_advance(self) -> bool:
        """True if sim advanced active_gate_index — triggers post-gate coast."""
        active = int(self.data.get("active_gate_index", 0) or 0)
        if active > self._last_active_gate:
            self._last_active_gate = active
            self._advancing = False
            self._stabilize_start = None
            self._post_gate_time = _time.monotonic()
            print(
                f"[vq2] sim gate advance active={active} — post-gate coast",
                flush=True,
            )
            return True
        return False

    def tick(self) -> None:
        if not self.data.get("armed", False):
            self._hover_level()
            return

        if self._check_sim_gate_advance():
            pass

        if self._post_gate_time is not None:
            elapsed = _time.monotonic() - self._post_gate_time
            if elapsed < POST_GATE_HOVER_S:
                self._mode_str = "post_gate"
                self._hover_level()
                return
            self._post_gate_time = None
            self._searching = True
            self._search_start_time = _time.monotonic()
            self._search_yaw_dir = 1.0
            print("[vq2] post-gate hover done — search", flush=True)

        imu = self.data.get("imu")
        roll, pitch = self._att.update(imu) if imu else (0.0, 0.0)
        self._alt.update(imu)

        gate_target = self.data.get("gate_target")
        cam = self.data.get("camera")
        if gate_target and gate_target.get("detected") and cam is not None:
            age = _time.monotonic() - cam.get("received_at", 0)
            if age < VISION_MAX_AGE_S:
                if self._searching:
                    self._searching = False
                    self._search_start_time = None
                    nx = float(gate_target.get("nx", 0.0))
                    print(
                        f"[vq2] gate acquired nx={nx:+.3f} — vision track",
                        flush=True,
                    )
                self._mode_str = "vision"
                self._fly_vision(gate_target, roll, pitch)
                return

        if self._searching:
            self._mode_str = "search"
            self._do_search(roll, pitch)
            return

        self._mode_str = "no_gate"
        self._advancing = False
        self._stabilize_start = None
        self._hover_level()

    def _fly_vision(self, gate_target: dict, roll: float, pitch: float) -> None:
        nx = float(gate_target.get("nx", 0.0))
        ny = float(gate_target.get("ny", 0.0))
        r_frac = float(gate_target.get("r_frac", 0.0))

        ny_offset = _clamp(
            ny * VISION_VY_GAIN, -VISION_MAX_ALT_ADJUST, VISION_MAX_ALT_ADJUST
        )
        z_target = self._hold_z + ny_offset
        centered = abs(nx) < VISION_CENTER_DEADBAND and abs(ny) < VISION_CENTER_DEADBAND

        yaw_cmd = float(_clamp(SIGN_YAW * VISION_YAW_GAIN * nx, -YAW_CLIP, YAW_CLIP))
        roll_cmd = self._level_roll_rate(roll)

        if not self._advancing and r_frac > 0.15:
            self._advancing = True
            self._stabilize_start = None
            print("[vq2] ADVANCE — gate large in frame", flush=True)

        if self._advancing:
            if abs(nx) > 0.7 or abs(ny) > 0.7:
                self._advancing = False
                self._stabilize_start = None
                pitch_cmd = self._level_pitch_rate(pitch)
                cruise = False
            elif r_frac >= VISION_PROXIMITY_R_FRAC:
                pitch_cmd = abs(CRUISE_PITCH_RATE) * 0.5
                cruise = False
            elif self._obstacle_blocking():
                pitch_cmd = 0.0
                cruise = False
            else:
                alignment = max(0.0, 1.0 - abs(nx))
                pitch_cmd = float(
                    _clamp(
                        CRUISE_PITCH_RATE * (0.35 + 0.65 * alignment),
                        -RATE_CLIP,
                        RATE_CLIP,
                    )
                )
                cruise = True
        else:
            pitch_cmd = self._level_pitch_rate(pitch)
            cruise = False
            if centered:
                if self._stabilize_start is None:
                    self._stabilize_start = _time.monotonic()
                elif _time.monotonic() - self._stabilize_start >= STABILIZE_HOLD_S:
                    self._advancing = True
                    self._stabilize_start = None
                    print("[vq2] ADVANCE — gate centered", flush=True)
            else:
                self._stabilize_start = None

        self._maybe_debug_log(nx, ny, r_frac)
        thrust = self._alt.thrust(z_target, cruise=cruise)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)

    def _do_search(self, roll: float, pitch: float) -> None:
        now = _time.monotonic()
        elapsed = now - (self._search_start_time or now)
        period_count = int(elapsed / SEARCH_SWEEP_PERIOD_S)
        new_dir = 1.0 if period_count % 2 == 0 else -1.0
        if new_dir != self._search_yaw_dir:
            self._search_yaw_dir = new_dir

        yaw_cmd = float(
            _clamp(
                SIGN_YAW * SEARCH_SWEEP_YAW_RATE * self._search_yaw_dir,
                -YAW_CLIP,
                YAW_CLIP,
            )
        )
        pitch_cmd = (
            0.0
            if elapsed < SEARCH_WARMUP_S
            else float(_clamp(SEARCH_FORWARD_PITCH, -RATE_CLIP, RATE_CLIP))
        )
        roll_cmd = self._level_roll_rate(roll)
        if pitch_cmd == 0.0:
            pitch_cmd = self._level_pitch_rate(pitch)
        thrust = self._alt.thrust(self._hold_z)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)

    def _obstacle_blocking(self) -> bool:
        obstacles = self.data.get("obstacles") or []
        return any(
            abs(o.get("nx", 0.0)) < OBSTACLE_CLEAR_ZONE and o.get("r_frac", 0.0) > 0.005
            for o in obstacles
        )

    def _hover_level(self) -> None:
        imu = self.data.get("imu")
        roll, pitch = self._att.update(imu) if imu else (0.0, 0.0)
        if imu:
            self._alt.update(imu)
        roll_cmd = self._level_roll_rate(roll)
        pitch_cmd = self._level_pitch_rate(pitch)
        thrust = self._alt.thrust(self._hold_z) if imu else HOVER_T
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(roll_cmd, pitch_cmd, 0.0, thrust)

    def _maybe_debug_log(self, nx: float, ny: float, r_frac: float) -> None:
        now = _time.monotonic()
        if now - self._last_debug_log < DEBUG_LOG_INTERVAL_S:
            return
        self._last_debug_log = now
        active = int(self.data.get("active_gate_index", 0) or 0)
        print(
            f"[vq2] {self._mode_str} active={active} adv={self._advancing} "
            f"nx={nx:+.2f} ny={ny:+.2f} r={r_frac:.3f}",
            flush=True,
        )
