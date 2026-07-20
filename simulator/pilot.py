"""Pilot — classical-vision gate racer on the attitude-quaternion wire.

Called each cycle by controller.update(). Reads shared_data (written by
mavlink_rx + vision_rx) and drives the controller.

This is the default `make sim` pilot. It was rewritten to adopt the control
core proven by IBVSPilot/GPPilot (the wire that actually flew the course),
because the old body-rate version backflipped at GO and could not steer:

  1. attitude_quat wire + hover 0.264  (was `attitude` body-rates + 0.5 hover,
     which the sim integrates with no auto-level -> tumble at launch)
  2. self-leveling via a complementary IMU tilt filter, commanding
     (desired - measured) degrees with the GP KP/KR/KY signs
  3. a SETTLE launch phase (level + see before flying) + command slew limiting
  4. aim at the gate OPENING (hole centre + inner-corner spread) with camera-tilt
     compensation, not the orange body blob

Reused wholesale from simulator/ibvs_pilot.py: _TiltFilter, _CmdSlew, the wire
constants/limits, AIM geometry, and the RIGHT/soft-launch safety idioms.
"""

from __future__ import annotations

import math
import time as _time

from simulator.ibvs_pilot import (
    CAM_TILT_RAD,
    CRUISE_PITCH_DEG,
    FX,
    HOVER_THRUST,
    IMG_H,
    KP,
    KR,
    LEAN_RAMP_S,
    MAX_BANK_DEG,
    PITCH_WIRE_MAX_DEG,
    RIGHT_THRUST,
    YAW_WIRE_MAX_DEG,
    _CmdSlew,
    _TiltFilter,
    _wrap,
)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
CONTROL_HZ = 60  # spec VADR-TS-003 4.4 caps command rate < 100 Hz

# Vision guidance (pixel servoing on the gate opening) --------------------------
VISION_MAX_AGE_S = 0.5  # ignore gate_target older than this
LOST_HOLD_S = 0.6  # keep servoing on the frozen target this long after loss
K_YAW = 1.0  # yaw_err_deg = K_YAW * deg(atan(ex))
# Cap the yaw command hard: a large yaw setpoint on the attitude_quat wire couples
# into roll and (with the roll-leveling loop) runs the drone over to 180° roll.
# Steer gently in yaw and let repeated frames walk the heading onto the gate.
YAW_MAX_DEG = 4.0
# Sign of the yaw command. If the drone yaws AWAY from a tracked gate (|ex| grows
# instead of shrinking), flip this to +1 — a wrong yaw sign saturates the steer
# and is what couples into the roll flip.
YAW_SIGN = -1.0
# Keep the approach lean gentle: a large forward lean flips the tilt filter into
# gyro-only mode (see HARD_LEAN_DEG below), which drifts over a multi-second
# approach and tumbles. Slower + flatter keeps the attitude estimate valid.
K_SIZE_DEG = 2.5  # forward lean (deg) when the opening is far/small
S_NEAR_PX = 200.0  # inner spread (px) where forward lean tapers to coast
LEAN_COAST_DEG = 1.5  # residual nose-down when near
HARD_LEAN_DEG = 3.0  # above this |pitch_des|, integrate gyro-only (accel invalid)

# Commit: a timed blind dash once the opening is close + centred ---------------
S_COMMIT_PX = 260.0  # inner spread (~1.85 m) that triggers the dash
S_DROP_COMMIT_PX = 190.0  # spread at last sight for a dropout-commit
EX_COMMIT = 0.10  # |ex| (normalized) gate to commit
EY_COMMIT = 0.16  # |ey| (normalized) gate to commit
COMMIT_LEAN_DEG = 2.5  # nose-down during the dash (gentler — was 4.0)
COMMIT_TRIM_DEG = 6.0  # yaw trim while the gate stays visible in COMMIT
T_COMMIT = 1.5  # dash duration (s) — shorter blind window
BRAKE_LEAN_DEG = 3.0  # nose-up while braking after the dash
T_BRAKE = 1.0  # brake duration (s)

# Altitude — closed loop when a reference exists (odometry z, else pressure_alt
# relative to the launch pad), open-loop launch climb when neither does. The old
# body-rate pilot climbed on a fixed 0.55 trim; here the trim is the measured
# hover (0.264) and the loop supplies the climb, so we never blow the pad.
Z_RACE_TARGET_NED = -3.0  # racing altitude to hold pre-gate (m, NED: up = neg)
KP_Z = 0.030  # thrust per metre of altitude error
KD_Z = 0.020  # thrust per m/s of vertical rate (damping)
Z_EY_GAIN = 3.0  # metres of altitude bias per unit pixel vertical error
TH_MIN, TH_MAX = 0.20, 0.50  # closed-loop authority (climb off pad, gentle down)
# No-reference blind mode: bounded around hover, but with enough headroom to hold
# a climb toward a high gate once tracking.
TH_MIN_BLIND, TH_MAX_BLIND = 0.245, 0.330
K_TH_EY = 0.10  # blind-mode thrust per unit normalized vertical error
# Blind launch: get airborne open-loop when there is no altitude reference.
# Must clearly overcome the pad — 0.33 (barely over hover 0.264) did not lift.
# Safe to be aggressive now because the launch self-levels (no backflip).
LAUNCH_THRUST = 0.48  # decisive climb off the pad
T_LAUNCH_BLIND = 3.0  # blind-climb duration (s) before handing to guidance

# SETTLE = the pre-flight COUNTDOWN hold. The drone stays on the pad (level +
# converging the tilt estimate) for the full race countdown so it never lifts off
# before GO — fixes the early start. Held at least COUNTDOWN_HOLD_S before LAUNCH.
COUNTDOWN_HOLD_S = 3.0  # race countdown: no flight until this elapses (no early start)
T_SETTLE_MIN = COUNTDOWN_HOLD_S
T_SETTLE_MAX = COUNTDOWN_HOLD_S + 2.0
SETTLE_ATT_TOL = 0.1  # rad: tilt estimate must agree with accel this closely

# Search sweep when nothing is in view -----------------------------------------
SCAN_YAW_DEG = 8.0
SCAN_FLIP_S = 6.0
POST_GATE_COAST_S = 2.0

# Telemetry fallback (make sim / TRAINING has odometry; VQ2 does not) ----------
TELEMETRY_YAW_MAX_DEG = 10.0
PASSED_GATE_NEAR_M = 3.0
PASSED_GATE_ANGLE_RAD = math.radians(45)

STATUS_LOG_INTERVAL_S = 0.5  # denser trace to read the ex/attitude trend
UNSAFE_TICKS_TO_RIGHT = 5


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class Pilot:
    """Gate-traversal pilot: classical HSV vision on the attitude_quat wire."""

    def __init__(self, controller, data):  # type: ignore[type-arg]
        self.controller = controller
        self.data = data
        self._gates_passed = 0
        self.tilt = _TiltFilter()
        self._slew = _CmdSlew(CONTROL_HZ)
        self._init_flight_state()
        self._mode_str = "init"
        controller.control_hz = CONTROL_HZ
        controller.set_control_mode("attitude_quat")
        controller.set_attitude_quat_deg(0.0, 0.0, 0.0, HOVER_THRUST)
        print("[pilot] ready — attitude_quat wire, hole-aim, self-level", flush=True)

    def _init_flight_state(self) -> None:
        self._phase = "SETTLE"
        self._settle_t0: float | None = None
        self._settle_synced = False
        self._fly_since: float | None = None
        # held gate target (pixels + normalized errors)
        self._ctr: tuple[float, float] | None = None
        self._ex = 0.0
        self._ey = 0.0
        self._spread = 0.0
        self._last_seen: float | None = None
        self._last_frame_id: int | None = None
        # commit / brake clocks
        self._commit_until = 0.0
        self._brake_until = 0.0
        # altitude reference (odometry z when present, else pressure_alt vs pad)
        self._pa_ref: float | None = None  # pad pressure_alt latched at first read
        self._z_prev: float | None = None
        self._z_prev_t: float | None = None
        self._vz_est = 0.0  # filtered vertical rate (m/s, NED: down = +)
        self._launch_t0: float | None = None
        # heading estimate (integrated yaw gyro) so yaw can be commanded RELATIVE
        # to the current heading — gravity gives no yaw reference. Latched to 0 at
        # the launch heading; drift is fine because steering is differential.
        self._yaw_est = 0.0  # rad
        self._yaw_prev_us: float | None = None
        # search
        self._scan_dir = 1.0
        self._scan_since: float | None = None
        # post-gate coast (sim active_gate_index advance)
        self._post_gate_until = 0.0
        self._last_active = 0
        # passed-gate rejection map (needs odometry)
        self._passed_positions: list[tuple[float, float, float]] = []
        # safety
        self._unsafe_ticks = 0
        self._last_tgt_pitch_deg = 0.0
        self._last_status_log = 0.0

    @property
    def gates_passed(self) -> int:
        return self._gates_passed

    def on_attempt_start(self) -> None:
        """make auto hook — re-settle at each GO."""
        self._reset()

    def reset_for_attempt(self) -> None:
        """Clear flight state after a sim reset between auto-flight retries."""
        self._reset()
        self.data.pop("gate_target", None)

    def _reset(self) -> None:
        self.tilt.reset()
        self._slew.reset()
        self._gates_passed = 0
        self._init_flight_state()
        self._mode_str = "reset"
        self.controller.set_control_mode("attitude_quat")
        self.controller.set_attitude_quat_deg(0.0, 0.0, 0.0, HOVER_THRUST)

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------
    def _alt_state(self, imu: dict | None, now: float):  # type: ignore[type-arg]
        """(z, vz) NED altitude + rate in metres, or None when no reference.

        Odometry z (TRAINING) is authoritative. Otherwise pressure_alt relative
        to the pad latched at first valid read (VQ2 in-race). VQ2 idle reports
        pressure_alt=NaN -> None -> the pilot falls back to the blind launch.
        """
        odo = self.data.get("odometry")
        if odo is not None and odo.get("z") is not None:
            z = float(odo["z"])
            vz = float(odo.get("vz", 0.0))
            self._observe_z(z, now)  # keep filtered vz warm for a later handoff
            return z, vz
        pa = imu.get("pressure_alt") if imu else None
        if pa is None or not math.isfinite(float(pa)):
            return None
        pa = float(pa)
        if self._pa_ref is None:
            self._pa_ref = pa
        z = -(pa - self._pa_ref)  # pad = 0, climbing = more negative (NED)
        self._observe_z(z, now)
        return z, self._vz_est

    def _observe_z(self, z: float, now: float) -> None:
        if self._z_prev is not None and self._z_prev_t is not None:
            dt = now - self._z_prev_t
            if dt > 0.02:
                r = (z - self._z_prev) / dt
                self._vz_est = 0.7 * self._vz_est + 0.3 * _clamp(r, -8.0, 8.0)
                self._z_prev, self._z_prev_t = z, now
        else:
            self._z_prev, self._z_prev_t = z, now

    def _integrate_yaw(self, imu: dict) -> None:  # type: ignore[type-arg]
        """Dead-reckon heading from the yaw gyro so yaw can be commanded relative
        to it. NaN-guarded; absolute drift is harmless (steering is differential)."""
        t_us = imu.get("time_us")
        gz = imu.get("gz")
        if t_us is None or gz is None or not math.isfinite(float(gz)):
            return
        t_us = float(t_us)
        if self._yaw_prev_us is not None:
            dt = (t_us - self._yaw_prev_us) * 1e-6
            if 0.0 < dt < 0.5:
                self._yaw_est = _wrap(self._yaw_est + float(gz) * dt)
        self._yaw_prev_us = t_us

    def _aim_ny(self) -> float:
        """Normalized vertical aim offset for the up-tilted camera. A gate at
        the drone's own altitude sits BELOW image centre; aiming at 0 makes the
        drone climb forever (IBVS AIM_V, expressed resolution-independently)."""
        return (FX / (IMG_H / 2.0)) * math.tan(CAM_TILT_RAD + self.tilt.pitch)

    def _ingest_vision(self, now: float) -> None:
        """Pull the freshest gate_target, preferring the hole over the body blob."""
        gt = self.data.get("gate_target")
        cam = self.data.get("camera")
        if not gt or not gt.get("detected") or cam is None:
            return
        if now - cam.get("received_at", 0.0) >= VISION_MAX_AGE_S:
            return
        fid = gt.get("frame_id")
        if fid is not None and fid == self._last_frame_id:
            return
        self._last_frame_id = fid
        if gt.get("has_hole"):
            ex = float(gt.get("hole_nx", 0.0))
            ny = float(gt.get("hole_ny", 0.0))
            spread = float(gt.get("spread_px", 0.0))
            u, v = float(gt.get("hole_u", 0.0)), float(gt.get("hole_v", 0.0))
        else:
            ex = float(gt.get("nx", 0.0))
            ny = float(gt.get("ny", 0.0))
            spread = float(gt.get("w_px", 0.0)) * 0.55  # inner opening ~55% of body
            u, v = float(gt.get("u_px", 0.0)), float(gt.get("v_px", 0.0))
        if not (math.isfinite(ex) and math.isfinite(ny) and math.isfinite(spread)):
            return
        self._ctr = (u, v)
        self._ex = ex
        self._ey = ny - self._aim_ny()  # camera-tilt compensated vertical error
        self._spread = spread
        self._last_seen = now

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def tick(self) -> None:
        now = _time.monotonic()
        imu = self.data.get("imu")

        # Attitude estimate (gated like IBVS: don't blend accel under hard accel;
        # hard-sync only quasi-static during SETTLE).
        alt = self._alt_state(imu, now)  # (z, vz) NED metres, or None (blind)
        quasi = abs(self._vz_est) < 0.5
        if imu:
            # Gyro-only only under a genuinely steep lean. COMMIT is now gentle
            # (small lean), so keep accel-correcting through it — dropping to
            # gyro-only over the multi-second dash is what drifted into a tumble.
            hard = abs(self._last_tgt_pitch_deg) > HARD_LEAN_DEG
            static_now = self._phase == "SETTLE" and quasi and self.tilt.gyro_mag < 0.05
            allow = (not hard) and (self._phase != "SETTLE" or static_now)
            self.tilt.update(imu, allow_accel=allow, boost=static_now)
            self._integrate_yaw(imu)

        if not self.data.get("armed", False):
            self._mode_str = "disarmed"
            self._phase = "SETTLE"
            self._settle_t0 = None
            self._settle_synced = False
            self._launch_t0 = None
            self._command(0.0, 0.0, 0.0, None, None, now)
            return

        # SETTLE — the pre-flight COUNTDOWN hold: sit level on the pad at hover
        # thrust and converge the tilt estimate for the full race countdown. No
        # climb here (LAUNCH owns takeoff), so the drone never lifts off early.
        if self._phase == "SETTLE":
            self._mode_str = "countdown"
            if self._settle_t0 is None:
                self._settle_t0 = now
            if not self._settle_synced:
                self._settle_synced = self.tilt.sync_to_accel()
            dt = now - self._settle_t0
            converged = (
                self.tilt.f_ok
                and abs(self.tilt.roll - self.tilt.roll_acc) < SETTLE_ATT_TOL
                and abs(self.tilt.pitch - self.tilt.pitch_acc) < SETTLE_ATT_TOL
            )
            # Never launch before the countdown elapses, even if already level.
            if (dt >= T_SETTLE_MIN and converged) or dt >= T_SETTLE_MAX:
                self._phase = "LAUNCH"
                self._launch_t0 = now
                self._fly_since = now
                print(f"[pilot] countdown done ({dt:.1f}s) — launching", flush=True)
            else:
                self._command(0.0, 0.0, 0.0, None, None, now)
                return

        # LAUNCH — get airborne. Closed loop climbs to racing altitude; with no
        # altitude reference (VQ2 pressure_alt=NaN) climb open-loop for a window.
        if self._phase == "LAUNCH":
            self._mode_str = "launch"
            launched = now - (self._launch_t0 or now)
            if alt is not None:
                if alt[0] <= Z_RACE_TARGET_NED + 0.5 or launched > 6.0:
                    self._phase = "SCAN"
                else:
                    self._command(0.0, 0.0, 0.0, None, alt, now)
                    return
            elif launched < T_LAUNCH_BLIND:
                self._command(
                    0.0, 0.0, 0.0, None, None, now, thrust_override=LAUNCH_THRUST
                )
                return
            else:
                self._phase = "SCAN"

        # Post-gate coast whenever the sim advances the active gate.
        active = int(self.data.get("active_gate_index", 0) or 0)
        if active > self._last_active:
            self._last_active = active
            self._gates_passed = max(self._gates_passed, active)
            self._post_gate_until = now + POST_GATE_COAST_S
            self._ctr = None
            self._last_seen = None
            self._phase = "SCAN"
            self._commit_until = 0.0
        if now < self._post_gate_until:
            self._mode_str = "post_gate"
            self._command(0.0, 0.0, 0.0, None, alt, now)
            return

        self._ingest_vision(now)

        # COMMIT / BRAKE are timed and override guidance.
        if self._phase == "COMMIT":
            if now < self._commit_until:
                self._mode_str = "commit"
                yaw = 0.0
                if self._last_seen is not None and now - self._last_seen < 0.3:
                    yaw = _clamp(
                        math.degrees(math.atan(self._ex)),
                        -COMMIT_TRIM_DEG,
                        COMMIT_TRIM_DEG,
                    )
                self._command(0.0, -COMMIT_LEAN_DEG, yaw, None, alt, now)
                return
            # A finished blind dash is NOT a confirmed pass — with no gate
            # telemetry we can't know we cleared it. Count passes only from the
            # sim's active_gate_index (synced in tick); just log the attempt.
            self._record_passed_position()
            self._phase = "BRAKE"
            self._brake_until = now + T_BRAKE
            self._ctr = None
            self._last_seen = None
            print("[pilot] dash complete (unconfirmed) — braking", flush=True)
        if self._phase == "BRAKE":
            if now < self._brake_until:
                self._mode_str = "brake"
                self._command(0.0, BRAKE_LEAN_DEG, 0.0, None, alt, now)
                return
            self._phase = "SCAN"

        # Vision tracking — the opening is fresh and not a gate we already flew.
        seen_age = None if self._last_seen is None else now - self._last_seen
        tracking = seen_age is not None and seen_age < LOST_HOLD_S
        if tracking and not self._is_passed_gate(self._ex):
            self._mode_str = "track"
            self._track(now, alt, fresh=seen_age < 0.2)
            return

        # Telemetry fallback — yaw toward the nearest un-passed gate (odometry).
        yaw_deg = self._telemetry_yaw_deg()
        if yaw_deg is not None:
            self._mode_str = "telemetry"
            self._command(0.0, CRUISE_PITCH_DEG, yaw_deg, None, alt, now)
            return

        # Nothing to fly toward — sweep.
        self._mode_str = "scan"
        self._scan(now, alt)

    # ------------------------------------------------------------------
    # Guidance branches (each produces desired attitude, not raw rates)
    # ------------------------------------------------------------------
    def _track(self, now: float, alt, fresh: bool) -> None:
        self._phase = "TRACK"
        centred = abs(self._ex) < EX_COMMIT and abs(self._ey) < EY_COMMIT
        dropped_close = (
            not fresh and self._spread >= S_DROP_COMMIT_PX and abs(self._ex) < EX_COMMIT
        )
        if (self._spread >= S_COMMIT_PX and centred and fresh) or dropped_close:
            self._phase = "COMMIT"
            self._commit_until = now + T_COMMIT
            print(f"[pilot] COMMIT S={self._spread:.0f} ex={self._ex:+.2f}", flush=True)
            self._command(0.0, -COMMIT_LEAN_DEG, 0.0, None, alt, now)
            return
        yaw_deg = _clamp(
            K_YAW * math.degrees(math.atan(self._ex)),
            -YAW_WIRE_MAX_DEG,
            YAW_WIRE_MAX_DEG,
        )
        lean = K_SIZE_DEG * max(0.0, 1.0 - self._spread / S_NEAR_PX)
        pitch_des = -max(lean, LEAN_COAST_DEG)
        self._command(0.0, pitch_des, yaw_deg, self._ey, alt, now)

    def _scan(self, now: float, alt) -> None:
        # Search in place: hover (no forward lean) and yaw-sweep to bring a gate
        # into view. Leaning forward while sweeping blindly curves the drone off
        # course (it drifts away / ends up facing the wrong way) with no position
        # feedback to recover; TRACK owns all forward motion, once a gate is seen.
        if self._scan_since is None:
            self._scan_since = now
        elif now - self._scan_since > SCAN_FLIP_S:
            self._scan_dir = -self._scan_dir
            self._scan_since = now
        self._command(0.0, 0.0, self._scan_dir * SCAN_YAW_DEG, None, alt, now)

    # ------------------------------------------------------------------
    # Telemetry fallback + passed-gate rejection (odometry only)
    # ------------------------------------------------------------------
    def _odometry_yaw(self, odo: dict) -> float:  # type: ignore[type-arg]
        qw, qx = odo.get("qw", 1.0), odo.get("qx", 0.0)
        qy, qz = odo.get("qy", 0.0), odo.get("qz", 0.0)
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def _telemetry_yaw_deg(self) -> float | None:
        odo = self.data.get("odometry")
        gates = self.data.get("track_gates")
        if not odo or not gates:
            return None
        ox, oy = odo.get("x", 0.0), odo.get("y", 0.0)
        best = None
        best_d = float("inf")
        for g in gates:
            pos = g.get("position_ned")
            if not pos or len(pos) < 3:
                continue
            if abs(pos[0]) + abs(pos[1]) + abs(pos[2]) < 0.01:
                continue
            dx, dy = pos[0] - ox, pos[1] - oy
            d = dx * dx + dy * dy
            if d < best_d and not self._is_passed_position(pos[0], pos[1]):
                best_d, best = d, (dx, dy)
        if best is None:
            return None
        bearing = math.atan2(best[1], best[0])
        err = _wrap(bearing - self._odometry_yaw(odo))
        return _clamp(math.degrees(err), -TELEMETRY_YAW_MAX_DEG, TELEMETRY_YAW_MAX_DEG)

    def _record_passed_position(self) -> None:
        odo = self.data.get("odometry")
        if odo is not None:
            self._passed_positions.append(
                (odo.get("x", 0.0), odo.get("y", 0.0), odo.get("z", 0.0))
            )

    def _is_passed_position(self, gx: float, gy: float) -> bool:
        for px, py, _pz in self._passed_positions:
            if math.hypot(gx - px, gy - py) < PASSED_GATE_NEAR_M:
                return True
        return False

    def _is_passed_gate(self, ex: float) -> bool:
        """True if the detection bearing matches a gate we already flew through."""
        odo = self.data.get("odometry")
        if odo is None or not self._passed_positions:
            return False
        dx0, dy0 = odo.get("x", 0.0), odo.get("y", 0.0)
        bearing = self._odometry_yaw(odo) + math.atan(ex)
        for px, py, _pz in self._passed_positions:
            dx, dy = px - dx0, py - dy0
            if math.hypot(dx, dy) < PASSED_GATE_NEAR_M:
                return True
            if abs(_wrap(bearing - math.atan2(dy, dx))) < PASSED_GATE_ANGLE_RAD:
                return True
        return False

    # ------------------------------------------------------------------
    # Wire: desired attitude (deg) -> attitude_quat command, with self-level,
    # soft-launch ramp, thrust, slew, and an inverted-attitude righting guard.
    # ------------------------------------------------------------------
    def _thrust(self, pixel_ey: float | None, alt) -> float:
        if alt is None:
            # No altitude reference: pixel-servo vertical if tracking, else hover.
            # Tight authority so a missing sensor can never run the drone away.
            t = HOVER_THRUST
            if pixel_ey is not None:  # ey>0 (gate below aim) -> we're high -> descend
                t -= K_TH_EY * pixel_ey
            return _clamp(t, TH_MIN_BLIND, TH_MAX_BLIND)
        z, vz = alt
        if pixel_ey is not None:
            z_target = z + Z_EY_GAIN * pixel_ey  # servo gate to the aim line
        else:
            z_target = Z_RACE_TARGET_NED  # climb to / hold racing altitude
        t = HOVER_THRUST + KP_Z * (z - z_target) + KD_Z * vz
        return _clamp(t, TH_MIN, TH_MAX)

    def _command(
        self,
        roll_des_deg: float,
        pitch_des_deg: float,
        yaw_err_deg: float,
        pixel_ey: float | None,
        alt,
        now: float,
        thrust_override: float | None = None,
    ) -> None:
        roll_deg = math.degrees(_wrap(self.tilt.roll))
        pitch_deg = math.degrees(_wrap(self.tilt.pitch))

        # Soft launch: no dive past cruise lean for LEAN_RAMP_S after SETTLE.
        if (
            self._fly_since is not None
            and (now - self._fly_since) < LEAN_RAMP_S
            and pitch_des_deg < CRUISE_PITCH_DEG
        ):
            pitch_des_deg = CRUISE_PITCH_DEG

        thrust = (
            thrust_override
            if thrust_override is not None
            else self._thrust(pixel_ey, alt)
        )
        pitch_cmd = _clamp(
            (pitch_des_deg - pitch_deg) * KP, -PITCH_WIRE_MAX_DEG, PITCH_WIRE_MAX_DEG
        )
        roll_cmd = _clamp((roll_des_deg - roll_deg) * KR, -MAX_BANK_DEG, MAX_BANK_DEG)
        # Yaw is an ABSOLUTE-heading setpoint on this wire, so steer relative to
        # the integrated heading: command "current heading + a bounded turn toward
        # the gate". yaw_err_deg==0 (settle/launch/hover) holds the heading.
        steer_deg = _clamp(yaw_err_deg * YAW_SIGN, -YAW_MAX_DEG, YAW_MAX_DEG)
        yaw_cmd = math.degrees(_wrap(self._yaw_est)) + steer_deg

        unsafe = math.cos(self.tilt.roll) * math.cos(self.tilt.pitch) < 0.1
        self._unsafe_ticks = self._unsafe_ticks + 1 if unsafe else 0
        self._last_tgt_pitch_deg = pitch_des_deg

        if self._unsafe_ticks >= UNSAFE_TICKS_TO_RIGHT:
            # Inverted/tumbling — level the quat setpoint, low thrust, no dash.
            self._commit_until = 0.0
            self._phase = "SCAN"
            r_cmd = _clamp((0.0 - roll_deg) * KR, -MAX_BANK_DEG, MAX_BANK_DEG)
            p_cmd = _clamp(
                (0.0 - pitch_deg) * KP, -PITCH_WIRE_MAX_DEG, PITCH_WIRE_MAX_DEG
            )
            hold_yaw = math.degrees(_wrap(self._yaw_est))
            self.controller.set_attitude_quat_deg(r_cmd, p_cmd, hold_yaw, RIGHT_THRUST)
        else:
            roll_cmd, pitch_cmd, _yaw_slew, thrust = self._slew.apply(
                roll_cmd, pitch_cmd, yaw_cmd, thrust
            )
            # yaw_cmd is an absolute heading (wraps at ±180); send it un-slewed so
            # the wrap discontinuity doesn't ramp the command through 360°.
            self.controller.set_attitude_quat_deg(roll_cmd, pitch_cmd, yaw_cmd, thrust)

        if now - self._last_status_log >= STATUS_LOG_INTERVAL_S:
            self._last_status_log = now
            z_s = "--" if alt is None else f"{alt[0]:+.1f}"
            print(
                f"[pilot] {self._mode_str}/{self._phase} ex={self._ex:+.2f} "
                f"ey={self._ey:+.2f} S={self._spread:.0f} passed={self._gates_passed} "
                f"pd={pitch_des_deg:+.1f} th={thrust:.3f} z={z_s} "
                f"roll={roll_deg:+.0f} pitch={pitch_deg:+.0f} "
                f"yaw={math.degrees(_wrap(self._yaw_est)):+.0f} ycmd={yaw_cmd:+.0f}",
                flush=True,
            )
