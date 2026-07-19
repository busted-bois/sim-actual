"""IBVS auto-flight pilot: fly the gate the camera SEES, in image space.

No position estimate. Inputs: YOLO pixels + HIGHRES_IMU (tilt CF + pressure).

Control (image-based visual servoing) on the 7d04af1 attitude_quat wire
(degree setpoints, not body rates — rates backflipped at GO):
  yaw     <- bearing to opening centre (deg)
  thrust  <- vertical pixel error vs AIM_V (20 deg cam) + pressure hold
  fwd     <- soft nose-down lean from apparent size (deg, GP-style clamps)
  COMMIT  <- timed blind dash when spread + centred
"""

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np

# Wire encoding from commit 7d04af1 GPPilot: degree attitude on the
# attitude-quaternion path. Body-rate SET_ATTITUDE_TARGET backflipped at GO.
HOVER_THRUST = 0.264
IBVS_CONTROL_HZ = 60
KP, KR, KY = 1.0, -1.0, -1.0  # same signs as GPPilot compute_guidance
PITCH_WIRE_MAX_DEG = 12.0
MAX_BANK_DEG = 10.0
YAW_WIRE_MAX_DEG = 12.0
CMD_SLEW_DEG_S = 90.0
THRUST_SLEW_PER_S = 1.0
LEAN_RAMP_S = 2.5  # after SETTLE: no dive past cruise pitch
CRUISE_PITCH_DEG = -2.0  # light nose-down (GP DESIRED_PITCH_DEG)
COMMIT_PITCH_DEG = -4.0
BRAKE_PITCH_DEG = 3.0
SCAN_YAW_DEG = 8.0
RIGHT_THRUST = 0.20

IMG_W, IMG_H = 640.0, 360.0
CX, CY, FX = 320.0, 180.0, 320.0
CAM_TILT_RAD = math.radians(20.0)
# A gate at the drone's own altitude sits BELOW the image centre by the camera
# up-tilt: v_aim = cy + fx*tan(20 deg) ~= 296 px. Aiming at cy instead makes
# the drone climb forever.
AIM_V = CY + FX * math.tan(CAM_TILT_RAD)

KP_CONF = 0.7  # keypoint confidence for "corner is real"
STATUS_LOG_INTERVAL_S = 1.0
NAV_LOG_INTERVAL_S = 0.1


class _CmdSlew:
    """Rate-limit attitude/thrust on the quat wire (same idea as GP CommandSlew)."""

    def __init__(self, hz: float = IBVS_CONTROL_HZ):
        self._max_deg = CMD_SLEW_DEG_S / hz
        self._max_thrust = THRUST_SLEW_PER_S / hz
        self._prev: tuple[float, float, float, float] | None = None

    def reset(self) -> None:
        self._prev = None

    def apply(
        self, roll: float, pitch: float, yaw: float, thrust: float
    ) -> tuple[float, float, float, float]:
        if self._prev is None:
            self._prev = (roll, pitch, yaw, thrust)
            return self._prev
        pr, pp, py, pt = self._prev
        m = self._max_deg
        out = (
            pr + float(np.clip(roll - pr, -m, m)),
            pp + float(np.clip(pitch - pp, -m, m)),
            py + float(np.clip(yaw - py, -m, m)),
            pt + float(np.clip(thrust - pt, -self._max_thrust, self._max_thrust)),
        )
        self._prev = out
        return out


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class _TiltFilter:
    """Complementary roll/pitch from HIGHRES_IMU: gyro integration corrected
    toward the accel gravity direction when the specific force looks like
    gravity (|f| near g). Every sample is NaN-guarded -- one bad packet must
    not poison the attitude the way it poisoned the EKF pose."""

    def __init__(self, alpha=0.02):
        self.alpha = alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.roll_acc = 0.0  # last accel-only tilt, logged so a single run
        self.pitch_acc = 0.0  # shows whether gyro and accel conventions agree
        self.f_ok = False  # last specific force was gravity-like (tilt valid)
        self.gyro_mag = 0.0  # |gyro| of the last sample (rotation = not static)
        self._last_us = None

    def reset(self):
        self.roll, self.pitch, self._last_us = 0.0, 0.0, None
        self.roll_acc, self.pitch_acc = 0.0, 0.0
        self.f_ok = False
        self.gyro_mag = 0.0

    def update(self, imu: dict, allow_accel: bool = True, boost: bool = False) -> None:
        """allow_accel=False integrates gyro only: under sustained commanded
        acceleration (commit dash) the specific force tilts away from gravity
        and the accel blend biases pitch backward -- over-lean, runaway speed,
        the late tumble of the first live IBVS runs. boost=True blends hard
        (quasi-static: the accel IS gravity, trust it)."""
        vals = [imu.get(k) for k in ("ax", "ay", "az", "gx", "gy", "gz", "time_us")]
        if any(v is None for v in vals) or not all(np.isfinite(float(v)) for v in vals):
            return
        ax, ay, az, gx, gy, gz, t_us = (float(v) for v in vals)
        self.gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
        if self._last_us is not None:
            dt = (t_us - self._last_us) * 1e-6
            if 0.0 < dt < 0.5:
                self.roll += gx * dt
                self.pitch += gy * dt
        self._last_us = t_us
        f = math.sqrt(ax * ax + ay * ay + az * az)
        self.f_ok = 8.0 < f < 12.0
        if self.f_ok:  # not accelerating hard: accel points along gravity
            self.roll_acc = math.atan2(-ay, -az)
            self.pitch_acc = math.atan2(ax, math.hypot(ay, az))
            if allow_accel:
                a = 0.25 if boost else self.alpha
                self.roll = (1 - a) * self.roll + a * self.roll_acc
                self.pitch = (1 - a) * self.pitch + a * self.pitch_acc

    def sync_to_accel(self) -> bool:
        """Hard-set the estimate to the accel tilt. Only valid quasi-static
        (spawn, resting) -- the caller gates that; returns True on success."""
        if not self.f_ok:
            return False
        self.roll, self.pitch = self.roll_acc, self.pitch_acc
        return True


class IBVSPilot:
    """Drop-in auto-loop pilot: pixel servoing on the detected gate."""

    def __init__(self, controller, data):
        self.controller = controller
        self.data = data
        self.p = dict(
            box_conf=0.5,  # min YOLO box confidence
            min_box_w=14.0,  # ignore boxes narrower than this (px, background)
            match_px=140.0,  # held-target update must be this close (px)
            lost_hold_t=0.5,  # keep servoing on the frozen error this long (s)
            k_yaw_ex=1.0,  # yaw_err_deg = k * deg(atan(ex))
            k_th_ey=0.10,  # thrust per unit normalized vertical error
            k_pa=0.020,  # thrust per metre of pressure-alt error (holds)
            k_pad=0.015,  # thrust per m/s of pressure-alt rate (damping)
            th_clip=0.035,  # thrust authority around hover (+/-)
            # Soft degree leans (7d04af1 GP wire). Former rad leans backflipped.
            k_size_deg=5.0,  # forward lean at S=0 (far), tapering to coast
            s_near=200.0,  # inner-corner spread (px) where lean tapers out
            lean_coast_deg=2.0,  # residual nose-down when near
            s_commit=260.0,  # spread (px, ~1.85 m) to trigger the blind dash
            s_drop_commit=190.0,  # spread at last sight for a dropout-commit
            ex_commit=0.08,  # |ex| gate to commit (normalized)
            ey_commit=0.14,  # |ey| gate to commit (normalized)
            commit_lean_deg=4.0,  # nose-down during blind dash
            t_commit=2.2,  # blind dash duration (s)
            commit_trim_deg=8.0,  # yaw trim while gate still visible in COMMIT
            t_brake=1.2,  # reverse-lean brake after the dash (s)
            brake_lean_deg=3.0,  # nose-up while braking
            scan_yaw_deg=SCAN_YAW_DEG,  # yaw sweep while scanning
            t_settle_min=2.0,  # post-reset SETTLE: level + see before flying
            t_settle_max=5.0,
            settle_climb=0.4,  # gentle climb (m) — was 1.5, blew pad early
            settle_att_tol=0.1,
            settle_frames=5,
            quasi_pa=0.3,
            down_cos=-0.3,
            down_t=1.0,
        )
        self.n_passed = 0
        self.phase = "SETTLE"  # never fly blind: orient + see, then chase
        self._settle_t0 = None
        self._settle_synced = False
        self._frames_seen = 0
        self._down_since = None
        self._down_logged = False
        self.tilt = _TiltFilter()
        self._cmd_slew = _CmdSlew()
        self._fly_since = None  # lean-ramp clock after SETTLE
        self._ctr = None  # (u, v) of the held gate centre
        self._ex = 0.0
        self._ey = 0.0
        self._spread = 0.0
        self._last_seen = None
        self._commit_until = 0.0
        self._brake_until = 0.0
        self._pa_hold = None  # pressure-alt to hold when not pixel-servoing
        self._pa_prev = None
        self._pa_rate = 0.0
        self._pa_prev_t = None
        self._last_frame_id = None
        self._unsafe_ticks = 0
        self._last_tgt_pitch_deg = 0.0
        self._scan_dir = 1.0
        self._scan_since = None
        self._last_status_log = 0.0
        self._nav_log = None
        self._nav_wr = None
        self._last_nav_log = 0.0
        controller.control_hz = IBVS_CONTROL_HZ
        controller.set_control_mode("attitude_quat")
        controller.set_attitude_quat_deg(0.0, 0.0, 0.0, HOVER_THRUST)
        print("[ibvs] pixel-servo pilot ready (attitude_quat wire)", flush=True)

    @property
    def gates_passed(self) -> int:
        return self.n_passed

    def _reset_state(self) -> None:
        self.n_passed = 0
        self.phase = "SETTLE"
        self._settle_t0 = None
        self._settle_synced = False
        self._frames_seen = 0
        self._down_since = None
        self._down_logged = False
        self.tilt.reset()
        self._cmd_slew.reset()
        self._fly_since = None
        self._ctr = None
        self._ex = 0.0
        self._ey = 0.0
        self._spread = 0.0
        self._last_seen = None
        self._commit_until = 0.0
        self._brake_until = 0.0
        self._pa_hold = None
        self._pa_prev = None
        self._pa_rate = 0.0
        self._pa_prev_t = None
        self._last_frame_id = None
        self._unsafe_ticks = 0
        self._last_tgt_pitch_deg = 0.0
        self._scan_dir = 1.0
        self._scan_since = None

    def on_attempt_start(self) -> None:
        self._reset_state()
        self._open_nav_log()
        self.controller.set_control_mode("attitude_quat")
        self.controller.set_attitude_quat_deg(0.0, 0.0, 0.0, HOVER_THRUST)

    def reset_for_attempt(self) -> None:
        self._reset_state()
        self._close_nav_log()
        self.data.pop("pose", None)
        self.controller.set_control_mode("attitude_quat")
        self.controller.set_attitude_quat_deg(0.0, 0.0, 0.0, HOVER_THRUST)

    # --- telemetry --------------------------------------------------------
    def _open_nav_log(self) -> None:
        self._close_nav_log()
        try:
            os.makedirs(os.path.join("rl", "data"), exist_ok=True)
            path = os.path.join(
                "rl", "data", time.strftime("nav_log_ibvs_%Y%m%d_%H%M%S.csv")
            )
            self._nav_log = open(path, "w", newline="")
            self._nav_wr = csv.writer(self._nav_log)
            self._nav_wr.writerow(
                "t phase ex ey spread roll pitch roll_acc pitch_acc n_passed".split()
            )
            print(f"[ibvs] nav log -> {path}", flush=True)
        except OSError as e:  # telemetry must never ground the pilot
            print(f"[ibvs] nav log unavailable: {e}", flush=True)
            self._nav_log, self._nav_wr = None, None

    def _close_nav_log(self) -> None:
        if self._nav_log is not None:
            try:
                self._nav_log.close()
            except OSError:
                pass
        self._nav_log, self._nav_wr = None, None

    # --- vision -----------------------------------------------------------
    def _gate_pixels(self, g):
        """(centre_u, centre_v, inner_spread_px) of one detection, from the
        confident INNER corners (the opening) when possible, else the box."""
        box = np.asarray(g.get("box"), float).reshape(-1)
        kxy = g.get("keypoints")
        kcf = g.get("keypoint_conf")
        if kxy is not None and kcf is not None:
            kxy = np.asarray(kxy, float)
            kcf = np.asarray(kcf, float)
            inner = kxy[:4]
            ok = kcf[:4] > KP_CONF
            if ok.sum() >= 2 and np.isfinite(inner[ok]).all():
                pts = inner[ok]
                ctr = pts.mean(axis=0)
                spread = float(pts[:, 0].max() - pts[:, 0].min())
                if ok.sum() < 3:  # spread of 2 corners can be degenerate
                    spread = max(spread, float(box[2] - box[0]) * 0.55)
                return float(ctr[0]), float(ctr[1]), spread
        # box fallback: inner opening is ~55% of the outer-frame box width
        return (
            float((box[0] + box[2]) / 2.0),
            float((box[1] + box[3]) / 2.0),
            float(box[2] - box[0]) * 0.55,
        )

    def _ingest(self, now: float) -> None:
        pose = self.data.get("pose")
        if not pose or pose.get("frame_id") == self._last_frame_id:
            return
        self._last_frame_id = pose.get("frame_id")
        self._frames_seen += 1  # fresh frames of the CURRENT world (SETTLE gate)
        cands = []
        for g in pose.get("gates") or []:
            if float(g.get("conf", 0.0) or 0.0) < self.p["box_conf"]:
                continue
            box = g.get("box")
            if box is None:
                continue
            b = np.asarray(box, float).reshape(-1)
            if not np.isfinite(b).all() or (b[2] - b[0]) < self.p["min_box_w"]:
                continue
            u, v, s = self._gate_pixels(g)
            if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(s)):
                continue
            cands.append((float(g["conf"]), u, v, s))
        if not cands:
            return
        if self._ctr is not None:
            # Track discipline: only the detection nearest the held centre may
            # update it -- chasing the most confident box each frame thrashes
            # between the near gate and background gates.
            best = min(
                cands,
                key=lambda c: math.hypot(c[1] - self._ctr[0], c[2] - self._ctr[1]),
            )
            if (
                math.hypot(best[1] - self._ctr[0], best[2] - self._ctr[1])
                > self.p["match_px"]
            ):
                return
        else:
            best = max(cands, key=lambda c: c[0])
        _, u, v, s = best
        self._ctr = (u, v)
        self._ex = (u - CX) / CX
        # The aim line moves with body pitch: leaning forward tilts the camera
        # down, the gate rides UP in the frame, and an uncompensated aim makes
        # the drone climb off the axis (measured ~0.7 m high at the plane).
        aim_v = CY + FX * math.tan(CAM_TILT_RAD + self.tilt.pitch)
        self._ey = (v - aim_v) / IMG_H
        self._spread = s
        self._last_seen = now

    # --- altitude (pressure) ----------------------------------------------
    def _update_pressure(self, imu: dict, now: float) -> float | None:
        pa = imu.get("pressure_alt") if imu else None
        if pa is None or not np.isfinite(float(pa)):
            return None
        pa = float(pa)
        if self._pa_prev is not None and self._pa_prev_t is not None:
            dt = now - self._pa_prev_t
            if dt > 0.02:
                r = (pa - self._pa_prev) / dt
                self._pa_rate = 0.7 * self._pa_rate + 0.3 * float(np.clip(r, -8, 8))
                self._pa_prev, self._pa_prev_t = pa, now
        else:
            self._pa_prev, self._pa_prev_t = pa, now
        if self._pa_hold is None:
            self._pa_hold = pa
        return pa

    def _thrust(self, pa: float | None, pixel_ey: float | None) -> float:
        p = self.p
        t = HOVER_THRUST
        if pixel_ey is not None:
            # gate below the aim line (ey>0) -> we are high -> descend
            t -= p["k_th_ey"] * pixel_ey
        elif pa is not None and self._pa_hold is not None:
            t += p["k_pa"] * (self._pa_hold - pa)
        if pa is not None:
            t -= p["k_pad"] * self._pa_rate  # damp vertical rate
        return float(
            np.clip(t, HOVER_THRUST - p["th_clip"], HOVER_THRUST + p["th_clip"])
        )

    # --- main loop ----------------------------------------------------------
    def tick(self) -> None:
        p = self.p
        now = time.monotonic()
        imu = self.data.get("imu")
        quasi = abs(self._pa_rate) < p["quasi_pa"]
        if imu:
            hard_accel = (
                self.phase == "COMMIT" or abs(self._last_tgt_pitch_deg) > 2.0
            )
            static_now = (
                self.phase in ("SETTLE", "DOWN") and quasi and self.tilt.gyro_mag < 0.05
            )
            allow = (not hard_accel) and (self.phase != "SETTLE" or static_now)
            self.tilt.update(imu, allow_accel=allow, boost=static_now)
        pa = self._update_pressure(imu, now)
        self._ingest(now)

        roll, pitch = _wrap(self.tilt.roll), _wrap(self.tilt.pitch)
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        # Desired absolute attitude (deg). Wire = GP-style error * KP/KR/KY.
        pitch_des_deg = 0.0
        roll_des_deg = 0.0
        yaw_err_deg = 0.0
        pixel_ey = None

        inverted_acc = (
            self.tilt.f_ok
            and quasi
            and math.cos(self.tilt.roll_acc) * math.cos(self.tilt.pitch_acc)
            < p["down_cos"]
        )
        if inverted_acc:
            if self._down_since is None:
                self._down_since = now
            elif now - self._down_since > p["down_t"] and self.phase != "DOWN":
                self.phase = "DOWN"
                self._commit_until = 0.0
                if not self._down_logged:
                    self._down_logged = True
                    print("[ibvs] GROUNDED inverted -- waiting for reset", flush=True)
        else:
            self._down_since = None
            if self.phase == "DOWN":
                self.phase = "SETTLE"
                self._settle_t0 = None
                self._settle_synced = False
                self._fly_since = None
        down = self.phase == "DOWN"

        # SETTLE: level quat + soft climb. No rate fighting the spawn tilt.
        settling = self.phase == "SETTLE"
        if settling:
            if self._settle_t0 is None:
                self._settle_t0 = now
                self._frames_seen = 0
                if pa is not None:
                    self._pa_hold = pa + p["settle_climb"]
            if not self._settle_synced:
                self._settle_synced = self.tilt.sync_to_accel()
            dt_settle = now - self._settle_t0
            converged = (
                self.tilt.f_ok
                and abs(roll - self.tilt.roll_acc) < p["settle_att_tol"]
                and abs(pitch - self.tilt.pitch_acc) < p["settle_att_tol"]
            )
            ready = (
                dt_settle >= p["t_settle_min"]
                and converged
                and self._frames_seen >= p["settle_frames"]
            )
            if ready or dt_settle >= p["t_settle_max"]:
                print(
                    f"[ibvs] SETTLE done in {dt_settle:.1f}s "
                    f"(frames={self._frames_seen} "
                    f"pitch={pitch_deg:+.0f}deg)",
                    flush=True,
                )
                self.phase = "SCAN"
                settling = False
                self._fly_since = now
                if pa is not None:
                    self._pa_hold = pa
            pitch_des_deg = 0.0
            roll_des_deg = 0.0
            yaw_err_deg = 0.0

        if not settling and not down and now < self._commit_until:
            self.phase = "COMMIT"
            pitch_des_deg = -p["commit_lean_deg"]
            if self._last_seen is not None and now - self._last_seen < 0.3:
                yaw_err_deg = float(
                    np.clip(
                        math.degrees(math.atan(self._ex)),
                        -p["commit_trim_deg"],
                        p["commit_trim_deg"],
                    )
                )
        elif self.phase == "COMMIT":
            self.n_passed += 1
            print(f"[ibvs] PASSED g{self.n_passed} (timed dash)", flush=True)
            self.phase = "BRAKE"
            self._brake_until = now + p["t_brake"]
            self._ctr = None
            self._last_seen = None
            self._pa_hold = self._pa_prev

        if self.phase == "BRAKE":
            if now < self._brake_until:
                pitch_des_deg = p["brake_lean_deg"]
            else:
                self.phase = "SCAN"

        if not settling and not down and self.phase not in ("COMMIT", "BRAKE"):
            seen_age = None if self._last_seen is None else now - self._last_seen
            tracking = seen_age is not None and seen_age < p["lost_hold_t"]
            if tracking:
                self.phase = "TRACK"
                centred = (
                    abs(self._ex) < p["ex_commit"] and abs(self._ey) < p["ey_commit"]
                )
                fresh = seen_age < 0.2
                dropped_close = (
                    not fresh
                    and self._spread >= p["s_drop_commit"]
                    and abs(self._ex) < p["ex_commit"]
                )
                if (
                    self._spread >= p["s_commit"] and centred and fresh
                ) or dropped_close:
                    self.phase = "COMMIT"
                    self._commit_until = now + p["t_commit"]
                    self._pa_hold = self._pa_prev
                    pitch_des_deg = -p["commit_lean_deg"]
                    print(
                        f"[ibvs] COMMIT S={self._spread:.0f} ex={self._ex:+.2f}",
                        flush=True,
                    )
                else:
                    yaw_err_deg = float(
                        np.clip(
                            p["k_yaw_ex"] * math.degrees(math.atan(self._ex)),
                            -YAW_WIRE_MAX_DEG,
                            YAW_WIRE_MAX_DEG,
                        )
                    )
                    lean = p["k_size_deg"] * max(
                        0.0, 1.0 - self._spread / p["s_near"]
                    )
                    pitch_des_deg = -max(lean, p["lean_coast_deg"])
                    pixel_ey = self._ey
            else:
                if self._last_seen is not None:
                    self._ctr = None
                self.phase = "SCAN"
                if self._scan_since is None:
                    self._scan_since = now
                elif now - self._scan_since > 6.0:
                    self._scan_dir = -self._scan_dir
                    self._scan_since = now
                yaw_err_deg = self._scan_dir * p["scan_yaw_deg"]
                pitch_des_deg = CRUISE_PITCH_DEG
                self._pa_hold = self._pa_hold if self._pa_hold is not None else pa

        # Soft launch: no dive past cruise for LEAN_RAMP_S after SETTLE.
        if (
            self._fly_since is not None
            and not settling
            and not down
            and (now - self._fly_since) < LEAN_RAMP_S
            and pitch_des_deg < CRUISE_PITCH_DEG
        ):
            pitch_des_deg = CRUISE_PITCH_DEG

        thrust = self._thrust(pa, pixel_ey)
        # 7d04af1 GP wire: degree error * KP/KR/KY on attitude_quat.
        pitch_cmd = float(
            np.clip(
                (pitch_des_deg - pitch_deg) * KP,
                -PITCH_WIRE_MAX_DEG,
                PITCH_WIRE_MAX_DEG,
            )
        )
        roll_cmd = float(
            np.clip((roll_des_deg - roll_deg) * KR, -MAX_BANK_DEG, MAX_BANK_DEG)
        )
        yaw_cmd = float(
            np.clip(yaw_err_deg * KY, -YAW_WIRE_MAX_DEG, YAW_WIRE_MAX_DEG)
        )

        unsafe = math.cos(roll) * math.cos(pitch) < 0.1
        if unsafe:
            self._unsafe_ticks += 1
        else:
            self._unsafe_ticks = 0

        if down:
            self.controller.set_attitude_quat_deg(0.0, 0.0, 0.0, 0.18)
        elif self._unsafe_ticks >= 5:
            self._commit_until = 0.0
            self.phase = "RIGHT"
            # Level quat setpoint; low thrust (rate righting backflipped).
            r_cmd = float(np.clip((0.0 - roll_deg) * KR, -MAX_BANK_DEG, MAX_BANK_DEG))
            p_cmd = float(
                np.clip((0.0 - pitch_deg) * KP, -PITCH_WIRE_MAX_DEG, PITCH_WIRE_MAX_DEG)
            )
            self.controller.set_attitude_quat_deg(r_cmd, p_cmd, 0.0, RIGHT_THRUST)
        else:
            roll_cmd, pitch_cmd, yaw_cmd, thrust = self._cmd_slew.apply(
                roll_cmd, pitch_cmd, yaw_cmd, thrust
            )
            self.controller.set_attitude_quat_deg(
                roll_cmd, pitch_cmd, yaw_cmd, thrust
            )

        if self.phase != "SCAN":
            self._scan_since = None
        self._last_tgt_pitch_deg = pitch_des_deg

        if now - self._last_status_log >= STATUS_LOG_INTERVAL_S:
            self._last_status_log = now
            print(
                f"[ibvs] {self.phase} ex={self._ex:+.2f} ey={self._ey:+.2f} "
                f"S={self._spread:.0f} passed={self.n_passed} "
                f"pd={pitch_des_deg:+.1f}",
                flush=True,
            )
        if self._nav_wr is not None and now - self._last_nav_log >= NAV_LOG_INTERVAL_S:
            self._last_nav_log = now
            self._nav_wr.writerow(
                [
                    round(now, 2),
                    self.phase,
                    round(self._ex, 3),
                    round(self._ey, 3),
                    round(self._spread, 1),
                    round(roll, 3),
                    round(pitch, 3),
                    round(self.tilt.roll_acc, 3),
                    round(self.tilt.pitch_acc, 3),
                    self.n_passed,
                ]
            )
