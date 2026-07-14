"""IBVS auto-flight pilot: fly the gate the camera SEES, in image space.

No position estimate anywhere. Measured 2026-07-02 (nav_log_auto_*.csv): the
VQ2 EKF pose was non-finite from the FIRST tick of every attempt, so any
world/map-based guidance never acquired a target and every run was 100% SCAN.
This pilot's only inputs are things that cannot go NaN-blind the same way:

  * YOLO detections (raw pixels: box + 8 corner keypoints) at ~30 Hz, and
  * HIGHRES_IMU (accel+gyro, 100 Hz) for a complementary-filter attitude (the
    sim has no auto-leveling; roll/pitch must be actively regulated) and a
    pressure-altitude hold (the only altitude reference we allow ourselves).

Control (image-based visual servoing; per fresh inference frame):
  yaw     <- bearing to the opening centre, atan((u-cx)/fx) (steering)
  thrust  <- vertical pixel error, offset for the 20 deg camera up-tilt
             (a gate at our own altitude sits BELOW the image centre), plus a
             pressure-alt-rate damper
  fwd     <- apparent gate size: small in frame -> lean forward, near -> coast
  COMMIT  <- gate wide enough + centred (or it just vanished at point-blank
             range while centred) -> timed blind dash straight through, count
             the pass, brake, scan for the next gate. Detection ALWAYS dies at
             point-blank range, so the last metres are flown blind on attitude
             hold alone.
"""

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np

from rl.fly2_course import (
    HOVER_T,
    K_ATT,
    K_YAW,
    RATE_CLIP,
    SIGN_PITCH,
    SIGN_ROLL,
    SIGN_YAW,
    YAW_CLIP,
)
from simulator.tilt_filter import TiltFilter

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
K_RIGHT = 1.5  # self-righting P gain (rad/s per rad of estimated tilt)
RIGHT_CLIP = 1.0  # self-righting rate authority (rad/s; > RATE_CLIP on purpose)
RIGHT_THRUST = 0.20  # low thrust while righting: hover thrust while inverted
#                      powers the drone into the ground


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


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
            k_yaw_ex=1.0,  # yaw_err = k * atan(ex) (bearing to the opening)
            k_th_ey=0.10,  # thrust per unit normalized vertical error
            k_pa=0.020,  # thrust per metre of pressure-alt error (holds)
            k_pad=0.015,  # thrust per m/s of pressure-alt rate (damping)
            th_clip=0.035,  # thrust authority around hover (+/-)
            k_size=0.10,  # forward lean at S=0 (far gate), tapering to coast
            s_near=200.0,  # inner-corner spread (px) where lean tapers out
            lean_coast=0.02,  # residual forward lean when near
            s_commit=260.0,  # spread (px, ~1.85 m) to trigger the blind dash
            s_drop_commit=190.0,  # spread at last sight for a dropout-commit
            ex_commit=0.08,  # |ex| gate to commit (normalized)
            ey_commit=0.14,  # |ey| gate to commit (normalized)
            commit_lean=0.08,  # forward lean during the blind dash
            t_commit=2.2,  # blind dash duration (s, ~2.6+ m)
            commit_trim=0.15,  # yaw-only trim clip while the gate is still
            # visible during COMMIT (guided commit -- translation stays blind)
            t_brake=1.2,  # reverse-lean brake after the dash (s)
            brake_lean=0.05,  # backward tilt while braking
            scan_yaw=0.35,  # yaw error injected while scanning
            t_settle_min=2.0,  # post-reset SETTLE: orient + see before flying
            t_settle_max=5.0,  # ...but never sit longer than this
            settle_climb=1.5,  # gentle climb (m of pressure alt) off the ramp
            settle_att_tol=0.1,  # filter-vs-accel agreement to leave SETTLE (rad)
            settle_frames=5,  # fresh YOLO frames of the NEW world required
            quasi_pa=0.3,  # |pressure-alt rate| below this = quasi-static (m/s)
            down_cos=-0.3,  # accel cos(roll)cos(pitch) below this = inverted
            down_t=1.0,  # ...sustained this long while static -> GROUNDED
        )
        self.n_passed = 0
        self.phase = "SETTLE"  # never fly blind: orient + see, then chase
        self._settle_t0 = None
        self._settle_synced = False
        self._frames_seen = 0
        self._down_since = None
        self._down_logged = False
        self.tilt = TiltFilter()
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
        self._last_tgt_pitch = 0.0
        self._scan_dir = 1.0
        self._scan_since = None
        self._last_status_log = 0.0
        self._nav_log = None
        self._nav_wr = None
        self._last_nav_log = 0.0
        controller.set_control_mode("attitude")
        controller.set_attitude_rates(0, 0, 0, HOVER_T)
        print("[ibvs] pixel-servo pilot ready (no position estimate)", flush=True)

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
        self._ctr = None
        self._last_seen = None
        self._commit_until = 0.0
        self._brake_until = 0.0
        self._pa_hold = None
        self._pa_prev = None
        self._pa_rate = 0.0
        self._pa_prev_t = None
        self._last_frame_id = None
        self._unsafe_ticks = 0
        self._last_tgt_pitch = 0.0
        self._scan_dir = 1.0
        self._scan_since = None

    def on_attempt_start(self) -> None:
        self._reset_state()
        self._open_nav_log()

    def reset_for_attempt(self) -> None:
        self._reset_state()
        self._close_nav_log()
        self.data.pop("pose", None)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, 0, 0, HOVER_T)

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
        t = HOVER_T
        if pixel_ey is not None:
            # gate below the aim line (ey>0) -> we are high -> descend
            t -= p["k_th_ey"] * pixel_ey
        elif pa is not None and self._pa_hold is not None:
            t += p["k_pa"] * (self._pa_hold - pa)
        if pa is not None:
            t -= p["k_pad"] * self._pa_rate  # damp vertical rate
        return float(np.clip(t, HOVER_T - p["th_clip"], HOVER_T + p["th_clip"]))

    # --- main loop ----------------------------------------------------------
    def tick(self) -> None:
        p = self.p
        now = time.monotonic()
        imu = self.data.get("imu")
        quasi = abs(self._pa_rate) < p["quasi_pa"]  # not moving vertically
        if imu:
            # Gyro-only while accelerating hard: the accel blend would read
            # the commanded forward lean as a backward-pitch error. Blend
            # HARD only while we KNOW we are static: SETTLE/DOWN, no vertical
            # rate AND a silent gyro (rotation = airborne/leveling, where the
            # accel measures the thrust axis, not gravity -- boosting there
            # erased the captured ramp tilt / shortened the dash in sim).
            hard_accel = self.phase == "COMMIT" or abs(self._last_tgt_pitch) > 0.05
            static_now = (
                self.phase in ("SETTLE", "DOWN") and quasi and self.tilt.gyro_mag < 0.05
            )
            # During SETTLE: blend ONLY while static (on the ramp the accel is
            # gravity-true); once lifting/leveling go pure gyro so the truth
            # captured by the sync is not eroded before flight begins.
            allow = (not hard_accel) and (self.phase != "SETTLE" or static_now)
            self.tilt.update(imu, allow_accel=allow, boost=static_now)
        pa = self._update_pressure(imu, now)
        self._ingest(now)

        # Wrapped: after a tumble the integrated angles can be wound past
        # +/-pi (roll hit 9.7 rad live); the P loop must see the short arc.
        roll, pitch = _wrap(self.tilt.roll), _wrap(self.tilt.pitch)
        tgt_roll, tgt_pitch, yaw_err = 0.0, 0.0, 0.0
        pixel_ey = None

        # --- GROUNDED inverted: static and the accel (gravity, since we are
        # not moving) says upside down. Flying commands only thrash the wreck
        # -- go limp and let the race-monitor countdown reset the attempt.
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
            if self.phase == "DOWN":  # bounced/rolled upright: re-orient
                self.phase = "SETTLE"
                self._settle_t0 = None
                self._settle_synced = False
        down = self.phase == "DOWN"

        # --- SETTLE: orient + SEE the new environment before flying. Live
        # logs 2026-07-04: the drone spawns ~18 deg nose-down (pitch_acc
        # -0.311 from row 1) while the fresh filter reads level, and TRACK
        # engaged 0.3 s after reset -- it darted off un-oriented every time.
        settling = self.phase == "SETTLE"
        if settling:
            if self._settle_t0 is None:
                self._settle_t0 = now
                self._frames_seen = 0
                if pa is not None:  # gentle climb clear of the spawn ramp
                    self._pa_hold = pa + p["settle_climb"]
            if not self._settle_synced:
                # Static at GO: one accel sample IS the true attitude.
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
                    f"pitch={math.degrees(pitch):+.0f}deg)",
                    flush=True,
                )
                self.phase = "SCAN"
                settling = False
                if pa is not None:
                    self._pa_hold = pa  # hold the altitude we reached
            # while settling: targets stay level, thrust holds the climb;
            # detections are ingested (target may be pre-held) but not acted on
            roll, pitch = _wrap(self.tilt.roll), _wrap(self.tilt.pitch)

        if not settling and not down and now < self._commit_until:
            self.phase = "COMMIT"
            tgt_pitch = -p["commit_lean"]
            # Guided commit: while the gate is STILL visible, keep trimming
            # heading onto it (translation stays blind) -- fully blind dashes
            # crossed up to 0.57 m off-centre in the closed-loop sim.
            if self._last_seen is not None and now - self._last_seen < 0.3:
                yaw_err = float(
                    np.clip(math.atan(self._ex), -p["commit_trim"], p["commit_trim"])
                )
        elif self.phase == "COMMIT":
            # dash finished this tick: book the pass, start the brake
            self.n_passed += 1
            print(f"[ibvs] PASSED g{self.n_passed} (timed dash)", flush=True)
            self.phase = "BRAKE"
            self._brake_until = now + p["t_brake"]
            self._ctr = None
            self._last_seen = None
            self._pa_hold = self._pa_prev

        if self.phase == "BRAKE":
            if now < self._brake_until:
                tgt_pitch = p["brake_lean"]
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
                    # Blind dash: detection dies at point-blank range; fly the
                    # last metres on attitude + pressure hold alone.
                    self.phase = "COMMIT"
                    self._commit_until = now + p["t_commit"]
                    self._pa_hold = self._pa_prev
                    tgt_pitch = -p["commit_lean"]
                    print(
                        f"[ibvs] COMMIT S={self._spread:.0f} ex={self._ex:+.2f}",
                        flush=True,
                    )
                else:
                    yaw_err = float(
                        np.clip(p["k_yaw_ex"] * math.atan(self._ex), -0.5, 0.5)
                    )
                    lean = p["k_size"] * max(0.0, 1.0 - self._spread / p["s_near"])
                    tgt_pitch = -max(lean, p["lean_coast"])
                    pixel_ey = self._ey
            else:
                if self._last_seen is not None:
                    self._ctr = None  # stale: release the track
                self.phase = "SCAN"
                # Alternate the sweep direction: a one-way scan spiralled the
                # drone away from the course in the first live run.
                if self._scan_since is None:
                    self._scan_since = now
                elif now - self._scan_since > 6.0:
                    self._scan_dir = -self._scan_dir
                    self._scan_since = now
                yaw_err = self._scan_dir * p["scan_yaw"]
                self._pa_hold = self._pa_hold if self._pa_hold is not None else pa

        thrust = self._thrust(pa, pixel_ey)
        # ODOMETRY sign convention. EST_SIGNS was tried live 2026-07-02 and
        # the drone flew INVERTED -- the accel-anchored complementary filter
        # reports truth-convention angles, so the odometry signs (the config
        # of the smoothest run so far) are correct here.
        roll_cmd = float(
            np.clip(SIGN_ROLL * K_ATT * (tgt_roll - roll), -RATE_CLIP, RATE_CLIP)
        )
        pitch_cmd = float(
            np.clip(SIGN_PITCH * K_ATT * (tgt_pitch - pitch), -RATE_CLIP, RATE_CLIP)
        )
        yaw_cmd = float(np.clip(SIGN_YAW * K_YAW * yaw_err, -YAW_CLIP, YAW_CLIP))

        # Safety: flipped. The old freeze-at-hover cutoff recovered only
        # "half the time" (observed live) -- and hover thrust while inverted
        # powers the drone into the ground. Actively right instead, driving
        # the FILTER attitude (which tracked the flip through the gyro) back
        # to level via the shortest arc, thrust cut low. Note the accel tilt
        # is useless here: in flight the accelerometer measures the THRUST
        # axis, not gravity -- it reads near-level even upside down.
        unsafe = math.cos(roll) * math.cos(pitch) < 0.1
        if unsafe:
            self._unsafe_ticks += 1
        else:
            self._unsafe_ticks = 0
        if down:
            # Limp on the ground: zero rates, minimum thrust, wait for reset.
            self.controller.set_attitude_rates(0.0, 0.0, 0.0, 0.18)
        elif self._unsafe_ticks >= 5:
            self._commit_until = 0.0  # a tumble voids any pending commit
            self.phase = "RIGHT"
            r_w, p_w = _wrap(roll), _wrap(pitch)
            self.controller.set_attitude_rates(
                float(
                    np.clip(SIGN_ROLL * K_RIGHT * (0.0 - r_w), -RIGHT_CLIP, RIGHT_CLIP)
                ),
                float(
                    np.clip(SIGN_PITCH * K_RIGHT * (0.0 - p_w), -RIGHT_CLIP, RIGHT_CLIP)
                ),
                0.0,
                RIGHT_THRUST,
            )
        else:
            self.controller.set_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)
        if self.phase != "SCAN":
            self._scan_since = None
        self._last_tgt_pitch = tgt_pitch

        if now - self._last_status_log >= STATUS_LOG_INTERVAL_S:
            self._last_status_log = now
            print(
                f"[ibvs] {self.phase} ex={self._ex:+.2f} ey={self._ey:+.2f} "
                f"S={self._spread:.0f} passed={self.n_passed}",
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
