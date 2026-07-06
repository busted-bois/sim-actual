"""Qualification-mode state estimator — ESKF on HIGHRES_IMU + vision landmarks.

Under the VQ2 telemetry block the sim stops sending ODOMETRY / ATTITUDE /
LOCAL_POSITION_NED, leaving HIGHRES_IMU (accel+gyro+mag+baro) and the camera
as the only state sources. This module reconstructs pos/vel/attitude:

  Predict   : IMU strapdown (accel+gyro) at sensor rate.
  Update 1  : accel gravity-tilt — only when |accel|~g (quasi-static), sigma
              inflated by the dynamic excess.
  Update 2  : magnetometer attitude vs a boot-captured reference field
              (heading is RELATIVE to boot; declination irrelevant since the
              whole frame — gates, map, drone — is boot-anchored).
  Update 3  : baro altitude, z = -(pressure_alt - alt_boot)  (NED z down).
  Update 4  : vision landmark position p = gate_map - R_wb @ gate_body, fed
              by fly2 when a confirmed mapped gate is re-observed
              (innovation-gated; SLAM-lite — bounds drift vs the map, which
              is all VisionGuidance needs).

ESKF core ported from rl/ekf.py (copied, not imported — simulator/ stays free
of rl/, same pattern as gate_pnp's local intrinsics). Boot init on the ground:
accel -> roll/pitch, mag -> yaw=0 reference, baro -> z=0, p=v=0. Accel gravity
sign is auto-detected from the ground samples (flag logged).

Structure (testability refactor):
  EstimatorCore  — PURE and deterministic: no threads, no locks, no wall
                   clock (dt comes from imu["time_us"]). Every input is an
                   event consumed via apply(); the offline replay harness
                   (simulator/est_replay.py) drives ONLY this class.
  StateEstimator — thin thread-safe live adapter around EstimatorCore. Each
                   method records the event (if a recorder is attached) and
                   forwards it under one lock, so the recorded JSONL is the
                   exact stream the core consumed, in consumption order.

Thread model: on_imu() runs on the MAVLinkRX thread; update_landmark() and the
state getters run on the control loop — one lock covers all of it.

    uv run -m simulator.state_estimator --selftest    (or: make est-selftest)
"""

from __future__ import annotations

import argparse
import math
import threading

import numpy as np

GRAVITY = 9.81
G_WORLD = np.array([0.0, 0.0, GRAVITY])  # NED: gravity points +Z (down)


# ---- quaternion helpers (w,x,y,z) ------------------------------------------
def quat_mult(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_norm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0, 0, 0])


def quat_from_smallangle(dtheta):
    half = 0.5 * np.asarray(dtheta, float)
    return quat_norm(np.array([1.0, half[0], half[1], half[2]]))


def quat_to_R(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ]
    )


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return quat_norm(
        np.array(
            [
                cy * cp * cr + sy * sp * sr,
                cy * cp * sr - sy * sp * cr,
                cy * sp * cr + sy * cp * sr,
                sy * cp * cr - cy * sp * sr,
            ]
        )
    )


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class ESKF:
    """9-D error state [dp, dv, dtheta], body-frame attitude error (right-mult)."""

    def __init__(
        self,
        p0=None,
        v0=None,
        q0=None,
        sigma_accel=0.3,
        sigma_gyro=0.02,
        p0_std=0.5,
    ):
        self.p = np.zeros(3) if p0 is None else np.asarray(p0, float).copy()
        self.v = np.zeros(3) if v0 is None else np.asarray(v0, float).copy()
        self.q = (
            np.array([1.0, 0, 0, 0]) if q0 is None else quat_norm(np.asarray(q0, float))
        )
        self.P = np.eye(9) * (p0_std**2)
        self.sa, self.sg = sigma_accel, sigma_gyro

    # ---- prediction --------------------------------------------------------
    def predict(self, accel_body, gyro_body, dt):
        if dt <= 0 or dt > 0.5:
            return
        accel_body = np.asarray(accel_body, float)
        gyro_body = np.asarray(gyro_body, float)
        R = quat_to_R(self.q)
        a_world = R @ accel_body + G_WORLD

        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        self.q = quat_norm(quat_mult(self.q, quat_from_smallangle(gyro_body * dt)))

        A = np.zeros((9, 9))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 6:9] = -R @ skew(accel_body)
        A[6:9, 6:9] = -skew(gyro_body)
        F = np.eye(9) + A * dt

        Q = np.zeros((9, 9))
        Q[3:6, 3:6] = (self.sa**2) * dt * dt * np.eye(3)
        Q[6:9, 6:9] = (self.sg**2) * dt * dt * np.eye(3)
        self.P = F @ self.P @ F.T + Q

    # ---- generic update (Joseph form) --------------------------------------
    def _update(self, H, r, Rm):
        S = H @ self.P @ H.T + Rm
        Kk = self.P @ H.T @ np.linalg.inv(S)
        dx = Kk @ r
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = quat_norm(quat_mult(self.q, quat_from_smallangle(dx[6:9])))
        I_KH = np.eye(9) - Kk @ H
        self.P = I_KH @ self.P @ I_KH.T + Kk @ Rm @ Kk.T

    def update_position(self, p_meas, sigma=0.5):
        H = np.zeros((3, 9))
        H[:, 0:3] = np.eye(3)
        r = np.asarray(p_meas, float) - self.p
        self._update(H, r, (sigma**2) * np.eye(3))

    def update_z(self, z_meas, sigma=0.3):
        H = np.zeros((1, 9))
        H[0, 2] = 1.0
        r = np.array([z_meas - self.p[2]])
        self._update(H, r, np.array([[sigma**2]]))

    def update_gravity_tilt(self, accel_body, sigma):
        # At rest the accelerometer reads f = -R^T g. First-order in the
        # body-frame error: f = -u - skew(u) dtheta, u = R^T g.
        u = quat_to_R(self.q).T @ G_WORLD
        r = np.asarray(accel_body, float) - (-u)
        H = np.zeros((3, 9))
        H[:, 6:9] = -skew(u)
        self._update(H, r, (sigma**2) * np.eye(3))

    def update_mag(self, mag_unit_body, m_world_unit, sigma=0.1):
        # Predicted body field u = R^T m_w; h = u + skew(u) dtheta.
        u = quat_to_R(self.q).T @ m_world_unit
        r = np.asarray(mag_unit_body, float) - u
        H = np.zeros((3, 9))
        H[:, 6:9] = skew(u)
        self._update(H, r, (sigma**2) * np.eye(3))


class EstimatorCore:
    """Sensor plumbing + boot init around the ESKF.

    Pure and deterministic: consumes timestamped events, keeps no threads,
    locks, or wall-clock state. The same event sequence always produces the
    same pose — which is what makes offline replay exact.
    """

    def __init__(
        self,
        init_samples=100,
        tilt_gate=0.5,  # use accel tilt only when ||a|-g| < this (m/s^2)
        sigma_tilt=0.6,
        sigma_mag=0.15,
        sigma_baro=0.3,
        sigma_landmark=0.5,
        landmark_gate_m=3.0,
        c_thrust=36.33,  # thrust-accel per unit; 9.81/0.27 so the measured
        # hover thrust predicts EXACTLY zero net accel (36.0 left a standing
        # -0.09 m/s^2 residual = ~0.15 m/s phantom descent, no baro to catch)
        gyro_spike=15.0,  # skip samples with |gyro| beyond this (rad/s)
        drag_kd=0.6,  # linear drag (1/s): the real drone saturates at a few
        # m/s; without it the model integrates lean accel unboundedly
        contact_s=0.8,  # after a collision, hold translation this long: the
        # drone is in CONTACT and the free-air thrust model does not apply
        # (measured 2026-07-05: wall-grinding at 1 collision/s ratcheted the
        # z estimate +42 m in 75 s -- each inter-collision gap integrated a
        # phantom descent the contact was actually resisting)
    ):
        self.p = dict(
            init_samples=init_samples,
            tilt_gate=tilt_gate,
            sigma_tilt=sigma_tilt,
            sigma_mag=sigma_mag,
            sigma_baro=sigma_baro,
            sigma_landmark=sigma_landmark,
            landmark_gate_m=landmark_gate_m,
            c_thrust=c_thrust,
            gyro_spike=gyro_spike,
            drag_kd=drag_kd,
            contact_s=contact_s,
        )
        self._buf: list[dict] = []
        self.ekf: ESKF | None = None
        self.accel_sign = 1.0
        self.m_world_unit = None  # boot-anchored world mag direction (or None)
        self.baro_ok = False
        self.alt0 = 0.0
        self._last_t_us = None
        self._quiet_gyro_n = 0
        self._last_gyro_n = 0.0
        self._contact_until_us = None
        self.n_landmarks = 0
        self.n_landmarks_rejected = 0
        self.n_nan_skipped = 0
        self.n_spikes_skipped = 0
        self.lm_innov_sum = 0.0
        self.lm_innov_max = 0.0
        # Last commanded collective thrust (set by SimInterface on every send).
        # Measured live (2026-07-01): this sim's accelerometer is clean ONLY on
        # the ground; the instant thrust is applied it emits garbage (|f| 100s
        # of m/s^2, spikes to 1e4+). So in flight we predict velocity from the
        # COMMANDED thrust through the measured model a = R@[0,0,-c_thrust*T]+g
        # instead of integrating the accelerometer.
        self.thrust_cmd = 0.0

    @property
    def ready(self) -> bool:
        return self.ekf is not None

    def apply(self, ev: dict):
        """Single event-stream entry point — what replay drives.

        The live adapter records each event and calls the same methods, so a
        replayed log walks a byte-identical code path.
        """
        k = ev["k"]
        if k == "imu":
            self.on_imu(ev["d"])
        elif k == "thrust":
            self.set_thrust(ev["d"]["thrust"])
        elif k == "landmark":
            return self.update_landmark(ev["d"]["p_meas"])
        elif k == "collision":
            self.notify_collision()
        elif k == "reset":
            self.reset()
        # truth / est_ckpt / meta are observer events — never fed to the core.

    def set_thrust(self, thrust: float):
        self.thrust_cmd = float(thrust)

    def reset(self):
        """Forget everything and re-run ground init (call after a sim reset --
        the teleport invalidates the gyro-integrated attitude)."""
        self.ekf = None
        self._buf.clear()
        self._last_t_us = None
        self._last_gyro_n = 0.0
        self._contact_until_us = None
        self.thrust_cmd = 0.0
        self.m_world_unit = None
        self.baro_ok = False
        self.n_landmarks = 0
        self.n_landmarks_rejected = 0
        self.n_nan_skipped = 0
        self.n_spikes_skipped = 0
        self.lm_innov_sum = 0.0
        self.lm_innov_max = 0.0

    # ---- boot init on the ground -------------------------------------------
    def _init_from_buffer(self):
        f = np.array([[s["ax"], s["ay"], s["az"]] for s in self._buf]).mean(axis=0)
        g = np.array([[s["gx"], s["gy"], s["gz"]] for s in self._buf]).mean(axis=0)
        m = np.array([[s["mx"], s["my"], s["mz"]] for s in self._buf]).mean(axis=0)
        self.alt0 = float(np.mean([s["pressure_alt"] for s in self._buf]))

        # Specific force at rest should be f = -R^T g (fz ~ -9.81 level). If the
        # sim reports the opposite convention, flip all incoming accel.
        if f[2] > 0:
            self.accel_sign = -1.0
            f = -f
            print("[est] accel gravity sign FLIPPED (ground az > 0)", flush=True)
        roll = math.atan2(-f[1], -f[2])
        pitch = math.atan2(f[0], math.hypot(f[1], f[2]))
        q0 = quat_from_rpy(roll, pitch, 0.0)  # yaw = 0 anchors the boot frame

        # Live sim (probed 2026-07-01): mag and pressure are NaN in this sim --
        # both sensors must self-disable, else one NaN poisons the filter.
        mn = np.linalg.norm(m)
        if np.isfinite(mn) and mn > 1e-6:
            self.m_world_unit = quat_to_R(q0) @ (m / mn)
        else:
            print("[est] mag unusable at boot -- yaw rides on gyro", flush=True)
        self.baro_ok = bool(np.isfinite(self.alt0))
        if not self.baro_ok:
            print("[est] baro unusable -- z rides on accel + landmarks", flush=True)

        # Wide accel process noise: prediction runs on the thrust MODEL, whose
        # error (drag, thrust-gain calib) is far larger than an IMU's.
        self.ekf = ESKF(q0=q0, sigma_accel=2.0)
        print(
            f"[est] init: roll={math.degrees(roll):+.1f} pitch={math.degrees(pitch):+.1f}"
            f" alt0={self.alt0:.2f} |g_bias|={np.linalg.norm(g):.4f}"
            f" mag={'ok' if self.m_world_unit is not None else 'OFF'}"
            f" baro={'ok' if self.baro_ok else 'OFF'}",
            flush=True,
        )

    # ---- per-IMU-sample ------------------------------------------------------
    def on_imu(self, imu: dict):
        if self.ekf is None:
            self._buf.append(dict(imu))
            if len(self._buf) >= self.p["init_samples"]:
                self._init_from_buffer()
                self._last_t_us = imu["time_us"]
            return

        t_us = imu["time_us"]
        dt = (t_us - self._last_t_us) * 1e-6 if self._last_t_us is not None else 0.0
        self._last_t_us = t_us
        accel = self.accel_sign * np.array([imu["ax"], imu["ay"], imu["az"]])
        gyro = np.array([imu["gx"], imu["gy"], imu["gz"]])
        if not (np.isfinite(accel).all() and np.isfinite(gyro).all()):
            self.n_nan_skipped += 1
            return  # never let a NaN sample into the filter
        gn = float(np.linalg.norm(gyro))
        prev_gn, self._last_gyro_n = self._last_gyro_n, gn
        if gn > self.p["gyro_spike"] and prev_gn < self.p["gyro_spike"] / 3:
            # ISOLATED jump only: garbage spikes are single samples out of a
            # quiet stream. A real crash tumble ramps up over samples (and
            # can exceed 15 rad/s) -- dropping those samples made the
            # attitude lag reality through collisions (measured 2026-07-05:
            # 45 "spikes" skipped in a crash-heavy flight).
            self.n_spikes_skipped += 1
            return

        if self.thrust_cmd < 0.05:
            # Unpowered: the drone sits on the ground and the thrust model
            # can't see the ground reaction (it would predict free-fall).
            # ZUPT: freeze translation, integrate attitude from the gyro,
            # anchor tilt on the (clean-on-ground) accel.
            if 0 < dt <= 0.5:
                self.ekf.q = quat_norm(
                    quat_mult(self.ekf.q, quat_from_smallangle(gyro * dt))
                )
            self.ekf.v = np.zeros(3)
            excess = abs(float(np.linalg.norm(accel)) - GRAVITY)
            if excess < self.p["tilt_gate"]:
                sigma = self.p["sigma_tilt"] * (
                    1.0 + 4.0 * excess / self.p["tilt_gate"]
                )
                self.ekf.update_gravity_tilt(accel, sigma)
            return

        # Contact episode: within contact_s of a collision the drone is
        # touching something (wall grind / floor scrape emits collisions
        # every ~1 s) -- the free-air thrust model would integrate phantom
        # motion the contact force is actually resisting. Hold translation,
        # keep attitude on the gyro.
        if self._contact_until_us is not None and t_us < self._contact_until_us:
            if 0 < dt <= 0.5:
                self.ekf.q = quat_norm(
                    quat_mult(self.ekf.q, quat_from_smallangle(gyro * dt))
                )
            self.ekf.v = np.zeros(3)
            return

        # Powered flight: this sim's accel is garbage the moment thrust is
        # applied (measured live), so predict velocity from the COMMANDED
        # thrust (+ linear drag) instead of the accelerometer. No absolute
        # tilt reference here; attitude rides on the clean gyro + landmark
        # coupling.
        f_model = np.array([0.0, 0.0, -self.p["c_thrust"] * self.thrust_cmd])
        f_model -= quat_to_R(self.ekf.q).T @ (self.p["drag_kd"] * self.ekf.v)
        self.ekf.predict(f_model, gyro, dt)

        # Grounded-sanity clamp: a rigidly still gyro for ~2 s while the
        # model claims real speed means the drone is parked/crashed (a
        # flying quad always jitters) -- stop hallucinating motion.
        if float(np.linalg.norm(gyro)) < 2e-3:
            self._quiet_gyro_n += 1
            if self._quiet_gyro_n > 230 and float(np.linalg.norm(self.ekf.v)) > 2.0:
                self.ekf.v = np.zeros(3)
        else:
            self._quiet_gyro_n = 0

        if self.m_world_unit is not None:
            m = np.array([imu["mx"], imu["my"], imu["mz"]])
            mn = float(np.linalg.norm(m))
            if np.isfinite(mn) and mn > 1e-6:
                self.ekf.update_mag(m / mn, self.m_world_unit, self.p["sigma_mag"])

        if self.baro_ok:
            alt = imu["pressure_alt"]
            if np.isfinite(alt):
                self.ekf.update_z(-(alt - self.alt0), self.p["sigma_baro"])

    def notify_collision(self):
        """A collision kicks the real state in ways the thrust model cannot
        see. Zero velocity (closer than the pre-impact value) and open the
        translation covariance so the next landmark snaps us back. Attitude
        stays -- the gyro tracks through impacts. Also opens a contact
        window: translation holds until contact_s after the impact."""
        if self.ekf is None:
            return
        self.ekf.v = np.zeros(3)
        # Cap the inflation: repeated 1 Hz scrape collisions used to grow
        # P without bound, after which the innovation gate accepted 8 m
        # landmark yanks at full strength (measured 2026-07-05: a mid-turn
        # z yank of ~4 m corrupted the servo's propagated target).
        if float(np.trace(self.ekf.P[0:3, 0:3])) < 75.0:
            self.ekf.P[0:3, 0:3] += np.eye(3) * 25.0
        if float(np.trace(self.ekf.P[3:6, 3:6])) < 27.0:
            self.ekf.P[3:6, 3:6] += np.eye(3) * 9.0
        if self._last_t_us is not None:
            self._contact_until_us = self._last_t_us + int(self.p["contact_s"] * 1e6)

    # ---- vision landmark update ----------------------------------------------
    def update_landmark(self, p_meas) -> bool:
        if self.ekf is None:
            return False
        innov = float(np.linalg.norm(np.asarray(p_meas, float) - self.ekf.p))
        # Gate opens with the filter's own position uncertainty, else a
        # fixed gate can lock out recovery once dead-reckoning drift
        # exceeds it (fixes rejected forever).
        sigma_p = math.sqrt(max(1e-9, float(np.trace(self.ekf.P[0:3, 0:3])) / 3))
        gate = max(self.p["landmark_gate_m"], 3.0 * sigma_p)
        if innov > gate:
            self.n_landmarks_rejected += 1
            return False
        # Innovation stats: how far the estimate was from each accepted
        # vision fix -- the VQ2 proxy for estimation quality (no truth).
        self.lm_innov_sum += innov
        self.lm_innov_max = max(self.lm_innov_max, innov)
        # Robust update: a large innovation gets a proportionally larger
        # sigma, so no single fix can yank the estimate more than ~half the
        # innovation -- anchors frozen during collision-noisy moments feed
        # back their own error otherwise. Repeated consistent fixes still
        # converge at full weight once the innovation is small.
        sigma = self.p["sigma_landmark"]
        if innov > 2.0:
            sigma = max(sigma, 0.5 * innov)
        self.ekf.update_position(p_meas, sigma)
        self.n_landmarks += 1
        return True

    # ---- state access --------------------------------------------------------
    def pose(self):
        """(pos, vel, quat) copies, or None until boot init completes."""
        if self.ekf is None:
            return None
        return self.ekf.p.copy(), self.ekf.v.copy(), self.ekf.q.copy()


class StateEstimator:
    """Thread-safe live adapter around EstimatorCore.

    Records every event immediately before the core consumes it (both under
    one lock), so the JSONL log is the exact stream the core saw, in order —
    replaying it through EstimatorCore.apply() reproduces the live estimate
    bit-for-bit (verified live via the 1 Hz est_ckpt events).
    """

    def __init__(self, recorder=None, **params):
        self._lock = threading.Lock()
        self.core = EstimatorCore(**params)
        self.recorder = recorder
        self._next_ckpt_us = None
        if recorder is not None:
            recorder.log("meta", {"version": 1, "params": dict(self.core.p)})

    # ---- pass-throughs (selftest / diagnostics) ------------------------------
    @property
    def ready(self) -> bool:
        return self.core.ready

    @property
    def p(self) -> dict:
        return self.core.p

    @property
    def n_landmarks(self) -> int:
        return self.core.n_landmarks

    @property
    def n_landmarks_rejected(self) -> int:
        return self.core.n_landmarks_rejected

    @property
    def m_world_unit(self):
        return self.core.m_world_unit

    @property
    def baro_ok(self) -> bool:
        return self.core.baro_ok

    @property
    def thrust_cmd(self) -> float:
        return self.core.thrust_cmd

    @thrust_cmd.setter
    def thrust_cmd(self, thrust: float):
        # sim_interface assigns this on every command send (~90 Hz). Property
        # setter = the recording tap AND the fix for the old unlocked write.
        with self._lock:
            if self.recorder is not None:
                self.recorder.log("thrust", {"thrust": float(thrust)})
            self.core.set_thrust(thrust)

    # ---- event entry points ---------------------------------------------------
    def on_imu(self, imu: dict):
        with self._lock:
            if self.recorder is not None:
                self.recorder.log("imu", imu)
            self.core.on_imu(imu)
            # 1 Hz live-pose checkpoint: replay asserts it reproduces these.
            if self.recorder is not None and self.core.ekf is not None:
                t_us = imu["time_us"]
                if self._next_ckpt_us is None or t_us >= self._next_ckpt_us:
                    self._next_ckpt_us = t_us + 1_000_000
                    p, v, q = self.core.pose()
                    self.recorder.log(
                        "est_ckpt",
                        {
                            "time_us": t_us,
                            "p": [float(x) for x in p],
                            "v": [float(x) for x in v],
                            "q": [float(x) for x in q],
                        },
                    )

    def update_landmark(self, p_meas, raw: dict | None = None) -> bool:
        """raw: optional provenance ({map_p, gate_body, quat_used}) recorded
        alongside p_meas so replay can re-derive the landmark from a replayed
        attitude (--rederive-landmarks)."""
        with self._lock:
            ok = self.core.update_landmark(p_meas)
            if self.recorder is not None:
                d = {"p_meas": [float(x) for x in p_meas], "accepted": bool(ok)}
                if raw:
                    d.update(
                        {
                            k: [float(x) for x in v]
                            for k, v in raw.items()
                            if v is not None
                        }
                    )
                self.recorder.log("landmark", d)
            return ok

    def notify_collision(self):
        with self._lock:
            if self.recorder is not None:
                self.recorder.log("collision")
            self.core.notify_collision()

    def reset(self):
        """Forget everything and re-run ground init (call after a sim reset)."""
        with self._lock:
            if self.recorder is not None:
                self.recorder.log("reset")
            self._next_ckpt_us = None
            self.core.reset()

    def pose(self):
        """(pos, vel, quat) copies, or None until boot init completes."""
        with self._lock:
            return self.core.pose()


# ---------------------------------------------------------------------------
# Scenario selftests — every scenario drives the PURE EstimatorCore (no
# threads), seeded, deterministic. Each pins one behavior the live sim
# depends on; failures point at the exact broken channel.

_NAN = float("nan")
_DT = 1 / 150.0  # sim IMU rate


def _mk_imu(t_us, f, g, m=(_NAN, _NAN, _NAN), alt=_NAN):
    return dict(
        ax=f[0],
        ay=f[1],
        az=f[2],
        gx=g[0],
        gy=g[1],
        gz=g[2],
        mx=m[0],
        my=m[1],
        mz=m[2],
        abs_pressure=_NAN,
        pressure_alt=alt,
        temperature=_NAN,
        time_us=int(t_us),
    )


def _boot_core(rng, q_true=None, m_world=None, n=50, **params):
    """Ground-boot a core: n clean unpowered samples on (optionally tilted)
    ground. Returns a ready core with time_us at n-1."""
    core = EstimatorCore(init_samples=n, **params)
    R_t = quat_to_R(q_true) if q_true is not None else np.eye(3)
    for k in range(n):
        f = -R_t.T @ G_WORLD + rng.normal(0, 0.02, 3)
        m = (
            R_t.T @ m_world + rng.normal(0, 0.002, 3)
            if m_world is not None
            else (_NAN, _NAN, _NAN)
        )
        core.on_imu(_mk_imu(k, f, (0, 0, 0), m))
    assert core.ready
    return core


def _att_err_deg(q_true, q_est):
    qt = np.asarray(q_true, float)
    dq = quat_mult(np.array([qt[0], -qt[1], -qt[2], -qt[3]]), q_est)
    return math.degrees(2 * math.acos(min(1.0, abs(float(dq[0])))))


def _fly(core, rng, seconds, gyro_fn=None, lm_fn=None, spike_every=0, t0_us=50):
    """Drive a powered core with garbage accel (live-sim reality). gyro_fn(k)
    -> body rates; lm_fn(t) -> truth pos for a 5 Hz landmark or None."""
    n = int(seconds / _DT)
    for k in range(n):
        t = k * _DT
        g = np.asarray(gyro_fn(k), float) if gyro_fn else rng.normal(0, 0.002, 3)
        if spike_every and k % spike_every == 0:
            g = g + np.array([25.0, 0, 0])
        core.on_imu(_mk_imu(t0_us + int((t + _DT) * 1e6), rng.uniform(-400, 400, 3), g))
        if lm_fn and k % 30 == 0:  # 5 Hz
            p_true = lm_fn(t)
            if p_true is not None:
                core.update_landmark(np.asarray(p_true, float) + rng.normal(0, 0.15, 3))
    return core


def _st_boot_tilted_ground():
    """Boot init recovers ground tilt from accel to <1 deg."""
    rng = np.random.default_rng(7)
    q_true = quat_from_rpy(math.radians(5.0), math.radians(-3.0), 0.0)
    core = _boot_core(rng, q_true=q_true, m_world=np.array([0.32, -0.05, 0.41]))
    err = _att_err_deg(q_true, core.pose()[2])
    print(f"  boot attitude err = {err:.2f} deg")
    assert err < 1.0, "boot tilt init should match ground truth"
    assert core.m_world_unit is not None  # mag was finite -> enabled


def _st_nan_self_disable():
    """NaN mag/baro at boot self-disable; a NaN flight sample is skipped."""
    rng = np.random.default_rng(7)
    core = _boot_core(rng)  # all-NaN mag/baro
    assert core.m_world_unit is None and not core.baro_ok
    core.set_thrust(GRAVITY / core.p["c_thrust"])
    p0, _, q0 = core.pose()
    core.on_imu(_mk_imu(10_000_000, (_NAN, _NAN, _NAN), (0, 0, 0)))
    assert core.n_nan_skipped == 1
    p1, _, q1 = core.pose()
    assert np.array_equal(p0, p1) and np.array_equal(q0, q1), "NaN must be inert"
    print("  NaN sensors disabled at boot; NaN sample skipped inert")


def _st_zupt_ground():
    """Unpowered on the ground: translation pinned at 0, tilt tracks."""
    rng = np.random.default_rng(7)
    core = _boot_core(rng)
    rate = math.radians(3.0) / 10.0  # ground slowly tilts 3 deg over 10 s
    n = int(10.0 / _DT)
    for k in range(n):
        t = k * _DT
        q_t = quat_from_rpy(rate * (t + _DT), 0.0, 0.0)
        f = -quat_to_R(q_t).T @ G_WORLD + rng.normal(0, 0.02, 3)
        core.on_imu(_mk_imu(50 + int((t + _DT) * 1e6), f, (rate, 0, 0)))
    p, v, q = core.pose()
    err = _att_err_deg(quat_from_rpy(math.radians(3.0), 0, 0), q)
    print(
        f"  after 10 s tilting ground: |p|={np.linalg.norm(p):.2e} att_err={err:.2f} deg"
    )
    assert np.linalg.norm(p) < 1e-6 and np.linalg.norm(v) == 0.0, "ZUPT must pin"
    assert err < 1.0, "tilt must track the moving ground"


def _st_thrust_model_flight():
    """Live-sim profile: garbage accel, gyro spikes, NaN mag/baro; velocity
    from the commanded-thrust model; landmarks bound drift."""
    rng = np.random.default_rng(7)
    core = _boot_core(rng)
    core.set_thrust(GRAVITY / core.p["c_thrust"])  # hover -> model a_world ~ 0
    # Drag-consistent truth: coasting at hover thrust, v decays with drag_kd
    # (a constant-velocity truth would be dynamically inconsistent with the
    # model and the filter would misattribute the residual to attitude).
    kd = core.p["drag_kd"]
    v0 = np.array([2.0, 1.0, -0.5])
    T = 20.0
    n = int(T / _DT)
    att_errs, pos_errs = [], []
    for k in range(n):
        t = k * _DT
        p_rel = v0 / kd * (1.0 - math.exp(-kd * t))
        spike = 25.0 if k % 97 == 0 else 0.0
        core.on_imu(
            _mk_imu(
                50 + int((t + _DT) * 1e6),
                rng.uniform(-400, 400, 3),
                rng.normal(0, 0.002, 3) + np.array([spike, 0, 0]),
            )
        )
        if t > T / 2 and k % 30 == 0:  # 5 Hz landmark fixes in second half
            core.update_landmark(p_rel + rng.normal(0, 0.15, 3))
        pe, ve, qe = core.pose()
        att_errs.append(_att_err_deg(np.array([1.0, 0, 0, 0]), qe))
        pos_errs.append(np.linalg.norm(pe - p_rel))
    att_max = max(att_errs)
    drift_no_fix = pos_errs[n // 2 - 1]
    pos_final = float(np.mean(pos_errs[-150:]))
    pe, ve, qe = core.pose()
    assert np.isfinite(pe).all() and np.isfinite(ve).all() and np.isfinite(qe).all()
    print(
        f"  att_max={att_max:.2f}deg drift@{T / 2:.0f}s(no landmarks)={drift_no_fix:.1f}m"
        f" pos_final(with landmarks)={pos_final:.2f}m"
        f" (landmarks used={core.n_landmarks} rejected={core.n_landmarks_rejected})"
    )
    # Attitude wanders a little in flight (no absolute reference exists) --
    # a few deg is expected and fine for gate flying.
    assert att_max < 5.0, "attitude must ignore garbage accel + gyro spikes"
    assert pos_final < 0.5, "landmarks should bound position"
    assert pos_final < drift_no_fix, "landmarks should beat dead reckoning"


def _st_gyro_spike_rejection():
    """ISOLATED gyro garbage must not touch the attitude -- but a real crash
    tumble (rates ramping past the spike threshold) must be TRACKED."""
    rng = np.random.default_rng(7)
    core = _boot_core(rng)
    core.set_thrust(GRAVITY / core.p["c_thrust"])
    _fly(core, rng, 3.0, spike_every=50)
    err = _att_err_deg(np.array([1.0, 0, 0, 0]), core.pose()[2])
    print(f"  {core.n_spikes_skipped} isolated spikes skipped, att_err={err:.2f} deg")
    assert core.n_spikes_skipped == int(3.0 / _DT) // 50 + 1
    assert err < 0.5, "spikes must be rejected, not integrated"

    # Crash tumble: roll rate ramps 0 -> 20 rad/s (past the 15 rad/s gate)
    # and holds. The ramp means no sample is an isolated jump -- ALL must
    # integrate, or the attitude lags reality through the crash.
    core2 = _boot_core(rng)
    core2.set_thrust(0.3)
    n = int(1.0 / _DT)
    total = 0.0
    for k in range(n):
        t = k * _DT
        rate = min(20.0, 40.0 * t)  # ramp 0.5 s, hold at 20 rad/s
        total += rate * _DT
        core2.on_imu(
            _mk_imu(50 + int((t + _DT) * 1e6), rng.uniform(-400, 400, 3), (rate, 0, 0))
        )
    err = _att_err_deg(quat_from_rpy(_wrap_pi(total), 0, 0), core2.pose()[2])
    print(
        f"  20 rad/s tumble: {core2.n_spikes_skipped} skipped, "
        f"att err vs integrated truth = {err:.1f} deg"
    )
    assert core2.n_spikes_skipped == 0, "a ramping tumble is not a spike"
    assert err < 5.0, "crash rotation must be tracked, not dropped"


def _wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _st_yaw_sign():
    """Positive body yaw rate (gz>0) must INCREASE quat_to_yaw. Pins the
    estimator's integration convention to the one every consumer of
    snapshot.quat uses (simulator.transforms.quat_to_yaw / spec.quat_to_R).
    If flight 'turns the wrong way' and this passes, the bug is in the
    controller signs (EST_SIGNS) or guidance, NOT the estimator."""
    from simulator.transforms import quat_to_yaw

    rng = np.random.default_rng(7)
    core = _boot_core(rng)
    core.set_thrust(GRAVITY / core.p["c_thrust"])
    _fly(core, rng, 2.0, gyro_fn=lambda k: (0.0, 0.0, 0.5))
    yaw = quat_to_yaw(*core.pose()[2])
    print(f"  2 s of gz=+0.5 rad/s -> yaw = {yaw:+.3f} rad (expect +1.0)")
    assert abs(yaw - 1.0) < 0.05, "gz>0 must integrate to POSITIVE yaw"


def _st_z_channel_calibration():
    """c_thrust 10% off + no baro -> unbounded z drift (documents WHY altitude
    control fails on miscalibration); 5 Hz landmarks must re-bound it."""
    rng = np.random.default_rng(7)
    thrust = GRAVITY / 36.0  # what the controller sends believing c=36
    # Filter's model 10% hot: predicts climb that isn't happening.
    core = _boot_core(rng, c_thrust=36.0 * 1.1)
    core.set_thrust(thrust)
    _fly(core, rng, 5.0)  # truth: perfect hover at origin, no landmarks
    drift = abs(core.pose()[0][2])
    rate = drift / 5.0
    print(
        f"  10% c_thrust error, no landmarks: z drift {drift:.1f} m in 5 s ({rate:.2f} m/s)"
    )
    assert drift > 0.5, "miscalibration must show up (this documents the z bug)"

    core2 = _boot_core(rng, c_thrust=36.0 * 1.1)
    core2.set_thrust(thrust)
    zs = []
    n = int(10.0 / _DT)
    for k in range(n):
        t = k * _DT
        core2.on_imu(
            _mk_imu(
                50 + int((t + _DT) * 1e6),
                rng.uniform(-400, 400, 3),
                rng.normal(0, 0.002, 3),
            )
        )
        if k % 30 == 0:
            core2.update_landmark(rng.normal(0, 0.15, 3))
        zs.append(abs(core2.pose()[0][2]))
    print(f"  same error + 5 Hz landmarks: max |z| = {max(zs):.2f} m")
    assert max(zs) < 1.0, "landmarks must bound z despite thrust miscalibration"


def _st_landmark_recovery():
    """Dead-reckon far past the fixed landmark gate, then fixes arrive: the
    covariance-adaptive gate must accept and pull the estimate back (guards
    the 'rejected forever' fix)."""
    rng = np.random.default_rng(7)
    core = _boot_core(rng, c_thrust=36.0 * 1.03)  # 3% model error -> drift
    core.set_thrust(GRAVITY / 36.0)
    _fly(core, rng, 15.0)  # truth: hover at origin
    drift = float(np.linalg.norm(core.pose()[0]))
    assert drift > core.p["landmark_gate_m"], (
        "scenario needs drift beyond the fixed gate"
    )
    _fly(core, rng, 4.0, lm_fn=lambda t: np.zeros(3), t0_us=50 + int(16e6))
    err = float(np.linalg.norm(core.pose()[0]))
    print(
        f"  drift {drift:.1f} m (> gate {core.p['landmark_gate_m']}) -> after fixes {err:.2f} m"
        f" (accepted {core.n_landmarks}, rejected {core.n_landmarks_rejected})"
    )
    assert core.n_landmarks > 0, "adaptive gate must open with covariance"
    assert err < 0.5, "landmarks must recover the estimate"


def _st_collision_reset():
    """A collision teleports the real state past the landmark gate; the
    inflated covariance must let the next fixes re-anchor."""
    rng = np.random.default_rng(7)
    core = _boot_core(rng)
    core.set_thrust(GRAVITY / core.p["c_thrust"])
    _fly(core, rng, 5.0, lm_fn=lambda t: np.zeros(3))  # tracking at origin
    p_bounce = np.array([4.0, -2.0, 0.0])  # impact knocks the drone 4.5 m
    core.notify_collision()
    assert float(np.linalg.norm(core.pose()[1])) == 0.0, "collision zeroes velocity"
    _fly(core, rng, 3.0, lm_fn=lambda t: p_bounce, t0_us=50 + int(6e6))
    err = float(np.linalg.norm(core.pose()[0] - p_bounce))
    print(f"  post-collision 4.5 m jump re-anchored to {err:.2f} m")
    assert err < 0.5, "inflated covariance must accept the post-impact fixes"


def _st_wall_grind_contact():
    """Repeated collisions (wall/floor grind) must NOT ratchet the position:
    each contact window holds translation instead of integrating the free-air
    thrust model (measured live: +42 m z runaway in 75 s of grinding)."""
    rng = np.random.default_rng(7)
    core = _boot_core(rng)
    core.set_thrust(0.26)  # scan-bleed thrust: model would predict descent
    n = int(15.0 / _DT)
    for k in range(n):
        t = k * _DT
        core.on_imu(
            _mk_imu(
                50 + int((t + _DT) * 1e6),
                rng.uniform(-400, 400, 3),
                rng.normal(0, 0.02, 3),
            )
        )
        if k % 150 == 0:  # a collision every second, like a wall grind
            core.notify_collision()
    drift = float(np.linalg.norm(core.pose()[0]))
    print(f"  15 s of 1 Hz collisions at bleed thrust: |p| drift = {drift:.2f} m")
    assert drift < 1.5, "contact windows must stop the collision ratchet"


def _st_sim_reset():
    """reset() forgets everything (attitude, thrust, counters) and re-boots."""
    rng = np.random.default_rng(7)
    core = _boot_core(rng, q_true=quat_from_rpy(0.3, 0.1, 0.0))
    core.set_thrust(0.3)
    _fly(core, rng, 2.0, gyro_fn=lambda k: (0.2, 0.1, -0.3))
    core.reset()
    assert not core.ready and core.thrust_cmd == 0.0
    q2_true = quat_from_rpy(math.radians(-4.0), math.radians(2.0), 0.0)
    R_t = quat_to_R(q2_true)
    for k in range(core.p["init_samples"]):
        f = -R_t.T @ G_WORLD + rng.normal(0, 0.02, 3)
        core.on_imu(_mk_imu(k, f, (0, 0, 0)))
    err = _att_err_deg(q2_true, core.pose()[2])
    print(f"  re-boot after reset: attitude err {err:.2f} deg, counters cleared")
    assert core.ready and err < 1.0
    assert core.n_landmarks == 0 and core.n_spikes_skipped == 0


def _st_record_replay_roundtrip():
    """Determinism keystone: run the live adapter with a recorder, feed the
    recorded events into a fresh core via apply() -> identical state."""
    from simulator.est_recorder import ListRecorder

    rng = np.random.default_rng(7)
    rec = ListRecorder()
    est = StateEstimator(recorder=rec, init_samples=50)
    for k in range(50):
        est.on_imu(_mk_imu(k, -G_WORLD + rng.normal(0, 0.02, 3), (0, 0, 0)))
    est.thrust_cmd = GRAVITY / est.p["c_thrust"]
    n = int(5.0 / _DT)
    for k in range(n):
        t = k * _DT
        est.on_imu(
            _mk_imu(
                50 + int((t + _DT) * 1e6),
                rng.uniform(-400, 400, 3),
                rng.normal(0, 0.002, 3),
            )
        )
        if k == n // 2:
            est.notify_collision()
        if t > 2.0 and k % 30 == 0:
            est.update_landmark(rng.normal(0, 0.15, 3))
    p_live, v_live, q_live = est.pose()

    core = EstimatorCore(init_samples=50)
    for ev in rec.events:
        core.apply(ev)
    p_r, v_r, q_r = core.pose()
    dev = max(
        float(np.max(np.abs(p_r - p_live))),
        float(np.max(np.abs(v_r - v_live))),
        float(np.max(np.abs(q_r - q_live))),
    )
    print(f"  recorded {len(rec.events)} events; replay deviation = {dev:.2e}")
    assert dev < 1e-12, "replayed events must reproduce the live state exactly"


_SCENARIOS = [
    _st_boot_tilted_ground,
    _st_nan_self_disable,
    _st_zupt_ground,
    _st_thrust_model_flight,
    _st_gyro_spike_rejection,
    _st_yaw_sign,
    _st_z_channel_calibration,
    _st_landmark_recovery,
    _st_collision_reset,
    _st_wall_grind_contact,
    _st_sim_reset,
    _st_record_replay_roundtrip,
]


def _selftest():
    for fn in _SCENARIOS:
        print(f"[selftest] {fn.__name__.removeprefix('_st_')}")
        fn()
    print(f"[selftest] OK — {len(_SCENARIOS)} scenarios green")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
