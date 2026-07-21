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
CHI2_3 = 7.815  # 3-DOF Mahalanobis NIS gate at 95%


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
        self.n_rejected = 0

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
    def _update(
        self, H, r, Rm, gate: bool = False, chi2_thresh: float = CHI2_3
    ) -> bool:
        """Joseph-form update. If gate, reject when NIS r^T S^{-1} r >= chi2."""
        r = np.asarray(r, float)
        if not np.isfinite(r).all():
            return False
        S = H @ self.P @ H.T + Rm
        try:
            if gate:
                nu = float(r @ np.linalg.solve(S, r))
                if not np.isfinite(nu) or nu >= chi2_thresh:
                    self.n_rejected += 1
                    return False
            Kk = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        if not np.isfinite(Kk).all():
            return False
        dx = Kk @ r
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = quat_norm(quat_mult(self.q, quat_from_smallangle(dx[6:9])))
        I_KH = np.eye(9) - Kk @ H
        self.P = I_KH @ self.P @ I_KH.T + Kk @ Rm @ Kk.T
        return True

    def update_position(self, p_meas, sigma=0.5, gate: bool = True) -> bool:
        H = np.zeros((3, 9))
        H[:, 0:3] = np.eye(3)
        r = np.asarray(p_meas, float) - self.p
        return self._update(H, r, (sigma**2) * np.eye(3), gate=gate)

    def update_z(self, z_meas, sigma=0.3) -> bool:
        H = np.zeros((1, 9))
        H[0, 2] = 1.0
        r = np.array([z_meas - self.p[2]])
        return self._update(H, r, np.array([[sigma**2]]), gate=False)

    def update_gravity_tilt(self, accel_body, sigma) -> bool:
        # At rest the accelerometer reads f = -R^T g. First-order in the
        # body-frame error: f = -u - skew(u) dtheta, u = R^T g.
        u = quat_to_R(self.q).T @ G_WORLD
        r = np.asarray(accel_body, float) - (-u)
        H = np.zeros((3, 9))
        H[:, 6:9] = -skew(u)
        return self._update(H, r, (sigma**2) * np.eye(3), gate=False)

    def update_mag(self, mag_unit_body, m_world_unit, sigma=0.1) -> bool:
        # Predicted body field u = R^T m_w; h = u + skew(u) dtheta.
        u = quat_to_R(self.q).T @ m_world_unit
        r = np.asarray(mag_unit_body, float) - u
        H = np.zeros((3, 9))
        H[:, 6:9] = skew(u)
        return self._update(H, r, (sigma**2) * np.eye(3), gate=False)


class StateEstimator:
    """Sensor plumbing + boot init around the ESKF. Thread-safe."""

    def __init__(
        self,
        init_samples=100,
        tilt_gate=0.5,  # use accel tilt only when ||a|-g| < this (m/s^2)
        sigma_tilt=0.6,
        sigma_mag=0.15,
        sigma_baro=0.3,
        sigma_landmark=0.5,
        landmark_gate_m=3.0,
        c_thrust=36.0,  # measured: thrust-accel ~36 m/s^2 per thrust unit
        gyro_spike=15.0,  # skip samples with |gyro| beyond this (rad/s)
        drag_kd=0.6,  # linear drag (1/s): the real drone saturates at a few
        # m/s; without it the model integrates lean accel unboundedly
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
        )
        self._lock = threading.Lock()
        self._buf: list[dict] = []
        self.ekf: ESKF | None = None
        self.accel_sign = 1.0
        self.m_world_unit = None  # boot-anchored world mag direction (or None)
        self.baro_ok = False
        self.alt0 = 0.0
        self._last_t_us = None
        self._quiet_gyro_n = 0
        self.n_landmarks = 0
        self.n_landmarks_rejected = 0
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

    def reset(self):
        """Forget everything and re-run ground init (call after a sim reset --
        the teleport invalidates the gyro-integrated attitude)."""
        with self._lock:
            self.ekf = None
            self._buf.clear()
            self._last_t_us = None
            self.thrust_cmd = 0.0
            self.m_world_unit = None
            self.baro_ok = False
            self.n_landmarks = 0
            self.n_landmarks_rejected = 0

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

    # ---- per-IMU-sample (MAVLinkRX thread) ----------------------------------
    def on_imu(self, imu: dict):
        with self._lock:
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
                return  # never let a NaN sample into the filter
            if float(np.linalg.norm(gyro)) > self.p["gyro_spike"]:
                return  # transient garbage sample (observed live)

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
        stays -- the gyro tracks through impacts."""
        with self._lock:
            if self.ekf is None:
                return
            self.ekf.v = np.zeros(3)
            self.ekf.P[0:3, 0:3] += np.eye(3) * 25.0
            self.ekf.P[3:6, 3:6] += np.eye(3) * 9.0

    # ---- vision landmark update (control-loop thread) -----------------------
    def update_landmark(self, p_meas) -> bool:
        with self._lock:
            if self.ekf is None:
                return False
            # Mahalanobis NIS gate inside update_position (camera/landmark).
            if not self.ekf.update_position(
                p_meas, self.p["sigma_landmark"], gate=True
            ):
                self.n_landmarks_rejected += 1
                return False
            self.n_landmarks += 1
            return True

    # ---- state access --------------------------------------------------------
    def pose(self):
        """(pos, vel, quat) copies, or None until boot init completes."""
        with self._lock:
            if self.ekf is None:
                return None
            return self.ekf.p.copy(), self.ekf.v.copy(), self.ekf.q.copy()


# ---------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(7)
    m_world = np.array([0.32, -0.05, 0.41])  # arbitrary sim field

    # --- 1. boot init on tilted ground --------------------------------------
    roll_t, pitch_t = math.radians(5.0), math.radians(-3.0)
    q_true = quat_from_rpy(roll_t, pitch_t, 0.0)
    R_t = quat_to_R(q_true)
    est = StateEstimator(init_samples=50)
    for _ in range(50):
        f = -R_t.T @ G_WORLD + rng.normal(0, 0.02, 3)
        m = R_t.T @ m_world + rng.normal(0, 0.002, 3)
        est.on_imu(
            dict(
                ax=f[0],
                ay=f[1],
                az=f[2],
                gx=0,
                gy=0,
                gz=0,
                mx=m[0],
                my=m[1],
                mz=m[2],
                abs_pressure=1013.0,
                pressure_alt=105.0,
                temperature=20.0,
                time_us=0,
            )
        )
    assert est.ready
    _, _, q0 = est.pose()
    dq = quat_mult(np.array([q_true[0], -q_true[1], -q_true[2], -q_true[3]]), q0)
    att_err0 = math.degrees(2 * math.acos(min(1.0, abs(dq[0]))))
    print(f"[selftest] boot init attitude err = {att_err0:.2f} deg")
    assert att_err0 < 1.0, "boot tilt init should match ground truth"

    # --- 2. live-sim profile flight: thrust-model predict + landmarks --------
    # Matches what the sim actually sends (measured 2026-07-01): clean accel
    # ONLY on the ground, garbage accel + occasional gyro spikes in flight,
    # mag/baro NaN throughout. Truth: hover thrust, constant velocity.
    dt = 1 / 150.0
    T = 20.0
    n = int(T / dt)
    v0 = np.array([2.0, 1.0, -0.5])
    nan = float("nan")

    est2 = StateEstimator(init_samples=50)
    for k in range(50):  # ground init: level, unpowered, clean accel
        f = -G_WORLD + rng.normal(0, 0.02, 3)
        est2.on_imu(
            dict(
                ax=f[0],
                ay=f[1],
                az=f[2],
                gx=0,
                gy=0,
                gz=0,
                mx=nan,
                my=nan,
                mz=nan,
                abs_pressure=nan,
                pressure_alt=nan,
                temperature=nan,
                time_us=k,
            )
        )
    assert est2.ready and est2.m_world_unit is None and not est2.baro_ok

    est2.thrust_cmd = GRAVITY / est2.p["c_thrust"]  # hover -> model a_world ~ 0
    # Drag-consistent truth: coasting at hover thrust, v decays with drag_kd
    # (a constant-velocity truth would be dynamically inconsistent with the
    # model and the filter would misattribute the residual to attitude).
    kd = est2.p["drag_kd"]
    att_errs, pos_errs = [], []
    for k in range(n):
        t = k * dt
        p_rel = v0 / kd * (1.0 - math.exp(-kd * t))
        garbage = rng.uniform(-400, 400, 3)  # in-flight accel is unusable
        spike = 25.0 if k % 97 == 0 else 0.0  # occasional gyro garbage too
        est2.on_imu(
            dict(
                ax=garbage[0],
                ay=garbage[1],
                az=garbage[2],
                gx=rng.normal(0, 0.002) + spike,
                gy=rng.normal(0, 0.002),
                gz=rng.normal(0, 0.002),
                mx=nan,
                my=nan,
                mz=nan,
                abs_pressure=nan,
                pressure_alt=nan,
                temperature=nan,
                time_us=50 + int((t + dt) * 1e6),
            )
        )
        if t > T / 2 and k % 30 == 0:  # 5 Hz landmark fixes in second half
            est2.update_landmark(p_rel + rng.normal(0, 0.15, 3))
        pe, ve, qe = est2.pose()
        dq = quat_mult(np.array([1.0, 0, 0, 0]), qe)
        att_errs.append(math.degrees(2 * math.acos(min(1.0, abs(dq[0])))))
        pos_errs.append(np.linalg.norm(pe - p_rel))

    half = n // 2
    att_max = max(att_errs)
    drift_no_fix = pos_errs[half - 1]
    pos_final = float(np.mean(pos_errs[-150:]))
    pe, ve, qe = est2.pose()
    assert np.isfinite(pe).all() and np.isfinite(ve).all() and np.isfinite(qe).all()
    print(
        f"[selftest] live profile: att_max={att_max:.2f}deg "
        f"drift@{T / 2:.0f}s(no landmarks)={drift_no_fix:.1f}m "
        f"pos_final(with landmarks)={pos_final:.2f}m "
        f"(landmarks used={est2.n_landmarks} rejected={est2.n_landmarks_rejected})"
    )
    # Attitude wanders a little in flight (landmark updates trim it via the
    # thrust-direction coupling; no absolute reference exists) -- a few deg
    # is expected and fine for gate flying.
    assert att_max < 5.0, "attitude must ignore garbage accel + gyro spikes"
    assert pos_final < 0.5, "landmarks should bound position"
    assert pos_final < drift_no_fix, "landmarks should beat dead reckoning"
    print("[selftest] OK — thrust-model ESKF + landmark updates")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
