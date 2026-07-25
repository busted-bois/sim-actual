"""AndurilGP-style 400 Hz IMU estimation with ESKF covariance predict.

GyroAHRS attitude (complementary) + ESKF strapdown for pos/vel. Each IMU
tick runs the discrete covariance time update:

    Pk^- = Fk Pk-1 Fk^T + Q

In powered flight this sim's accelerometer is unreliable (same finding as
StateEstimator), so prediction uses the thrust+drag model; vision landmarks
correct position after predict (Mahalanobis-gated).

Daemon thread; GPPilot takes an atomic snapshot each control tick.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from collections import deque

import numpy as np

from simulator.gyro_ahrs import GyroAHRS
from simulator.state_estimator import ESKF, quat_from_rpy, quat_to_R

G = 9.81
ACC_SMOOTH_N = 5
ESTIMATION_POLL_HZ = 400
LAUNCH_PITCH_DEG = -17.8
ALPHA = 0.98  # complementary filter: gyro weight (accel gravity tilt gets 1-alpha)
# Process noise: wider than lab IMU — thrust model + sim error over 5 gates.
SIGMA_ACCEL = 2.0  # match StateEstimator (model error, not IMU noise)
SIGMA_GYRO = 0.03
SIGMA_POS = 0.05  # extra position random-walk each predict (m/s)/sqrt(s) scale
SIGMA_LANDMARK = 0.5
# Thrust → body specific force (measured live in StateEstimator).
C_THRUST = 36.0
DRAG_KD = 0.6  # 1/s linear drag in world frame, mapped to body
THRUST_POWERED = 0.05
MAX_SPEED_MPS = 8.0  # sanity clamp on dead-reckon / model speed


def _symmetrize(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)


class GPEstimation:
    """Background IMU propagator. Call start() after first IMU appears."""

    def __init__(self, data: dict, launch_pitch_deg: float = LAUNCH_PITCH_DEG):
        self.data = data
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._last_imu_ts_us: int | None = None
        self._launch_pitch_deg = launch_pitch_deg
        self.thrust_cmd = 0.0
        self.ahrs = GyroAHRS(initial_pitch_deg=launch_pitch_deg)
        self.ekf = ESKF(
            q0=quat_from_rpy(0.0, math.radians(launch_pitch_deg), 0.0),
            sigma_accel=SIGMA_ACCEL,
            sigma_gyro=SIGMA_GYRO,
            p0_std=0.5,
        )
        self.vel_ned = np.zeros(3)
        self.vel_body = np.zeros(3)
        self.pos_ned = np.zeros(3)
        self.rates_body = np.zeros(3)  # deg/s
        self._att_deg = (0.0, launch_pitch_deg, 0.0)
        self._acc_buf: deque[tuple[float, float, float]] = deque(maxlen=ACC_SMOOTH_N)
        self.n_landmarks = 0
        self.n_landmarks_rejected = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="gp-estimation"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def set_thrust(self, thrust: float) -> None:
        """Latest commanded collective (control thread → predict model)."""
        with self._lock:
            self.thrust_cmd = float(np.clip(thrust, 0.0, 1.0))

    def reset(self) -> None:
        with self._lock:
            self._last_imu_ts_us = None
            self.thrust_cmd = 0.0
            self.ahrs = GyroAHRS(initial_pitch_deg=self._launch_pitch_deg)
            self.ekf = ESKF(
                q0=quat_from_rpy(0.0, math.radians(self._launch_pitch_deg), 0.0),
                sigma_accel=SIGMA_ACCEL,
                sigma_gyro=SIGMA_GYRO,
                p0_std=0.5,
            )
            self.vel_ned[:] = 0.0
            self.vel_body[:] = 0.0
            self.pos_ned[:] = 0.0
            self.rates_body[:] = 0.0
            self._att_deg = (0.0, self._launch_pitch_deg, 0.0)
            self._acc_buf = deque(maxlen=ACC_SMOOTH_N)
            self.n_landmarks = 0
            self.n_landmarks_rejected = 0

    def zero_velocity(self) -> None:
        """Clear speed after impact; keep position, open translation cov."""
        with self._lock:
            self.vel_ned[:] = 0.0
            self.vel_body[:] = 0.0
            self.ekf.v[:] = 0.0
            # Do NOT wipe ekf.p / pos_ned — mid-course collision must not
            # teleport the filter to origin (StateEstimator.notify_collision).
            self.ekf.P[0:3, 0:3] += np.eye(3) * 25.0
            self.ekf.P[3:6, 3:6] += np.eye(3) * 9.0
            self.ekf.P = _symmetrize(self.ekf.P)

    def on_gate_passed(self) -> None:
        """Next gate landmark should re-anchor — open translation cov a bit."""
        with self._lock:
            self.ekf.P[0:3, 0:3] += np.eye(3) * 4.0
            self.ekf.P[3:6, 3:6] += np.eye(3) * 2.0
            self.ekf.P = _symmetrize(self.ekf.P)

    def update_landmark(
        self, p_meas: np.ndarray, range_m: float | None = None
    ) -> bool:
        """Vision world-position update after predict (Mahalanobis-gated)."""
        with self._lock:
            sigma = SIGMA_LANDMARK
            if range_m is not None and np.isfinite(range_m):
                # PnP / pinhole worsens with range.
                sigma = SIGMA_LANDMARK * (1.0 + 0.12 * max(0.0, float(range_m) - 2.0))
            ok = self.ekf.update_position(
                np.asarray(p_meas, float), sigma, gate=True
            )
            if not ok:
                self.n_landmarks_rejected += 1
                return False
            self.n_landmarks += 1
            self.ekf.P = _symmetrize(self.ekf.P)
            self.pos_ned[:] = self.ekf.p
            self.vel_ned[:] = self.ekf.v
            self._clamp_speed()
            self._refresh_vel_body()
            return True

    def snapshot(self) -> dict:
        """Atomic copy of attitude / velocity / position for the control tick."""
        with self._lock:
            return {
                "att_deg": self._att_deg,
                "quat": self.ahrs.quaternion,
                "pos_ned": self.pos_ned.copy(),
                "vel_ned": self.vel_ned.copy(),
                "vel_body": self.vel_body.copy(),
                "rates_body_dps": self.rates_body.copy(),
                "P_trace": float(np.trace(self.ekf.P)),
                "n_landmarks": int(self.n_landmarks),
                "n_landmarks_rejected": int(self.n_landmarks_rejected),
                "thrust_cmd": float(self.thrust_cmd),
            }

    def _loop(self) -> None:
        interval = 1.0 / ESTIMATION_POLL_HZ
        last_tb = 0.0
        while self._running:
            imu = self.data.get("imu")
            if imu is not None:
                try:
                    self._process_imu(imu)
                except Exception:
                    now = time.monotonic()
                    if now - last_tb >= 5.0:
                        traceback.print_exc()
                        last_tb = now
            time.sleep(interval)

    def _process_imu(self, imu: dict) -> None:
        ts_us = int(imu.get("time_us") or imu.get("time_usec") or 0)
        with self._lock:
            if self._last_imu_ts_us is None:
                self._last_imu_ts_us = ts_us
                return
            if ts_us == self._last_imu_ts_us:
                return
            dt = max(0.0005, min(0.1, (ts_us - self._last_imu_ts_us) * 1e-6))
            self._last_imu_ts_us = ts_us

            # Gyro signs inverted vs NED in this simulator (AndurilGP empirical).
            gx = -float(imu.get("gx", imu.get("xgyro", 0.0)))
            gy = -float(imu.get("gy", imu.get("ygyro", 0.0)))
            gz = -float(imu.get("gz", imu.get("zgyro", 0.0)))
            ax = float(imu.get("ax", imu.get("xacc", 0.0)))
            ay = float(imu.get("ay", imu.get("yacc", 0.0)))
            az = float(imu.get("az", imu.get("zacc", 0.0)))
            if not all(
                math.isfinite(v) for v in (gx, gy, gz, ax, ay, az, dt)
            ):
                return

            self._acc_buf.append((ax, ay, az))
            n = len(self._acc_buf)
            ax_s = sum(s[0] for s in self._acc_buf) / n
            ay_s = sum(s[1] for s in self._acc_buf) / n
            az_s = sum(s[2] for s in self._acc_buf) / n

            self._att_deg = self.ahrs.update_with_accel(
                gx, gy, gz, ax_s, ay_s, az_s, dt, ALPHA
            )
            self.rates_body[0] = math.degrees(gx)
            self.rates_body[1] = math.degrees(gy)
            self.rates_body[2] = math.degrees(gz)

            # AHRS owns nominal attitude; ESKF predicts p/v + Pk^- = FPF^T+Q.
            q_ahrs = np.asarray(self.ahrs.quaternion, float).copy()
            self.ekf.q = q_ahrs
            gyro = np.array([gx, gy, gz], dtype=float)

            if self.thrust_cmd < THRUST_POWERED:
                # On pad / unpowered: don't free-fall on thrust model; hold
                # translation, keep cov growing slowly so first landmark works.
                self.ekf.v[:] = 0.0
                self.ekf.P[0:3, 0:3] += (SIGMA_POS**2) * dt * np.eye(3)
                self.ekf.P[3:6, 3:6] += (self.ekf.sa**2) * dt * dt * np.eye(3)
                self.ekf.P[6:9, 6:9] += (self.ekf.sg**2) * dt * dt * np.eye(3)
            else:
                # Powered: thrust+drag model (accel in this sim is garbage).
                R = quat_to_R(q_ahrs)
                f_model = np.array(
                    [0.0, 0.0, -C_THRUST * self.thrust_cmd], dtype=float
                )
                f_model -= R.T @ (DRAG_KD * self.ekf.v)
                self.ekf.predict(f_model, gyro, dt)
                self.ekf.P[0:3, 0:3] += (SIGMA_POS**2) * dt * np.eye(3)

            # Keep AHRS as nominal attitude (predict also gyro-integrates q).
            self.ekf.q = q_ahrs
            self.ekf.P = _symmetrize(self.ekf.P)
            if not np.isfinite(self.ekf.P).all():
                self.ekf.P = np.eye(9) * 4.0
            self._clamp_speed()
            self.pos_ned[:] = self.ekf.p
            self.vel_ned[:] = self.ekf.v
            self._refresh_vel_body()

    def _clamp_speed(self) -> None:
        speed = float(np.linalg.norm(self.ekf.v))
        if speed > MAX_SPEED_MPS:
            self.ekf.v *= MAX_SPEED_MPS / speed

    def _refresh_vel_body(self) -> None:
        qw, qx, qy, qz = self.ahrs.quaternion
        vN, vE, vD = self.vel_ned
        self.vel_body[0] = (
            (1 - 2 * (qy * qy + qz * qz)) * vN
            + 2 * (qx * qy + qw * qz) * vE
            + 2 * (qx * qz - qw * qy) * vD
        )
        self.vel_body[1] = (
            2 * (qx * qy - qw * qz) * vN
            + (1 - 2 * (qx * qx + qz * qz)) * vE
            + 2 * (qy * qz + qw * qx) * vD
        )
        self.vel_body[2] = (
            2 * (qx * qz + qw * qy) * vN
            + 2 * (qy * qz - qw * qx) * vE
            + (1 - 2 * (qx * qx + qy * qy)) * vD
        )
