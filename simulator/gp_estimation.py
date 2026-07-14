"""AndurilGP-style 400 Hz IMU strapdown estimation.

GyroAHRS attitude + SMA-smoothed accel + NED dead-reckoning. Runs on a
daemon thread; GPPilot takes an atomic snapshot each control tick.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np

from simulator.gyro_ahrs import GyroAHRS

G = 9.81
ACC_SMOOTH_N = 5
ESTIMATION_POLL_HZ = 400
LAUNCH_PITCH_DEG = -17.8


class GPEstimation:
    """Background IMU propagator. Call start() after first IMU appears."""

    def __init__(self, data: dict, launch_pitch_deg: float = LAUNCH_PITCH_DEG):
        self.data = data
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._last_imu_ts_us: int | None = None
        self.ahrs = GyroAHRS(initial_pitch_deg=launch_pitch_deg)
        self.vel_ned = np.zeros(3)
        self.vel_body = np.zeros(3)
        self.pos_ned = np.zeros(3)
        self.rates_body = np.zeros(3)  # deg/s
        self._att_deg = (0.0, launch_pitch_deg, 0.0)
        self._acc_buf: deque[tuple[float, float, float]] = deque(maxlen=ACC_SMOOTH_N)
        self._launch_pitch_deg = launch_pitch_deg

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

    def reset(self) -> None:
        with self._lock:
            self._last_imu_ts_us = None
            self.ahrs = GyroAHRS(initial_pitch_deg=self._launch_pitch_deg)
            self.vel_ned[:] = 0.0
            self.vel_body[:] = 0.0
            self.pos_ned[:] = 0.0
            self.rates_body[:] = 0.0
            self._att_deg = (0.0, self._launch_pitch_deg, 0.0)
            self._acc_buf = deque(maxlen=ACC_SMOOTH_N)

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
            }

    def _loop(self) -> None:
        interval = 1.0 / ESTIMATION_POLL_HZ
        while self._running:
            data_lock = self.data.get("lock")
            imu = None
            if data_lock is not None:
                with data_lock:
                    imu = self.data.get("imu")
            else:
                imu = self.data.get("imu")
            if imu is not None:
                self._process_imu(imu)
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
            self._att_deg = self.ahrs.update(gx, gy, gz, dt)
            self.rates_body[0] = math.degrees(gx)
            self.rates_body[1] = math.degrees(gy)
            self.rates_body[2] = math.degrees(gz)

            ax = float(imu.get("ax", imu.get("xacc", 0.0)))
            ay = float(imu.get("ay", imu.get("yacc", 0.0)))
            az = float(imu.get("az", imu.get("zacc", 0.0)))
            self._acc_buf.append((ax, ay, az))
            n = len(self._acc_buf)
            ax_s = sum(s[0] for s in self._acc_buf) / n
            ay_s = sum(s[1] for s in self._acc_buf) / n
            az_s = sum(s[2] for s in self._acc_buf) / n
            self._integrate(ax_s, ay_s, az_s, dt)

    def _integrate(self, ax: float, ay: float, az: float, dt: float) -> None:
        qw, qx, qy, qz = self.ahrs.quaternion
        sf_n = (
            (1 - 2 * (qy * qy + qz * qz)) * ax
            + 2 * (qx * qy - qw * qz) * ay
            + 2 * (qx * qz + qw * qy) * az
        )
        sf_e = (
            2 * (qx * qy + qw * qz) * ax
            + (1 - 2 * (qx * qx + qz * qz)) * ay
            + 2 * (qy * qz - qw * qx) * az
        )
        sf_d = (
            2 * (qx * qz - qw * qy) * ax
            + 2 * (qy * qz + qw * qx) * ay
            + (1 - 2 * (qx * qx + qy * qy)) * az
        )

        a_n, a_e, a_d = sf_n, sf_e, sf_d + G
        self.vel_ned[0] += a_n * dt
        self.vel_ned[1] += a_e * dt
        self.vel_ned[2] += a_d * dt
        self.pos_ned[0] += self.vel_ned[0] * dt
        self.pos_ned[1] += self.vel_ned[1] * dt
        self.pos_ned[2] += self.vel_ned[2] * dt

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
