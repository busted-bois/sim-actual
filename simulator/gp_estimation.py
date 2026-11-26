"""AndurilGP-style 400 Hz IMU strapdown estimation.

GyroAHRS attitude + SMA-smoothed accel + NED dead-reckoning. Runs on a
daemon thread; GPPilot takes an atomic snapshot each control tick.
"""

from __future__ import annotations

import math
import os
import threading
import time
import traceback
from collections import deque

import numpy as np

from simulator.gyro_ahrs import GyroAHRS

G = 9.81
ACC_SMOOTH_N = 5
ESTIMATION_POLL_HZ = 400
LAUNCH_PITCH_DEG = -17.8
# Gyro sign convention differs by sim build: VQ2 delivers gyro inverted vs NED
# (-1, AndurilGP-empirical), but VQ1 delivers it un-inverted (+1, measured
# against VQ1 ATTITUDE truth, R^2=1.0 all axes). Wrong sign spins the integrated
# attitude the wrong way -> phantom yaw -> fly-away. Default -1 preserves VQ2.
GYRO_SIGN = float(os.environ.get("GP_GYRO_SIGN", "-1.0"))


class GPEstimation:
    """Background IMU propagator. Call start() after first IMU appears."""

    def __init__(self, data: dict, launch_pitch_deg: float = LAUNCH_PITCH_DEG,
                 gyro_sign: float | None = None):
        self.data = data
        # Read the gyro sign at CONSTRUCTION (runtime), not import, so setting
        # GP_GYRO_SIGN before the pilot is built takes effect (VQ1=+1, VQ2=-1).
        self._gyro_sign = (
            gyro_sign if gyro_sign is not None
            else float(os.environ.get("GP_GYRO_SIGN", "-1.0"))
        )
        # Per-axis accel sign, sim-dependent (GP_ACC_SIGN="x,y,z"). VQ1 flips the
        # forward axis ("-1,1,1", measured vs ATTITUDE truth); VQ2 default "1,1,1".
        # Wrong ax sign corrupts forward velocity and (in the EKF) inverts pitch.
        self._acc_sign = tuple(
            float(v) for v in os.environ.get("GP_ACC_SIGN", "1,1,1").split(",")
        )
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

    def zero_velocity(self) -> None:
        """Clear dead-reckoned speed without reseeding AHRS (post-GO hygiene)."""
        with self._lock:
            self.vel_ned[:] = 0.0
            self.vel_body[:] = 0.0
            self.pos_ned[:] = 0.0

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
        last_tb = 0.0
        while self._running:
            # RX swaps in a fresh imu dict per message (atomic ref under the
            # GIL), so reading it needs no lock.
            imu = self.data.get("imu")
            if imu is not None:
                try:
                    self._process_imu(imu)
                except Exception:
                    # One malformed IMU sample must not kill this thread —
                    # snapshot() would silently freeze at the last value and
                    # the pilot would fly a stale attitude.
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

            # Gyro sign is sim-dependent (GP_GYRO_SIGN): -1 for VQ2, +1 for VQ1.
            s = self._gyro_sign
            gx = s * float(imu.get("gx", imu.get("xgyro", 0.0)))
            gy = s * float(imu.get("gy", imu.get("ygyro", 0.0)))
            gz = s * float(imu.get("gz", imu.get("zgyro", 0.0)))
            self._att_deg = self.ahrs.update(gx, gy, gz, dt)
            self.rates_body[0] = math.degrees(gx)
            self.rates_body[1] = math.degrees(gy)
            self.rates_body[2] = math.degrees(gz)

            ax = self._acc_sign[0] * float(imu.get("ax", imu.get("xacc", 0.0)))
            ay = self._acc_sign[1] * float(imu.get("ay", imu.get("yacc", 0.0)))
            az = self._acc_sign[2] * float(imu.get("az", imu.get("zacc", 0.0)))
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
