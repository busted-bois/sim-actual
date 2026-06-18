from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


GRAVITY_NED_MPS2 = 9.81


@dataclass
class ImuPrediction:
    dt_s: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    pos_ned: tuple[float, float, float]
    vel_ned: tuple[float, float, float]
    accel_ned: tuple[float, float, float]
    covariance_trace: float


def _rotation_body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


class ImuKalmanPredictor:
    """IMU-only prediction model (no external correction yet).

    State: [pn, pe, pd, vn, ve, vd, roll, pitch, yaw]
    """

    def __init__(self) -> None:
        self._x = np.zeros((9, 1), dtype=float)
        self._P = np.eye(9, dtype=float) * 0.05
        self._gyro_noise = 0.02
        self._accel_noise = 0.7

    def predict(
        self,
        accel_body_mps2: tuple[float, float, float],
        gyro_body_rps: tuple[float, float, float],
        dt_s: float,
    ) -> ImuPrediction:
        dt_s = max(1e-4, min(0.05, dt_s))

        # Integrate attitude from gyro.
        self._x[6, 0] += gyro_body_rps[0] * dt_s
        self._x[7, 0] += gyro_body_rps[1] * dt_s
        self._x[8, 0] += gyro_body_rps[2] * dt_s

        # Use accel tilt to reduce gyro drift in roll/pitch (IMU-only complementary).
        ax, ay, az = accel_body_mps2
        roll_acc = math.atan2(ay, max(1e-6, az))
        pitch_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az) + 1e-6)
        self._x[6, 0] = 0.98 * self._x[6, 0] + 0.02 * roll_acc
        self._x[7, 0] = 0.98 * self._x[7, 0] + 0.02 * pitch_acc

        roll = float(self._x[6, 0])
        pitch = float(self._x[7, 0])
        yaw = float(self._x[8, 0])
        rot_bn = _rotation_body_to_ned(roll, pitch, yaw)

        # Body specific force -> NED acceleration; add gravity in +Down axis.
        accel_ned = rot_bn @ np.array([[ax], [ay], [az]], dtype=float)
        accel_ned[2, 0] += GRAVITY_NED_MPS2

        # Integrate velocity/position.
        self._x[3:6, 0:1] += accel_ned * dt_s
        self._x[0:3, 0:1] += self._x[3:6, 0:1] * dt_s

        # Covariance prediction.
        F = np.eye(9, dtype=float)
        F[0, 3] = dt_s
        F[1, 4] = dt_s
        F[2, 5] = dt_s
        Q = np.eye(9, dtype=float) * 1e-5
        Q[3:6, 3:6] *= (self._accel_noise**2) * dt_s
        Q[6:9, 6:9] *= (self._gyro_noise**2) * dt_s
        self._P = F @ self._P @ F.T + Q

        return ImuPrediction(
            dt_s=dt_s,
            roll_rad=roll,
            pitch_rad=pitch,
            yaw_rad=yaw,
            pos_ned=(float(self._x[0, 0]), float(self._x[1, 0]), float(self._x[2, 0])),
            vel_ned=(float(self._x[3, 0]), float(self._x[4, 0]), float(self._x[5, 0])),
            accel_ned=(
                float(accel_ned[0, 0]),
                float(accel_ned[1, 0]),
                float(accel_ned[2, 0]),
            ),
            covariance_trace=float(np.trace(self._P)),
        )
