"""Pure gyroscope attitude integrator (AndurilGP GyroAHRS).

No accelerometer correction — thrust corrupts specific force during
manoeuvres. Seed from known launch-ramp pitch, then integrate gyro only.
"""

from __future__ import annotations

import math

import numpy as np


def euler_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """ZYX Euler (rad, NED) -> quaternion [w, x, y, z]."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


class GyroAHRS:
    """Quaternion gyro integrator. NED ZYX Euler: +roll RWD, +pitch nose up."""

    def __init__(self, initial_pitch_deg: float = 0.0, initial_roll_deg: float = 0.0):
        self.q = np.array(
            euler_to_quat(
                math.radians(initial_roll_deg),
                math.radians(initial_pitch_deg),
                0.0,
            ),
            dtype=np.float64,
        )

    def update(
        self, gx: float, gy: float, gz: float, dt: float
    ) -> tuple[float, float, float]:
        """Integrate one gyro sample (rad/s, sign-corrected). Returns Euler deg."""
        qw, qx, qy, qz = self.q
        h = 0.5 * dt
        qw_n = qw + h * (-qx * gx - qy * gy - qz * gz)
        qx_n = qx + h * (qw * gx + qy * gz - qz * gy)
        qy_n = qy + h * (qw * gy - qx * gz + qz * gx)
        qz_n = qz + h * (qw * gz + qx * gy - qy * gx)
        n = math.sqrt(qw_n**2 + qx_n**2 + qy_n**2 + qz_n**2)
        self.q = np.array([qw_n, qx_n, qy_n, qz_n], dtype=np.float64) / n
        return self.euler_deg()

    def euler_deg(self) -> tuple[float, float, float]:
        qw, qx, qy, qz = self.q
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        sp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
        pitch = math.asin(sp)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    @property
    def quaternion(self) -> np.ndarray:
        return self.q.copy()
