"""Attitude from HIGHRES_IMU when the sim no longer sends ATTITUDE/ODOMETRY."""

from __future__ import annotations

import math


def euler_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Aerospace roll/pitch/yaw (rad) -> quaternion (w, x, y, z)."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


class ImuAttitude:
    """Gyro integration with accel tilt correction (body FRD)."""

    def __init__(self, tilt_alpha: float = 0.04):
        self.tilt_alpha = tilt_alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self._last_t_us: int | None = None

    def update(
        self,
        time_us: int,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
    ) -> dict:
        if self._last_t_us is not None:
            dt = (time_us - self._last_t_us) * 1e-6
            if 0.0 < dt < 0.5:
                self.roll += gx * dt
                self.pitch += gy * dt
                self.yaw += gz * dt

        roll_acc = math.atan2(ay, az)
        pitch_acc = math.atan2(-ax, math.hypot(ay, az))
        a = self.tilt_alpha
        self.roll = (1.0 - a) * self.roll + a * roll_acc
        self.pitch = (1.0 - a) * self.pitch + a * pitch_acc

        self._last_t_us = time_us
        return {
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll_speed": gx,
            "pitch_speed": gy,
            "yaw_speed": gz,
        }
