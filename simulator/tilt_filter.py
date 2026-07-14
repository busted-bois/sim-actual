"""Complementary roll/pitch from HIGHRES_IMU (shared by IBVS + fly2).

Gyro integrate; blend accel gravity tilt only when |f|~g. Under thrust the
accel measures the thrust axis, not gravity — leave allow_accel=False then.
"""

from __future__ import annotations

import math

import numpy as np


class TiltFilter:
    def __init__(self, alpha: float = 0.02):
        self.alpha = alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.roll_acc = 0.0
        self.pitch_acc = 0.0
        self.f_ok = False
        self.gyro_mag = 0.0
        self._last_us = None

    def reset(self) -> None:
        self.roll = self.pitch = 0.0
        self._last_us = None
        self.roll_acc = self.pitch_acc = 0.0
        self.f_ok = False
        self.gyro_mag = 0.0

    def update(self, imu: dict, allow_accel: bool = True, boost: bool = False) -> None:
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
        if self.f_ok:
            self.roll_acc = math.atan2(-ay, -az)
            self.pitch_acc = math.atan2(ax, math.hypot(ay, az))
            if allow_accel:
                a = 0.25 if boost else self.alpha
                self.roll = (1 - a) * self.roll + a * self.roll_acc
                self.pitch = (1 - a) * self.pitch + a * self.pitch_acc

    def sync_to_accel(self) -> bool:
        if not self.f_ok:
            return False
        self.roll, self.pitch = self.roll_acc, self.pitch_acc
        return True
