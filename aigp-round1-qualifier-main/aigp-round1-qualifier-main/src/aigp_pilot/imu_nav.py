from __future__ import annotations

import math

import numpy as np


class ImuNavigator:
    """Attitude from HIGHRES_IMU gyro (+ mag yaw). VQ2 has no position telemetry."""

    def __init__(self) -> None:
        self.att = np.zeros(3, dtype=float)  # roll, pitch, yaw
        self.ready = False
        self._last_t_us: int | None = None

    def update(self, msg) -> None:
        t_us = int(msg.time_usec)
        if self._last_t_us is None:
            self._last_t_us = t_us
            self.ready = True
            return

        dt = (t_us - self._last_t_us) * 1e-6
        self._last_t_us = t_us
        if dt <= 0.0 or dt > 0.5:
            return

        gx, gy, gz = float(msg.xgyro), float(msg.ygyro), float(msg.zgyro)
        if not all(math.isfinite(v) for v in (gx, gy, gz)):
            return

        self.att[0] += gx * dt
        self.att[1] += gy * dt
        self.att[2] += gz * dt

        xmag, ymag = float(msg.xmag), float(msg.ymag)
        if math.isfinite(xmag) and math.isfinite(ymag) and (xmag * xmag + ymag * ymag) > 1e-12:
            yaw_mag = math.atan2(ymag, xmag)
            blend = min(1.0, 2.0 * dt)
            delta = (yaw_mag - self.att[2] + math.pi) % (2 * math.pi) - math.pi
            self.att[2] += blend * delta
