"""Velocity estimate from HIGHRES_IMU accel integration only (no optical flow).

The IMU reports specific force; with a level-flight assumption the gravity
component sits on body z (NED: sensor reads ~ -g on z when level), so linear
acceleration = specific_force + (0, 0, +g).

Integration drifts - it is reset on every gate pass event and clamped to
zero whenever the magnitude diverges past SPEED_CLAMP_MPS.

Publishes into shared_data["state"]:
    speed_mps    - |v| for the hover check
    velocity_ned - (vx, vy, vz) for dead reckoning
"""

import math

GRAVITY_MPS2 = 9.80665
SPEED_CLAMP_MPS = 15.0  # estimate above this is divergence - reset to zero


class VelocityEstimator:
    def __init__(self, data):
        self.data = data
        self.velocity = [0.0, 0.0, 0.0]
        self._last_imu_time_us = None
        data["state"] = {"speed_mps": 0.0, "velocity_ned": (0.0, 0.0, 0.0)}

    def tick(self):
        """Integrate the latest IMU sample. Call each controller tick."""
        imu = self.data.get("imu")
        if not imu:
            return

        t_us = imu["time_boot_us"]
        if self._last_imu_time_us is None:
            self._last_imu_time_us = t_us
            return
        if t_us <= self._last_imu_time_us:
            return  # no new sample since last tick
        dt = (t_us - self._last_imu_time_us) * 1e-6
        self._last_imu_time_us = t_us

        ax, ay, az = imu["accel"]
        # remove gravity assuming level flight (attitude compensation later)
        az += GRAVITY_MPS2

        self.velocity[0] += ax * dt
        self.velocity[1] += ay * dt
        self.velocity[2] += az * dt

        speed = math.sqrt(sum(v * v for v in self.velocity))
        if speed > SPEED_CLAMP_MPS:
            self.reset()
            speed = 0.0

        self.data["state"] = {
            "speed_mps": speed,
            "velocity_ned": tuple(self.velocity),
        }

    def reset(self):
        """Zero the integrator - on gate pass or divergence."""
        self.velocity = [0.0, 0.0, 0.0]
        self.data["state"] = {"speed_mps": 0.0, "velocity_ned": (0.0, 0.0, 0.0)}
