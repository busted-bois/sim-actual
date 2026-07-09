"""Post-commit dead reckoning.

After the tracker commits, fly a fixed forward velocity (body-forward) and
accumulate distance from the IMU-integrated velocity estimate until the
next gate becomes visible in perception. No timeout - t_min-only design.
"""

import math

DEAD_RECKON_SPEED_MPS = 1.0  # commanded forward speed while blind


class DeadReckoner:
    def __init__(self, data):
        self.data = data
        self.active = False
        self.distance_m = 0.0
        self._last_tick_time = None

    def start(self):
        self.active = True
        self.distance_m = 0.0
        self._last_tick_time = None

    def stop(self):
        self.active = False
        self._last_tick_time = None

    def tick(self, now):
        """Accumulate travelled distance from the velocity estimate.
        Returns the commanded forward speed (m/s) while active."""
        if not self.active:
            return 0.0
        if self._last_tick_time is not None:
            dt = max(0.0, now - self._last_tick_time)
            state = self.data.get("state", {})
            vx, vy, vz = state.get("velocity_ned", (0.0, 0.0, 0.0))
            self.distance_m += math.sqrt(vx * vx + vy * vy + vz * vz) * dt
        self._last_tick_time = now
        return DEAD_RECKON_SPEED_MPS
