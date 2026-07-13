"""Blow-up detector and recovery actions."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from flightlab.state import Cmd, State

DEG60 = math.radians(60.0)
GYRO_LIMIT = 8.0
POSE_GAP_S = 0.5
ALT_HIGH_M = 60.0
ALT_LOW_M = 0.2
TAKEOFF_LATCH_M = 0.5  # low-alt trip arms only after first climbing past this
HORIZ_LIMIT_M = 100.0
TILT_TRIP_S = 0.2
GYRO_TRIP_S = 0.5
HOVER_THRUST = 0.27


@dataclass
class SafetyResult:
    tripped: bool = False
    reason: str = ""
    recovery_cmd: Cmd = field(default_factory=lambda: Cmd(0.0, 0.0, 0.0, HOVER_THRUST))


class SafetyMonitor:
    def __init__(self) -> None:
        self._tilt_t0: float | None = None
        self._gyro_t0: float | None = None
        self._was_armed = False
        self._arm_mono: float | None = None
        self._airborne = False  # latched once alt first exceeds TAKEOFF_LATCH_M

    def reset(self) -> None:
        """Re-anchor the grace window. Call immediately before flight phases
        (after race GO + arm) — calling it right after reset_sim lets the
        up-to-45 s race-GO wait silently burn the whole grace window."""
        self._tilt_t0 = None
        self._gyro_t0 = None
        self._was_armed = False
        self._arm_mono = time.monotonic()
        self._airborne = False

    def check(self, s: State) -> SafetyResult:
        now = s.t_mono
        arm_grace = self._arm_mono is not None and (now - self._arm_mono) < 3.0
        tilt = max(abs(s.roll), abs(s.pitch))
        if tilt > DEG60:
            if self._tilt_t0 is None:
                self._tilt_t0 = now
            elif now - self._tilt_t0 >= TILT_TRIP_S:
                return SafetyResult(True, "tilt>60deg", _level_hover())
        else:
            self._tilt_t0 = None

        alt = -s.pos_ned[2]
        if s.alt_trusted:
            if alt > TAKEOFF_LATCH_M:
                self._airborne = True
            if not arm_grace:
                if alt > ALT_HIGH_M:
                    return SafetyResult(True, "alt>60m", _level_hover())
                # The drone spawns on the ground (alt ~0): the low-alt floor
                # only arms after the first real climb, else it trips at tick 1.
                if self._airborne and alt < ALT_LOW_M:
                    return SafetyResult(True, "alt<0.2m", _level_hover())

        if abs(s.pos_ned[0]) > HORIZ_LIMIT_M or abs(s.pos_ned[1]) > HORIZ_LIMIT_M:
            return SafetyResult(True, "horiz>100m", _level_hover())

        gyro_mag = math.hypot(*s.gyro)
        if gyro_mag > GYRO_LIMIT:
            if self._gyro_t0 is None:
                self._gyro_t0 = now
            elif now - self._gyro_t0 >= GYRO_TRIP_S:
                return SafetyResult(True, "gyro>8rad/s", _level_hover())
        else:
            self._gyro_t0 = None

        if s.pose_age_s > POSE_GAP_S:
            return SafetyResult(True, "pose_gap>0.5s", _level_hover())

        if self._was_armed and not s.armed:
            return SafetyResult(True, "unexpected_disarm", _level_hover())
        self._was_armed = s.armed

        return SafetyResult()


def _level_hover() -> Cmd:
    return Cmd(0.0, 0.0, 0.0, HOVER_THRUST)
