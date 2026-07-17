"""Blow-up monitor + kill sequence."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from flightlab.bus import HOVER_THRUST, Bus
from flightlab.state import State

ATT_LIMIT_RAD = math.radians(60.0)
ATT_HOLD_S = 0.2
ALT_MAX_M = 60.0
ALT_MIN_M = 0.2  # only after airborne (spawn starts ~0)
AIRBORNE_ALT_M = 1.0
NE_LIMIT_M = 100.0
GYRO_LIMIT = 8.0  # rad/s
GYRO_HOLD_S = 0.5
POSE_GAP_S = 0.5
DISARM_HOLD_S = 0.4  # debounce brief armed HB flicker / stale buffer
KILL_HOLD_S = 2.0


@dataclass
class SafetyResult:
    tripped: bool = False
    reason: str = ""


@dataclass
class SafetyMonitor:
    """Active in every test. Call check() each tick; handle_trip() on trip."""

    expect_near_ground: bool = False
    armed_expected: bool = True
    _att_bad_since: float | None = None
    _gyro_bad_since: float | None = None
    _disarm_bad_since: float | None = None
    _ever_had_pose: bool = False
    _was_airborne: bool = False
    trips: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self._att_bad_since = None
        self._gyro_bad_since = None
        self._disarm_bad_since = None
        self._ever_had_pose = False
        self._was_airborne = False
        self.expect_near_ground = False
        self.armed_expected = True

    def check(self, s: State) -> SafetyResult:
        now = s.t
        if s.has_pose:
            self._ever_had_pose = True

        # Pose gap
        if self._ever_had_pose and s.pose_age > POSE_GAP_S:
            return self._trip(f"pose_gap={s.pose_age:.2f}s")

        # Unexpected disarm (debounced — ignore brief HB flicker / stale buffer)
        if self.armed_expected and self._ever_had_pose and not s.armed:
            if self._disarm_bad_since is None:
                self._disarm_bad_since = now
            elif now - self._disarm_bad_since >= DISARM_HOLD_S:
                return self._trip("unexpected_disarm")
        else:
            self._disarm_bad_since = None

        if not s.has_pose:
            return SafetyResult()

        n, e, z = s.pos
        alt = -z  # NED → altitude above spawn/ground approx
        if alt >= AIRBORNE_ALT_M:
            self._was_airborne = True

        # Attitude
        if abs(s.roll) > ATT_LIMIT_RAD or abs(s.pitch) > ATT_LIMIT_RAD:
            if self._att_bad_since is None:
                self._att_bad_since = now
            elif now - self._att_bad_since >= ATT_HOLD_S:
                return self._trip(
                    f"attitude |r|={math.degrees(abs(s.roll)):.0f} "
                    f"|p|={math.degrees(abs(s.pitch)):.0f} deg"
                )
        else:
            self._att_bad_since = None

        # Altitude — max always; min only after we were airborne (not spawn)
        if alt > ALT_MAX_M:
            return self._trip(f"alt={alt:.1f}m > {ALT_MAX_M}")
        if self._was_airborne and not self.expect_near_ground and alt < ALT_MIN_M:
            return self._trip(f"alt={alt:.2f}m unplanned low")

        # Lateral drift
        if abs(n) > NE_LIMIT_M or abs(e) > NE_LIMIT_M:
            return self._trip(f"ne=({n:.1f},{e:.1f})")

        # Gyro
        gx, gy, gz = s.gyro
        if max(abs(gx), abs(gy), abs(gz)) > GYRO_LIMIT:
            if self._gyro_bad_since is None:
                self._gyro_bad_since = now
            elif now - self._gyro_bad_since >= GYRO_HOLD_S:
                return self._trip(f"gyro={max(abs(gx), abs(gy), abs(gz)):.1f}")
        else:
            self._gyro_bad_since = None

        return SafetyResult()

    def _trip(self, reason: str) -> SafetyResult:
        self.trips.append(reason)
        return SafetyResult(tripped=True, reason=reason)

    def handle_trip(self, bus: Bus, reason: str) -> None:
        """Level + hover 2 s → disarm → reset."""
        print(f"[safety] BLOWUP: {reason} — kill sequence", flush=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < KILL_HOLD_S:
            bus.drain()
            bus.send(0.0, 0.0, 0.0, HOVER_THRUST)
            time.sleep(1.0 / 90.0)
        bus.disarm()
        time.sleep(0.3)
        bus.reset()
        time.sleep(2.0)
