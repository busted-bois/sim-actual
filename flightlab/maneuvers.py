"""Maneuver primitives: hold, step, ramp, timed vz schedules, waypoints."""

from __future__ import annotations

import math
from dataclasses import dataclass

from flightlab.protocol import Target
from flightlab.state import State

# Waypoint P-loop gains (position -> lean target). Sign convention matches
# rl/fly2_course: tgt_roll = -k*right_err, tgt_pitch = -k*forward_err.
WP_KP = 0.05  # rad lean per m position error
WP_KD = 0.12  # rad lean per m/s velocity (damping)
WP_LEAN_MAX = math.radians(8.0)  # TILT-validated sag limit
WP_TOL_M = 0.75
WP_TIMEOUT_S = 25.0


@dataclass
class Segment:
    """One timed segment of a maneuver schedule."""

    duration: float
    z: float | None = None
    vz: float | None = None
    lean_roll: float = 0.0
    lean_pitch: float = 0.0
    label: str = ""


class Schedule:
    """Piecewise-constant setpoint schedule over time since start."""

    def __init__(self, segments: list[Segment]) -> None:
        self.segments = segments
        self._bounds: list[tuple[float, float, Segment]] = []
        t = 0.0
        for seg in segments:
            self._bounds.append((t, t + seg.duration, seg))
            t += seg.duration
        self.total = t

    def target_at(self, elapsed: float) -> Target | None:
        if elapsed < 0 or elapsed >= self.total:
            return None
        for t0, t1, seg in self._bounds:
            if t0 <= elapsed < t1:
                return Target(
                    z=seg.z,
                    vz=seg.vz,
                    lean_roll=seg.lean_roll,
                    lean_pitch=seg.lean_pitch,
                )
        return None

    def label_at(self, elapsed: float) -> str:
        for t0, t1, seg in self._bounds:
            if t0 <= elapsed < t1:
                return seg.label
        return ""

    def done(self, elapsed: float) -> bool:
        return elapsed >= self.total


def hold(z: float, duration: float, label: str = "hold") -> Schedule:
    return Schedule([Segment(duration=duration, z=z, vz=0.0, label=label)])


def step_then_hold(
    z0: float,
    z1: float,
    hold0: float,
    hold1: float,
    label0: str = "pre",
    label1: str = "step",
) -> Schedule:
    return Schedule(
        [
            Segment(duration=hold0, z=z0, vz=0.0, label=label0),
            Segment(duration=hold1, z=z1, vz=0.0, label=label1),
        ]
    )


def altitude_steps(z_base: float, delta: float = 5.0, hold_s: float = 5.0) -> Schedule:
    """+delta hold, then −delta (back to base), each hold_s. NED: climb = −z."""
    z_up = z_base - delta  # climb
    return Schedule(
        [
            Segment(duration=hold_s, z=z_up, vz=0.0, label="climb_step"),
            Segment(duration=hold_s, z=z_base, vz=0.0, label="descend_step"),
        ]
    )


def vz_rate_schedule(
    rates: list[float],
    each_s: float = 3.0,
    z_hold: float | None = None,
) -> Schedule:
    """Timed vz setpoints. If z_hold set, also pin altitude softly via z target None."""
    segs = [
        Segment(duration=each_s, z=z_hold, vz=vz, label=f"vz={vz:+.1f}") for vz in rates
    ]
    return Schedule(segs)


def soft_land(z_start: float, vz_descend: float = 1.5) -> Schedule:
    """Descend at +vz (NED down) from z_start. Duration long; suite ends on settle."""
    # Generous upper bound; runner aborts on ground settle.
    return Schedule(
        [
            Segment(
                duration=60.0,
                z=None,
                vz=vz_descend,
                label="soft_land",
            )
        ]
    )


class WaypointTracker:
    """Per-tick lean targets toward a NED waypoint (z held via baro law).

    Body-frame P on position with velocity damping; leans capped at the
    TILT-validated 8 deg so tilt compensation covers the altitude loss.
    """

    def __init__(
        self,
        wp: tuple[float, float, float],
        tol_m: float = WP_TOL_M,
        z_tol_m: float = 1.0,
        timeout_s: float = WP_TIMEOUT_S,
        label: str = "wp",
    ) -> None:
        self.wp = (float(wp[0]), float(wp[1]), float(wp[2]))
        self.tol_m = tol_m
        self.z_tol_m = z_tol_m
        self.timeout_s = timeout_s
        self.label = label
        self.last_dist = float("inf")

    def dist(self, s: State) -> float:
        return math.hypot(self.wp[0] - s.pos[0], self.wp[1] - s.pos[1])

    def reached(self, s: State) -> bool:
        # Altitude gate too: on climb courses the next gate is up to ~11 m
        # higher — pushing through before the climb finishes hits the frame.
        return self.dist(s) <= self.tol_m and abs(s.pos[2] - self.wp[2]) <= self.z_tol_m

    def target_at(self, s: State) -> Target:
        dn = self.wp[0] - s.pos[0]
        de = self.wp[1] - s.pos[1]
        self.last_dist = math.hypot(dn, de)
        cy, sy = math.cos(s.yaw), math.sin(s.yaw)
        fwd_err = dn * cy + de * sy
        right_err = -dn * sy + de * cy
        fwd_vel = s.vel[0] * cy + s.vel[1] * sy
        right_vel = -s.vel[0] * sy + s.vel[1] * cy
        lean_pitch = _clip(-(WP_KP * fwd_err - WP_KD * fwd_vel), WP_LEAN_MAX)
        lean_roll = _clip(-(WP_KP * right_err - WP_KD * right_vel), WP_LEAN_MAX)
        return Target(
            z=self.wp[2],
            vz=0.0,
            lean_roll=lean_roll,
            lean_pitch=lean_pitch,
        )


def _clip(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def lean_hold(
    z: float,
    lean_rad: float,
    duration: float,
    axis: str = "roll",
) -> Schedule:
    lean_roll = lean_rad if axis == "roll" else 0.0
    lean_pitch = lean_rad if axis == "pitch" else 0.0
    return Schedule(
        [
            Segment(
                duration=duration,
                z=z,
                vz=0.0,
                lean_roll=lean_roll,
                lean_pitch=lean_pitch,
                label=f"lean_{axis}",
            )
        ]
    )
