"""Maneuver primitives: hold, step, ramp, timed vz schedules."""

from __future__ import annotations

from dataclasses import dataclass

from flightlab.protocol import Target


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
