"""Test maneuver helpers and phase targets."""

from __future__ import annotations

import math
from dataclasses import dataclass

from flightlab.state import Target

HOVER_Z_NED = -3.0


@dataclass
class Phase:
    name: str
    duration_s: float
    target: Target
    open_loop_rates: tuple[float, float, float] | None = None
    open_loop_thrust: float | None = None


def hover_target(
    z: float = HOVER_Z_NED, roll: float = 0.0, pitch: float = 0.0
) -> Target:
    return Target(roll=roll, pitch=pitch, yaw=0.0, z=z)


def sign_id_phases(
    z: float = HOVER_Z_NED, roll: float = 0.0, pitch: float = 0.0
) -> list[Phase]:
    """B0: +0.2 rad/s pulse 0.3 s per axis at hover."""
    pulse = 0.2
    dur = 0.3
    settle = 1.5
    axes = [
        ("roll", (pulse, 0.0, 0.0)),
        ("pitch", (0.0, pulse, 0.0)),
        ("yaw", (0.0, 0.0, pulse)),
    ]
    phases: list[Phase] = [
        Phase("hover_settle", 2.0, hover_target(z, roll, pitch)),
    ]
    for name, rates in axes:
        phases.append(
            Phase(
                f"sign_{name}_pulse",
                dur,
                hover_target(z, roll, pitch),
                open_loop_rates=rates,
            )
        )
        phases.append(
            Phase(
                f"sign_{name}_settle",
                settle,
                hover_target(z, roll, pitch),
            )
        )
    return phases


def rate_tracking_phase(z: float = HOVER_Z_NED) -> list[Phase]:
    return [
        Phase("hover_settle", 1.0, hover_target(z)),
        Phase(
            "rate_roll_cmd",
            1.0,
            hover_target(z),
            open_loop_rates=(0.3, 0.0, 0.0),
        ),
        Phase("rate_recover", 1.0, hover_target(z)),
    ]


def angle_step_phases(
    axis: str, step_deg: float = 8.0, z: float = HOVER_Z_NED
) -> list[Phase]:
    step = math.radians(step_deg)
    tgt = hover_target(z)
    if axis == "roll":
        tgt = Target(roll=step, pitch=0.0, yaw=0.0, z=z)
    elif axis == "pitch":
        tgt = Target(roll=0.0, pitch=step, yaw=0.0, z=z)
    return [
        Phase("hover_settle", 1.5, hover_target(z)),
        Phase(f"step_{axis}", 3.0, tgt),
        Phase("hold", 2.0, tgt),
    ]


def hover_jitter_phase(duration_s: float = 20.0, z: float = HOVER_Z_NED) -> list[Phase]:
    return [Phase("hover_jitter", duration_s, hover_target(z))]


def disturbance_phase(z: float = HOVER_Z_NED) -> list[Phase]:
    return [
        Phase("hover_settle", 1.5, hover_target(z)),
        Phase(
            "disturb_roll",
            0.3,
            hover_target(z),
            open_loop_rates=(0.8, 0.0, 0.0),
        ),
        Phase("recover", 3.0, hover_target(z)),
    ]
