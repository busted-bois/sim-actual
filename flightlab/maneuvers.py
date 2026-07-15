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
    # Damp body rates toward zero using the signs measured so far
    # (ctx.sign_accum); axes without a measured sign get zero command.
    rate_null: bool = False


def hover_target(
    z: float = HOVER_Z_NED, roll: float = 0.0, pitch: float = 0.0
) -> Target:
    return Target(roll=roll, pitch=pitch, yaw=0.0, z=z)


def sign_doublet_phases(
    axis: str,
    z: float = HOVER_Z_NED,
    pulse: float = 0.2,
    dur: float = 0.3,
    gap: float = 0.4,
) -> list[Phase]:
    """B0: open-loop ± rate doublet on one axis (net rate ~ 0, so tilt does
    not accumulate across axes the way single pulses did)."""
    i = ("roll", "pitch", "yaw").index(axis)
    pos = tuple(pulse if j == i else 0.0 for j in range(3))
    neg = tuple(-pulse if j == i else 0.0 for j in range(3))
    zero = (0.0, 0.0, 0.0)
    tgt = hover_target(z)
    return [
        Phase(f"sign_{axis}_base0", gap, tgt, open_loop_rates=zero),
        Phase(f"sign_{axis}_pulse_pos", dur, tgt, open_loop_rates=pos),
        Phase(f"sign_{axis}_mid", gap, tgt, open_loop_rates=zero),
        Phase(f"sign_{axis}_pulse_neg", dur, tgt, open_loop_rates=neg),
        Phase(f"sign_{axis}_base1", gap, tgt, open_loop_rates=zero),
    ]


def rate_null_phase(z: float = HOVER_Z_NED, duration_s: float = 0.5) -> Phase:
    """Damp residual body rates between B0 axes (uses measured signs only)."""
    return Phase("rate_null", duration_s, hover_target(z), rate_null=True)


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
