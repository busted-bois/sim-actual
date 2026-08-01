"""Minimal always-on lateral leveler (P on roll/pitch → rates)."""

from __future__ import annotations

import math

# Measured odometry sign conventions (rl/fly2_course.py).
SIGN_ROLL = -1.0
SIGN_PITCH = +1.0

LEVEL_GAIN = 3.0  # rad/s per rad (odometry)
LEVEL_CLIP = 0.5  # rad/s


def level_rates(
    roll: float,
    pitch: float,
    lean_roll: float = 0.0,
    lean_pitch: float = 0.0,
    soft: bool = False,
) -> tuple[float, float]:
    """P controller toward (lean_roll, lean_pitch) attitude targets.

    soft=True (VQ2 / ESKF): send zero rates. Boot pitch bias (±20°) + any
    non-zero P made pitch run to 70° and trip safety. Vertical tests only
    need thrust; plant holds attitude under zero rates (no auto-level).
    """
    if soft:
        return 0.0, 0.0
    roll_cmd = SIGN_ROLL * LEVEL_GAIN * (lean_roll - roll)
    pitch_cmd = SIGN_PITCH * LEVEL_GAIN * (lean_pitch - pitch)
    roll_cmd = float(max(-LEVEL_CLIP, min(LEVEL_CLIP, roll_cmd)))
    pitch_cmd = float(max(-LEVEL_CLIP, min(LEVEL_CLIP, pitch_cmd)))
    return roll_cmd, pitch_cmd


def tilt_comp_factor(roll: float, pitch: float) -> float:
    """1 / (cos(roll)*cos(pitch)), clamped away from singularity."""
    c = math.cos(roll) * math.cos(pitch)
    c = max(0.2, abs(c))
    return 1.0 / c
