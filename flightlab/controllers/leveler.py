"""Minimal always-on lateral leveler (P on roll/pitch → rates)."""

from __future__ import annotations

import math

# Measured odometry sign conventions (rl/fly2_course.py).
SIGN_ROLL = -1.0
SIGN_PITCH = +1.0

LEVEL_GAIN = 3.0  # rad/s per rad (odometry)
LEVEL_CLIP = 0.5  # rad/s
# Soft ESKF: boot pitch bias + gain 3 → dive; still need soft lean for TILT.
EST_LEVEL_GAIN = 0.6
EST_LEVEL_CLIP = 0.30
LEAN_EPS = 1e-3


def level_rates(
    roll: float,
    pitch: float,
    lean_roll: float = 0.0,
    lean_pitch: float = 0.0,
    soft: bool = False,
) -> tuple[float, float]:
    """P controller toward (lean_roll, lean_pitch) attitude targets.

    soft=True (VQ2 / ESKF): zero rates when lean≈0 (boot bias → pitch blowup).
    Soft P toward lean when lean targets set (TILT / gate approach).
    """
    if soft and abs(lean_roll) + abs(lean_pitch) <= LEAN_EPS:
        return 0.0, 0.0
    gain = EST_LEVEL_GAIN if soft else LEVEL_GAIN
    clip = EST_LEVEL_CLIP if soft else LEVEL_CLIP
    roll_cmd = SIGN_ROLL * gain * (lean_roll - roll)
    pitch_cmd = SIGN_PITCH * gain * (lean_pitch - pitch)
    roll_cmd = float(max(-clip, min(clip, roll_cmd)))
    pitch_cmd = float(max(-clip, min(clip, pitch_cmd)))
    return roll_cmd, pitch_cmd


def tilt_comp_factor(roll: float, pitch: float) -> float:
    """1 / (cos(roll)*cos(pitch)), clamped away from singularity."""
    c = math.cos(roll) * math.cos(pitch)
    c = max(0.2, abs(c))
    return 1.0 / c
