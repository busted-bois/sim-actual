"""Lightweight geometric attitude controller for the VQ2 RL policy.

The policy emits (roll_target, pitch_target, yaw_rate, thrust) in [-1,1]^4. This
maps the roll/pitch ANGLE targets to body-rate commands with a simple
proportional law against the current estimated attitude, then hands
(roll_rate, pitch_rate, yaw_rate, thrust) to the sim's SET_ATTITUDE_TARGET rate
path (`SimInterface.send_attitude_rates` -> `controller._send_attitude_rates`),
which is the command form the VQ2 sim actuates. NEVER outputs motor commands.

Kept pure/stateless (current attitude passed in) so it unit-tests offline.
"""

from __future__ import annotations

import os

import numpy as np

from rl import spec

# Policy roll/pitch target of ±1 maps to this tilt (rad). ~0.35 rad ≈ 20° —
# enough bank to race, not enough nose-up to rocket-climb out of the course.
MAX_TILT_RAD = float(os.environ.get("VQ2_MAX_TILT", "0.35"))
# Angle-error -> rate gain (1/s). Saturates into the ±0.6 rad/s plant caps.
KP_ATT = float(5.0)

# Physical thrust band after mapping [-1,1] -> [0,1]. Hover ~0.27; GP pilot
# lives in ~[0.18, 0.50]. Cap well below full-throttle so the policy cannot
# climb/accel out of the race corridor.
THRUST_MIN = float(os.environ.get("VQ2_THRUST_MIN", "0.18"))
THRUST_MAX = float(os.environ.get("VQ2_THRUST_MAX", "0.40"))


def action_to_rates(action, cur_roll_rad: float, cur_pitch_rad: float):
    """(roll_t, pitch_t, yaw_rate, thrust) in [-1,1] ->
       (roll_rate, pitch_rate, yaw_rate, thrust) for send_attitude_rates.

    roll/pitch are ANGLE targets P-controlled to rates vs the current attitude;
    yaw is a direct rate; thrust maps [-1,1] -> [0,1] then clamps to the racing
    band [THRUST_MIN, THRUST_MAX]."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    roll_target = a[0] * MAX_TILT_RAD
    pitch_target = a[1] * MAX_TILT_RAD
    yaw_rate = float(a[2] * spec.MAX_YAW_RATE)
    thrust = float(np.clip((a[3] + 1.0) * 0.5, THRUST_MIN, THRUST_MAX))
    roll_rate = float(np.clip(KP_ATT * (roll_target - cur_roll_rad),
                              -spec.MAX_ROLL_RATE, spec.MAX_ROLL_RATE))
    pitch_rate = float(np.clip(KP_ATT * (pitch_target - cur_pitch_rad),
                               -spec.MAX_PITCH_RATE, spec.MAX_PITCH_RATE))
    return roll_rate, pitch_rate, yaw_rate, thrust


if __name__ == "__main__":  # quick sanity
    # Level drone, command full roll-right target -> should saturate roll_rate +.
    rr, pr, yr, th = action_to_rates([1.0, 0.0, 0.0, 0.0], 0.0, 0.0)
    assert abs(rr - spec.MAX_ROLL_RATE) < 1e-9 and abs(pr) < 1e-9
    assert abs(th - THRUST_MAX) < 1e-9, th   # a3=0 -> 0.5, clamped to max
    # Full thrust command still hard-capped.
    _, _, _, th_hi = action_to_rates([0.0, 0.0, 0.0, 1.0], 0.0, 0.0)
    assert abs(th_hi - THRUST_MAX) < 1e-9, th_hi
    _, _, _, th_lo = action_to_rates([0.0, 0.0, 0.0, -1.0], 0.0, 0.0)
    assert abs(th_lo - THRUST_MIN) < 1e-9, th_lo
    # Already at target angle -> zero rate.
    rr2, _, _, _ = action_to_rates([1.0, 0.0, 0.0, 1.0], MAX_TILT_RAD, 0.0)
    assert abs(rr2) < 1e-9, rr2
    print("[vq2.controller] selftest OK", rr, pr, yr, th, f"thrust=[{THRUST_MIN},{THRUST_MAX}]")
