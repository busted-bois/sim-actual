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

import numpy as np

from rl import spec

# Policy roll/pitch target of ±1 maps to this tilt (rad). ~0.5 rad = 28.6°, a
# reasonable aggressive-but-stable bank for a racer inside the ±0.6 rad/s plant.
MAX_TILT_RAD = float(0.5)
# Angle-error -> rate gain (1/s). At 0.5 rad error this asks 2.5 rad/s, clamped
# to the ±0.6 rad/s cap, so the controller saturates toward the target then eases.
KP_ATT = float(5.0)


def action_to_rates(action, cur_roll_rad: float, cur_pitch_rad: float):
    """(roll_t, pitch_t, yaw_rate, thrust) in [-1,1] ->
       (roll_rate, pitch_rate, yaw_rate, thrust) for send_attitude_rates.

    roll/pitch are ANGLE targets P-controlled to rates vs the current attitude;
    yaw is a direct rate; thrust maps [-1,1] -> [0,1]."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    roll_target = a[0] * MAX_TILT_RAD
    pitch_target = a[1] * MAX_TILT_RAD
    yaw_rate = float(a[2] * spec.MAX_YAW_RATE)
    thrust = float((a[3] + 1.0) * 0.5)  # [-1,1] -> [0,1]
    roll_rate = float(np.clip(KP_ATT * (roll_target - cur_roll_rad),
                              -spec.MAX_ROLL_RATE, spec.MAX_ROLL_RATE))
    pitch_rate = float(np.clip(KP_ATT * (pitch_target - cur_pitch_rad),
                               -spec.MAX_PITCH_RATE, spec.MAX_PITCH_RATE))
    return roll_rate, pitch_rate, yaw_rate, thrust


if __name__ == "__main__":  # quick sanity
    # Level drone, command full roll-right target -> should saturate roll_rate +.
    rr, pr, yr, th = action_to_rates([1.0, 0.0, 0.0, 0.0], 0.0, 0.0)
    assert abs(rr - spec.MAX_ROLL_RATE) < 1e-9 and abs(pr) < 1e-9
    assert abs(th - 0.5) < 1e-9, th
    # Already at target angle -> zero rate.
    rr2, _, _, _ = action_to_rates([1.0, 0.0, 0.0, 1.0], MAX_TILT_RAD, 0.0)
    assert abs(rr2) < 1e-9, rr2
    print("[vq2.controller] selftest OK", rr, pr, yr, th)
