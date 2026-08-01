"""Action -> command mapping for the VQ2 RL policy.

CRITICAL: the policy must actuate through the SAME channel the proven GP pilot
flies on. The GP pilot runs the Controller in "attitude_quat" mode and sends
ABSOLUTE attitude as degrees -- `set_attitude_quat_deg(roll_deg, pitch_deg,
yaw_deg, thrust)` -> `_send_attitude_quat` (euler->quaternion SET_ATTITUDE_TARGET,
body rates masked out). It clamps bank to MAX_BANK_DEG=14deg.

So the policy's (roll, pitch, yaw, thrust) in [-1,1]^4 map to absolute attitude
DEGREES, sent via `SimInterface.send_attitude_quat_deg`. This is the exact form
the demos are recorded in (see `rl.vq2.log_demos`), so BC reproduces the GP pilot
faithfully. (The previous version emitted body RATES on a different wire -- a
channel the GP pilot never used and whose numbers meant something else entirely,
which is why the cloned policy didn't fly toward gates.)

Kept pure/stateless so it unit-tests offline.
"""

from __future__ import annotations

import os

import numpy as np

# Policy roll/pitch target of ±1 maps to this absolute tilt (deg). Covers the GP
# pilot's ±14deg bank clamp with margin; pitch leans are smaller (slow cruise).
MAX_TILT_DEG = float(20.0)
# Policy yaw of ±1 maps to this absolute yaw command (deg) on the quat wire. The
# GP pilot's yaw commands are modest (~±12deg in cruise); 45 covers reorient.
MAX_YAW_DEG = float(45.0)


def action_to_attitude(action):
    """(roll, pitch, yaw, thrust) in [-1,1] -> (roll_deg, pitch_deg, yaw_deg,
    thrust in [0,1]) for `SimInterface.send_attitude_quat_deg` -- the same
    absolute-attitude channel the GP pilot uses."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    roll_deg = float(a[0] * MAX_TILT_DEG)
    pitch_deg = float(a[1] * MAX_TILT_DEG)
    yaw_deg = float(a[2] * MAX_YAW_DEG)
    thrust = float((a[3] + 1.0) * 0.5)  # [-1,1] -> [0,1]
    return roll_deg, pitch_deg, yaw_deg, thrust


# --- residual RL: the policy's action is a SMALL correction ADDED to the GP
# base command, not a full attitude. Kept small so it can't destabilize the
# proven controller; the base (GP feedback law) provides stability, the policy
# only nudges the line. Tunable via VQ2_RES_* for Stage 3.
RES_TILT_DEG = float(os.environ.get("VQ2_RES_TILT_DEG", "6.0"))   # ± roll/pitch nudge
RES_YAW_DEG = float(os.environ.get("VQ2_RES_YAW_DEG", "8.0"))     # ± yaw nudge
RES_THRUST = float(os.environ.get("VQ2_RES_THRUST", "0.08"))      # ± thrust nudge


def action_to_residual(action):
    """(roll, pitch, yaw, thrust) in [-1,1] -> a SMALL (deg, deg, deg, thrust)
    correction to ADD onto the GP pilot's base command."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return (float(a[0] * RES_TILT_DEG), float(a[1] * RES_TILT_DEG),
            float(a[2] * RES_YAW_DEG), float(a[3] * RES_THRUST))


def gp_cmd_to_action(roll_deg, pitch_deg, yaw_deg, thrust):
    """Inverse of `action_to_attitude`: the GP pilot's absolute-degree command
    -> normalized action in [-1,1]^4. Used to label BC demos so the cloned policy
    reproduces the GP pilot's exact command."""
    return np.array([
        np.clip(roll_deg / MAX_TILT_DEG, -1.0, 1.0),
        np.clip(pitch_deg / MAX_TILT_DEG, -1.0, 1.0),
        np.clip(yaw_deg / MAX_YAW_DEG, -1.0, 1.0),
        np.clip(2.0 * float(thrust) - 1.0, -1.0, 1.0),
    ], dtype=np.float32)


if __name__ == "__main__":  # quick sanity
    rd, pd, yd, th = action_to_attitude([1.0, -0.5, 0.0, 0.0])
    assert abs(rd - MAX_TILT_DEG) < 1e-9 and abs(pd + 0.5 * MAX_TILT_DEG) < 1e-9
    assert abs(th - 0.5) < 1e-9, th
    # round-trip: command -> action -> command is identity within clamp range.
    a = gp_cmd_to_action(14.0, -3.0, 10.0, 0.27)
    rd2, pd2, yd2, th2 = action_to_attitude(a)
    assert abs(rd2 - 14.0) < 1e-4 and abs(pd2 + 3.0) < 1e-4 and abs(yd2 - 10.0) < 1e-4
    assert abs(th2 - 0.27) < 1e-4, th2
    print("[vq2.controller] selftest OK", round(rd, 2), round(pd, 2), round(yd, 2), th)
