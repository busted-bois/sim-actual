"""Geometric inner loop: velocity setpoint -> attitude-rate command.

Generalizes the cascaded P-control inner loop already proven in ``rl.fly2``:
given a desired world-frame (NED) velocity + yaw rate, compute the
roll/pitch/yaw-rate + thrust command that pursues it. ``fly2.py``'s own
pursuit law is a special case of this (its v_des is always "toward the
current gate"); here the desired velocity is an arbitrary input, so an RL
policy (or anything else) can sit on top and choose where to go.

This is a pure function of its arguments only -- no sim/global state -- so
it can be called identically from training (``rl.env``, against the
internal physics model) and deployment (``rl.deploy``, against the live
sim). That's the whole point: the policy's outer-loop decisions must see the
same inner loop in both places, or training doesn't transfer.

Reuses ``fly2.py``'s exact forward/right body-frame decomposition and its
empirically-measured sign conventions (SIGN_ROLL/PITCH/YAW) -- those were
tuned against the real sim and are trusted as correct; this module composes
them differently, it doesn't re-derive them.

    uv run -m rl.geo_control --selftest
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeoGains:
    """Tunable gains. Defaults match rl.fly2's measured/tuned values for the
    live sim. Pass a different instance to match the internal training
    model's parameters (e.g. spec.HOVER_THRUST=0.5 there vs 0.27 here)."""

    hover_thrust: float = 0.27  # throttle [0,1] to hover (sim-measured, fly2.HOVER_T)
    k_lean: float = 0.08  # forward velocity-error (m/s) -> lean angle (rad)
    k_lat: float = 0.11  # lateral velocity-error (m/s) -> roll angle (rad)
    lean_max: float = 0.18  # rad, forward-lean clip
    brake_max: float = 0.07  # rad, backward-lean (braking) clip
    roll_max: float = 0.16  # rad
    k_att: float = 0.6  # attitude-angle P -> rate
    k_vz: float = 0.03  # vertical velocity-error (m/s) -> thrust
    rate_clip: float = 0.30  # rad/s, roll/pitch rate clip
    yaw_clip: float = 0.8  # rad/s, yaw rate clip
    thrust_min: float = 0.18
    thrust_max: float = 0.5
    # Measured command-sign conventions (pitch normal; roll + yaw inverted) -- see fly2.py.
    sign_roll: float = -1.0
    sign_pitch: float = 1.0
    sign_yaw: float = -1.0


def velocity_to_action(
    v_des_ned: np.ndarray,
    yaw_rate_des: float,
    vel_ned: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
    gains: GeoGains = GeoGains(),
) -> np.ndarray:
    """Velocity/yaw-rate setpoint -> [roll_rate, pitch_rate, yaw_rate, thrust].

    Units match ``sim_interface.send_attitude_rates``: rates in rad/s,
    thrust in [0,1]. ``v_des_ned``/``vel_ned`` are world-frame NED m/s,
    ``roll``/``pitch``/``yaw`` and ``yaw_rate_des`` are radians / rad-s^-1.
    """
    v_des_ned = np.asarray(v_des_ned, dtype=float)
    vel_ned = np.asarray(vel_ned, dtype=float)

    verr_x = v_des_ned[0] - vel_ned[0]
    verr_y = v_des_ned[1] - vel_ned[1]

    # Body-frame forward/right decomposition of a world (x,y) vector -- same
    # rotation fly2.py uses for its dx/dy-to-gate (its e_cross is exactly
    # this "right" component applied to the position vector instead of a
    # velocity-error vector).
    forward_err = verr_x * math.cos(yaw) + verr_y * math.sin(yaw)
    right_err = verr_x * math.sin(yaw) - verr_y * math.cos(yaw)

    lean = float(np.clip(gains.k_lean * forward_err, -gains.brake_max, gains.lean_max))
    tgt_pitch = -lean  # forward = negative pitch (measured, see fly2.py)
    tgt_roll = float(np.clip(gains.k_lat * right_err, -gains.roll_max, gains.roll_max))

    roll_cmd = float(
        np.clip(
            gains.sign_roll * gains.k_att * (tgt_roll - roll),
            -gains.rate_clip,
            gains.rate_clip,
        )
    )
    pitch_cmd = float(
        np.clip(
            gains.sign_pitch * gains.k_att * (tgt_pitch - pitch),
            -gains.rate_clip,
            gains.rate_clip,
        )
    )
    yaw_cmd = float(
        np.clip(gains.sign_yaw * yaw_rate_des, -gains.yaw_clip, gains.yaw_clip)
    )

    # NED: vz bigger = descending faster. If we're descending faster than
    # desired (vel_ned[2] > v_des_ned[2]) we need MORE thrust to correct, so
    # the error is (actual - desired), same polarity as fly2's altitude PD.
    vz_err = vel_ned[2] - v_des_ned[2]
    thrust = float(
        np.clip(
            gains.hover_thrust + gains.k_vz * vz_err,
            gains.thrust_min,
            gains.thrust_max,
        )
    )

    return np.array([roll_cmd, pitch_cmd, yaw_cmd, thrust], dtype=float)


def _selftest():
    """Sanity checks -- not a byte-for-byte reproduction of fly2.py (its
    v_des is a scalar speed magnitude, not a velocity vector), just checks
    the generalized version agrees in sign/behavior on the cases fly2.py's
    own comments document as measured truth."""
    g = GeoGains()

    # Hover: no velocity error, no yaw rate, level attitude -> near-zero
    # rates and thrust pinned at hover (clipped into [thrust_min, thrust_max]).
    a = velocity_to_action(
        np.array([0.0, 0.0, 0.0]), 0.0, np.array([0.0, 0.0, 0.0]), 0.0, 0.0, 0.0, g
    )
    assert np.allclose(a[:3], 0.0), f"expected zero rates at hover, got {a[:3]}"
    assert abs(a[3] - g.hover_thrust) < 1e-9, f"expected hover thrust, got {a[3]}"

    # Forward-speed-up command (heading north, yaw=0) must produce NEGATIVE
    # pitch rate -- this is the measured fact fly2.py's header documents.
    a = velocity_to_action(
        np.array([2.0, 0.0, 0.0]), 0.0, np.array([0.0, 0.0, 0.0]), 0.0, 0.0, 0.0, g
    )
    assert a[1] < 0, f"forward accel should command negative pitch rate, got {a[1]}"
    assert abs(a[0]) < 1e-9 and abs(a[2]) < 1e-9, "pure forward cmd should not roll/yaw"

    # Climbing (more negative desired vz than current) must increase thrust
    # above hover -- NED altitude-sign sanity check.
    a = velocity_to_action(
        np.array([0.0, 0.0, -1.0]), 0.0, np.array([0.0, 0.0, 0.0]), 0.0, 0.0, 0.0, g
    )
    assert a[3] > g.hover_thrust, f"climb command should raise thrust, got {a[3]}"

    # Yaw-rate passthrough respects the measured sign inversion + clip.
    a = velocity_to_action(
        np.array([0.0, 0.0, 0.0]), 1.0, np.array([0.0, 0.0, 0.0]), 0.0, 0.0, 0.0, g
    )
    assert a[2] == -min(1.0, g.yaw_clip), (
        f"expected inverted+clipped yaw rate, got {a[2]}"
    )

    print("[geo_control] selftest OK", flush=True)


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
