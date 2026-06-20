"""Geometric cascaded controller (attitude-rate + thrust).

Position/velocity -> desired acceleration -> desired thrust vector + attitude
-> body-rate commands. Outputs the SAME normalized 4-D action the policy uses
(spec.scale_action), so it serves three roles:

  * a scripted "expert" to validate the env physics (Module 7),
  * an optional warm-start / fallback for deployment (Module 8),
  * a sanity baseline to compare the learned policy against.

Frames: world NED, body FRD, thrust acts along -body_z (up).
"""

from __future__ import annotations

import numpy as np

from rl import spec
from rl.env import THRUST_ACCEL

G_WORLD = np.array([0.0, 0.0, spec.GRAVITY])


def _vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def geometric_action(
    p, v, q, target, *, kp_p=1.6, kd_p=2.6, kp_att=8.0, max_speed=6.0
) -> np.ndarray:
    """Return normalized action [-1,1]^4 driving the drone toward `target`."""
    p = np.asarray(p, float)
    v = np.asarray(v, float)
    R = spec.quat_to_R(np.asarray(q, float))  # body->world

    # 1. Desired acceleration (PD on position, capped approach speed).
    to_t = target - p
    d = np.linalg.norm(to_t)
    v_des = (to_t / d) * min(max_speed, kp_p * d) if d > 1e-6 else np.zeros(3)
    a_des = kd_p * (v_des - v)

    # 2. Desired thrust vector (world). thrust_world = a_des - g, points up.
    thrust_world = a_des - G_WORLD
    T = np.linalg.norm(thrust_world)
    if T < 1e-6:
        thrust_world = -G_WORLD
        T = np.linalg.norm(thrust_world)
    thrust_cmd = float(np.clip(T / THRUST_ACCEL, 0.0, 1.0))

    # 3. Desired attitude: body-down axis opposes thrust; yaw faces target.
    b3 = -thrust_world / T  # body z (down) in world
    psi = np.arctan2(to_t[1], to_t[0])
    b1_ref = np.array([np.cos(psi), np.sin(psi), 0.0])
    b2 = np.cross(b3, b1_ref)
    nb2 = np.linalg.norm(b2)
    if nb2 < 1e-6:
        b1_ref = np.array([1.0, 0.0, 0.0])
        b2 = np.cross(b3, b1_ref)
        nb2 = np.linalg.norm(b2)
    b2 /= nb2
    b1 = np.cross(b2, b3)
    R_des = np.column_stack([b1, b2, b3])

    # 4. Attitude error -> body-rate command (in body frame).
    eR = 0.5 * _vee(R_des.T @ R - R.T @ R_des)
    omega_cmd = -kp_att * eR  # [roll,pitch,yaw] rate

    a = np.array(
        [
            omega_cmd[0] / spec.MAX_ROLL_RATE,
            omega_cmd[1] / spec.MAX_PITCH_RATE,
            omega_cmd[2] / spec.MAX_YAW_RATE,
            thrust_cmd * 2.0 - 1.0,  # [0,1] -> [-1,1]
        ]
    )
    return np.clip(a, -1.0, 1.0).astype(np.float32)
