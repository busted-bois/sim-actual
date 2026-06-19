"""Shared spec for the whole pipeline — intrinsics, frames, gate geometry,
projection, action/observation layout. Single source of truth so Modules
2/4/6/7/8 agree on conventions.

Frames
------
World  : NED   (x=North, y=East, z=Down).
Body   : FRD   (x=Forward, y=Right, z=Down). odometry quaternion (w,x,y,z)
                rotates Body -> World, matching simulator.transforms.quat_to_yaw.
Camera : optical (x=Right, y=Down, z=Forward along optical axis). Mounted on
                body, tilted CAM_TILT_DEG *up* from body-forward.

Action space (resolved with user): attitude-rate + thrust, NOT velocity
setpoints (this sim ignores SET_POSITION_TARGET velocity).
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Camera intrinsics (fixed, given). fx=fy=320, cx=320, cy=180 => 640x360 frame.
# ----------------------------------------------------------------------------
FX = 320.0
FY = 320.0
CX = 320.0
CY = 180.0
IMG_W = 640
IMG_H = 360

K = np.array(
    [
        [FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DIST = np.zeros((5,), dtype=np.float64)  # no lens distortion (given)

CAM_TILT_DEG = 20.0  # camera pitched up from body-forward

# ----------------------------------------------------------------------------
# Gate geometry — square opening, side GATE_SIZE_M (meters). Value matches the
# w/h the sim broadcasts in the gate map (data/gate_map.json: 2.72 for all 6).
# PnP (Module 4) and mask projection (Module 2) must use the REAL size or range
# estimates scale by the wrong factor and the pass-through lateral gate is off.
# ----------------------------------------------------------------------------
GATE_SIZE_M = 2.72
GATE_HALF = GATE_SIZE_M / 2.0

# Gate-local frame: +x = through-gate normal (travel dir), +y = right (width),
# +z = down (height). Corners ordered TL, TR, BR, BL with "top" = NED-up (-z).
GATE_CORNERS_LOCAL = np.array(
    [
        [0.0, -GATE_HALF, -GATE_HALF],  # TL
        [0.0, +GATE_HALF, -GATE_HALF],  # TR
        [0.0, +GATE_HALF, +GATE_HALF],  # BR
        [0.0, -GATE_HALF, +GATE_HALF],  # BL
    ],
    dtype=np.float64,
)

# ----------------------------------------------------------------------------
# Control / action space — attitude-rate + thrust.
# ----------------------------------------------------------------------------
MAX_ROLL_RATE = 4.0  # rad/s
MAX_PITCH_RATE = 4.0
MAX_YAW_RATE = 3.0
THRUST_MIN = 0.0
THRUST_MAX = 1.0
# Hover thrust on the REAL sim. 0.27 is a validated closed-loop trim (fly2.py
# HOVER_T=0.27 + altitude PID cleared all 6 gates); open-loop sysID leaves the
# true value uncertain (~0.19-0.33). Training randomizes around it (rl.env).
HOVER_THRUST = 0.27

# Policy emits 4 values in [-1, 1]; scale_action() maps to physical commands.
ACTION_DIM = 4
ACTION_SCALE = np.array(
    [MAX_ROLL_RATE, MAX_PITCH_RATE, MAX_YAW_RATE, 1.0], dtype=np.float64
)

OBS_DIM = 24

GRAVITY = 9.81

# ----------------------------------------------------------------------------
# 24-D observation layout (Module 6). Indices are the contract for 6/7/8.
# ----------------------------------------------------------------------------
OBS_LAYOUT = {
    "to_gate_body": slice(0, 3),  # unit vector to current gate (body frame)
    "dist_to_gate": slice(3, 4),  # meters (raw)
    "gate_normal_body": slice(4, 7),  # gate through-axis in body frame
    "vel_body": slice(7, 10),  # velocity in body frame (m/s)
    "ang_vel": slice(10, 13),  # body angular rates roll/pitch/yaw (rad/s)
    "gravity_body": slice(13, 16),  # gravity dir in body frame (encodes roll/pitch)
    "yaw_align": slice(16, 17),  # heading error to gate-normal bearing (rad)
    "to_next_gate_body": slice(17, 20),  # unit vector to next gate (body)
    "dist_to_next_gate": slice(20, 21),  # meters
    "last_action": slice(21, 24),  # last roll/pitch/yaw-rate command (normalized)
}


# ----------------------------------------------------------------------------
# Rotations
# ----------------------------------------------------------------------------
def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix (Body -> World)."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def R_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def R_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


# Camera->Body rotation: base maps optical(z-fwd,x-right,y-down) onto
# body(x-fwd,y-right,z-down), then pitch the optical axis up by CAM_TILT_DEG.
_R_BODY_CAM_BASE = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
R_BODY_CAM = R_y(np.radians(CAM_TILT_DEG)) @ _R_BODY_CAM_BASE
R_CAM_BODY = R_BODY_CAM.T


def world_to_cam(
    p_world: np.ndarray, drone_pos: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Transform world point(s) (...,3) into camera frame."""
    R_wb = quat_to_R(q)  # body->world
    rel = np.atleast_2d(p_world) - drone_pos  # world delta
    # Row-vector form: p_body_row = (R_wb.T @ rel_col)^T = rel_row @ R_wb.
    p_body = rel @ R_wb  # world->body
    # p_cam_row = (R_CAM_BODY @ p_body_col)^T = p_body_row @ R_CAM_BODY.T = @ R_BODY_CAM.
    p_cam = p_body @ R_BODY_CAM  # body->cam
    return p_cam


def project(p_world: np.ndarray, drone_pos: np.ndarray, q: np.ndarray):
    """Project world point(s) to pixel coords. Returns (pixels Nx2, in_front mask)."""
    p_cam = world_to_cam(p_world, drone_pos, q)
    z = p_cam[:, 2]
    in_front = z > 1e-3
    zc = np.where(in_front, z, 1.0)
    u = FX * p_cam[:, 0] / zc + CX
    v = FY * p_cam[:, 1] / zc + CY
    return np.stack([u, v], axis=1), in_front


def gate_corners_world(gate_pos: np.ndarray, gate_quat: np.ndarray) -> np.ndarray:
    """4 gate corners (TL,TR,BR,BL) in world NED."""
    R = quat_to_R(gate_quat)
    return gate_pos + GATE_CORNERS_LOCAL @ R.T


def scale_action(a: np.ndarray) -> np.ndarray:
    """Map normalized policy action [-1,1]^4 -> (roll_rate,pitch_rate,yaw_rate,thrust)."""
    a = np.clip(a, -1.0, 1.0)
    roll = a[0] * MAX_ROLL_RATE
    pitch = a[1] * MAX_PITCH_RATE
    yaw = a[2] * MAX_YAW_RATE
    thrust = (a[3] + 1.0) * 0.5  # [-1,1] -> [0,1]
    return np.array([roll, pitch, yaw, thrust], dtype=np.float64)
