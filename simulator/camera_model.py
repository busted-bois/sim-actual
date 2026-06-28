"""Pinhole camera geometry and IBVS interaction matrix for the live sim.

Intrinsics match SPEC / rl.spec (640x360, fx=fy=320, 20 deg camera tilt up).
"""

from __future__ import annotations

import math

import numpy as np

# Fixed intrinsics — keep in sync with rl/spec.py
FX = 320.0
FY = 320.0
CX = 320.0
CY = 180.0
IMG_W = 640
IMG_H = 360
CAM_TILT_DEG = 20.0
GATE_SIZE_M = 2.72

DEFAULT_JACOBIAN_BLEND = 0.25

_R_BODY_CAM_BASE = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def _R_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


R_BODY_CAM = _R_y(math.radians(CAM_TILT_DEG)) @ _R_BODY_CAM_BASE
R_CAM_BODY = R_BODY_CAM.T


def pixel_bearing(
    u: float,
    v: float,
    fx: float = FX,
    fy: float = FY,
    cx: float = CX,
    cy: float = CY,
) -> tuple[float, float]:
    """Horizontal and vertical bearing (rad) from principal point."""
    return math.atan2(u - cx, fx), math.atan2(v - cy, fy)


def range_from_width(
    width_px: float, gate_width_m: float, focal_px: float = FX
) -> float | None:
    if width_px < 1.0:
        return None
    return gate_width_m * focal_px / width_px


def interaction_matrix(
    u: float, v: float, depth_m: float, fx: float = FX, fy: float = FY
) -> np.ndarray:
    """2x6 image Jacobian L_v for pixel feature (u,v) with camera-frame depth Z.

    Twist order: [vx, vy, vz, wx, wy, wz] in camera frame (optical: x-right, y-down, z-fwd).
    s_dot = L @ v where s = [u, v]^T.
    """
    z = max(depth_m, 0.5)
    x = (u - CX) / fx
    y = (v - CY) / fy
    return np.array(
        [
            [-fx / z, 0.0, x / z, x * y / fx, -(fx + x * x * fx) / fx, y],
            [0.0, -fy / z, y / z, (fy + y * y * fy) / fy, -x * y / fy, -x],
        ],
        dtype=np.float64,
    )


def ibvs_twist(
    u: float,
    v: float,
    depth_m: float,
    u_star: float = CX,
    v_star: float = CY,
    gain: float = 0.8,
    fx: float = FX,
    fy: float = FY,
) -> np.ndarray:
    """Camera-frame twist that drives pixel error toward zero (damped least squares)."""
    e = np.array([u - u_star, v - v_star], dtype=np.float64)
    L = interaction_matrix(u, v, depth_m, fx, fy)
    lam = 0.01
    L_pinv = L.T @ np.linalg.inv(L @ L.T + lam * np.eye(2))
    return -gain * (L_pinv @ e)


def twist_to_body_rates(twist_cam: np.ndarray) -> tuple[float, float, float]:
    """Map camera twist to body yaw rate (rad/s) and vertical velocity hint (m/s).

    Returns (yaw_rate, vz_body_ned, lateral_body_y).
    NED body: x-fwd, y-right, z-down.
    """
    v_cam = twist_cam[0:3]
    w_cam = twist_cam[3:6]
    v_body = R_CAM_BODY @ v_cam
    w_body = R_CAM_BODY @ w_cam
    yaw_rate = float(w_body[2])
    vz_ned = float(v_body[2])
    return yaw_rate, vz_ned, float(v_body[1])


def _range_yaw_scale(range_m: float | None) -> float:
    if range_m is None or range_m < 0.5:
        return 1.0
    return max(0.4, min(1.0, range_m / 15.0))


def blend_jacobian_p(
    nx: float,
    ny: float,
    u: float,
    v: float,
    confidence: float,
    yaw_gain: float,
    vy_gain: float,
    blend: float = DEFAULT_JACOBIAN_BLEND,
    range_m: float | None = None,
) -> tuple[float, float]:
    """Blend P-control with pinhole bearing; adaptive + range-scaled yaw."""
    scale = _range_yaw_scale(range_m)
    yaw_p = yaw_gain * nx * scale
    alt_p = vy_gain * ny

    if confidence < 0.3:
        return yaw_p, alt_p

    bearing_h, _elevation = pixel_bearing(u, v)
    yaw_j = yaw_gain * bearing_h * scale

    nx_equiv = abs(nx) * math.pi
    adapt = min(1.0, abs(bearing_h) / max(nx_equiv, 1e-6))
    w = blend * min(confidence, 1.0) * adapt
    yaw = (1.0 - w) * yaw_p + w * yaw_j
    return yaw, alt_p


def blend_ibvs_pseudo(
    nx: float,
    ny: float,
    u: float,
    v: float,
    depth_m: float,
    confidence: float,
    yaw_gain: float,
    vy_gain: float,
    ibvs_gain: float = 0.8,
    blend: float = 0.6,
) -> tuple[float, float]:
    """Full IBVS pseudo-inverse blend (kept for regression benchmarks)."""
    yaw_p = yaw_gain * nx
    alt_p = vy_gain * ny

    if depth_m <= 0.5 or confidence < 0.3:
        return yaw_p, alt_p

    twist = ibvs_twist(u, v, depth_m, gain=ibvs_gain)
    yaw_ibvs, vz_body, _ = twist_to_body_rates(twist)
    alt_ibvs = -vz_body * 2.0

    w = blend * min(confidence, 1.0)
    yaw = (1.0 - w) * yaw_p + w * yaw_ibvs
    alt = (1.0 - w) * alt_p + w * alt_ibvs
    return yaw, alt


def blend_ibvs_p(
    nx: float,
    ny: float,
    u: float,
    v: float,
    depth_m: float,
    confidence: float,
    yaw_gain: float,
    vy_gain: float,
    ibvs_gain: float = 0.8,
    blend: float = 0.6,
) -> tuple[float, float]:
    """Alias for blend_jacobian_p (depth arg ignored)."""
    del depth_m, ibvs_gain, blend
    return blend_jacobian_p(nx, ny, u, v, confidence, yaw_gain, vy_gain)
