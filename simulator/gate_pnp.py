"""Tier 2 gate PnP from image corners and known gate geometry."""

from __future__ import annotations

import math

import cv2
import numpy as np

from simulator.flight_config import (
    CAM_CX,
    CAM_CY,
    CAM_FX,
    CAM_FY,
    CAM_TILT_UP_DEG,
    GATE_INNER_MM,
    PNP_MIN_CORNERS,
)


def _gate_object_points():
    half = 0.5 * GATE_INNER_MM / 1000.0
    # Gate frame: X forward through opening, Y right, Z down (NED-like gate plane).
    return np.array(
        [
            [0.0, -half, -half],
            [0.0, half, -half],
            [0.0, half, half],
            [0.0, -half, half],
        ],
        dtype=np.float64,
    )


def _camera_matrix():
    return np.array(
        [[CAM_FX, 0.0, CAM_CX], [0.0, CAM_FY, CAM_CY], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _camera_to_body_rotation(pitch_up_degrees=CAM_TILT_UP_DEG):
    pitch = math.radians(-pitch_up_degrees)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]],
        dtype=np.float64,
    )


def solve_gate_pnp(corners_px, pitch_up_degrees=CAM_TILT_UP_DEG):
    """Return camera-frame tvec (m) and yaw correction (rad), or None."""
    if corners_px is None or len(corners_px) < PNP_MIN_CORNERS:
        return None

    image_points = np.array(corners_px[:4], dtype=np.float64)
    object_points = _gate_object_points()
    camera_matrix = _camera_matrix()
    dist = np.zeros(5, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    tvec = tvec.reshape(3)
    rvec = rvec.reshape(3)
    rot_cam, _ = cv2.Rodrigues(rvec)
    rot_body = _camera_to_body_rotation(pitch_up_degrees) @ rot_cam
    yaw_correction = math.atan2(rot_body[1, 0], rot_body[0, 0])
    range_m = float(max(0.5, tvec[2]))
    return {
        "tvec": (float(tvec[0]), float(tvec[1]), range_m),
        "yaw_correction": float(yaw_correction),
        "range_m": range_m,
        "lateral_m": float(tvec[0]),
    }


def order_corners(points):
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(points, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = pts[order]
    # Rotate so first point is top-left (smallest x+y).
    sums = ordered[:, 0] + ordered[:, 1]
    shift = int(np.argmin(sums))
    return [tuple(map(float, pt)) for pt in np.roll(ordered, -shift, axis=0)]
