"""Optional vision pose fusion for the RL deploy EKF."""

from __future__ import annotations

import math

import numpy as np

from rl import spec

YAW_SIGMA_RAD = 0.35  # base yaw-measurement uncertainty at confidence=1.0


def fuse_gate_target_position(
    ekf,
    gate_target: dict,
    gate_world_pos: np.ndarray,
    drone_quat: np.ndarray | None,
) -> bool:
    """Loosely fuse metric range from gate_target into the EKF position update.

    Uses bearing/range from the live gate_estimator path (wired through vision_rx)
    to derive a world-position measurement: p = gate - R_wb @ p_gate_body.
    Returns True when an update was applied.
    """
    range_m = gate_target.get("range_m")
    confidence = float(gate_target.get("confidence", 0.0) or 0.0)
    if range_m is None or range_m < 0.5 or confidence < 0.5 or drone_quat is None:
        return False

    bearing = float(gate_target.get("bearing_rad", 0.0))
    elevation = float(gate_target.get("elevation_rad", 0.0))
    # Gate centroid direction in camera frame -> body frame.
    p_cam = np.array(
        [
            range_m * np.tan(bearing),
            range_m * np.tan(elevation),
            range_m,
        ],
        dtype=np.float64,
    )
    R_wb = spec.quat_to_R(np.asarray(drone_quat, float))
    p_body = spec.R_BODY_CAM @ p_cam
    p_meas = np.asarray(gate_world_pos, float) - R_wb @ p_body
    sigma = 0.8 * (1.1 - min(confidence, 1.0))
    return bool(ekf.update_position(p_meas, sigma=sigma, gate=True))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rpy_from_quat(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def fuse_pnp_gate(ekf, det: dict, gate_world_pos, max_pred_err_m: float = 6.0) -> bool:
    """Fuse a YOLO+PnP gate detection against the active gate's world position.

    gate_pos_body is a full 3D body-frame measurement, so the implied drone
    position is p = gate_world - R_wb @ gate_body — far stronger than the HSV
    bearing/range path. Also anchors yaw the same way as fuse_gate_bearing_yaw.
    Outliers are rejected by the ESKF Mahalanobis NIS gate (r^T S^{-1} r).
    max_pred_err_m is unused (kept for call-site compat). Returns True when a
    position update ran.
    """
    del max_pred_err_m  # NIS gate replaces Euclidean pred-error check
    pose = det.get("pose") or {}
    gate_body = pose.get("gate_pos_body")
    conf = float(det.get("conf", 0.0) or 0.0)
    if gate_body is None or conf < 0.5:
        return False
    reproj = float(pose.get("reproj_px", 1e9))
    range_m = float(pose.get("range_m", 0.0))
    if reproj > 10.0 or not (0.5 < range_m < 40.0):
        return False

    g = np.asarray(gate_world_pos, float)
    R_wb = spec.quat_to_R(np.asarray(ekf.q, float))
    p_meas = g - R_wb @ np.asarray(gate_body, float)
    p_prior = ekf.p.copy()
    sigma = 1.2 - min(conf, 0.9)  # 0.3–0.7 m by detection confidence
    if not ekf.update_position(p_meas, sigma=sigma, gate=True):
        return False

    # Yaw anchor: world bearing to the gate vs the body-frame PnP bearing.
    dx, dy = g[0] - p_prior[0], g[1] - p_prior[1]
    if math.hypot(dx, dy) >= 0.5:
        yaw_bearing = float(
            pose.get("yaw_bearing", math.atan2(gate_body[1], gate_body[0]))
        )
        yaw_meas = _wrap(math.atan2(dy, dx) - yaw_bearing)
        roll, pitch, _ = _rpy_from_quat(ekf.q)
        q_meas = _quat_from_rpy(roll, pitch, yaw_meas)
        ekf.update_attitude(q_meas, sigma=YAW_SIGMA_RAD * (1.1 - min(conf, 1.0)))
    return True


def fuse_gate_bearing_yaw(
    ekf,
    gate_target: dict,
    drone_pos_ned: np.ndarray,
    gate_world_pos: np.ndarray,
) -> bool:
    """Anchor yaw from (world bearing to gate) vs. the measured camera bearing.

    `bearing_rad` is ~the body-frame horizontal angle to the gate (camera is
    only pitch-tilted, no yaw offset), so combined with the known gate world
    position and the current position estimate it gives an absolute yaw
    reference — unlike `fuse_gate_target_position`, which reuses the filter's
    own (possibly wrong) orientation and can't correct it. Keeps the filter's
    current roll/pitch and only corrects yaw via `ekf.update_attitude`.
    Returns True when an update was applied.
    """
    if not gate_target.get("detected"):
        return False
    range_m = gate_target.get("range_m")
    confidence = float(gate_target.get("confidence", 0.0) or 0.0)
    if range_m is None or range_m < 0.5 or confidence < 0.5:
        return False

    p = np.asarray(drone_pos_ned, float)
    g = np.asarray(gate_world_pos, float)
    dx, dy = g[0] - p[0], g[1] - p[1]
    if math.hypot(dx, dy) < 0.5:
        return False

    bearing = float(gate_target.get("bearing_rad", 0.0))
    world_bearing = math.atan2(dy, dx)
    yaw_meas = _wrap(world_bearing - bearing)

    roll, pitch, _ = _rpy_from_quat(ekf.q)
    q_meas = _quat_from_rpy(roll, pitch, yaw_meas)
    sigma = YAW_SIGMA_RAD * (1.1 - min(confidence, 1.0))
    ekf.update_attitude(q_meas, sigma=sigma)
    return True
