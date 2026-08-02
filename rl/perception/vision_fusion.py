"""Optional vision pose fusion for the RL deploy EKF."""

from __future__ import annotations

import math
import os
import warnings

import numpy as np

from rl.core import spec

YAW_SIGMA_RAD = 0.35  # base yaw-measurement uncertainty at confidence=1.0


def gate_pose_from_image(
    img_bgr,
    weights: str | None = None,
    device: str | None = None,
    drone_quat: np.ndarray | None = None,
    gate_world_pos: np.ndarray | None = None,
):
    """GateNet -> mask -> PnP pose for one BGR frame, or ``None`` if untrained.

    Lazy image-to-pose entry: only imports/constructs ``GateNetInfer`` when a
    weights file is actually present. Repo weights are absent by default (the
    user trains them), so in that state it warns once and returns ``None``
    rather than raising — letting the EKF coast on dead-reckoning. Does not
    touch the ``fuse_*`` APIs.
    """
    from rl.perception.gatenet import WEIGHTS_PATH
    from rl.perception.pnp import pose_from_mask

    path = WEIGHTS_PATH if weights is None else weights
    if not os.path.isfile(path):
        warnings.warn(
            f"GateNet weights not found at {path}; skipping image pose recovery",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    from rl.perception.gatenet import GateNetInfer

    mask = GateNetInfer(path, device=device).mask(img_bgr)
    return pose_from_mask(mask, drone_quat=drone_quat, gate_world_pos=gate_world_pos)


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
    ekf.update_position(p_meas, sigma=sigma)
    return True


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


def _extract_pnp(det: dict):
    if not isinstance(det, dict):
        return None
    pose = det.get("pose") or {}
    if not isinstance(pose, dict):
        return None
    gate_body = pose.get("gate_pos_body")
    try:
        conf = float(det.get("conf", 0.0) or 0.0)
        reproj = float(pose.get("reproj_px", 1e9))
        range_m = float(pose.get("range_m", 0.0))
    except (TypeError, ValueError):
        return None
    if gate_body is None:
        return None
    try:
        gb = np.asarray(gate_body, float)
        if gb.shape != (3,) or not np.isfinite(gb).all():
            return None
    except (ValueError, TypeError):
        return None
    if conf < 0.5 or not math.isfinite(conf):
        return None
    if not (math.isfinite(reproj) and math.isfinite(range_m)):
        return None
    if reproj > 10.0 or not (0.5 < range_m < 40.0):
        return None
    return gb, conf, reproj, range_m


def _fuse_pnp_position_yaw(ekf, gate_body, conf, gate_world_pos):
    gb = np.asarray(gate_body, float)
    gw = np.asarray(gate_world_pos, float)
    if gb.shape != (3,) or gw.shape != (3,):
        return False
    if not (np.isfinite(gb).all() and np.isfinite(gw).all()):
        return False
    R_wb = spec.quat_to_R(np.asarray(ekf.q, float))
    p_meas = gw - R_wb @ gb
    p_prior = ekf.p.copy()
    sigma = 1.2 - min(conf, 0.9)
    ekf.update_position(p_meas, sigma=sigma)

    dx, dy = gw[0] - p_prior[0], gw[1] - p_prior[1]
    if math.hypot(dx, dy) >= 0.5:
        yaw_bearing = math.atan2(gb[1], gb[0])
        yaw_meas = _wrap(math.atan2(dy, dx) - yaw_bearing)
        roll, pitch, _ = _rpy_from_quat(ekf.q)
        q_meas = _quat_from_rpy(roll, pitch, yaw_meas)
        ekf.update_attitude(q_meas, sigma=YAW_SIGMA_RAD * (1.1 - min(conf, 1.0)))
    return True


def fuse_pnp_gate(ekf, det: dict, gate_world_pos, max_pred_err_m: float = 6.0) -> bool:
    """Fuse a YOLO+PnP gate detection against the active gate's world position.

    gate_pos_body is a full 3D body-frame measurement, so the implied drone
    position is p = gate_world - R_wb @ gate_body — far stronger than the HSV
    bearing/range path. Also anchors yaw the same way as fuse_gate_bearing_yaw.
    Gated on prediction error so a detection of the WRONG gate (two gates in
    frame) cannot poison the filter. Returns True when a position update ran.
    Uses adaptive gate: max(max_pred_err_m, ekf.innovation_gate(max_pred_err_m)).
    """
    extracted = _extract_pnp(det)
    if extracted is None:
        return False
    gate_body, conf, _, _ = extracted

    gw = np.asarray(gate_world_pos, float)
    if gw.shape != (3,) or not np.isfinite(gw).all():
        return False
    R_wb = spec.quat_to_R(np.asarray(ekf.q, float))
    p_meas = gw - R_wb @ np.asarray(gate_body, float)
    innov = float(np.linalg.norm(p_meas - ekf.p))

    gate = max(max_pred_err_m, ekf.innovation_gate(max_pred_err_m))
    if innov > gate:
        return False

    return _fuse_pnp_position_yaw(ekf, gate_body, conf, gw)


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


def step_pnp_fusion(ekf, det, gate_world_pos, dt, max_coast, gate_floor=3.0):
    try:
        gw = np.asarray(gate_world_pos, float)
        if gw.shape != (3,) or not np.isfinite(gw).all():
            ekf.coast(1, dt)
            return "miss"
    except (ValueError, TypeError):
        ekf.coast(1, dt)
        return "miss"

    if not ekf.healthy() or ekf.coast_count >= max_coast:
        if not np.isfinite(ekf.q).all():
            ekf.coast(1, dt)
            return "rejected"
        extracted = _extract_pnp(det) if det else None
        if extracted is not None:
            gate_body, conf, _, _ = extracted
            ekf.reset_on_detection(pnp_gate_body=gate_body, gate_world_pos=gw)
            _fuse_pnp_position_yaw(ekf, gate_body, conf, gw)
            return "reset"
        ekf.coast(1, dt)
        return "miss"

    extracted = _extract_pnp(det) if det else None
    if extracted is None:
        ekf.coast(1, dt)
        return "miss"

    gate_body, conf, _, _ = extracted
    R_wb = spec.quat_to_R(np.asarray(ekf.q, float))
    p_meas = gw - R_wb @ np.asarray(gate_body, float)
    innov = float(np.linalg.norm(p_meas - ekf.p))
    gate = max(gate_floor, ekf.innovation_gate(gate_floor))

    if innov > gate:
        ekf.coast(1, dt)
        return "rejected"

    accepted = _fuse_pnp_position_yaw(ekf, gate_body, conf, gw)
    return "update" if accepted else "rejected"
