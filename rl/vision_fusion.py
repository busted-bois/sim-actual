"""Optional vision pose fusion for the RL deploy EKF."""

from __future__ import annotations

import numpy as np

from rl import spec


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
    p_body = spec.R_CAM_BODY @ p_cam
    p_meas = np.asarray(gate_world_pos, float) - R_wb @ p_body
    sigma = 0.8 * (1.1 - min(confidence, 1.0))
    ekf.update_position(p_meas, sigma=sigma)
    return True
