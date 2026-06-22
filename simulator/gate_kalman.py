"""Kalman filter for gate pose tracking in world (NED) frame.

The gate detector supplies visual measurements; this filter smooths them,
reduces uncertainty, and predicts through brief detection dropouts.
"""

from __future__ import annotations

import cv2
import numpy as np

from rl import spec
from rl.ekf import quat_from_smallangle, quat_inv, quat_mult, quat_norm
from rl.pnp import estimate_pose

MAX_REPROJ_ERR_PX = 15.0
BASE_POS_SIGMA_M = 0.2
BASE_ATT_SIGMA_RAD = 0.08


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array(
            [
                0.25 / s,
                (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s,
            ]
        )
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array(
            [
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
            ]
        )
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array(
            [
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
            ]
        )
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array(
        [
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        ]
    )


def measure_gate_world(
    corners: np.ndarray,
    drone_pos: np.ndarray,
    drone_quat: np.ndarray,
) -> dict | None:
    """PnP corners + drone pose -> gate position/orientation measurement."""
    pose = estimate_pose(corners.astype(np.float64))
    if pose is None:
        return None

    reproj_err = float(pose["reproj_err_px"])
    if reproj_err > MAX_REPROJ_ERR_PX:
        return None

    R_wb = spec.quat_to_R(np.asarray(drone_quat, dtype=float))
    gate_pos_body = np.asarray(pose["gate_pos_body"], dtype=float)
    p_meas = np.asarray(drone_pos, dtype=float) + R_wb @ gate_pos_body

    R_gate_cam, _ = cv2.Rodrigues(np.asarray(pose["rvec"], dtype=float))
    R_gate_world = R_wb @ spec.R_BODY_CAM @ R_gate_cam
    q_meas = quat_norm(_matrix_to_quat(R_gate_world))

    return {
        "pos_ned": p_meas,
        "quat": q_meas,
        "reproj_err_px": reproj_err,
        "range_m": float(pose["range_m"]),
    }


class GateKalmanFilter:
    """Tracks gate pose (position + orientation) with predict/update steps."""

    def __init__(self) -> None:
        self.p = np.zeros(3, dtype=float)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.P_pos = np.eye(3, dtype=float) * 4.0
        self.P_att = np.eye(3, dtype=float) * 0.2
        self._initialized = False

    def seed(self, pos_ned: np.ndarray, quat: np.ndarray) -> None:
        self.p = np.asarray(pos_ned, dtype=float).copy()
        self.q = quat_norm(np.asarray(quat, dtype=float))
        self._initialized = True

    def predict(self, dt_s: float) -> None:
        if dt_s <= 0:
            return
        self.P_pos += (0.02**2) * dt_s * np.eye(3)
        self.P_att += (0.01**2) * dt_s * np.eye(3)

    def update(
        self,
        p_meas: np.ndarray,
        q_meas: np.ndarray,
        sigma_pos: float,
        sigma_att: float,
    ) -> None:
        p_meas = np.asarray(p_meas, dtype=float)
        q_meas = quat_norm(np.asarray(q_meas, dtype=float))

        Rm = (sigma_pos**2) * np.eye(3)
        S = self.P_pos + Rm
        K = self.P_pos @ np.linalg.inv(S)
        self.p = self.p + K @ (p_meas - self.p)
        self.P_pos = (np.eye(3) - K) @ self.P_pos

        dq = quat_mult(quat_inv(self.q), q_meas)
        if dq[0] < 0:
            dq = -dq
        dtheta = 2.0 * dq[1:4]
        R_a = (sigma_att**2) * np.eye(3)
        S_a = self.P_att + R_a
        K_a = self.P_att @ np.linalg.inv(S_a)
        dtheta_corr = K_a @ dtheta
        self.q = quat_norm(quat_mult(self.q, quat_from_smallangle(dtheta_corr)))
        self.P_att = (np.eye(3) - K_a) @ self.P_att
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def covariance_trace(self) -> float:
        return float(np.trace(self.P_pos) + np.trace(self.P_att))

    def state(self) -> dict:
        return {
            "pos_ned": self.p.copy(),
            "quat": self.q.copy(),
            "P_trace": self.covariance_trace(),
            "initialized": self._initialized,
        }
