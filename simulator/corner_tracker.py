"""Per-corner Kalman tracking in image space.

Smooths noisy gate corners, predicts positions between frames, coasts through
brief dropouts, and rejects outlier detections far from the predicted location.
"""

from __future__ import annotations

import time as _time

import numpy as np

from rl.pnp import order_quad

MAX_OUTLIER_DIST_PX = 80.0
MAX_COAST_FRAMES = 20
BASE_MEAS_SIGMA_PX = 4.0
PROCESS_POS_VAR = 25.0  # px^2/s
PROCESS_VEL_VAR = 400.0  # (px/s)^2/s


class _PixelKalman2D:
    """Constant-velocity Kalman filter for one corner (u, v, du, dv)."""

    def __init__(self) -> None:
        self.x = np.zeros(4, dtype=float)
        self.P = np.eye(4, dtype=float) * 400.0
        self._initialized = False

    def predict(self, dt_s: float) -> np.ndarray:
        if dt_s <= 0:
            return self.x[:2].copy()
        F = np.eye(4)
        F[0, 2] = dt_s
        F[1, 3] = dt_s
        q_pos = PROCESS_POS_VAR * dt_s
        q_vel = PROCESS_VEL_VAR * dt_s
        Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[:2].copy()

    def update(self, z: np.ndarray, sigma_px: float) -> bool:
        z = np.asarray(z, dtype=float).reshape(2)
        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        R = (sigma_px**2) * np.eye(2)
        innov = z - H @ self.x
        S = H @ self.P @ H.T + R
        mahal = float(innov.T @ np.linalg.inv(S) @ innov)
        if mahal > 9.21:  # chi2 2-dof 99%
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(4) - K @ H) @ self.P
        self._initialized = True
        return True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()


class CornerTracker:
    """Tracks four gate corners with independent 2-D Kalman filters."""

    def __init__(self, data: dict) -> None:
        self.data = data
        self._filters = [_PixelKalman2D() for _ in range(4)]
        self._last_mono = _time.monotonic()
        self._coast_frames = 0
        self._updates = 0
        self._outliers = 0
        self._mean_flow_px_s = (0.0, 0.0)
        self._last_gate_idx = -1

    def _maybe_reset_for_gate(self) -> None:
        idx = int(self.data.get("pilot_course_gate_index", 0))
        if idx != self._last_gate_idx:
            self._filters = [_PixelKalman2D() for _ in range(4)]
            self._coast_frames = 0
            self._last_gate_idx = idx

    def process(
        self,
        corners_px: tuple[tuple[float, float], ...] | None,
        frame_id: int,
        reproj_err_px: float | None = None,
    ) -> np.ndarray | None:
        self._maybe_reset_for_gate()
        now = _time.monotonic()
        dt_s = max(1e-3, now - self._last_mono)
        self._last_mono = now

        predicted = np.array(
            [f.predict(dt_s) for f in self._filters], dtype=float
        )

        measured: np.ndarray | None = None
        if corners_px is not None and len(corners_px) >= 4:
            raw = order_quad(np.asarray(corners_px, dtype=float))
            measured = self._associate(raw, predicted)

        sigma = BASE_MEAS_SIGMA_PX
        if reproj_err_px is not None:
            sigma *= 1.0 + float(reproj_err_px) / 5.0

        n_updated = 0
        if measured is not None:
            for i, filt in enumerate(self._filters):
                if filt.update(measured[i], sigma):
                    n_updated += 1
                else:
                    self._outliers += 1
            if n_updated >= 2:
                self._coast_frames = 0
                self._updates += 1
                flows = [
                    f.velocity()
                    for f in self._filters
                    if f.initialized
                ]
                if flows:
                    mean_v = np.mean(flows, axis=0)
                    self._mean_flow_px_s = (float(mean_v[0]), float(mean_v[1]))
            else:
                self._coast_frames += 1
        else:
            self._coast_frames += 1

        initialized = any(f.initialized for f in self._filters)
        smoothed = np.array([f.position() for f in self._filters], dtype=float)
        tracking = initialized and self._coast_frames <= MAX_COAST_FRAMES

        self.data["corner_track"] = {
            "initialized": initialized,
            "tracking": tracking,
            "corners_px": tuple(
                (float(x), float(y)) for x, y in smoothed.reshape(4, 2)
            ),
            "predicted_px": tuple(
                (float(x), float(y)) for x, y in predicted.reshape(4, 2)
            ),
            "coast_frames": self._coast_frames,
            "updates": self._updates,
            "outliers_rejected": self._outliers,
            "mean_flow_px_s": self._mean_flow_px_s,
            "frame_id": frame_id,
        }

        if not tracking:
            return None
        return smoothed

    def _associate(
        self, measured: np.ndarray, predicted: np.ndarray
    ) -> np.ndarray | None:
        """Match measured quad to predicted corners; reject global outliers."""
        if not any(f.initialized for f in self._filters):
            return measured

        dists = np.linalg.norm(measured - predicted, axis=1)
        if float(dists.max()) > MAX_OUTLIER_DIST_PX * 2:
            return None

        out = measured.copy()
        for i in range(4):
            if dists[i] > MAX_OUTLIER_DIST_PX:
                out[i] = predicted[i]
                self._outliers += 1
        return out

    def search_roi(self) -> tuple[int, int, int, int] | None:
        """Bounding box (+pad) around predicted corners for narrowed detection."""
        ct = self.data.get("corner_track")
        if not ct or not ct.get("tracking"):
            return None
        pts = np.asarray(ct["predicted_px"], dtype=float).reshape(4, 2)
        pad = 60
        x0 = int(max(0, pts[:, 0].min() - pad))
        y0 = int(max(0, pts[:, 1].min() - pad))
        x1 = int(min(639, pts[:, 0].max() + pad))
        y1 = int(min(359, pts[:, 1].max() + pad))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1
