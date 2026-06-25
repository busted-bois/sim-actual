"""Module 4 — Corner detection + PnP pose estimation.

From a gate segmentation mask: extract the 4 opening corners, solve PnP
against the known square gate (side spec.GATE_SIZE_M) using the fixed
intrinsics, and produce a
vision-based pose estimate:

  * gate position in camera and body frames (range + bearings),
  * a world-position measurement for the drone (needs the matched gate's
    world pose + the drone's current orientation) — this is what the EKF
    (Module 5) consumes as its vision update.

    uv run -m rl.pnp --selftest
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from rl import spec

# Planar square object points for IPPE_SQUARE: x-right, y-up, z-out of gate.
# Order matches the canonical image order TL, TR, BR, BL.
_OBJ = np.array(
    [
        [-spec.GATE_HALF, +spec.GATE_HALF, 0.0],
        [+spec.GATE_HALF, +spec.GATE_HALF, 0.0],
        [+spec.GATE_HALF, -spec.GATE_HALF, 0.0],
        [-spec.GATE_HALF, -spec.GATE_HALF, 0.0],
    ],
    dtype=np.float64,
)

MIN_MASK_AREA = 400


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 image points as TL, TR, BR, BL (image coords, y down)."""
    pts = pts.astype(np.float64).reshape(4, 2)
    s = pts.sum(1)
    d = pts[:, 1] - pts[:, 0]  # y - x
    return np.array(
        [
            pts[np.argmin(s)],  # TL  (min x+y)
            pts[np.argmin(d)],  # TR  (min y-x)
            pts[np.argmax(s)],  # BR  (max x+y)
            pts[np.argmax(d)],  # BL  (max y-x)
        ]
    )


def detect_corners(
    mask: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    """Largest gate blob -> 4 ordered corners, or None.

    Optional roi=(x0, y0, x1, y1) restricts search to predicted corner region.
    """
    m = mask
    offset = (0, 0)
    if roi is not None:
        x0, y0, x1, y1 = roi
        m = m[y0:y1, x0:x1]
        offset = (x0, y0)
        if m.size == 0:
            return None
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = (m > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_MASK_AREA:
        return None

    # Try polygon approximation down to a quad; fall back to min-area rect.
    peri = cv2.arcLength(c, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08, 0.10):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float64)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64)
    quad = order_quad(quad)
    if offset != (0, 0):
        quad[:, 0] += offset[0]
        quad[:, 1] += offset[1]
    return quad


def filter_outlier_corners(
    corners: np.ndarray,
    predicted: np.ndarray | None,
    max_dist_px: float = 80.0,
) -> np.ndarray:
    """Drop corners whose innovation exceeds max_dist_px from prediction."""
    corners = order_quad(corners.astype(np.float64))
    if predicted is None:
        return corners
    pred = predicted.astype(np.float64).reshape(4, 2)
    dists = np.linalg.norm(corners - pred, axis=1)
    out = corners.copy()
    for i in range(4):
        if dists[i] > max_dist_px:
            out[i] = pred[i]
    return out


def estimate_pose(
    corners: np.ndarray,
    drone_quat: np.ndarray | None = None,
    gate_world_pos: np.ndarray | None = None,
) -> dict | None:
    """PnP from ordered corners -> gate-relative pose (+ optional world fix)."""
    img_pts = corners.astype(np.float64).reshape(4, 1, 2)
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(_OBJ, img_pts, spec.K, spec.DIST, flags=flag)
    if not ok:
        return None

    gate_pos_cam = tvec.reshape(3)
    if gate_pos_cam[2] <= 0:  # gate must be in front
        return None

    # Reprojection error (detection quality / EKF measurement-noise gate).
    proj, _ = cv2.projectPoints(_OBJ, rvec, tvec, spec.K, spec.DIST)
    reproj_err = float(np.linalg.norm(proj.reshape(4, 2) - corners, axis=1).mean())

    gate_pos_body = spec.R_BODY_CAM @ gate_pos_cam
    rng = float(np.linalg.norm(gate_pos_cam))
    yaw_bearing = float(np.arctan2(gate_pos_body[1], gate_pos_body[0]))
    horiz = float(np.hypot(gate_pos_body[0], gate_pos_body[1]))
    pitch_bearing = float(np.arctan2(-gate_pos_body[2], horiz))

    out = {
        "gate_pos_cam": gate_pos_cam,
        "gate_pos_body": gate_pos_body,
        "range_m": rng,
        "yaw_bearing": yaw_bearing,
        "pitch_bearing": pitch_bearing,
        "reproj_err_px": reproj_err,
        "rvec": rvec.reshape(3),
        "tvec": gate_pos_cam,
    }

    # World-position measurement for the EKF: p_drone = gate_world - R_wb @ gate_body.
    if drone_quat is not None and gate_world_pos is not None:
        R_wb = spec.quat_to_R(np.asarray(drone_quat, dtype=np.float64))
        out["drone_pos_world"] = (
            np.asarray(gate_world_pos, float) - R_wb @ gate_pos_body
        )
    return out


def pose_from_mask(
    mask: np.ndarray,
    drone_quat: np.ndarray | None = None,
    gate_world_pos: np.ndarray | None = None,
) -> dict | None:
    corners = detect_corners(mask)
    if corners is None:
        return None
    out = estimate_pose(corners, drone_quat, gate_world_pos)
    if out is not None:
        out["corners"] = corners
    return out


def _selftest():
    from rl.dataset import project_gate_mask

    rng = np.random.default_rng(1)
    errs, dpx = [], []
    for _ in range(200):
        # Random drone pose, gate ahead-ish.
        dpos = rng.uniform(-2, 2, 3)
        yaw = rng.uniform(-0.3, 0.3)
        dq = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        gpos = dpos + np.array(
            [rng.uniform(4, 8), rng.uniform(-1, 1), rng.uniform(-1, 1)]
        )
        gq = np.array([0.0, 0.0, 0.0, 1.0])  # facing -North
        gate = [
            {"id": 0, "pos": gpos.tolist(), "quat": gq.tolist(), "w": 1.5, "h": 1.5}
        ]
        mask = project_gate_mask((spec.IMG_H, spec.IMG_W), dpos, dq, gate)
        if mask.max() == 0:
            continue
        est = pose_from_mask(mask, drone_quat=dq, gate_world_pos=gpos)
        assert est is not None
        true_range = np.linalg.norm(gpos - dpos)
        errs.append(abs(est["range_m"] - true_range))
        dpx.append(np.linalg.norm(est["drone_pos_world"] - dpos))
    errs, dpx = np.array(errs), np.array(dpx)
    print(
        f"[selftest] n={len(errs)} range_err mean={errs.mean():.3f}m "
        f"max={errs.max():.3f}m"
    )
    print(f"[selftest] drone world-pos err mean={dpx.mean():.3f}m max={dpx.max():.3f}m")
    assert errs.mean() < 0.25, "range estimate should be accurate"
    # Fixed metric bound, not scaled by gate size: world-pos error is set by
    # corner pixel-quantization (~range/focal), independent of gate size.
    # Measured mean ~0.28 m at the real 2.72 m gate (fixed seed).
    assert dpx.mean() < 0.30, "world position recovery should be accurate"
    print("[selftest] OK — PnP recovers range + drone world position")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
